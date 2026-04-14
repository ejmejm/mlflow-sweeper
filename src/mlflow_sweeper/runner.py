"""Sweep execution/orchestration for mlflow-sweeper."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
import hashlib
import itertools
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Literal

from filelock import FileLock
import mlflow
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
import optuna

from optuna.distributions import CategoricalDistribution
from optuna.trial import FrozenTrial, TrialState

from mlflow_sweeper.config import (
    AtomicParam,
    CategoricalParam,
    FloatRangeParam,
    IntRangeParam,
    ParamSpec,
    SweepConfig,
)
from mlflow_sweeper.optimize import optimize_study
from mlflow_sweeper.plots import generate_plots
from mlflow_sweeper.samplers.grid import GridSampler
from mlflow_sweeper.samplers.random import RandomSampler


logger = logging.getLogger(__name__)


# Marker for "no metric value returned by user fn" — distinguishes from a
# real None (which means "fall back to MLflow lookup").
_NO_RETURN_VALUE: Any = object()


class TrialRunError(Exception):
    """Exception raised when a trial run fails."""
    pass


class TrialRunAbortError(Exception):
    """Exception raised when a trial run fails."""
    pass


def extract_error_trace(output_lines: list[str]) -> str:
    """Extract the Python error traceback from subprocess output lines.
    
    Searches for the standard Python traceback header and returns all lines
    from that point to the end of the output.
    
    Args:
        output_lines: List of output lines from the subprocess.
        
    Returns:
        The extracted error traceback, or a message indicating none was found.
    """
    for i, line in enumerate(output_lines):
        if line.startswith("Traceback (most recent call last):"):
            return "".join(output_lines[i:]).rstrip()
    return "(no error trace detected)"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the sweep runner."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type = str,
        nargs = "+",
        help = (
            "Path(s) to YAML or JSON file(s) containing sweep configuration. "
            "A sweep will be run for each config file provided."
        ),
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type = int,
        default = None,
        help = (
            "Number of trials to perform for the sweep. If None, will perform "
            "until the study is marked as complete."
        ),
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type = int,
        default = 1,
        help = "Number of parallel jobs in this call of the script.",
    )
    parser.add_argument(
        "--delete",
        action = "store_true",
        help = "Delete all data associated with the MLflow sweep and Optuna study.",
    )
    parser.add_argument(
        "--abort-on-fail",
        action = "store_true",
        help = "Terminate the sweep if an error occurs in any trial.",
        default = False,
    )
    parser.add_argument(
        "--log-params",
        action = "store_true",
        default = False,
        help = (
            "Log individual sweep parameter values to the MLflow trial run. "
            "Disabled by default so that the subprocess can log its own resolved "
            "parameter values without hitting MLflow's param-immutability restriction."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action = "store_true",
        default = False,
        help = "Disable automatic plot generation after sweep completion.",
    )
    parser.add_argument(
        "--allow-param-change",
        action = "store_true",
        default = False,
        help = (
            "Permit the sweep config's parameters to differ from the existing "
            "sweep. When set, valid trials are migrated and the sweep continues. "
            "If the config has not changed, a warning is emitted and the sweep "
            "runs normally."
        ),
    )
    return parser.parse_args()


def get_study_name(config: SweepConfig) -> str:
    """Optuna study name for this sweep."""
    return f"{config.experiment}/{config.sweep_name}"


def get_lock_dir(config: SweepConfig) -> str:
    return os.path.join(config.output_dir, 'locks')


def make_sampler(config: SweepConfig) -> optuna.samplers.BaseSampler:
    """Create the Optuna sampler for this sweep."""
    if config.algorithm == "grid":
        search_space = {name: spec.to_list() for name, spec in config.param_specs.items()}
        return GridSampler(search_space, max_retry_count=config.spec.max_retry)
    elif config.algorithm == "random":
        grid_search_space = None
        if config.spec.grid_params:
            grid_search_space = {
                name: config.param_specs[name].to_list()
                for name in config.spec.grid_params
            }
        return RandomSampler(
            n_runs=config.spec.n_runs,
            max_retry_count=config.spec.max_retry,
            grid_search_space=grid_search_space,
        )
    raise ValueError(f"Invalid sweep algorithm: {config.algorithm}")


def _optuna_study_lock_path(config: SweepConfig) -> str:
    """File lock path used to guard study creation."""
    study_name = get_study_name(config)
    lock_id = hashlib.md5(f"{study_name}:{config.optuna_storage}".encode("utf-8")).hexdigest()
    return os.path.join(get_lock_dir(config), f"study_{lock_id}.lock")


def _mlflow_db_lock_path(config: SweepConfig) -> str:
    """File lock path used to guard MLflow database initialization."""
    lock_id = hashlib.md5(config.mlflow_storage.encode("utf-8")).hexdigest()
    return os.path.join(get_lock_dir(config), f"mlflow_db_{lock_id}.lock")


def init_study(config: SweepConfig, max_retries: int = 4, retry_delay: float = 4.0) -> optuna.Study:
    """Initialize (or load) an Optuna study for this sweep."""
    study_name = get_study_name(config)

    for attempt in range(max_retries):
        try:
            with FileLock(_optuna_study_lock_path(config)):
                study = optuna.create_study(
                    study_name = study_name,
                    sampler = make_sampler(config),
                    storage = config.optuna_storage,
                    direction = config.spec.direction,
                    load_if_exists = True,
                )
            return study
        except Exception as e:
            if "already exists" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database conflict on attempt {attempt + 1}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise


def check_study_is_complete(study: optuna.Study) -> bool:
    """Check if the study is complete."""
    if hasattr(study.sampler, 'is_exhausted'):
        return study.sampler.is_exhausted(study)
    raise ValueError(f"Invalid study sampler: {study.sampler}")


def get_param_values_for_trial(
    trial: optuna.Trial, param_specs: dict[str, ParamSpec]
) -> dict[str, Any]:
    """Resolve all parameter values for a trial."""
    return {name: spec.suggest(trial) for name, spec in param_specs.items()}


def _resolve_metric(
    return_value: Any,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    trial_run_id: str,
) -> float:
    """Resolve the optimization metric for a finished trial.

    Args:
        return_value: Value the body returned. May be a number (used directly),
            a ``dict[str, float]`` of metrics (those are logged then the
            ``config.spec.metric`` entry is returned), or ``None``/the
            ``_NO_RETURN_VALUE`` sentinel (fall back to fetching the metric
            from the trial's MLflow run, matching the subprocess path).
    """
    if isinstance(return_value, (int, float)) and not isinstance(return_value, bool):
        return float(return_value)

    if isinstance(return_value, dict):
        # Log every numeric entry, then pick the optimization metric.
        numeric_metrics = {k: float(v) for k, v in return_value.items()}
        if numeric_metrics:
            with mlflow.start_run(run_id=trial_run_id):
                mlflow.log_metrics(numeric_metrics)
        if config.spec.metric is None:
            return 0.0
        if config.spec.metric not in numeric_metrics:
            raise ValueError(
                f"Optimization metric {config.spec.metric} not found in returned dict "
                f"for trial run {trial_run_id}!"
            )
        return numeric_metrics[config.spec.metric]

    if config.spec.metric is None:
        return 0.0

    trial_run = mlflow_client.get_run(trial_run_id)
    summary_metrics = trial_run.data.metrics
    if config.spec.metric not in summary_metrics:
        raise ValueError(
            f"Optimization metric {config.spec.metric} not found in trial run {trial_run_id}!"
        )
    return float(summary_metrics[config.spec.metric])


def _run_trial_with_mlflow(
    trial: optuna.Trial,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    args: argparse.Namespace,
    parent_run_id: str,
    *,
    body: Callable[[dict[str, Any], str], Any],
    label_fn: Callable[[dict[str, Any]], str],
) -> float:
    """Shared MLflow lifecycle around a single trial.

    Creates the trial's MLflow child run, lets ``body`` execute the work
    (subprocess or in-process callable), then ends the run with the right
    status and resolves the optimization metric.
    """
    param_values = get_param_values_for_trial(trial, config.param_specs)
    label = label_fn(param_values)

    active = mlflow.active_run()
    if active is not None:
        logger.debug("Active MLflow run: %s.", active.info.run_id)

    # Use explicit parent tag instead of nested=True, because with n_jobs > 1
    # the parent run's thread-local active run context is not visible in worker
    # threads, causing nested=True to create top-level runs instead.
    trial_run = mlflow.start_run(
        experiment_id = mlflow.get_experiment_by_name(config.experiment).experiment_id,
        tags = {
            "sweep_name": config.sweep_name,
            "mlflow.parentRunId": parent_run_id,
        },
    )
    trial_run_id = trial_run.info.run_id
    trial.set_user_attr('mlflow_run_id', trial_run_id)

    mlflow.log_param("full_command", label)
    if args.log_params:
        mlflow.log_params(param_values)

    logger.info("Sweep run #%s.", trial.number)
    logger.info("Created trial MLflow run: %s.", trial_run_id)
    logger.info("Running new trial: `%s`", label)

    state = RunStatus.FINISHED
    return_value: Any = _NO_RETURN_VALUE
    try:
        return_value = body(param_values, trial_run_id)
    except (TrialRunError, TrialRunAbortError):
        state = RunStatus.FAILED
        raise
    except KeyboardInterrupt:
        state = RunStatus.KILLED
        raise
    finally:
        mlflow.end_run(RunStatus.to_string(state))

    return _resolve_metric(return_value, config, mlflow_client, trial_run_id)


def _subprocess_body(
    config: SweepConfig,
    args: argparse.Namespace,
) -> Callable[[dict[str, Any], str], Any]:
    """Body that executes a trial by spawning ``config.command`` as a subprocess."""
    def _body(param_values: dict[str, Any], trial_run_id: str) -> Any:
        params_str_list = [f"{name}={value}" for name, value in param_values.items()]
        command_list = config.command.split() + params_str_list

        environ = os.environ.copy()
        environ["MLFLOW_RUN_ID"] = trial_run_id
        environ["MLFLOW_TRACKING_URI"] = config.mlflow_storage

        proc: subprocess.Popen[str] | None = None
        output_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                command_list,
                env = environ,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                bufsize = 1,
            )

            faded = "\033[2;37m"  # dim grey
            reset = "\033[0m"
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_lines.append(line)
                    print(f"{faded}{line}{reset}", end="", flush=True)

            proc.wait()
            logger.info(f"Trial run {trial_run_id} finished with exit code {proc.returncode}.")

            if proc.returncode != 0:
                error_trace = extract_error_trace(output_lines)
                error_msg = (
                    f"Trial run {trial_run_id} failed with exit code {proc.returncode}!\n\n"
                    f"{error_trace}"
                )
                if args.abort_on_fail:
                    raise TrialRunAbortError(error_msg)
                raise TrialRunError(error_msg)

        except KeyboardInterrupt:
            if proc is not None:
                proc.send_signal(signal.SIGINT)
                # Wait up to 60s, then hard-kill.
                for _ in range(60):
                    proc.poll()
                    if proc.returncode is not None:
                        break
                    time.sleep(1)
                if proc.returncode is None:
                    proc.kill()
            raise
        finally:
            if output_lines:
                with tempfile.TemporaryDirectory() as tmpdir:
                    log_path = os.path.join(tmpdir, "stdout.log")
                    with open(log_path, "w") as f:
                        f.writelines(output_lines)
                    mlflow.log_artifact(log_path)
        return None
    return _body


def _callable_body(
    fn: Callable[..., Any],
    args: argparse.Namespace,
) -> Callable[[dict[str, Any], str], Any]:
    """Body that executes a trial by calling a user-provided Python function.

    The function is invoked with ``**param_values``.  An MLflow run is
    already active when ``fn`` runs; the user must not call
    ``start_run``/``end_run``.  We assert the active-run-stack depth is
    unchanged afterwards to surface contract violations early.
    """
    from mlflow.tracking import fluent as _fluent

    def _body(param_values: dict[str, Any], trial_run_id: str) -> Any:
        stack_before = list(_fluent._active_run_stack.get())
        try:
            return fn(**param_values)
        except (TrialRunError, TrialRunAbortError, KeyboardInterrupt):
            raise
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = (
                f"Trial run {trial_run_id} failed with {type(e).__name__}: {e}\n\n{tb}"
            )
            if args.abort_on_fail:
                raise TrialRunAbortError(error_msg) from e
            raise TrialRunError(error_msg) from e
        finally:
            stack_after = list(_fluent._active_run_stack.get())
            if [r.info.run_id for r in stack_before] != [r.info.run_id for r in stack_after]:
                # Reset the stack so subsequent trials aren't poisoned, but
                # raise so the user sees the bug.
                _fluent._active_run_stack.set(stack_before)
                raise RuntimeError(
                    "User function modified the MLflow active-run stack. "
                    "Do not call mlflow.start_run() / mlflow.end_run() inside "
                    "the function passed to run_sweep(); the runner manages "
                    "the trial's MLflow run for you."
                )
    return _body


def run_experiment(
    trial: optuna.Trial,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    args: argparse.Namespace,
    parent_run_id: str = "",
) -> float:
    """Run a single trial as a subprocess and log it under MLflow."""
    def _label(param_values: dict[str, Any]) -> str:
        params_str_list = [f"{name}={value}" for name, value in param_values.items()]
        return " ".join(config.command.split() + params_str_list)

    return _run_trial_with_mlflow(
        trial,
        config,
        mlflow_client,
        args,
        parent_run_id,
        body=_subprocess_body(config, args),
        label_fn=_label,
    )


def start_mlflow_parent_run(
    client: MlflowClient, config: SweepConfig, optuna_study_name: str
) -> mlflow.ActiveRun:
    """Create or reuse a single MLflow parent run for this sweep."""
    lock_id = hashlib.md5(
        f"{config.mlflow_storage}:{optuna_study_name}".encode("utf-8")
    ).hexdigest()
    lock_path = os.path.join(get_lock_dir(config), f"mlflow_sweep_parent_{lock_id}.lock")

    with FileLock(lock_path):
        runs = mlflow.search_runs(
            experiment_names = [config.experiment],
            filter_string = (
                f'tags.sweep_name = "{config.sweep_name}" '
                f'AND tags.optuna_study_name = "{optuna_study_name}"'
            ),
            output_format = "list",
        )
        assert (
            len(runs) <= 1
        ), f"Multiple parent runs found for sweep {config.sweep_name} and study {optuna_study_name}!"

        if len(runs) == 0:
            parent_run = mlflow.start_run(
                run_name = config.sweep_name,
                tags = {
                    "sweep_name": config.sweep_name,
                    "optuna_study_name": optuna_study_name,
                },
            )
            logger.info("Creating new parent MLflow run: %s.", config.sweep_name)
            return parent_run

        run_id = runs[0].info.run_id
        logger.info("Using existing parent MLflow run: %s.", run_id)
        return mlflow.start_run(run_id = run_id)


def sync_mlflow_and_optuna(
    study: optuna.Study,
    mlflow_client: MlflowClient,
    config: SweepConfig,
) -> None:
    """Sync the Optuna study with the MLFlow parent run, voiding any Optuna trials that no longer have a corresponding MLFlow run."""
    
    # Get all of the valid MLFlow runs for the given experiment.
    experiment = mlflow_client.get_experiment_by_name(config.experiment)
    if experiment is None:
        logger.warning("Could not find MLflow experiment: %s.", config.experiment)
        return study
    mlflow_runs = mlflow_client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string = f'tags.sweep_name = "{config.sweep_name}"',
    )
    valid_run_ids = [run.info.run_id for run in mlflow_runs]
    
    # Void any Optuna trials that no longer have a corresponding MLFlow run.
    # Optuna doesn't provide a method to directly delete them, and deleting and recreating the whole study
    # could cause problems if other processes are accessing the study.
    with FileLock(_optuna_study_lock_path(config)):
        voided_trial_ids = study._storage.get_study_user_attrs(study._study_id).get('voided_trial_ids', [])
        optuna_trials = study.get_trials()
        for trial in optuna_trials:
            if trial.user_attrs.get('mlflow_run_id') not in valid_run_ids and trial._trial_id not in voided_trial_ids:
                voided_trial_ids.append(trial._trial_id)
        study._storage.set_study_user_attr(study._study_id, 'voided_trial_ids', voided_trial_ids)


def _trial_matches_config(trial_params: dict[str, Any], param_specs: dict[str, ParamSpec]) -> bool:
    """Check if a trial's params are valid under the given param specs.

    AtomicParam values are excluded from the comparison because Optuna trials
    only contain params registered via ``trial.suggest_*`` calls, and
    ``AtomicParam.suggest()`` returns its value directly without calling
    any suggest method.
    """
    swept_specs = {name: spec for name, spec in param_specs.items()
                   if not isinstance(spec, AtomicParam)}
    if set(trial_params.keys()) != set(swept_specs.keys()):
        return False
    for name, value in trial_params.items():
        spec = swept_specs[name]
        if isinstance(spec, CategoricalParam):
            if value not in spec.values:
                return False
        elif isinstance(spec, IntRangeParam):
            if not (spec.low <= value <= spec.high):
                return False
        elif isinstance(spec, FloatRangeParam):
            if not (spec.low <= value <= spec.high):
                return False
    return True


def _reparent_mlflow_runs(
    mlflow_client: MlflowClient,
    config: SweepConfig,
    old_parent_run_id: str,
    new_parent_run_id: str,
) -> None:
    """Re-parent all trial runs (and their children) from old parent to new parent."""
    experiment = mlflow_client.get_experiment_by_name(config.experiment)
    if experiment is None:
        return

    # Find all runs directly parented to the old parent.
    trial_runs = mlflow_client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string = f'tags.mlflow.parentRunId = "{old_parent_run_id}"',
    )
    for run in trial_runs:
        mlflow_client.set_tag(run.info.run_id, "mlflow.parentRunId", new_parent_run_id)

    # Also check for grandchildren parented to trial runs — they point to the
    # trial run (not the parent), so they should already be fine.  But verify
    # none were accidentally parented to the old parent.
    logger.info("Re-parented %d trial runs to new parent.", len(trial_runs))


def update_sweep_params(
    config: SweepConfig,
    old_study: optuna.Study,
    mlflow_client: MlflowClient,
) -> optuna.Study:
    """Delete old Optuna study and create a new one, migrating valid trials.

    Only trials whose params match the new config are migrated into the new
    Optuna study.  MLflow run migration (re-parenting) is handled separately
    by the caller.
    """
    study_name = get_study_name(config)

    # 1. Collect old trials (COMPLETE and FAIL, non-voided).
    old_trials = old_study.get_trials(states=[TrialState.COMPLETE, TrialState.FAIL])
    voided_ids = old_study._storage.get_study_user_attrs(
        old_study._study_id
    ).get('voided_trial_ids', [])
    old_trials = [t for t in old_trials if t._trial_id not in voided_ids]

    # 2. Filter to trials valid under new config.
    valid_trials = [t for t in old_trials if _trial_matches_config(t.params, config.param_specs)]
    logger.info(
        "Migrating %d of %d trials to updated config (%d dropped).",
        len(valid_trials), len(old_trials), len(old_trials) - len(valid_trials),
    )

    # 3. Delete old Optuna study.
    with FileLock(_optuna_study_lock_path(config)):
        optuna.delete_study(study_name=study_name, storage=config.optuna_storage)

    # 4. Create new study with new sampler.
    new_study = init_study(config)
    new_sampler = new_study.sampler

    # 5. Add valid trials to new study with correct system attrs.
    for old_trial in valid_trials:
        if config.algorithm == "grid":
            # Compute grid_id by finding the matching point in the shuffled grid.
            # AtomicParam values are NOT in Optuna trial params (they bypass
            # trial.suggest_*), so fill them in from the spec.
            param_names = list(new_sampler._search_space.keys())
            combo = tuple(
                config.param_specs[name].value if isinstance(config.param_specs[name], AtomicParam)
                else old_trial.params[name]
                for name in param_names
            )
            try:
                grid_id = new_sampler._all_grids.index(combo)
            except ValueError:
                logger.warning("Trial %d params don't match any grid point; skipping.", old_trial.number)
                continue
            # Distributions must match trial params keys (which exclude atomics).
            distributions = {
                name: CategoricalDistribution(choices=new_sampler._search_space[name])
                for name in param_names
                if not isinstance(config.param_specs[name], AtomicParam)
            }
            system_attrs: dict[str, Any] = {
                'search_space': new_sampler._search_space,
                'grid_id': grid_id,
            }
        elif config.algorithm == "random":
            distributions = old_trial.distributions
            system_attrs = {'random_group': True}
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

        user_attrs = {k: v for k, v in old_trial.user_attrs.items()}
        # Pass only `values` (not `value`) — FrozenTrial rejects both at once.
        # For FAIL trials, values may be None.
        frozen = optuna.trial.create_trial(
            state=old_trial.state,
            values=old_trial.values,
            params=old_trial.params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
        )
        new_study.add_trial(frozen)

    return new_study


def _setup_sweep_storage(
    config: SweepConfig,
) -> tuple[optuna.Study, MlflowClient, str]:
    """Initialize Optuna + MLflow storage and start (or resume) the parent run.

    Returns (study, mlflow_client, parent_run_id).  The parent run is left
    *active* on the calling thread's MLflow run stack — callers are
    responsible for stripping it before launching workers.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    study = init_study(config)

    mlflow.set_tracking_uri(config.mlflow_storage)
    with FileLock(_mlflow_db_lock_path(config)):
        mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
        mlflow.set_experiment(config.experiment)
        start_mlflow_parent_run(mlflow_client, config, study.study_name)

    sync_mlflow_and_optuna(study, mlflow_client, config)
    parent_run_id = mlflow.active_run().info.run_id
    return study, mlflow_client, parent_run_id


def _tracked_config_dict(config: SweepConfig) -> dict[str, Any]:
    """Subset of the SweepConfig that should trigger a re-sweep when changed."""
    structured_config = OmegaConf.structured(config)
    dict_config = OmegaConf.to_container(structured_config, throw_on_missing=True)
    return {k: v for k, v in dict_config.items() if k not in ('plots', 'param_specs')}


def _handle_config_change(
    config: SweepConfig,
    study: optuna.Study,
    mlflow_client: MlflowClient,
    parent_run_id: str,
    allow_param_change: bool,
) -> tuple[optuna.Study, str, bool]:
    """Detect a config change vs. the parent run; migrate if allowed.

    Returns (study, parent_run_id, did_migrate).  The active MLflow run on the
    current thread points to the (possibly new) parent run on return.
    """
    tracked_config = _tracked_config_dict(config)
    try:
        mlflow.log_params(tracked_config)
        return study, parent_run_id, False
    except MlflowException as e:
        if "Changing param values is not allowed" not in str(e):
            raise
        if not allow_param_change:
            logger.error(
                "Sweep config has changed since the last run. To run with the new "
                "config, either:\n"
                "  1. Delete the existing sweep and start fresh, or\n"
                "  2. Rename the sweep (change 'sweep_name' in your config), or\n"
                "  3. Pass allow_param_change=True (CLI: --allow-param-change) to "
                "update the sweep in-place."
            )
            raise SystemExit(1) from e

        logger.info("Config changed. Migrating sweep params in-place...")
        old_parent_run_id = parent_run_id

        # End current parent run context before migration.
        mlflow.end_run()

        # Migrate Optuna study (delete old, create new, add matching trials).
        new_study = update_sweep_params(config, study, mlflow_client)

        # Delete old MLflow parent and create a new one.
        mlflow_client.delete_run(old_parent_run_id)
        start_mlflow_parent_run(mlflow_client, config, new_study.study_name)
        new_parent_run_id = mlflow.active_run().info.run_id

        # Re-parent all existing trial runs to the new parent.
        _reparent_mlflow_runs(mlflow_client, config, old_parent_run_id, new_parent_run_id)

        # Log new config on the fresh parent.
        mlflow.log_params(tracked_config)

        logger.info("Param migration complete.")
        return new_study, new_parent_run_id, True


def _strip_parent_from_active_stack(parent_run_id: str) -> None:
    """Remove the parent run from the thread-local MLflow active-run stack.

    The parent run stays alive on the server; its terminal status is managed
    explicitly via ``mlflow_client.set_terminated()``.  Stripping it here
    lets child trials call ``mlflow.start_run()`` without ``nested=True``
    and avoids accidental nesting when worker threads inherit the stack.
    """
    _stack = mlflow.tracking.fluent._active_run_stack
    _stack.set([r for r in _stack.get() if r.info.run_id != parent_run_id])


def _finalize_parent_run(
    study: optuna.Study,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    parent_run_id: str,
    no_plots: bool,
) -> None:
    """Generate plots (if enabled), then mark the parent run FINISHED/RUNNING."""
    if check_study_is_complete(study):
        if not no_plots:
            # generate_plots needs an active run for mlflow.active_run() and
            # artifact logging.  The parent may already be active (if no
            # workers were spawned and the stack was never stripped) or
            # inactive (post-optimization).  Normalize by ending any active
            # run, then resuming the parent fresh.
            active = mlflow.active_run()
            if active is not None:
                mlflow.end_run()
            mlflow.start_run(run_id=parent_run_id)
            try:
                generate_plots(study, config)
            finally:
                mlflow.end_run()
        mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.FINISHED))
    else:
        # Make sure no run is left dangling on the active stack.
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.RUNNING))


def _drive_optimization(
    study: optuna.Study,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    parent_run_id: str,
    run_fn: Callable[[optuna.Trial], float],
    *,
    n_trials: int | None,
    n_jobs: int,
    no_plots: bool,
) -> None:
    """Drive ``optimize_study`` and manage the parent run's terminal status."""
    _strip_parent_from_active_stack(parent_run_id)

    try:
        optimize_study(study, run_fn, n_trials=n_trials, n_jobs=n_jobs, catch=(TrialRunError,))
    except KeyboardInterrupt:
        # Make sure no run is left dangling on the active stack.
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.KILLED))
        raise
    except Exception:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.FAILED))
        raise

    # Only finalize (set FINISHED/RUNNING + plots) on the success path —
    # the error paths above already set the terminal status.
    _finalize_parent_run(study, config, mlflow_client, parent_run_id, no_plots)
    logger.info("Sweep %s completed.", config.sweep_name)


def _build_cli_args_namespace(
    *,
    n_trials: int | None,
    n_jobs: int,
    abort_on_fail: bool,
    log_params: bool,
    no_plots: bool,
    allow_param_change: bool,
) -> argparse.Namespace:
    """Build a Namespace mirroring the CLI args, for sharing with subprocess body."""
    return argparse.Namespace(
        n_trials = n_trials,
        n_jobs = n_jobs,
        abort_on_fail = abort_on_fail,
        log_params = log_params,
        no_plots = no_plots,
        allow_param_change = allow_param_change,
    )


def _run_sweep_cli(args: argparse.Namespace, config: SweepConfig) -> None:
    """CLI driver: runs a sweep using the configured shell command (subprocess)."""
    if not config.command:
        raise ValueError(
            "SweepConfig.command is required for the CLI/subprocess sweep path."
        )

    study, mlflow_client, parent_run_id = _setup_sweep_storage(config)
    study, parent_run_id, did_migrate = _handle_config_change(
        config, study, mlflow_client, parent_run_id,
        allow_param_change=args.allow_param_change,
    )
    if args.allow_param_change and not did_migrate:
        logger.warning(
            "--allow-param-change had no effect: the sweep config has not changed."
        )

    if check_study_is_complete(study):
        logger.info("Study is complete. No more trials to run.")
        _finalize_parent_run(study, config, mlflow_client, parent_run_id, args.no_plots)
        return

    run_fn = partial(
        run_experiment,
        config = config,
        mlflow_client = mlflow_client,
        args = args,
        parent_run_id = parent_run_id,
    )
    _drive_optimization(
        study, config, mlflow_client, parent_run_id, run_fn,
        n_trials=args.n_trials, n_jobs=args.n_jobs, no_plots=args.no_plots,
    )


def run_sweep(
    config: SweepConfig,
    fn: Callable[..., Any],
    *,
    n_trials: int | None = None,
    n_jobs: int = 1,
    executor: Literal["thread", "process"] = "process",
    abort_on_fail: bool = False,
    log_params: bool = False,
    no_plots: bool = False,
    allow_param_change: bool = False,
) -> None:
    """Run a full sweep using a Python callable for each trial.

    Args:
        config: Parsed sweep configuration.
        fn: Per-trial Python function. Called with ``**param_values``; must
            not call ``mlflow.start_run``/``end_run`` (the runner manages
            the trial's MLflow run).  May return a ``float`` (used as the
            optimization metric), a ``dict[str, float]`` of metrics
            (logged then ``config.spec.metric`` is used), or ``None`` (the
            metric is read off the MLflow run after the function returns).
        n_trials: Total trials cap; ``None`` = run until the study is exhausted.
        n_jobs: Worker count.  When ``1`` the trial runs inline regardless of
            ``executor``.  When ``>1`` use either a thread or process pool.
        executor: ``"thread"`` or ``"process"`` (default).  Process pool gives
            real parallelism but requires ``fn`` to be importable
            (no closures/lambdas).
        abort_on_fail: If True, the first failing trial terminates the sweep.
        log_params: Mirror the CLI flag — log per-trial params on the trial run.
        no_plots: Skip plot generation after the sweep completes.
        allow_param_change: If True, permit the config's parameters to differ
            from the existing sweep; valid trials are migrated.  Warns if
            the config has not actually changed.
    """
    if executor not in ("thread", "process"):
        raise ValueError(f"executor must be 'thread' or 'process', got {executor!r}")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")

    # Validate process-pool serialization BEFORE we touch MLflow / Optuna,
    # so a bad function (lambda, closure) doesn't leave an orphan parent run.
    if executor == "process" and n_jobs > 1:
        from mlflow_sweeper.process_executor import validate_fn_picklable
        validate_fn_picklable(fn)

    args = _build_cli_args_namespace(
        n_trials=n_trials, n_jobs=n_jobs, abort_on_fail=abort_on_fail,
        log_params=log_params, no_plots=no_plots,
        allow_param_change=allow_param_change,
    )

    study, mlflow_client, parent_run_id = _setup_sweep_storage(config)
    study, parent_run_id, did_migrate = _handle_config_change(
        config, study, mlflow_client, parent_run_id,
        allow_param_change=allow_param_change,
    )
    if allow_param_change and not did_migrate:
        logger.warning(
            "allow_param_change=True had no effect: the sweep config has not changed."
        )

    if check_study_is_complete(study):
        logger.info("Study is complete. No more trials to run.")
        _finalize_parent_run(study, config, mlflow_client, parent_run_id, no_plots)
        return

    use_process_pool = executor == "process" and n_jobs > 1
    if use_process_pool:
        from mlflow_sweeper.process_executor import run_with_process_pool

        _strip_parent_from_active_stack(parent_run_id)
        try:
            run_with_process_pool(
                config = config,
                fn = fn,
                n_jobs = n_jobs,
                n_trials = n_trials,
                parent_run_id = parent_run_id,
                args_namespace = args,
            )
        except KeyboardInterrupt:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.KILLED))
            raise
        except Exception:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow_client.set_terminated(parent_run_id, RunStatus.to_string(RunStatus.FAILED))
            raise

        # Success path: reload the study so we see trials added by workers,
        # then finalize (plots + FINISHED).
        study = optuna.load_study(
            study_name=get_study_name(config),
            storage=config.optuna_storage,
            sampler=make_sampler(config),
        )
        _finalize_parent_run(study, config, mlflow_client, parent_run_id, no_plots)
        logger.info("Sweep %s completed.", config.sweep_name)
        return

    # Serial / thread-pool path.
    run_fn = partial(
        _run_trial_with_mlflow,
        config = config,
        mlflow_client = mlflow_client,
        args = args,
        parent_run_id = parent_run_id,
        body = _callable_body(fn, args),
        label_fn = _callable_label_fn(fn),
    )
    _drive_optimization(
        study, config, mlflow_client, parent_run_id, run_fn,
        n_trials=n_trials, n_jobs=n_jobs, no_plots=no_plots,
    )


def _callable_label_fn(fn: Callable[..., Any]) -> Callable[[dict[str, Any]], str]:
    """Build a label string for the 'full_command' MLflow param of a callable trial."""
    qualname = getattr(fn, "__qualname__", repr(fn))
    module = getattr(fn, "__module__", "<unknown>")
    def _label(param_values: dict[str, Any]) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in param_values.items())
        return f"{module}.{qualname}({params_str})"
    return _label


def delete_sweep(config: SweepConfig) -> None:
    """Delete all Optuna + MLflow artifacts associated with a sweep config."""
    study_name = get_study_name(config)
    try:
        optuna.delete_study(study_name=study_name, storage=config.optuna_storage)
        logger.info("Deleted Optuna study: %s.", study_name)
    except KeyError:
        logger.warning("Could not find Optuna study: %s.", study_name)

    # Protect MLflow database initialization from concurrent access by multiple processes.
    with FileLock(_mlflow_db_lock_path(config)):
        mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    experiment = mlflow_client.get_experiment_by_name(config.experiment)
    if experiment is None:
        logger.warning("Could not find MLflow experiment: %s.", config.experiment)
        return

    runs = mlflow_client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string = f'tags.sweep_name = "{config.sweep_name}"',
    )
    # Also find child runs of each sweep run so they get deleted too.
    child_runs = []
    for run in runs:
        child_runs.extend(mlflow_client.search_runs(
            experiment_ids = [experiment.experiment_id],
            filter_string = f'tags.mlflow.parentRunId = "{run.info.run_id}"',
        ))
    seen = {run.info.run_id for run in runs}
    child_runs = [r for r in child_runs if r.info.run_id not in seen]
    all_runs = list(runs) + child_runs
    for run in all_runs:
        mlflow_client.delete_run(run.info.run_id)
    logger.info("Deleted %s associated MLflow runs (%s child runs).", len(runs), len(child_runs))
