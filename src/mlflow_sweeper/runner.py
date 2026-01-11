"""Sweep execution/orchestration for mlflow-sweeper."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import logging
import os
import signal
import subprocess
import time
from typing import Any

from filelock import FileLock
import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
import optuna

from mlflow_sweeper.config import ParamSpec, SweepConfig
from mlflow_sweeper.samplers.grid import GridSampler
from mlflow_sweeper.optimize import optimize_study


logger = logging.getLogger(__name__)


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
        "--abort_on_fail",
        action = "store_true",
        help = "Terminate the sweep if an error occurs in any trial.",
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
        return GridSampler(search_space)
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


def init_study(config: SweepConfig) -> optuna.Study:
    """Initialize (or load) an Optuna study for this sweep."""
    study_name = get_study_name(config)

    with FileLock(_optuna_study_lock_path(config)):
        study = optuna.create_study(
            study_name = study_name,
            sampler = make_sampler(config),
            storage = config.optuna_storage,
            direction = config.spec.direction,
            load_if_exists = True,
        )
    return study


def check_study_is_complete(study: optuna.Study) -> bool:
    """Check if the study is complete."""
    if isinstance(study.sampler, GridSampler):
        return study.sampler.is_exhausted(study)
    raise ValueError(f"Invalid study sampler: {study.sampler}")


def get_param_values_for_trial(
    trial: optuna.Trial, param_specs: dict[str, ParamSpec]
) -> dict[str, Any]:
    """Resolve all parameter values for a trial."""
    return {name: spec.suggest(trial) for name, spec in param_specs.items()}


def run_experiment(
    trial: optuna.Trial,
    config: SweepConfig,
    mlflow_client: MlflowClient,
    args: argparse.Namespace,
) -> float:
    """Run a single trial as a subprocess and log it under MLflow."""
    param_values = get_param_values_for_trial(trial, config.param_specs)
    params_str_list = [f"{name}={value}" for name, value in param_values.items()]
    command_list = config.command.split() + params_str_list
    full_command_str = " ".join(command_list)

    active = mlflow.active_run()
    if active is not None:
        logger.debug("Active MLflow run: %s.", active.info.run_id)

    trial_run = mlflow.start_run(
        tags = {"sweep_name": config.sweep_name},
        nested = True,
    )
    trial_run_id = trial_run.info.run_id
    trial.set_user_attr('mlflow_run_id', trial_run_id)

    mlflow.log_param("full_command", full_command_str)
    mlflow.log_params(param_values)
    
    environ = os.environ.copy()
    environ["MLFLOW_RUN_ID"] = trial_run_id
    environ["MLFLOW_TRACKING_URI"] = config.mlflow_storage

    logger.info("Sweep run #%s.", trial.number)
    logger.info("Created trial MLflow run: %s.", trial_run_id)
    logger.info("Running new trial with command: `%s`", full_command_str)

    proc: subprocess.Popen[str] | None = None
    try:
        state = RunStatus.FAILED
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
                print(f"{faded}{line}{reset}", end="")

        proc.wait()
        if proc.returncode != 0 and args.abort_on_fail:
            raise RuntimeError(
                f"Trial run {trial_run_id} failed with exit code {proc.returncode}!")
        elif proc.returncode == 0:
            state = RunStatus.FINISHED
    
    except KeyboardInterrupt:
        state = RunStatus.KILLED
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
        mlflow.end_run(RunStatus.to_string(state))

    assert proc is not None
    logger.info("Trial run %s finished with exit code %s.", trial_run_id, proc.returncode)

    if config.spec.metric is None:
        return 0.0

    trial_run = mlflow_client.get_run(trial_run_id)
    summary_metrics = trial_run.data.metrics
    if config.spec.metric not in summary_metrics:
        raise ValueError(
            f"Optimization metric {config.spec.metric} not found in trial run {trial_run_id}!"
        )
    return float(summary_metrics[config.spec.metric])


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
) -> optuna.Study:
    """Sync the Optuna study with the MLFlow parent run, deleting any Optuna trials that no longer have a corresponding MLFlow run."""
    
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
    
    # Delete any Optuna trials that no longer have a corresponding MLFlow run.
    # This actually happens by creating a new study and copying over the valid existing trials.
    optuna_trials = study.get_trials()
    valid_optuna_trials = []
    for trial in optuna_trials:
        if trial.user_attrs.get('mlflow_run_id') in valid_run_ids:
            valid_optuna_trials.append(trial)
            
    optuna.delete_study(study_name=study.study_name, storage=study._storage)
    new_study = init_study(config)
    
    new_study.add_trials(valid_optuna_trials)
    
    return new_study


def run_sweep(args: argparse.Namespace, config: SweepConfig) -> None:
    """Run a full sweep for a single config."""
    os.makedirs(config.output_dir, exist_ok=True)

    study = init_study(config)

    mlflow.set_tracking_uri(config.mlflow_storage)
    # Protect MLflow database initialization from concurrent access by multiple processes.
    with FileLock(_mlflow_db_lock_path(config)):
        mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
        mlflow.set_experiment(config.experiment)
        start_mlflow_parent_run(mlflow_client, config, study.study_name)
    
    # This will delete any Optuna trials that no longer exist in MLFlow.
    study = sync_mlflow_and_optuna(study, mlflow_client, config)
    
    if check_study_is_complete(study):
        logger.info("Study is complete. No more trials to run.")
        return

    logger.info("Running sweep: %s/%s", config.experiment, config.sweep_name)
    
    structured_config = OmegaConf.structured(config)
    dict_config = OmegaConf.to_container(structured_config, throw_on_missing=True)
    mlflow.log_params(dict_config)

    run_fn = partial(
        run_experiment,
        config = config,
        mlflow_client = mlflow_client,
        args = args,
    )
    
    # Update the status of the parent MLFlow run based on the status of the Optuna study.
    try:
        optimize_study(study, run_fn, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt as e:
        mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))
        raise e
    except Exception as e:
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))
        raise e
    finally:
        if check_study_is_complete(study):
            mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))
        else:
            mlflow.end_run(RunStatus.to_string(RunStatus.RUNNING))

    logger.info("Sweep %s completed.", config.sweep_name)


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
    for run in runs:
        mlflow_client.delete_run(run.info.run_id)
    logger.info("Deleted %s associated MLflow runs.", len(runs))
