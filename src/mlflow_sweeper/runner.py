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
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import optuna

from mlflow_sweeper.config import ParamSpec, config_params_to_spec_dict


logger = logging.getLogger(__name__)


def get_study_name(config: DictConfig) -> str:
    """Optuna study name for this sweep."""
    return f"{config.experiment}/{config.sweep_name}"


def make_sampler(
    config: DictConfig, param_specs: dict[str, ParamSpec]
) -> optuna.samplers.BaseSampler:
    """Create the Optuna sampler for this sweep."""
    if config.algorithm == "grid":
        search_space = {name: spec.to_list() for name, spec in param_specs.items()}
        return optuna.samplers.GridSampler(search_space)
    raise ValueError(f"Invalid sweep algorithm: {config.algorithm}")


def _optuna_study_lock_path(config: DictConfig) -> str:
    """File lock path used to guard study creation."""
    study_name = get_study_name(config)
    storage = str(config.optuna_storage)
    lock_id = hashlib.md5(f"{study_name}:{storage}".encode("utf-8")).hexdigest()
    return os.path.join(config.output_dir, f"study_{lock_id}.lock")


def _optuna_direction(config: DictConfig) -> str:
    """Optuna optimization direction for the study."""
    if "spec" not in config:
        return "minimize"
    if "direction" not in config.spec:
        return "minimize"
    if config.spec.direction is None:
        return "minimize"
    return str(config.spec.direction)


def init_study(config: DictConfig) -> tuple[optuna.Study, dict[str, ParamSpec]]:
    """Initialize (or load) an Optuna study for this sweep."""
    study_name = get_study_name(config)
    param_specs = config_params_to_spec_dict(config)

    with FileLock(_optuna_study_lock_path(config)):
        study = optuna.create_study(
            study_name = study_name,
            sampler = make_sampler(config, param_specs),
            storage = config.optuna_storage,
            direction = _optuna_direction(config),
            load_if_exists = True,
        )
    return study, param_specs


def get_param_values_for_trial(
    trial: optuna.Trial, param_specs: dict[str, ParamSpec]
) -> dict[str, Any]:
    """Resolve all parameter values for a trial."""
    return {name: spec.suggest(trial) for name, spec in param_specs.items()}


def run_experiment(
    trial: optuna.Trial,
    config: DictConfig,
    param_specs: dict[str, ParamSpec],
    mlflow_client: MlflowClient,
) -> float:
    """Run a single trial as a subprocess and log it under MLflow."""
    param_values = get_param_values_for_trial(trial, param_specs)
    params_str_list = [f"{name}={value}" for name, value in param_values.items()]
    command_list = str(config.command).split() + params_str_list
    full_command_str = " ".join(command_list)

    active = mlflow.active_run()
    if active is not None:
        logger.debug("Active MLflow run: %s.", active.info.run_id)

    trial_run = mlflow.start_run(
        tags = {"sweep_name": str(config.sweep_name)},
        nested = True,
    )
    trial_run_id = trial_run.info.run_id

    mlflow.log_param("full_command", full_command_str)

    environ = os.environ.copy()
    environ["MLFLOW_RUN_ID"] = trial_run_id

    logger.info("Sweep run #%s.", trial.number)
    logger.info("Created trial MLflow run: %s.", trial_run_id)
    logger.info("Running new trial with command: `%s`", full_command_str)

    proc: subprocess.Popen[str] | None = None
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
                print(f"{faded}{line}{reset}", end="")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Trial run {trial_run_id} failed with exit code {proc.returncode}!"
            )
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
        mlflow.end_run()

    assert proc is not None
    logger.info("Trial run %s finished with exit code %s.", trial_run_id, proc.returncode)

    optimization_metric = config.get("spec", {}).get("metric")
    if optimization_metric is None:
        return 0.0

    trial_run = mlflow_client.get_run(trial_run_id)
    summary_metrics = trial_run.data.metrics
    if optimization_metric not in summary_metrics:
        raise ValueError(
            f"Optimization metric {optimization_metric} not found in trial run {trial_run_id}!"
        )
    return float(summary_metrics[optimization_metric])


def start_mlflow_parent_run(
    client: MlflowClient, config: DictConfig, optuna_study_name: str
) -> mlflow.ActiveRun:
    """Create or reuse a single MLflow parent run for this sweep."""
    lock_id = hashlib.md5(
        f"{config.mlflow_storage}:{optuna_study_name}".encode("utf-8")
    ).hexdigest()
    lock_path = os.path.join(config.output_dir, f"mlflow_sweep_parent_{lock_id}.lock")

    with FileLock(lock_path):
        runs = mlflow.search_runs(
            experiment_names = [str(config.experiment)],
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
                run_name = str(config.sweep_name),
                tags = {
                    "sweep_name": str(config.sweep_name),
                    "optuna_study_name": optuna_study_name,
                },
            )
            logger.info("Creating new parent MLflow run: %s.", config.sweep_name)
            return parent_run

        run_id = runs[0].info.run_id
        logger.info("Using existing parent MLflow run: %s.", run_id)
        return mlflow.start_run(run_id = run_id)


def run_sweep(args: argparse.Namespace, config: DictConfig) -> None:
    """Run a full sweep for a single config."""
    os.makedirs(config.output_dir, exist_ok=True)

    study, param_specs = init_study(config)

    logger.info("Running sweep: %s/%s", config.experiment, config.sweep_name)
    mlflow.set_tracking_uri(config.mlflow_storage)
    mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    mlflow.set_experiment(str(config.experiment))
    start_mlflow_parent_run(mlflow_client, config, study.study_name)

    dict_config = OmegaConf.to_container(config, throw_on_missing=True)
    if isinstance(dict_config, dict):
        # NOTE: mlflow.log_params expects a flat dict; this is kept for backward
        # compatibility with the previous script behavior.
        mlflow.log_params(dict_config)

    run_fn = partial(
        run_experiment,
        config = config,
        param_specs = param_specs,
        mlflow_client = mlflow_client,
    )
    study.optimize(run_fn, n_trials=args.n_trials, n_jobs=args.n_jobs)

    mlflow.end_run()
    logger.info("Sweep %s completed.", config.sweep_name)


def delete_sweep(config: DictConfig) -> None:
    """Delete all Optuna + MLflow artifacts associated with a sweep config."""
    study_name = get_study_name(config)
    try:
        optuna.delete_study(study_name=study_name, storage=config.optuna_storage)
        logger.info("Deleted Optuna study: %s.", study_name)
    except KeyError:
        logger.warning("Could not find Optuna study: %s.", study_name)

    mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    experiment = mlflow_client.get_experiment_by_name(str(config.experiment))
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

