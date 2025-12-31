"""Sweep runner.

This module runs parameter sweeps (currently grid search) by executing a user-provided
command for each trial and logging results to MLflow. It uses Optuna to enumerate and
coordinate trials and supports both single-agent runs and multi-agent/distributed
execution against shared storages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
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
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna


CONFIG_REQUIRED_FIELDS = [
    "experiment",
    "sweep_name",
    "command",
    "algorithm",
    "parameters",
    "optuna_storage",
    "mlflow_storage",
    "output_dir",
]
VALID_ALGORITHMS = ["grid"]


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        logging.DEBUG: "\033[90m",  # grey
        logging.INFO: "\033[38;5;141m",  # purple/blue
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 (format is fine here)
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


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
    return parser.parse_args()


def validate_config(config: DictConfig) -> None:
    """Validate required config fields and basic constraints."""
    for field in CONFIG_REQUIRED_FIELDS:
        assert field in config, f"'{field}' field is required in the config!"

    assert (
        config.algorithm in VALID_ALGORITHMS
    ), f"'algorithm' must be one of {VALID_ALGORITHMS}!"

    if "spec" in config and "direction" in config.spec:
        assert config.spec.direction in [
            "minimize",
            "maximize",
        ], "'direction' must be one of ['minimize', 'maximize']!"


@dataclass(frozen=True)
class ParamSpec(ABC):
    """Abstract parameter specification used to create grid search spaces."""

    name: str

    @abstractmethod
    def to_categorical(self) -> "CategoricalParam":
        """Convert parameter to a categorical parameter if possible."""

    def to_list(self) -> list[Any]:
        """Return a list of all possible values (grid search)."""
        return self.to_categorical().values

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggest a value for the given Optuna trial."""


@dataclass(frozen=True)
class AtomicParam(ParamSpec):
    """Single fixed parameter value."""

    value: Any

    def to_categorical(self) -> "CategoricalParam":
        return CategoricalParam(self.name, [self.value])

    def suggest(self, trial: optuna.Trial) -> Any:  # noqa: ARG002
        return self.value


@dataclass(frozen=True)
class CategoricalParam(ParamSpec):
    """Explicit categorical parameter choices."""

    values: list[Any]

    def to_categorical(self) -> "CategoricalParam":
        return self

    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_categorical(self.name, self.values)


@dataclass(frozen=True)
class IntRangeParam(ParamSpec):
    """Integer range parameter (supports log/step)."""

    low: int
    high: int
    log: bool = False
    step: int = 1

    def to_categorical(self) -> CategoricalParam:
        if self.log:
            raise ValueError(
                "Integer ranges in log space cannot be converted to CategoricalParam!"
            )
        return CategoricalParam(self.name, list(range(self.low, self.high, self.step)))

    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_int(
            self.name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass(frozen=True)
class FloatRangeParam(ParamSpec):
    """Float range parameter (supports log/step)."""

    low: float
    high: float
    log: bool = False
    step: float | None = None

    def to_categorical(self) -> CategoricalParam:
        if self.log:
            raise ValueError(
                "Float ranges in log space cannot be converted to CategoricalParam!"
            )
        if self.step is None:
            raise ValueError(
                f"FloatRangeParam '{self.name}' must specify 'step' to build a grid."
            )

        # Include `high` when it lands on the grid (within float tolerance).
        values = np.arange(self.low, self.high + (self.step / 2.0), self.step).tolist()
        return CategoricalParam(self.name, values)

    def suggest(self, trial: optuna.Trial) -> Any:
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )


def validate_param_dict_entry(name: str, entry: dict[str, Any]) -> None:
    """Validate a typed parameter spec dict.

    Args:
        name: Name of the parameter.
        entry: Specification dictionary for the parameter.

    Supported types:
        - 'atomic': requires 'type' and 'value'
        - 'int_range'/'float_range': requires 'type', 'low', 'high'; optional 'log', 'step'
        - 'categorical': requires 'type' and 'values'
    """
    if "type" not in entry:
        raise ValueError("Parameter must specify 'type' in the config!")

    if entry["type"] not in ["atomic", "int_range", "float_range", "categorical"]:
        raise ValueError(
            f"Invalid parameter type: {entry['type']}; "
            "must be one of ['atomic', 'int_range', 'float_range', 'categorical']!"
        )

    if entry["type"] == "atomic":
        allowed_keys = {"type", "value"}
        missing = [k for k in ["type", "value"] if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(f"Atomic parameter '{name}' missing required fields: {missing}")
        if extraneous:
            raise ValueError(f"Atomic parameter '{name}' has extraneous fields: {extraneous}")

    elif entry["type"] in ["int_range", "float_range"]:
        required_keys = {"type", "low", "high"}
        optional_keys = {"log", "step"}
        allowed_keys = required_keys.union(optional_keys)
        missing = [k for k in required_keys if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(
                f"{entry['type'].capitalize()} parameter '{name}' missing required fields: {missing}"
            )
        if extraneous:
            raise ValueError(
                f"{entry['type'].capitalize()} parameter '{name}' has extraneous fields: {extraneous}"
            )

    elif entry["type"] == "categorical":
        allowed_keys = {"type", "values"}
        missing = [k for k in ["type", "values"] if k not in entry]
        extraneous = [k for k in entry if k not in allowed_keys]
        if missing:
            raise ValueError(
                f"Categorical parameter '{name}' missing required fields: {missing}"
            )
        if extraneous:
            raise ValueError(
                f"Categorical parameter '{name}' has extraneous fields: {extraneous}"
            )


def param_from_config_entry(name: str, value: Any) -> ParamSpec:
    """Create a ParamSpec from a name and a config entry.

    Args:
        name: The parameter name.
        value: The parameter value or specification.

    Returns:
        A parameter specification dataclass.
    """
    if isinstance(value, dict):
        validate_param_dict_entry(name, value)

        if value["type"] == "atomic":
            return AtomicParam(name = name, value = value["value"])
        if value["type"] == "int_range":
            return IntRangeParam(
                name = name,
                low = value["low"],
                high = value["high"],
                log = value.get("log", False),
                step = value.get("step", 1),
            )
        if value["type"] == "float_range":
            return FloatRangeParam(
                name = name,
                low = value["low"],
                high = value["high"],
                log = value.get("log", False),
                step = value.get("step"),
            )
        if value["type"] == "categorical":
            return CategoricalParam(name = name, values = value["values"])

    if isinstance(value, list):
        return CategoricalParam(name = name, values = value)

    return AtomicParam(name = name, value = value)


def config_params_to_spec_dict(config: DictConfig) -> dict[str, ParamSpec]:
    """Convert config.parameters into a dict of parameter specifications."""
    params_dict = OmegaConf.to_container(config.parameters, throw_on_missing=True)
    assert isinstance(params_dict, dict)

    param_specs: dict[str, ParamSpec] = {}
    for name, value in params_dict.items():
        param_specs[name] = param_from_config_entry(name, value)
    return param_specs


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


def configure_logging() -> None:
    """Configure colored, human-friendly logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main() -> None:
    """CLI entrypoint."""
    configure_logging()

    args = parse_args()

    configs: list[DictConfig] = []
    for config_path in args.config:
        config = OmegaConf.load(config_path)
        configs.append(config)

    if args.delete:
        for config in configs:
            logger.info("Deleting sweep: %s/%s...", config.experiment, config.sweep_name)
            delete_sweep(config)
        return

    for config_path, config in zip(args.config, configs, strict=True):
        logger.info("Validating config: %s", config_path)
        validate_config(config)

    for config in configs:
        run_sweep(args, config)


if __name__ == "__main__":
    main()

