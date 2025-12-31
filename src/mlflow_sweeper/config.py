"""Configuration + parameter parsing for mlflow-sweeper.

This module is intentionally limited to:
- CLI argument parsing
- loading + validating sweep configs
- parsing `parameters:` into typed parameter specifications
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from typing import Any

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


def load_configs(config_paths: list[str]) -> list[DictConfig]:
    """Load sweep configs from one or more YAML/JSON paths."""
    return [OmegaConf.load(path) for path in config_paths]


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

