"""Configuration + parameter parsing for mlflow-sweeper.

This module is intentionally limited to:
- loading + validating sweep configs
- parsing `parameters:` into typed parameter specifications
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.study import StudyDirection


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


def load_configs(config_paths: list[str]) -> list[DictConfig]:
    """Load sweep configs from one or more YAML/JSON paths."""
    return [OmegaConf.load(path) for path in config_paths]


def validate_config(config: DictConfig) -> None:
    """Validate required config fields and basic constraints.

    Args:
        config: A DictConfig loaded from a YAML/JSON file.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    # Check required fields
    missing = [f for f in CONFIG_REQUIRED_FIELDS if f not in config]
    if missing:
        raise ValueError(f"Required fields missing from config: {missing}")

    # Validate algorithm
    if config.algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"'algorithm' must be one of {VALID_ALGORITHMS}!")

    # Validate direction if present
    if "spec" in config and "direction" in config.spec:
        if config.spec.direction not in ["minimize", "maximize", None]:
            raise ValueError("'direction' must be one of ['minimize', 'maximize', None]!")


def _parse_direction(direction_str: str | None) -> StudyDirection:
    """Parse a direction string into a StudyDirection.

    Args:
        direction_str: The direction string from config ('minimize', 'maximize', or None).

    Returns:
        The corresponding StudyDirection.

    Raises:
        ValueError: If the direction value is invalid.
    """
    if direction_str is None:
        return StudyDirection.MINIMIZE
    elif direction_str.lower() == "minimize":
        return StudyDirection.MINIMIZE
    elif direction_str.lower() == "maximize":
        return StudyDirection.MAXIMIZE
    else:
        raise ValueError(f"Invalid optimization direction: {direction_str}")


def optuna_direction(config: DictConfig) -> StudyDirection:
    """Get Optuna optimization direction from config.

    Args:
        config: A DictConfig loaded from a YAML/JSON file.

    Returns:
        The StudyDirection for the optimization.

    Raises:
        ValueError: If the direction value is invalid.
    """
    if "spec" not in config or "direction" not in config.spec:
        return StudyDirection.NOT_SET
    return _parse_direction(config.spec.direction)


@dataclass
class SpecConfig:
    """Parsed spec configuration for sweep optimization."""

    direction: StudyDirection = StudyDirection.MINIMIZE
    max_retry: int = 3
    metric: str | None = None

    @classmethod
    def from_dict_config(cls, spec: DictConfig | None) -> "SpecConfig":
        """Create a SpecConfig from the spec section of a config.

        Args:
            spec: The spec section of a DictConfig, or None if not present.

        Returns:
            A SpecConfig instance with parsed values.
        """
        if spec is None:
            return cls()

        kwargs: dict[str, Any] = {}

        if "direction" in spec:
            kwargs["direction"] = _parse_direction(spec.direction)

        if "max_retry" in spec:
            kwargs["max_retry"] = spec.max_retry

        if "metric" in spec:
            kwargs["metric"] = spec.metric

        return cls(**kwargs)


@dataclass
class SweepConfig:
    """Parsed and validated sweep configuration."""

    # Required fields
    experiment: str
    sweep_name: str
    command: str
    optuna_storage: str
    mlflow_storage: str
    algorithm: str
    parameters: dict[str, Any]
    param_specs: dict[str, ParamSpec]
    spec: SpecConfig

    # Optional spec (contains direction, max_retry, metric)
    output_dir: str = "output"

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> "SweepConfig":
        """Create a SweepConfig from a DictConfig.

        Args:
            config: A DictConfig loaded from a YAML/JSON file.

        Returns:
            A validated SweepConfig instance.

        Raises:
            ValueError: If required fields are missing or values are invalid.
        """
        validate_config(config)

        # Convert parameters to dict
        params = OmegaConf.to_container(config.parameters, throw_on_missing=True)
        assert isinstance(params, dict)
        
        kwargs = {}
        if "output_dir" in config:
            kwargs["output_dir"] = config.output_dir

        return cls(
            experiment = config.experiment,
            sweep_name = config.sweep_name,
            command = config.command,
            optuna_storage = config.optuna_storage,
            mlflow_storage = config.mlflow_storage,
            algorithm = config.algorithm,
            parameters = params,
            param_specs = config_params_to_spec_dict(config.parameters),
            spec = SpecConfig.from_dict_config(config.get("spec")),
            **kwargs,
        )


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


def config_params_to_spec_dict(parameters: dict[str, Any] | DictConfig) -> dict[str, ParamSpec]:
    """Convert config.parameters into a dict of parameter specifications."""
    params_dict = OmegaConf.to_container(parameters, throw_on_missing=True)
    assert isinstance(params_dict, dict)

    param_specs: dict[str, ParamSpec] = {}
    for name, value in params_dict.items():
        param_specs[name] = param_from_config_entry(name, value)
    return param_specs

