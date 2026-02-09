"""Tests for algorithm-specific spec field validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from mlflow_sweeper.config import validate_config


def _make_config(**overrides):
    """Build a minimal valid DictConfig, with optional overrides."""
    base = {
        "experiment": "test-exp",
        "sweep_name": "test-sweep",
        "command": "echo hello",
        "algorithm": "grid",
        "parameters": {"x": [1, 2]},
        "optuna_storage": "sqlite:///test.db",
        "mlflow_storage": "sqlite:///mlflow.db",
        "output_dir": "output",
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_grid_rejects_random_spec_fields():
    config = _make_config(algorithm="grid", spec={"n_runs": 10})
    with pytest.raises(ValueError, match="only valid for algorithm 'random'"):
        validate_config(config)


def test_unknown_spec_field_rejected():
    config = _make_config(algorithm="random", spec={"n_run": 5})
    with pytest.raises(ValueError, match="Unrecognized spec field 'n_run'"):
        validate_config(config)


def test_random_accepts_its_spec_fields():
    config = _make_config(
        algorithm="random",
        spec={"n_runs": 10, "grid_params": ["x"], "direction": "minimize"},
    )
    validate_config(config)  # should not raise
