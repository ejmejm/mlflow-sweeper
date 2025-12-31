"""Minimal sweep "training" script used by tests.

This script is executed by `mlflow-sweeper` as a subprocess. It:
- parses CLI args of the form `name=value`
- prints out each sweep variable and its value
- resumes the MLflow run provided via `MLFLOW_RUN_ID`
- logs parameters to MLflow
"""

from __future__ import annotations

import os
import sys
from typing import Any

import mlflow
import omegaconf


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict using dot-separated keys."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def parse_kv_args(args: list[str]) -> dict[str, Any]:
    """Parse `name=value` tokens from argv."""
    parsed: dict[str, Any] = {}
    for token in args:
        if "=" not in token:
            raise ValueError(f"Expected name=value token, got: {token!r}")
        name, raw = token.split("=", 1)
        parsed[name] = raw
    return parsed


def main() -> None:
    params = parse_kv_args(sys.argv[1:])
    config = omegaconf.OmegaConf.create(params)

    # Print each sweep variable value (useful for debugging).
    for key, value in params.items():
        print(f"{key}={value}")

    # Initialize MLflow and log params exactly as required by the tests.
    try:
        # MLflow 3.8+ exposes DEFAULT_TRACKING_URI (not MLFLOW_DEFAULT_TRACKING_URI).
        from mlflow.tracking._tracking_service.utils import (  # type: ignore
            DEFAULT_TRACKING_URI as MLFLOW_DEFAULT_TRACKING_URI,
        )
    except Exception:  # pragma: no cover
        MLFLOW_DEFAULT_TRACKING_URI = "file:///tmp/mlruns"

    project = str(config.get("project", "test-project"))
    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", MLFLOW_DEFAULT_TRACKING_URI))
    if os.environ.get("MLFLOW_RUN_ID") is None:
        mlflow.set_experiment(project)
    mlflow.start_run()
    raw_dict_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    assert isinstance(raw_dict_config, dict)
    flat_config = flatten_dict(raw_dict_config)
    mlflow.log_params(flat_config)

    mlflow.end_run()


if __name__ == "__main__":
    main()
