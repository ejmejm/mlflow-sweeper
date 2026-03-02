"""Test script that logs a synthetic metric to MLflow.

Parses `name=value` CLI args, resumes the MLflow run from MLFLOW_RUN_ID,
and logs a deterministic metric derived from the parameter values so that
tests can verify plot contents.

The metric is computed as the sum of all numeric parameter values.
"""

from __future__ import annotations

import os
import sys

import mlflow


def parse_kv_args(args: list[str]) -> dict[str, str]:
    """Parse `name=value` tokens from argv."""
    parsed: dict[str, str] = {}
    for token in args:
        if "=" not in token:
            raise ValueError(f"Expected name=value token, got: {token!r}")
        name, raw = token.split("=", 1)
        parsed[name] = raw
    return parsed


def main() -> None:
    params = parse_kv_args(sys.argv[1:])

    mlflow_storage = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_storage is None:
        raise ValueError("Expected MLFLOW_TRACKING_URI to be set.")

    mlflow.set_tracking_uri(str(mlflow_storage))
    mlflow.start_run()

    # Compute a deterministic metric from numeric params
    metric_value = 0.0
    for value in params.values():
        try:
            metric_value += float(value)
        except (ValueError, TypeError):
            pass

    mlflow.log_metric("loss", metric_value)
    mlflow.log_metric("accuracy", 1.0 / (1.0 + metric_value))
    mlflow.end_run()


if __name__ == "__main__":
    main()
