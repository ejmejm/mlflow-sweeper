"""Test script that creates a child MLflow run under the current trial run."""

from __future__ import annotations

import os
import sys

import mlflow


def main() -> None:
    params = {}
    for token in sys.argv[1:]:
        if "=" in token:
            name, value = token.split("=", 1)
            params[name] = value

    mlflow_storage = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_storage is None:
        raise ValueError("Expected MLFLOW_TRACKING_URI to be set.")

    mlflow.set_tracking_uri(str(mlflow_storage))
    trial_run_id = os.environ.get("MLFLOW_RUN_ID")

    # Resume the trial run briefly to get its experiment_id.
    mlflow.start_run()
    experiment_id = mlflow.active_run().info.experiment_id
    mlflow.end_run()

    # Create a child run under the trial run.
    child_run = mlflow.start_run(
        experiment_id=experiment_id,
        tags={"mlflow.parentRunId": trial_run_id},
        run_name="child",
    )
    mlflow.log_param("child_param", "hello")
    mlflow.end_run()


if __name__ == "__main__":
    main()
