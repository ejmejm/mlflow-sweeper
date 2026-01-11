"""Pytest fixtures for mlflow-sweeper tests."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
import optuna
import pytest


@dataclasses.dataclass(frozen=True)
class SweepHarness:
    """Shared harness for tests: unique storages under /tmp."""

    repo_root: str
    root_dir: str
    output_dir: str
    optuna_storage: str
    mlflow_storage: str
    assets_dir: str

    experiment: str
    sweep_name: str

    def write_config(self, *, parameters: dict[str, Any]) -> str:
        """Write a sweep YAML config and return its path."""
        config_path = os.path.join(self.root_dir, "sweep.yaml")
        config = {
            "experiment": self.experiment,
            "sweep_name": self.sweep_name,
            "command": f"{sys.executable} {os.path.join(self.assets_dir, 'print_sweep_vars.py')}",
            "algorithm": "grid",
            "parameters": parameters,
            "optuna_storage": self.optuna_storage,
            "mlflow_storage": self.mlflow_storage,
            "output_dir": self.output_dir,
        }
        OmegaConf.save(config=OmegaConf.create(config), f=config_path)
        return config_path

    def run_cli(self, config_path: str, *args: str) -> None:
        """Run the sweep CLI for a config path."""
        cmd = [sys.executable, os.path.join(self.repo_root, "sweep.py"), config_path, *args]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n\nOutput:\n{proc.stdout}")

    def mlflow_client(self) -> MlflowClient:
        """Create an MLflow client for this harness's tracking URI."""
        return MlflowClient(tracking_uri=self.mlflow_storage)

    def optuna_study_name(self) -> str:
        """Return the Optuna study name for this sweep."""
        return f"{self.experiment}/{self.sweep_name}"

    def list_mlflow_runs(self) -> list[mlflow.entities.Run]:
        """List ACTIVE runs associated with this sweep."""
        client = self.mlflow_client()
        experiment = client.get_experiment_by_name(self.experiment)
        if experiment is None:
            return []
        return client.search_runs(
            experiment_ids = [experiment.experiment_id],
            filter_string = f'tags.sweep_name = "{self.sweep_name}"',
            run_view_type = mlflow.entities.ViewType.ACTIVE_ONLY,
        )

    def get_parent_run_id(self) -> str | None:
        """Return the parent run ID (tagged with optuna_study_name), if any."""
        for run in self.list_mlflow_runs():
            if "optuna_study_name" in run.data.tags:
                return run.info.run_id
        return None

    def list_trial_runs(self) -> list[mlflow.entities.Run]:
        """List ACTIVE trial (nested) runs for this sweep."""
        runs = self.list_mlflow_runs()
        return [r for r in runs if "optuna_study_name" not in r.data.tags]

    def load_optuna_study(self) -> optuna.Study:
        """Load the Optuna study for this sweep."""
        return optuna.load_study(
            study_name=self.optuna_study_name(),
            storage=self.optuna_storage,
        )


@pytest.fixture()
def sweep_harness() -> Iterator[SweepHarness]:
    """Create a unique, temporary harness rooted in /tmp and clean it up."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root_dir = tempfile.mkdtemp(prefix="mlflow_sweeper_test_", dir="/tmp")
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    optuna_db = os.path.join(root_dir, "optuna.db")
    optuna_storage = f"sqlite:///{optuna_db}"
    mlflow_db = os.path.join(root_dir, "mlflow.db")
    mlflow_storage = f"sqlite:///{mlflow_db}"

    harness = SweepHarness(
        repo_root = repo_root,
        root_dir = root_dir,
        output_dir = output_dir,
        optuna_storage = optuna_storage,
        mlflow_storage = mlflow_storage,
        assets_dir = os.path.join(repo_root, "tests", "assets"),
        experiment = f"test-exp-{uuid.uuid4().hex}",
        sweep_name = f"test-sweep-{uuid.uuid4().hex}",
    )
    try:
        yield harness
    finally:
        # Best-effort cleanup; tests always create under /tmp so leaks are contained.
        try:
            shutil.rmtree(root_dir)
        except Exception:
            pass
