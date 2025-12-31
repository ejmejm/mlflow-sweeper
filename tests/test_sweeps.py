"""Basic integration tests for mlflow-sweeper."""

from __future__ import annotations

import itertools
import subprocess
import sys

import mlflow
import optuna

from tests.conftest import SweepHarness


def _trial_param(run: mlflow.entities.Run, name: str) -> str | None:
    value = run.data.params.get(name)
    if value is None:
        return None
    return str(value)


def _expected_combinations(parameters: dict[str, list[str]], keys: list[str]) -> set[tuple[str, ...]]:
    values = [parameters[k] for k in keys]
    return set(itertools.product(*values))


def _seen_combinations(
    trial_runs: list[mlflow.entities.Run], keys: list[str]
) -> set[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    for run in trial_runs:
        combo: list[str] = []
        for k in keys:
            value = _trial_param(run, k)
            if value is None:
                break
            combo.append(value)
        if len(combo) == len(keys):
            seen.add(tuple(combo))
    return seen


def _assert_all_combinations_seen(
    *, trial_runs: list[mlflow.entities.Run], keys: list[str], parameters: dict[str, list[str]]
) -> None:
    expected = _expected_combinations(parameters, keys)
    seen = _seen_combinations(trial_runs, keys)
    missing = expected.difference(seen)
    assert not missing, f"Missing combinations for {keys}: {sorted(missing)}"


def _assert_parent_and_get_trial_runs(*, harness: SweepHarness) -> list[mlflow.entities.Run]:
    runs = harness.list_mlflow_runs()
    parents = [r for r in runs if "optuna_study_name" in r.data.tags]
    assert len(parents) == 1, f"Expected exactly 1 parent run, found {len(parents)}."
    return [r for r in runs if "optuna_study_name" not in r.data.tags]


def test_grid_runs_all_4_combinations(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "color": ["red", "blue"],
        "shape": ["square", "circle"],
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            # Passed through to the subprocess script to satisfy the required MLflow init.
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) >= 4

    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["color", "shape"], parameters=grid_params
    )


def test_delete_sweep_removes_optuna_and_mlflow_runs(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={
            "a": ["0", "1"],
            "b": ["x", "y"],
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1")

    # Sanity: Optuna study exists and MLflow has runs.
    _ = sweep_harness.load_optuna_study()
    assert len(sweep_harness.list_mlflow_runs()) > 0

    sweep_harness.run_cli(config_path, "--delete")

    # Optuna study deleted.
    try:
        optuna.load_study(
            study_name=sweep_harness.optuna_study_name(),
            storage=sweep_harness.optuna_storage,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected Optuna study to be deleted, but it still exists.")

    # MLflow runs deleted (ACTIVE view should be empty).
    assert sweep_harness.list_mlflow_runs() == []


def test_parallel_inprocess_jobs_does_not_double_runs(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "x": ["0", "1"],
        "y": ["0", "1"],
        "z": ["0", "1"],  # 2*2*2 = 8 combos
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "2")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    # Duplicates are allowed, but we should not have >= 2x the base grid size.
    assert 0 < len(trial_runs) < 16
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )


def test_parallel_two_processes_does_not_double_runs(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "x": ["0", "1"],
        "y": ["0", "1"],
        "z": ["0", "1"],  # 2*2*2 = 8 combos
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    cmd = [
        sys.executable,
        f"{sweep_harness.repo_root}/sweep.py",
        config_path,
        "-n",
        "100",
        "-j",
        "1",
    ]

    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    out1, _ = proc1.communicate(timeout=300)
    out2, _ = proc2.communicate(timeout=300)

    if proc1.returncode != 0:
        raise RuntimeError(f"Runner 1 failed.\n\nOutput:\n{out1}")
    if proc2.returncode != 0:
        raise RuntimeError(f"Runner 2 failed.\n\nOutput:\n{out2}")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    assert 0 < len(trial_runs) < 16
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )
