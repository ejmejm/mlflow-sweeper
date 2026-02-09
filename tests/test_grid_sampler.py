"""Integration tests for the grid sampler."""

from __future__ import annotations

import os
import subprocess
import sys

import optuna

from tests.conftest import SweepHarness
from tests.helpers import (
    _assert_all_combinations_seen,
    _assert_all_runs_finished,
    _assert_parent_and_get_trial_runs,
    _capture_and_print_interleaved,
    _trial_param,
)


def test_grid_runs_all_4_combinations(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "color": ["red", "blue"],
        "shape": ["square", "circle"],
    }
    config_path = sweep_harness.write_config(parameters=grid_params)

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 4

    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["color", "shape"], parameters=grid_params
    )


def test_delete_sweep_removes_optuna_and_mlflow_runs(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters = {
            "a": ["0", "1"],
            "b": ["x", "y"],
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
    config_path = sweep_harness.write_config(parameters=grid_params)

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "4")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    # Grid search should produce exactly 8 combinations (2*2*2).
    assert len(trial_runs) == 8
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )

    _assert_all_runs_finished(trial_runs)


def test_parallel_two_processes_does_not_double_runs(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "x": ["0", "1"],
        "y": ["0", "1"],
        "z": ["0", "1"],  # 2*2*2 = 8 combos
    }
    config_path = sweep_harness.write_config(parameters=grid_params)

    cmd = [
        sys.executable,
        f"{sweep_harness.repo_root}/sweep.py",
        config_path,
        "-n",
        "100",
        "-j",
        "1",
    ]

    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Capture and print output interleaved in original order
    print("=== Runner 1 Output ===", file=sys.__stdout__)
    sys.__stdout__.flush()
    out1, err1 = _capture_and_print_interleaved(proc1, timeout=300)

    print("=== Runner 2 Output ===", file=sys.__stdout__)
    sys.__stdout__.flush()
    out2, err2 = _capture_and_print_interleaved(proc2, timeout=300)

    print(f"Runner 1 exit code: {proc1.returncode}", file=sys.__stdout__)
    print(f"Runner 2 exit code: {proc2.returncode}", file=sys.__stdout__)
    sys.__stdout__.flush()

    if proc1.returncode != 0:
        raise RuntimeError(
            f"Runner 1 failed with exit code {proc1.returncode}.\n\n"
            f"Stdout:\n{out1}\n\nStderr:\n{err1}"
        )
    if proc2.returncode != 0:
        raise RuntimeError(
            f"Runner 2 failed with exit code {proc2.returncode}.\n\n"
            f"Stdout:\n{out2}\n\nStderr:\n{err2}"
        )

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    # Grid search should produce exactly 8 combinations (2*2*2).
    assert len(trial_runs) == 8
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )
    _assert_all_runs_finished(trial_runs)


def test_failed_runs_are_retried(sweep_harness: SweepHarness) -> None:
    max_retry = 2
    grid_params = {
        "should_fail": ["true", "false"],
    }
    config_path = sweep_harness.write_config(
        parameters=grid_params,
        command=f"{sys.executable} {os.path.join(sweep_harness.assets_dir, 'maybe_fail.py')}",
        spec={"max_retry": max_retry},
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    finished_runs = [r for r in trial_runs if r.info.status == "FINISHED"]
    failed_runs = [r for r in trial_runs if r.info.status == "FAILED"]

    # should_fail=false succeeds on first attempt: 1 FINISHED run.
    assert len(finished_runs) == 1
    # should_fail=true fails every time: 1 original + max_retry retries = 3 FAILED runs.
    assert len(failed_runs) == max_retry + 1

    # Total trial runs = 1 (success) + 3 (failures).
    assert len(trial_runs) == 1 + (max_retry + 1)

    # All FINISHED runs have should_fail=false.
    for run in finished_runs:
        assert _trial_param(run, "should_fail") == "false"

    # All FAILED runs have should_fail=true.
    for run in failed_runs:
        assert _trial_param(run, "should_fail") == "true"


def test_resumed_sweep_retries_failed_runs(sweep_harness: SweepHarness) -> None:
    """Resuming a sweep should continue retrying failed combos, not skip them."""
    max_retry = 2
    grid_params = {"should_fail": ["true"]}  # 1 combo, always fails
    config_path = sweep_harness.write_config(
        parameters=grid_params,
        command=f"{sys.executable} {os.path.join(sweep_harness.assets_dir, 'maybe_fail.py')}",
        spec={"max_retry": max_retry},
    )

    # First run: only 1 trial allowed, so only the first attempt happens.
    sweep_harness.run_cli(config_path, "-n", "1", "-j", "1")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 1
    assert trial_runs[0].info.status == "FAILED"

    # Second run (resume): should retry the failed combo up to max_retry more times.
    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    failed_runs = [r for r in trial_runs if r.info.status == "FAILED"]

    # Total: 1 original + max_retry retries = 3 FAILED runs.
    assert len(failed_runs) == max_retry + 1
