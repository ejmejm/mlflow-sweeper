"""Integration tests for the random sampler."""

from __future__ import annotations

import os
import sys
from collections import defaultdict

from tests.conftest import SweepHarness
from tests.helpers import (
    _assert_all_runs_finished,
    _assert_parent_and_get_trial_runs,
    _trial_param,
)


def test_random_runs_set_number_of_trials(sweep_harness: SweepHarness) -> None:
    """Random sampler with n_runs=5 should produce exactly 5 trials."""
    params = {
        "color": ["red", "blue", "green"],
        "shape": ["square", "circle", "triangle"],
    }
    config_path = sweep_harness.write_config(
        parameters=params,
        algorithm="random",
        spec={"n_runs": 5, "max_retry": 0},
    )

    sweep_harness.run_cli(config_path, "-n", "100")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 5
    _assert_all_runs_finished(trial_runs)

    # All param values should be valid choices.
    for run in trial_runs:
        assert _trial_param(run, "color") in ["red", "blue", "green"]
        assert _trial_param(run, "shape") in ["square", "circle", "triangle"]

    # Parent should be FINISHED since n_runs is exhausted.
    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    parent_run = client.get_run(parent_run_id)
    assert parent_run.info.status == "FINISHED"


def test_random_unlimited_runs(sweep_harness: SweepHarness) -> None:
    """Random sampler without n_runs should run all requested trials and leave parent RUNNING."""
    params = {
        "color": ["red", "blue", "green"],
        "shape": ["square", "circle"],
    }
    config_path = sweep_harness.write_config(
        parameters=params,
        algorithm="random",
        spec={"max_retry": 0},
    )

    sweep_harness.run_cli(config_path, "-n", "8")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 8

    # All child runs should be finished.
    _assert_all_runs_finished(trial_runs)

    # Parent should be RUNNING since there's no n_runs cap.
    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    parent_run = client.get_run(parent_run_id)
    assert parent_run.info.status == "RUNNING"


def test_random_with_n_runs_parallel(sweep_harness: SweepHarness) -> None:
    """Random sampler with n_runs and parallel jobs should not exceed n_runs."""
    params = {
        "color": ["red", "blue", "green"],
        "shape": ["square", "circle"],
    }
    config_path = sweep_harness.write_config(
        parameters=params,
        algorithm="random",
        spec={"n_runs": 6, "max_retry": 0},
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "4")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 6

    # Parent should be FINISHED since n_runs is exhausted.
    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    parent_run = client.get_run(parent_run_id)
    assert parent_run.info.status == "FINISHED"


def test_random_with_grid_params(sweep_harness: SweepHarness) -> None:
    """Random sampler with grid_params should expand grid combos for each random sample."""
    params = {
        "lr": ["0.1", "0.01", "0.001"],
        "seed": ["0", "1", "2"],
    }
    config_path = sweep_harness.write_config(
        parameters=params,
        algorithm="random",
        spec={"n_runs": 2, "grid_params": ["seed"], "max_retry": 0},
    )

    sweep_harness.run_cli(config_path, "-n", "100")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # 2 random groups x 3 seeds = 6 total trials
    assert len(trial_runs) == 6

    # Each unique lr value should have all 3 seeds
    lr_to_seeds: dict[str, set[str]] = defaultdict(set)
    for run in trial_runs:
        lr = _trial_param(run, "lr")
        seed = _trial_param(run, "seed")
        assert lr is not None
        assert seed is not None
        lr_to_seeds[lr].add(seed)

    # We should have exactly 2 unique lr values (n_runs=2)
    assert len(lr_to_seeds) == 2
    for lr, seeds in lr_to_seeds.items():
        assert seeds == {"0", "1", "2"}, f"lr={lr} missing seeds: expected {{0,1,2}}, got {seeds}"


def test_random_grid_params_retries_failed_combos(sweep_harness: SweepHarness) -> None:
    """Failed grid combos in random sampler should be retried up to max_retry times."""
    max_retry = 1
    params = {
        "lr": ["0.1", "0.01", "0.001"],
        "should_fail": ["true", "false"],
    }
    config_path = sweep_harness.write_config(
        parameters=params,
        algorithm="random",
        command=f"{sys.executable} {os.path.join(sweep_harness.assets_dir, 'maybe_fail.py')}",
        spec={"n_runs": 1, "grid_params": ["should_fail"], "max_retry": max_retry},
    )

    sweep_harness.run_cli(config_path, "-n", "100")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    finished_runs = [r for r in trial_runs if r.info.status == "FINISHED"]
    failed_runs = [r for r in trial_runs if r.info.status == "FAILED"]

    # should_fail=false succeeds: 1 FINISHED run.
    assert len(finished_runs) == 1
    # should_fail=true fails every time: 1 original + max_retry retries = 2 FAILED runs.
    assert len(failed_runs) == max_retry + 1

    # Total: 1 success + 2 failures = 3.
    assert len(trial_runs) == 1 + (max_retry + 1)
