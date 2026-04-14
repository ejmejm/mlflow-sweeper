"""Integration tests for the --allow-param-change feature."""

from __future__ import annotations

import os
import subprocess
import sys

import optuna

from tests.conftest import SweepHarness
from tests.helpers import _assert_parent_and_get_trial_runs, _trial_param


def test_error_message_hints_allow_param_change(sweep_harness: SweepHarness) -> None:
    """Changed config error should mention --allow-param-change."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    config_path = sweep_harness.write_config(parameters={"x": [1, 2, 3]})
    cmd = [sys.executable, os.path.join(sweep_harness.repo_root, "sweep.py"), config_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode != 0
    assert "--allow-param-change" in proc.stdout


def test_grid_expand_values(sweep_harness: SweepHarness) -> None:
    """Expanding grid values should keep old runs and add new ones in one call."""
    initial_params = {"color": ["red", "blue"], "shape": ["square"]}
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    # Verify initial state: 2 trial runs.
    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 2
    old_trial_run_ids = {r.info.run_id for r in trial_runs_before}

    # Expand the grid and migrate-then-run with --allow-param-change.
    expanded_params = {"color": ["red", "blue"], "shape": ["square", "circle"]}
    config_path = sweep_harness.write_config(parameters=expanded_params)
    sweep_harness.run_cli(config_path, "--allow-param-change")

    # After one call: migration + new trials = 4 trial runs total.
    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 4

    # Old trial run IDs should still be present.
    new_trial_run_ids = {r.info.run_id for r in trial_runs_after}
    assert old_trial_run_ids.issubset(new_trial_run_ids)


def test_grid_expand_with_atomic_params(sweep_harness: SweepHarness) -> None:
    """Atomic params (fixed values) should not prevent trial migration."""
    initial_params = {
        "fixed_flag": True,
        "fixed_name": "experiment_1",
        "lr": [0.01, 0.1],
        "layers": [2, 4],
    }
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 4  # 2 lr × 2 layers

    # Expand lr with a new value — migrate + run new trials.
    expanded_params = {
        "fixed_flag": True,
        "fixed_name": "experiment_1",
        "lr": [0.001, 0.01, 0.1],
        "layers": [2, 4],
    }
    config_path = sweep_harness.write_config(parameters=expanded_params)
    sweep_harness.run_cli(config_path, "--allow-param-change")

    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 6  # 4 old + 2 new


def test_grid_shrink_values(sweep_harness: SweepHarness) -> None:
    """Shrinking grid values should drop non-matching trials from Optuna but keep them in MLflow."""
    initial_params = {"color": ["red", "blue", "green"], "shape": ["square"]}
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 3

    # Shrink: drop "green".
    shrunk_params = {"color": ["red", "blue"], "shape": ["square"]}
    config_path = sweep_harness.write_config(parameters=shrunk_params)
    sweep_harness.run_cli(config_path, "--allow-param-change")

    # All 3 trial runs still visible in MLflow (the dropped one is just not in
    # the new Optuna study).  No new trials are added because the new grid is
    # already exhausted by the migrated trials.
    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 3

    # Optuna study should have only 2 matching trials.
    study = sweep_harness.load_optuna_study()
    complete_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete_trials) == 2


def test_grid_add_new_param(sweep_harness: SweepHarness) -> None:
    """Adding a new param means no old trials match; sweep then runs all new trials."""
    initial_params = {"color": ["red", "blue"]}
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 2

    # Add a new parameter — migrate (no trials match) + run new trials.
    new_params = {"color": ["red", "blue"], "size": ["small", "large"]}
    config_path = sweep_harness.write_config(parameters=new_params)
    sweep_harness.run_cli(config_path, "--allow-param-change")

    # 2 old (re-parented) + 4 new = 6 total in MLflow.
    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 6


def test_trial_runs_reparented(sweep_harness: SweepHarness) -> None:
    """All trial runs should point to the new parent after update."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    old_parent_id = sweep_harness.get_parent_run_id()
    assert old_parent_id is not None

    config_path = sweep_harness.write_config(parameters={"x": [1, 2, 3]})
    sweep_harness.run_cli(config_path, "--allow-param-change")

    new_parent_id = sweep_harness.get_parent_run_id()
    assert new_parent_id is not None
    assert new_parent_id != old_parent_id

    # Every trial run should point to the new parent.
    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    for run in trial_runs:
        assert run.data.tags.get("mlflow.parentRunId") == new_parent_id


def test_children_of_trials_preserved(sweep_harness: SweepHarness) -> None:
    """Children of trial runs should remain accessible after update."""
    child_script = os.path.join(sweep_harness.assets_dir, "create_child_run.py")
    config_path = sweep_harness.write_config(
        parameters={"x": [1]},
        command=f"{sys.executable} {child_script}",
    )
    sweep_harness.run_cli(config_path)

    # There should be 1 trial run with 1 child.
    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 1
    trial_run_id = trial_runs[0].info.run_id

    client = sweep_harness.mlflow_client()
    experiment = client.get_experiment_by_name(sweep_harness.experiment)
    children_before = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{trial_run_id}"',
    )
    assert len(children_before) == 1

    # Migrate + run any new trials.
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2]},
        command=f"{sys.executable} {child_script}",
    )
    sweep_harness.run_cli(config_path, "--allow-param-change")

    # The old trial's child should still point to the old trial run.
    children_after = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{trial_run_id}"',
    )
    assert len(children_after) == 1
    assert children_after[0].info.run_id == children_before[0].info.run_id


def test_failed_trials_migrated(sweep_harness: SweepHarness) -> None:
    """Failed trials should be migrated so retry logic continues correctly."""
    fail_script = os.path.join(sweep_harness.assets_dir, "maybe_fail.py")
    params = {"should_fail": ["true", "false"]}
    config_path = sweep_harness.write_config(
        parameters=params,
        command=f"{sys.executable} {fail_script}",
        spec={"max_retry": 2},
    )
    sweep_harness.run_cli(config_path)

    study_before = sweep_harness.load_optuna_study()
    fail_trials = study_before.get_trials(states=[optuna.trial.TrialState.FAIL])
    assert len(fail_trials) > 0, "Expected at least one failed trial"

    # Add a new value — migrate + run new trials.
    expanded_params = {"should_fail": ["true", "false", "maybe"]}
    config_path = sweep_harness.write_config(
        parameters=expanded_params,
        command=f"{sys.executable} {fail_script}",
        spec={"max_retry": 2},
    )
    sweep_harness.run_cli(config_path, "--allow-param-change")

    # The new study should have migrated the failed trials.
    study_after = sweep_harness.load_optuna_study()
    fail_trials_after = study_after.get_trials(states=[optuna.trial.TrialState.FAIL])
    assert len(fail_trials_after) >= len(fail_trials)


def test_allow_param_change_no_change_warns_and_runs(sweep_harness: SweepHarness) -> None:
    """--allow-param-change with unchanged config should warn but still complete normally."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    parent_id_before = sweep_harness.get_parent_run_id()

    # Run again with --allow-param-change but same config.
    cmd = [sys.executable, os.path.join(sweep_harness.repo_root, "sweep.py"),
           config_path, "--allow-param-change"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode == 0
    assert "had no effect" in proc.stdout.lower()

    # Parent should be the same (no migration).
    parent_id_after = sweep_harness.get_parent_run_id()
    assert parent_id_before == parent_id_after
