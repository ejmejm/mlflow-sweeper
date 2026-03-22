"""Integration tests for the --update-params feature."""

from __future__ import annotations

import os
import subprocess
import sys

import optuna

from tests.conftest import SweepHarness
from tests.helpers import _assert_parent_and_get_trial_runs, _trial_param


def test_error_message_hints_update_params(sweep_harness: SweepHarness) -> None:
    """Changed config error should mention --update-params."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    config_path = sweep_harness.write_config(parameters={"x": [1, 2, 3]})
    cmd = [sys.executable, os.path.join(sweep_harness.repo_root, "sweep.py"), config_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode != 0
    assert "--update-params" in proc.stdout


def test_grid_expand_values(sweep_harness: SweepHarness) -> None:
    """Expanding grid values should keep old runs and add new ones after a subsequent run."""
    initial_params = {"color": ["red", "blue"], "shape": ["square"]}
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    # Verify initial state: 2 trial runs.
    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 2
    old_trial_run_ids = {r.info.run_id for r in trial_runs_before}

    # Expand the grid and migrate with --update-params (no new trials run).
    expanded_params = {"color": ["red", "blue"], "shape": ["square", "circle"]}
    config_path = sweep_harness.write_config(parameters=expanded_params)
    sweep_harness.run_cli(config_path, "--update-params")

    # Still only 2 trial runs — migration doesn't run new trials.
    trial_runs_mid = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_mid) == 2

    # Now run the sweep normally to fill in the remaining trials.
    sweep_harness.run_cli(config_path)

    # Should now have 4 trial runs total (2 migrated + 2 new).
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

    # Expand lr with a new value.
    expanded_params = {
        "fixed_flag": True,
        "fixed_name": "experiment_1",
        "lr": [0.001, 0.01, 0.1],
        "layers": [2, 4],
    }
    config_path = sweep_harness.write_config(parameters=expanded_params)
    sweep_harness.run_cli(config_path, "--update-params")

    # Optuna should have 4 migrated trials (all old combos still valid).
    study = sweep_harness.load_optuna_study()
    complete_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete_trials) == 4

    # Run sweep to fill in remaining 2 new combos (lr=0.001 × layers=[2,4]).
    sweep_harness.run_cli(config_path)

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
    sweep_harness.run_cli(config_path, "--update-params")

    # All 3 trial runs still visible in MLflow.
    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 3

    # Optuna study should have only 2 matching trials.
    study = sweep_harness.load_optuna_study()
    complete_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete_trials) == 2


def test_grid_add_new_param(sweep_harness: SweepHarness) -> None:
    """Adding a new param means no old trials match; subsequent run creates all new trials."""
    initial_params = {"color": ["red", "blue"]}
    config_path = sweep_harness.write_config(parameters=initial_params)
    sweep_harness.run_cli(config_path)

    trial_runs_before = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_before) == 2

    # Add a new parameter — migrate only (no trials match).
    new_params = {"color": ["red", "blue"], "size": ["small", "large"]}
    config_path = sweep_harness.write_config(parameters=new_params)
    sweep_harness.run_cli(config_path, "--update-params")

    # Optuna should have 0 migrated trials (param names don't match).
    study = sweep_harness.load_optuna_study()
    assert len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) == 0

    # 2 old runs still in MLflow (re-parented to new parent).
    trial_runs_mid = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_mid) == 2

    # Now run to create the 4 new trials.
    sweep_harness.run_cli(config_path)

    # 2 old + 4 new = 6 total in MLflow.
    trial_runs_after = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs_after) == 6


def test_trial_runs_reparented(sweep_harness: SweepHarness) -> None:
    """All trial runs should point to the new parent after update."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    old_parent_id = sweep_harness.get_parent_run_id()
    assert old_parent_id is not None

    config_path = sweep_harness.write_config(parameters={"x": [1, 2, 3]})
    sweep_harness.run_cli(config_path, "--update-params")

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

    # Update params (migration only).
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2]},
        command=f"{sys.executable} {child_script}",
    )
    sweep_harness.run_cli(config_path, "--update-params")

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

    # Update params (add a new value to trigger change) — migration only.
    expanded_params = {"should_fail": ["true", "false", "maybe"]}
    config_path = sweep_harness.write_config(
        parameters=expanded_params,
        command=f"{sys.executable} {fail_script}",
        spec={"max_retry": 2},
    )
    sweep_harness.run_cli(config_path, "--update-params")

    # The new study should have migrated the failed trials.
    study_after = sweep_harness.load_optuna_study()
    fail_trials_after = study_after.get_trials(states=[optuna.trial.TrialState.FAIL])
    assert len(fail_trials_after) >= len(fail_trials)


def test_update_params_no_change_noop(sweep_harness: SweepHarness) -> None:
    """--update-params with unchanged config should detect nothing changed."""
    config_path = sweep_harness.write_config(parameters={"x": [1, 2]})
    sweep_harness.run_cli(config_path)

    parent_id_before = sweep_harness.get_parent_run_id()

    # Run again with --update-params but same config.
    cmd = [sys.executable, os.path.join(sweep_harness.repo_root, "sweep.py"),
           config_path, "--update-params"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode == 0
    assert "not changed" in proc.stdout.lower()

    # Parent should be the same (no migration needed).
    parent_id_after = sweep_harness.get_parent_run_id()
    assert parent_id_before == parent_id_after
