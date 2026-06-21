"""Integration tests for the sensitivity sampler.

A sensitivity sweep holds every parameter at its baseline (the value in
``parameters``) and varies one parameter at a time over the candidates listed in the
``sensitivity`` block. The trial set is one baseline run plus, per varied parameter,
one run per non-default candidate value: ``1 + Σ (candidates_i − 1)`` runs.
"""

from __future__ import annotations

import os
import sys

import optuna
import pytest

from mlflow_sweeper.samplers.sensitivity import SensitivitySampler
from tests.assets import programmatic_funcs as pf
from tests.conftest import SweepHarness
from tests.helpers import (
    _assert_all_runs_finished,
    _assert_parent_and_get_trial_runs,
    _trial_param,
)


def _seen(trial_runs, keys: list[str]) -> set[tuple[str, ...]]:
    """Set of value tuples (as strings) seen across trial runs for ``keys``."""
    return {tuple(_trial_param(r, k) for k in keys) for r in trial_runs}


def test_sensitivity_baseline_plus_one_at_a_time_count(sweep_harness: SweepHarness) -> None:
    """Baseline + one-at-a-time, with a third parameter held fixed."""
    config_path = sweep_harness.write_config(
        parameters={"a": 1, "b": 10, "c": 5},
        algorithm="sensitivity",
        sensitivity={"a": [1, 2, 3], "b": [10, 20]},
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # 1 baseline + (3-1) for a + (2-1) for b = 4 runs.
    assert len(trial_runs) == 4
    _assert_all_runs_finished(trial_runs)

    assert _seen(trial_runs, ["a", "b"]) == {
        ("1", "10"),  # baseline
        ("2", "10"),  # vary a
        ("3", "10"),  # vary a
        ("1", "20"),  # vary b
    }
    # c never appears in `sensitivity`, so it stays fixed at its baseline.
    for run in trial_runs:
        assert _trial_param(run, "c") == "5"


def test_sensitivity_count_formula(sweep_harness: SweepHarness) -> None:
    """Three varied params: count == 1 + Σ(len_i − 1), exactly one all-defaults run."""
    config_path = sweep_harness.write_config(
        parameters={"a": 1, "b": 1, "c": 1},
        algorithm="sensitivity",
        sensitivity={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2, 3, 4]},
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # 1 + (3-1) + (2-1) + (4-1) = 7
    assert len(trial_runs) == 7
    _assert_all_runs_finished(trial_runs)

    baseline_runs = [
        r for r in trial_runs
        if (_trial_param(r, "a"), _trial_param(r, "b"), _trial_param(r, "c")) == ("1", "1", "1")
    ]
    assert len(baseline_runs) == 1


def test_sensitivity_default_not_in_candidates(sweep_harness: SweepHarness) -> None:
    """A default that is not among the candidates is still used for the baseline."""
    config_path = sweep_harness.write_config(
        parameters={"a": 1, "b": 5},
        algorithm="sensitivity",
        sensitivity={"a": [2, 3]},
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # baseline a=1 plus a=2, a=3 = 3 runs.
    assert len(trial_runs) == 3
    assert {_trial_param(r, "a") for r in trial_runs} == {"1", "2", "3"}
    for run in trial_runs:
        assert _trial_param(run, "b") == "5"


def test_sensitivity_dedup_default_in_candidates(sweep_harness: SweepHarness) -> None:
    """When the default is among the candidates it is not run twice."""
    config_path = sweep_harness.write_config(
        parameters={"a": 2},
        algorithm="sensitivity",
        sensitivity={"a": [1, 2, 3]},
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # baseline a=2 plus a=1, a=3 (default a=2 de-duped) = 3 runs, not 4.
    assert len(trial_runs) == 3
    assert {_trial_param(r, "a") for r in trial_runs} == {"1", "2", "3"}


def test_sensitivity_no_varied_params_errors(sweep_harness: SweepHarness) -> None:
    """A sensitivity sweep without a `sensitivity` section is rejected."""
    with pytest.raises(ValueError, match="sensitivity"):
        sweep_harness.build_config(
            parameters={"a": 1, "b": 2},
            algorithm="sensitivity",
        )


def test_sensitivity_parameters_multivalue_errors(sweep_harness: SweepHarness) -> None:
    """A multi-valued entry in `parameters` is rejected for a sensitivity sweep."""
    with pytest.raises(ValueError, match="single base values"):
        sweep_harness.build_config(
            parameters={"a": [1, 2]},
            algorithm="sensitivity",
            sensitivity={"a": [1, 2, 3]},
        )


def test_sensitivity_failed_runs_are_retried(sweep_harness: SweepHarness) -> None:
    """A failing variation is retried up to max_retry times; the baseline succeeds."""
    max_retry = 2
    config_path = sweep_harness.write_config(
        parameters={"should_fail": "false"},
        algorithm="sensitivity",
        sensitivity={"should_fail": ["false", "true"]},
        command=f"{sys.executable} {os.path.join(sweep_harness.assets_dir, 'maybe_fail.py')}",
        spec={"max_retry": max_retry},
    )

    # --log-params so the runner records `should_fail` on each trial run (maybe_fail.py
    # does not log params itself).
    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1", "--log-params")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    finished_runs = [r for r in trial_runs if r.info.status == "FINISHED"]
    failed_runs = [r for r in trial_runs if r.info.status == "FAILED"]

    # baseline should_fail=false succeeds: 1 FINISHED run.
    assert len(finished_runs) == 1
    # variation should_fail=true fails every time: 1 original + max_retry retries.
    assert len(failed_runs) == max_retry + 1
    assert len(trial_runs) == 1 + (max_retry + 1)

    for run in finished_runs:
        assert _trial_param(run, "should_fail") == "false"
    for run in failed_runs:
        assert _trial_param(run, "should_fail") == "true"


def test_sensitivity_parallel_no_double_runs(sweep_harness: SweepHarness) -> None:
    """Parallel jobs must not double-run sensitivity points."""
    config_path = sweep_harness.write_config(
        parameters={"x": 0, "y": 0, "z": 0},
        algorithm="sensitivity",
        sensitivity={"x": [0, 1], "y": [0, 1], "z": [0, 1]},
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "4")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # baseline + one variation per param = 1 + 3 = 4.
    assert len(trial_runs) == 4
    _assert_all_runs_finished(trial_runs)

    seen = _seen(trial_runs, ["x", "y", "z"])
    assert seen == {("0", "0", "0"), ("1", "0", "0"), ("0", "1", "0"), ("0", "0", "1")}
    assert len(seen) == len(trial_runs)  # no duplicates


def test_sensitivity_programmatic(sweep_harness: SweepHarness) -> None:
    """Programmatic path: correct trial count and a FINISHED parent run."""
    sweep_harness.run_programmatic(
        fn=pf.quadratic,
        parameters={"x": 1, "y": 2},
        algorithm="sensitivity",
        sensitivity={"x": [1, 2, 3]},
    )

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 3

    study = sweep_harness.load_optuna_study()
    complete = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete) == 3
    assert {int(t.params["x"]) for t in complete} == {1, 2, 3}

    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    assert client.get_run(parent_run_id).info.status == "FINISHED"


def test_sensitivity_with_range_param(sweep_harness: SweepHarness) -> None:
    """A typed int_range candidate spec is expanded via to_list()."""
    config_path = sweep_harness.write_config(
        parameters={"depth": 2, "lr": 0.01},
        algorithm="sensitivity",
        sensitivity={"depth": {"type": "int_range", "low": 2, "high": 5, "step": 1}},
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # int_range(2, 5, 1) -> [2, 3, 4]; default depth=2 -> baseline 2 plus 3, 4 = 3 runs.
    assert len(trial_runs) == 3
    assert {_trial_param(r, "depth") for r in trial_runs} == {"2", "3", "4"}


def test_sensitivity_allow_param_change_expands_scope(sweep_harness: SweepHarness) -> None:
    """Expanding the sensitivity candidate values + --allow-param-change migrates cleanly.

    Regression test: the candidate values live in the `sensitivity` block (merged into
    param_specs), so config-change detection must track that block. Otherwise the change
    goes unnoticed, no migration happens, and new trials collide with the old trials'
    CategoricalDistribution choices.
    """
    # Initial scope: vary `a` over [1, 2] off baseline a=1.
    config_path = sweep_harness.write_config(
        parameters={"a": 1, "b": 5},
        algorithm="sensitivity",
        sensitivity={"a": [1, 2]},
    )
    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 2  # baseline a=1 + variation a=2
    assert {_trial_param(r, "a") for r in trial_runs} == {"1", "2"}

    # Expand the candidate values for `a`; re-run in-place with migration.
    config_path = sweep_harness.write_config(
        parameters={"a": 1, "b": 5},
        algorithm="sensitivity",
        sensitivity={"a": [1, 2, 3, 4]},
    )
    sweep_harness.run_cli(config_path, "--allow-param-change")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    # baseline a=1 + a=2,3,4 = 4 trials; the 2 originals migrated, 2 new ran.
    assert len(trial_runs) == 4
    assert {_trial_param(r, "a") for r in trial_runs} == {"1", "2", "3", "4"}
    for run in trial_runs:
        assert _trial_param(run, "b") == "5"


def test_build_sensitivity_grids_unit() -> None:
    """Unit test of the enumeration: baseline + one-at-a-time, no MLflow/Optuna run."""
    sampler = SensitivitySampler(
        search_space={"a": [1, 2, 3], "b": [10, 20]},
        defaults={"a": 1, "b": 10},
    )
    assert sampler._n_min_trials == 4
    assert set(sampler._all_grids) == {(1, 10), (2, 10), (3, 10), (1, 20)}
    assert sampler._param_names == ["a", "b"]
