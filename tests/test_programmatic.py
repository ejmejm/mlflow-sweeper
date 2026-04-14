"""Integration tests for the programmatic ``run_sweep(config, fn, ...)`` API."""

from __future__ import annotations

import logging
import time

import optuna
import pytest

from mlflow_sweeper.runner import run_sweep
from tests.assets import programmatic_funcs as pf
from tests.conftest import SweepHarness
from tests.helpers import _assert_parent_and_get_trial_runs


# Lifecycle tests run for both single-worker (in-process) and multi-worker
# parallel execution, across both pool flavors.  Process-pool is the headline
# new feature; thread-pool stays in the matrix to guarantee no behavioral
# divergence between the two.
EXECUTOR_MATRIX = [
    pytest.param("thread", 1, id="thread-n1"),
    pytest.param("thread", 4, id="thread-n4"),
    pytest.param("process", 1, id="process-n1"),
    pytest.param("process", 2, id="process-n2"),
]


def _assert_invariants(harness: SweepHarness, expected_trial_count: int) -> None:
    """Cross-cutting invariants that should hold after every programmatic sweep."""
    trial_runs = _assert_parent_and_get_trial_runs(harness=harness)
    assert len(trial_runs) == expected_trial_count, (
        f"Expected {expected_trial_count} trial runs, got {len(trial_runs)}: "
        f"{[r.info.run_id for r in trial_runs]}"
    )
    parent_id = harness.get_parent_run_id()
    assert parent_id is not None
    for run in trial_runs:
        assert run.data.tags.get("mlflow.parentRunId") == parent_id, (
            f"Trial {run.info.run_id} has wrong parent: "
            f"{run.data.tags.get('mlflow.parentRunId')} (expected {parent_id})"
        )

    import mlflow
    assert mlflow.active_run() is None, (
        "An MLflow run is still active after run_sweep returned."
    )


# ---------------------------------------------------------------------------
# Lifecycle / executor matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("executor,n_jobs", EXECUTOR_MATRIX)
def test_grid_completes_all_combinations(
    sweep_harness: SweepHarness, executor: str, n_jobs: int,
) -> None:
    sweep_harness.run_programmatic(
        fn=pf.quadratic,
        parameters={"x": [1, 2, 3, 4]},
        executor=executor,
        n_jobs=n_jobs,
    )
    _assert_invariants(sweep_harness, expected_trial_count=4)
    study = sweep_harness.load_optuna_study()
    complete = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    assert len(complete) == 4


@pytest.mark.parametrize("executor,n_jobs", EXECUTOR_MATRIX)
def test_random_n_runs_cap(
    sweep_harness: SweepHarness, executor: str, n_jobs: int,
) -> None:
    sweep_harness.run_programmatic(
        fn=pf.quadratic,
        parameters={"x": [1, 2, 3, 4, 5, 6, 7, 8]},
        algorithm="random",
        spec={"n_runs": 3},
        executor=executor,
        n_jobs=n_jobs,
    )
    _assert_invariants(sweep_harness, expected_trial_count=3)


def test_metric_returned_directly(sweep_harness: SweepHarness) -> None:
    sweep_harness.run_programmatic(
        fn=pf.quadratic,
        parameters={"x": [1, 2, 3, 4]},
        spec={"direction": "minimize"},
    )
    study = sweep_harness.load_optuna_study()
    assert study.best_value == 0.0
    assert int(study.best_params["x"]) == 3


def test_metric_logged_via_mlflow(sweep_harness: SweepHarness) -> None:
    sweep_harness.run_programmatic(
        fn=pf.log_metric_fn,
        parameters={"x": [1, 2, 3, 4]},
        spec={"direction": "minimize", "metric": "loss"},
    )
    study = sweep_harness.load_optuna_study()
    assert study.best_value == 0.0
    assert int(study.best_params["x"]) == 3


def test_failing_trial_without_abort(sweep_harness: SweepHarness) -> None:
    sweep_harness.run_programmatic(
        fn=pf.failing_when_x_is_5,
        parameters={"x": [1, 5, 7]},
        spec={"max_retry": 1},
    )

    # All three trial runs are present in MLflow; the failing one is FAILED.
    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    statuses = {int(r.data.params.get("x", "0")): r.info.status for r in trial_runs
                if "x" in r.data.params}
    # x=5 should have produced (max_retry+1) failed runs.
    failed_runs = [r for r in trial_runs if r.info.status == "FAILED"]
    finished_runs = [r for r in trial_runs if r.info.status == "FINISHED"]
    assert len(failed_runs) >= 1, "Expected at least one FAILED run"
    assert len(finished_runs) == 2, f"Expected 2 FINISHED runs, got {len(finished_runs)}"


def test_failing_trial_with_abort_on_fail(sweep_harness: SweepHarness) -> None:
    with pytest.raises(Exception):  # TrialRunAbortError, but it's internal
        sweep_harness.run_programmatic(
            fn=pf.failing_when_x_is_5,
            parameters={"x": [5, 1, 7]},  # x=5 first so it aborts immediately
            abort_on_fail=True,
        )

    parent_id = sweep_harness.get_parent_run_id()
    assert parent_id is not None
    parent = sweep_harness.mlflow_client().get_run(parent_id)
    assert parent.info.status == "FAILED"


# ---------------------------------------------------------------------------
# Concurrency-specific
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("executor,n_jobs", [
    pytest.param("thread", 4, id="thread-n4"),
    pytest.param("process", 2, id="process-n2"),
])
def test_no_metric_crosstalk(
    sweep_harness: SweepHarness, executor: str, n_jobs: int,
) -> None:
    """Each trial's `loss` must equal x^2 for that trial's `x` — proves no
    cross-thread/process metric attribution bugs."""
    grid = [1, 2, 3, 4, 5, 6, 7, 8]
    sweep_harness.run_programmatic(
        fn=pf.slow_quadratic,
        parameters={"x": grid},
        executor=executor,
        n_jobs=n_jobs,
        spec={"direction": "minimize", "metric": "loss"},
    )
    _assert_invariants(sweep_harness, expected_trial_count=len(grid))
    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    for run in trial_runs:
        x = float(run.data.params["x"])
        loss = run.data.metrics["loss"]
        assert loss == x * x, (
            f"Run {run.info.run_id} has loss={loss}, expected {x*x} (x={x})"
        )


def test_process_pool_actually_uses_processes(sweep_harness: SweepHarness) -> None:
    sweep_harness.run_programmatic(
        fn=pf.slow_quadratic,
        parameters={"x": [1, 2, 3, 4, 5, 6, 7, 8]},
        executor="process",
        n_jobs=2,
        spec={"direction": "minimize", "metric": "loss"},
    )
    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    pids = {int(r.data.params["worker_pid"]) for r in trial_runs
            if "worker_pid" in r.data.params}
    assert len(pids) >= 2, f"Expected >=2 worker pids, got {pids}"


def test_thread_pool_completes_without_deadlock(sweep_harness: SweepHarness) -> None:
    """Thread pool with sleeping trials should finish in reasonable time.

    Mostly catches deadlocks (file-lock contention, MLflow stack hazards).
    Strict speedup assertions are flaky because MLflow's sqlite logging
    serializes and dwarfs the GIL-released sleep budget.
    """
    start = time.time()
    sweep_harness.run_programmatic(
        fn=pf.slow_quadratic,
        parameters={"x": [1, 2, 3, 4, 5, 6, 7, 8]},
        executor="thread",
        n_jobs=4,
        spec={"direction": "minimize", "metric": "loss"},
    )
    elapsed = time.time() - start
    # 8 trials × 200ms sleep = 1.6s sequential; loose 4× upper bound catches
    # gross deadlocks without flaking on logging overhead.
    assert elapsed < 6.4, f"Wall time {elapsed:.2f}s suggests a deadlock"


# ---------------------------------------------------------------------------
# Contract / error path tests
# ---------------------------------------------------------------------------

def test_lambda_with_process_pool_raises_helpful_error(sweep_harness: SweepHarness) -> None:
    config = sweep_harness.build_config(parameters={"x": [1, 2]})
    with pytest.raises(ValueError, match="thread"):
        run_sweep(config, lambda **p: float(p["x"]), executor="process", n_jobs=2)

    # No parent run should have been created.
    assert sweep_harness.get_parent_run_id() is None


def test_user_violates_active_run_contract(sweep_harness: SweepHarness) -> None:
    with pytest.raises(RuntimeError, match="active-run stack"):
        sweep_harness.run_programmatic(
            fn=pf.violates_active_run_contract,
            parameters={"x": [1]},
            abort_on_fail=True,
        )


# ---------------------------------------------------------------------------
# allow_param_change semantics
# ---------------------------------------------------------------------------

def test_allow_param_change_runs_new_trials(sweep_harness: SweepHarness) -> None:
    """Migrate-then-continue: changed config produces new trials in one call."""
    sweep_harness.run_programmatic(fn=pf.quadratic, parameters={"x": [1, 2]})
    _assert_invariants(sweep_harness, expected_trial_count=2)

    sweep_harness.run_programmatic(
        fn=pf.quadratic,
        parameters={"x": [1, 2, 3, 4]},
        allow_param_change=True,
    )
    _assert_invariants(sweep_harness, expected_trial_count=4)


def test_allow_param_change_no_change_warns(
    sweep_harness: SweepHarness, caplog: pytest.LogCaptureFixture,
) -> None:
    sweep_harness.run_programmatic(fn=pf.quadratic, parameters={"x": [1, 2]})

    with caplog.at_level(logging.WARNING):
        sweep_harness.run_programmatic(
            fn=pf.quadratic,
            parameters={"x": [1, 2]},
            allow_param_change=True,
        )

    assert any("had no effect" in m for m in caplog.messages)


def test_param_change_without_flag_aborts(sweep_harness: SweepHarness) -> None:
    sweep_harness.run_programmatic(fn=pf.quadratic, parameters={"x": [1, 2]})

    with pytest.raises(SystemExit):
        sweep_harness.run_programmatic(
            fn=pf.quadratic,
            parameters={"x": [1, 2, 3]},
            allow_param_change=False,
        )
