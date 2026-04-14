"""Top-level callables for tests of run_sweep's programmatic path.

These functions live in a real importable module so process-pool tests
can reference them by ``module + qualname`` (closures/lambdas are
rejected by ``validate_fn_picklable``).
"""

from __future__ import annotations

import os
import time

import mlflow


def quadratic(**params: object) -> float:
    """Returns (x - 3)^2; metric returned directly."""
    x = float(params["x"])
    return (x - 3.0) ** 2


def log_metric_fn(**params: object) -> None:
    """Logs ``loss`` to MLflow and returns None (metric resolved via MLflow)."""
    x = float(params["x"])
    mlflow.log_metric("loss", (x - 3.0) ** 2)


def failing_when_x_is_5(**params: object) -> float:
    """Raises for x==5, otherwise returns float(x)."""
    x = int(params["x"])
    if x == 5:
        raise RuntimeError("boom")
    return float(x)


def slow_quadratic(**params: object) -> float:
    """Sleeps 200ms, logs ``x``, ``loss`` and ``worker_pid``, returns x^2.

    Logs ``x`` explicitly so the no-metric-crosstalk test can recover it
    from each MLflow run regardless of the ``log_params`` flag.
    """
    x = float(params["x"])
    time.sleep(0.2)
    mlflow.log_param("x", x)
    mlflow.log_metric("loss", x * x)
    mlflow.log_param("worker_pid", os.getpid())
    return x * x


def violates_active_run_contract(**params: object) -> float:
    """Calls mlflow.start_run() to test the runner's stack-depth check."""
    mlflow.start_run(nested=True)
    return float(params["x"])
