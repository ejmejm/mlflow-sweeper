"""Process-pool execution for programmatic sweeps.

Drives a sweep across N independent worker processes that share the
configured Optuna storage and MLflow tracking URI.  Workers never create
the parent MLflow run — they tag every trial run with the
``mlflow.parentRunId`` set up by the main process.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from importlib import import_module
import logging
import multiprocessing
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import optuna


logger = logging.getLogger(__name__)


def validate_fn_picklable(fn: Callable[..., Any]) -> None:
    """Validate that ``fn`` is importable from a top-level module attribute.

    Raises a clear ``ValueError`` *before* any pool/worker is spawned, so
    the user sees a useful message instead of a low-level pickling error
    deep inside ``concurrent.futures``.
    """
    serialized = _serialize_fn(fn)  # raises with helpful message if invalid
    # Round-trip to confirm we can resolve back to the same object.
    resolved = _deserialize_fn(serialized)
    if resolved is not fn:
        raise ValueError(
            f"Process executor requires a top-level function; "
            f"got {fn!r} (resolves to a different object on import). "
            "Use executor='thread' for closures/lambdas."
        )


def _serialize_fn(fn: Callable[..., Any]) -> tuple[str, str]:
    """Convert ``fn`` to a (module, qualname) pair for cross-process re-import."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module is None or qualname is None or "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(
            f"Process executor requires a top-level function; got {fn!r}. "
            "Use executor='thread' for closures/lambdas."
        )
    return module, qualname


def _deserialize_fn(ref: tuple[str, str]) -> Callable[..., Any]:
    module_name, qualname = ref
    mod = import_module(module_name)
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _split_trials(n_trials: int | None, n_jobs: int) -> list[int | None]:
    """Divide ``n_trials`` across workers; ``None`` -> all-None (run-until-done)."""
    if n_trials is None:
        return [None] * n_jobs
    base, rem = divmod(n_trials, n_jobs)
    return [base + (1 if i < rem else 0) for i in range(n_jobs)]


def _worker_main(
    config: Any,                 # SweepConfig (avoid runner import cycle in type hint)
    fn_ref: tuple[str, str],
    parent_run_id: str,
    n_trials: int | None,
    args_dict: dict[str, Any],
) -> None:
    """Entry point executed inside each worker process."""
    # Imported inside the worker so the module is available after fork/spawn.
    from mlflow_sweeper.optimize import optimize_study
    from mlflow_sweeper.runner import (
        TrialRunError,
        _callable_body,
        _callable_label_fn,
        _run_trial_with_mlflow,
        get_study_name,
        make_sampler,
    )

    mlflow.set_tracking_uri(config.mlflow_storage)
    mlflow.set_experiment(config.experiment)

    study = optuna.load_study(
        study_name=get_study_name(config),
        storage=config.optuna_storage,
        sampler=make_sampler(config),
    )

    fn = _deserialize_fn(fn_ref)
    mlflow_client = MlflowClient(tracking_uri=config.mlflow_storage)
    args = argparse.Namespace(**args_dict)

    run_fn = partial(
        _run_trial_with_mlflow,
        config=config,
        mlflow_client=mlflow_client,
        args=args,
        parent_run_id=parent_run_id,
        body=_callable_body(fn, args),
        label_fn=_callable_label_fn(fn),
    )
    optimize_study(study, run_fn, n_trials=n_trials, n_jobs=1, catch=(TrialRunError,))


def run_with_process_pool(
    *,
    config: Any,                 # SweepConfig
    fn: Callable[..., Any],
    n_jobs: int,
    n_trials: int | None,
    parent_run_id: str,
    args_namespace: argparse.Namespace,
) -> None:
    """Run ``optimize_study`` across ``n_jobs`` worker processes."""
    fn_ref = _serialize_fn(fn)
    chunks = _split_trials(n_trials, n_jobs)
    args_dict = vars(args_namespace)

    # ``spawn`` keeps each worker's MLflow / Optuna state clean (no inherited
    # active runs, no shared sqlite connections).  Required on Windows; safe
    # everywhere else.
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
        futures = [
            pool.submit(_worker_main, config, fn_ref, parent_run_id, chunk, args_dict)
            for chunk in chunks
        ]
        try:
            for f in as_completed(futures):
                f.result()  # propagates TrialRunAbortError, KeyboardInterrupt, etc.
        except BaseException:
            # Cancel any pending futures on the way out.
            for f in futures:
                f.cancel()
            raise
