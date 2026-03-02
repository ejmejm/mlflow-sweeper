"""Tests for the unified locking module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import tempfile
import threading

import pytest
import sqlalchemy as sa

from mlflow_sweeper.locking import (
    DatabaseLock,
    FileLockWrapper,
    LockManager,
    _locks_table,
    _metadata,
)


@pytest.fixture()
def db_engine():
    """Create a temporary SQLite database engine with the locks table."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        engine = sa.create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
        _metadata.create_all(engine, checkfirst=True)
        yield engine
        engine.dispose()


@pytest.fixture()
def lock_dir():
    """Create a temporary directory for file locks."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


### DatabaseLock ──────────────────────────────────────────────


def test_database_lock_acquire_release(db_engine):
    lock = DatabaseLock("test_lock", db_engine)
    lock.acquire()

    # Verify row exists
    with db_engine.connect() as conn:
        row = conn.execute(
            _locks_table.select().where(_locks_table.c.name == "test_lock")
        ).fetchone()
        assert row is not None

    lock.release()

    # Verify row is gone
    with db_engine.connect() as conn:
        row = conn.execute(
            _locks_table.select().where(_locks_table.c.name == "test_lock")
        ).fetchone()
        assert row is None


def test_database_lock_mutual_exclusion(db_engine):
    lock_a = DatabaseLock("shared", db_engine)
    lock_a.acquire()

    result = {}

    def try_acquire():
        lock_b = DatabaseLock("shared", db_engine)
        try:
            lock_b.acquire(timeout=0.3)
            result["acquired"] = True
            lock_b.release()
        except TimeoutError:
            result["timeout"] = True

    t = threading.Thread(target=try_acquire)
    t.start()
    t.join(timeout=5)

    assert result.get("timeout") is True
    lock_a.release()


def test_database_lock_different_names_independent(db_engine):
    lock_a = DatabaseLock("lock_a", db_engine)
    lock_a.acquire()

    result = {}

    def try_acquire_b():
        lock_b = DatabaseLock("lock_b", db_engine)
        try:
            lock_b.acquire(timeout=1.0)
            result["acquired"] = True
            lock_b.release()
        except TimeoutError:
            result["timeout"] = True

    t = threading.Thread(target=try_acquire_b)
    t.start()
    t.join(timeout=5)

    assert result.get("acquired") is True
    lock_a.release()


def test_database_lock_stale_cleanup(db_engine):
    # Insert a stale lock row directly
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=9999)
    with db_engine.begin() as conn:
        conn.execute(
            _locks_table.insert().values(
                name="stale_lock",
                holder="dead_process",
                acquired_at=stale_time,
            )
        )

    # Should succeed because stale lock gets cleaned up
    lock = DatabaseLock("stale_lock", db_engine, stale_timeout=60.0)
    lock.acquire(timeout=1.0)
    lock.release()


def test_database_lock_context_manager(db_engine):
    with DatabaseLock("ctx_lock", db_engine) as lock:
        with db_engine.connect() as conn:
            row = conn.execute(
                _locks_table.select().where(_locks_table.c.name == "ctx_lock")
            ).fetchone()
            assert row is not None

    # After exiting context, lock should be released
    with db_engine.connect() as conn:
        row = conn.execute(
            _locks_table.select().where(_locks_table.c.name == "ctx_lock")
        ).fetchone()
        assert row is None


def test_database_lock_release_on_exception(db_engine):
    try:
        with DatabaseLock("exc_lock", db_engine):
            raise ValueError("test error")
    except ValueError:
        pass

    # Lock should still be released
    with db_engine.connect() as conn:
        row = conn.execute(
            _locks_table.select().where(_locks_table.c.name == "exc_lock")
        ).fetchone()
        assert row is None


# ── FileLockWrapper ───────────────────────────────────────────


def test_file_lock_creates_file(lock_dir):
    lock = FileLockWrapper("simple_name", lock_dir)
    lock.acquire()
    assert os.path.exists(os.path.join(lock_dir, "simple_name.lock"))
    lock.release()


def test_file_lock_name_sanitization(lock_dir):
    lock = FileLockWrapper("study_init:exp/sweep", lock_dir)
    lock.acquire()
    assert os.path.exists(os.path.join(lock_dir, "study_init_exp_sweep.lock"))
    lock.release()


def test_file_lock_context_manager(lock_dir):
    with FileLockWrapper("ctx_test", lock_dir):
        assert os.path.exists(os.path.join(lock_dir, "ctx_test.lock"))


# ── LockManager ──────────────────────────────────────────────


def test_lock_manager_selects_database_for_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        mgr = LockManager(
            mlflow_storage=f"sqlite:///{db_path}",
            optuna_storage=f"sqlite:///{db_path}",
        )
        assert mgr._backend == "database"


def test_lock_manager_selects_database_for_postgresql():
    mgr = LockManager(
        mlflow_storage="postgresql://user:pass@host:5432/db",
        optuna_storage="sqlite:///test.db",
    )
    assert mgr._backend == "database"


def test_lock_manager_falls_back_to_optuna_storage():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "optuna.db")
        mgr = LockManager(
            mlflow_storage="http://mlflow-server:5000",
            optuna_storage=f"sqlite:///{db_path}",
            lock_dir=tmp,
        )
        assert mgr._backend == "database"
        assert mgr._db_uri == f"sqlite:///{db_path}"


def test_lock_manager_falls_back_to_file():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = LockManager(
            mlflow_storage="./mlruns",
            optuna_storage="./optuna",
            lock_dir=tmp,
        )
        assert mgr._backend == "file"


def test_lock_manager_file_requires_lock_dir():
    with pytest.raises(ValueError, match="lock_dir is required"):
        LockManager(
            mlflow_storage="./mlruns",
            optuna_storage="./optuna",
        )


def test_lock_manager_database_roundtrip():
    """End-to-end: LockManager creates DB locks that acquire/release correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        mgr = LockManager(
            mlflow_storage=f"sqlite:///{db_path}",
            optuna_storage=f"sqlite:///{db_path}",
        )
        with mgr.lock("roundtrip_test"):
            # Lock is held — verify via a second lock attempt with short timeout
            result = {}

            def try_acquire():
                lock = mgr.lock("roundtrip_test")
                try:
                    lock.acquire(timeout=0.3)
                    result["acquired"] = True
                    lock.release()
                except TimeoutError:
                    result["timeout"] = True

            t = threading.Thread(target=try_acquire)
            t.start()
            t.join(timeout=5)
            assert result.get("timeout") is True


def test_lock_manager_file_roundtrip():
    """End-to-end: LockManager creates file locks that work."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = LockManager(
            mlflow_storage="./mlruns",
            optuna_storage="./optuna",
            lock_dir=tmp,
        )
        with mgr.lock("file_test"):
            assert os.path.exists(os.path.join(tmp, "file_test.lock"))
