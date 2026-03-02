"""Unified locking system for mlflow-sweeper.

Provides named locks with automatic backend selection:
- DatabaseLock: uses a SQLAlchemy table for coordination (SQLite, PostgreSQL, MySQL)
- FileLockWrapper: wraps filelock.FileLock for non-database storage
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

import sqlalchemy as sa
from filelock import FileLock

logger = logging.getLogger(__name__)

_metadata = sa.MetaData()
_locks_table = sa.Table(
    "mlflow_sweeper_locks",
    _metadata,
    sa.Column("name", sa.String(512), primary_key=True),
    sa.Column("holder", sa.String(64), nullable=False),
    sa.Column("acquired_at", sa.DateTime, nullable=False),
)


class Lock(ABC):
    """Abstract base for named locks."""

    @abstractmethod
    def acquire(self, timeout: float = -1) -> None:
        """Acquire the lock. timeout=-1 means wait forever."""

    @abstractmethod
    def release(self) -> None:
        """Release the lock."""

    def __enter__(self) -> Lock:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class DatabaseLock(Lock):
    """Named lock backed by a row in mlflow_sweeper_locks."""

    def __init__(self, name: str, engine: sa.engine.Engine, stale_timeout: float = 20.0) -> None:
        self._name = name
        self._engine = engine
        self._holder = uuid.uuid4().hex
        self._stale_timeout = stale_timeout
        self._acquired = False

    def _cleanup_stale(self, conn: Any) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._stale_timeout)
        result = conn.execute(
            _locks_table.delete().where(
                sa.and_(
                    _locks_table.c.name == self._name,
                    _locks_table.c.acquired_at < cutoff,
                )
            )
        )
        if result.rowcount > 0:
            logger.warning("Cleaned up %d stale lock(s) for '%s'.", result.rowcount, self._name)

    def acquire(self, timeout: float = -1) -> None:
        delay = 0.005
        max_delay = 0.05
        start = time.monotonic()
        attempts = 0

        while True:
            attempts += 1
            try:
                with self._engine.begin() as conn:
                    self._cleanup_stale(conn)
                    conn.execute(
                        _locks_table.insert().values(
                            name=self._name,
                            holder=self._holder,
                            acquired_at=datetime.now(timezone.utc),
                        )
                    )
                self._acquired = True
                return
            except sa.exc.IntegrityError:
                pass

            if timeout >= 0 and (time.monotonic() - start) >= timeout:
                raise TimeoutError(f"Could not acquire lock '{self._name}' within {timeout}s")

            jitter = delay * 0.1 * (hash(self._holder) % 10)
            time.sleep(delay + jitter)
            delay = min(delay * 2, max_delay)

    def release(self) -> None:
        if not self._acquired:
            return
        with self._engine.begin() as conn:
            result = conn.execute(
                _locks_table.delete().where(
                    sa.and_(
                        _locks_table.c.name == self._name,
                        _locks_table.c.holder == self._holder,
                    )
                )
            )
            if result.rowcount == 0:
                logger.warning("Lock '%s' was not held by this process on release.", self._name)
        self._acquired = False


class FileLockWrapper(Lock):
    """Named lock backed by filelock.FileLock."""

    _UNSAFE_CHARS = re.compile(r'[:/\\<>"|?*\s]')

    def __init__(self, name: str, lock_dir: str) -> None:
        safe_name = self._UNSAFE_CHARS.sub("_", name)
        self._lock = FileLock(os.path.join(lock_dir, f"{safe_name}.lock"))

    def acquire(self, timeout: float = -1) -> None:
        self._lock.acquire(timeout=timeout if timeout >= 0 else None)

    def release(self) -> None:
        self._lock.release()


class LockManager:
    """Factory for named locks. Auto-selects backend from storage URIs."""

    _DB_PREFIXES = ("sqlite://", "postgresql://", "postgresql+", "mysql://", "mysql+")

    def __init__(
        self,
        mlflow_storage: str,
        optuna_storage: str,
        lock_dir: str | None = None,
        stale_timeout: float = 20.0,
    ) -> None:
        self._stale_timeout = stale_timeout
        self._engine: sa.engine.Engine | None = None
        self._lock_dir = lock_dir

        if self._is_db_uri(mlflow_storage):
            self._backend = "database"
            self._db_uri = mlflow_storage
        elif self._is_db_uri(optuna_storage):
            self._backend = "database"
            self._db_uri = optuna_storage
        else:
            self._backend = "file"
            self._db_uri = None

        if self._backend == "file":
            if lock_dir is None:
                raise ValueError("lock_dir is required when no database storage URI is available")
            os.makedirs(lock_dir, exist_ok=True)
            logger.warning(
                "Neither mlflow_storage ('%s') nor optuna_storage ('%s') is a database URI. "
                "Falling back to file-based locks in '%s'. "
                "This only works when all agents share the same filesystem.",
                mlflow_storage, optuna_storage, lock_dir,
            )

    @classmethod
    def _is_db_uri(cls, uri: str) -> bool:
        return any(uri.startswith(prefix) for prefix in cls._DB_PREFIXES)

    def _get_engine(self) -> sa.engine.Engine:
        if self._engine is None:
            assert self._db_uri is not None
            kwargs: dict[str, Any] = {}
            if self._db_uri.startswith("sqlite://"):
                kwargs["connect_args"] = {"timeout": 30}
            self._engine = sa.create_engine(self._db_uri, **kwargs)
            try:
                _metadata.create_all(self._engine, checkfirst=True)
            except sa.exc.OperationalError:
                # Another process may have created the table concurrently.
                # Verify it exists; if so, the race was harmless.
                insp = sa.inspect(self._engine)
                if not insp.has_table(_locks_table.name):
                    raise
        return self._engine

    def lock(self, name: str) -> Lock:
        """Create a named lock. Only same-name locks contend with each other."""
        if self._backend == "database":
            return DatabaseLock(name, self._get_engine(), self._stale_timeout)
        assert self._lock_dir is not None
        return FileLockWrapper(name, self._lock_dir)

