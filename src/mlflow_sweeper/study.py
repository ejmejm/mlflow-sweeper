"""Custom Study class with file-locked ask() to prevent race conditions."""

from __future__ import annotations

import hashlib
import os

from filelock import FileLock
import optuna


class LockedStudy(optuna.Study):
    """Study class with file-locked ask() to prevent race conditions in parallel runs."""

    def __init__(self, study: optuna.Study, output_dir: str) -> None:
        """Wrap an existing Optuna study with file-locked ask().

        Args:
            study: The Optuna study to wrap.
            output_dir: Directory for lock files.
        """
        # Copy all internal state from the original study
        # This allows us to extend Study while preserving all its functionality
        self.__dict__.update(study.__dict__)
        self._wrapped_study = study
        self._output_dir = output_dir
        self._lock_path = self._get_lock_path()

    def _get_lock_path(self) -> str:
        """Get the lock path for this study."""
        study_name = self.study_name
        storage = str(self._storage)
        lock_id = hashlib.md5(f"{study_name}:{storage}".encode("utf-8")).hexdigest()
        # Ensure output directory exists
        os.makedirs(self._output_dir, exist_ok=True)
        return os.path.join(self._output_dir, f"study_ask_{lock_id}.lock")

    def ask(self) -> optuna.Trial:
        """Create a new trial with file locking to prevent race conditions."""
        # Acquire file lock to prevent race conditions in parallel runs
        # The lock ensures only one thread/process can call ask() at a time
        # Use a timeout to avoid deadlocks, but long enough to handle normal operations
        lock = FileLock(self._lock_path, timeout=300)
        with lock:
            # Call the parent class's ask method using the unbound method
            # Since we copied __dict__, self has all the same internal state
            # This ensures we use self's state (which shares storage with the original study)
            return optuna.Study.ask(self)


def wrap_study(study: optuna.Study, output_dir: str) -> LockedStudy:
    """Convert an Optuna study to a LockedStudy.

    Args:
        study: The Optuna study to wrap.
        output_dir: Directory for lock files.

    Returns:
        A LockedStudy instance wrapping the original study.
    """
    return LockedStudy(study, output_dir)
