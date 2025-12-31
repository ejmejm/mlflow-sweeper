"""Backward-compatible wrapper for the packaged CLI.

Prefer installing the package and running:

    mlflow-sweep path/to/sweep.yaml
"""

from __future__ import annotations

import pathlib
import sys


def main() -> None:
    # Allow running `python sweep.py` from the repo without installing by
    # adding the local `src/` directory to the import path.
    repo_root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))

    from mlflow_sweeper.sweep import main as packaged_main

    packaged_main()


if __name__ == "__main__":
    main()