"""Minimal script that exits with code 1 when should_fail=true."""

from __future__ import annotations

import sys


def main() -> None:
    params = {}
    for token in sys.argv[1:]:
        if "=" in token:
            name, value = token.split("=", 1)
            params[name] = value

    if params.get("should_fail") == "true":
        sys.exit(1)


if __name__ == "__main__":
    main()
