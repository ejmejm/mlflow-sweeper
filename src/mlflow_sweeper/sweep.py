"""CLI entrypoint for mlflow-sweeper.

Code is split into:
- `mlflow_sweeper.config`: config + parameter parsing/validation
- `mlflow_sweeper.runner`: sweep execution/orchestration
"""

from __future__ import annotations

import logging

from mlflow_sweeper.config import load_configs, parse_args, validate_config
from mlflow_sweeper.runner import delete_sweep, run_sweep


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        logging.DEBUG: "\033[90m",  # grey
        logging.INFO: "\033[38;5;141m",  # purple/blue
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 (format is fine here)
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure colored, human-friendly logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def main() -> None:
    """CLI entrypoint."""
    configure_logging()

    args = parse_args()

    configs = load_configs(args.config)

    if args.delete:
        for config in configs:
            logger.info("Deleting sweep: %s/%s...", config.experiment, config.sweep_name)
            delete_sweep(config)
        return

    for config_path, config in zip(args.config, configs, strict=True):
        logger.info("Validating config: %s", config_path)
        validate_config(config)

    for config in configs:
        run_sweep(args, config)


if __name__ == "__main__":
    main()

