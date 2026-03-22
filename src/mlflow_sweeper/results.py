"""CLI tool to query and display sweep results from MLflow.

Prints a ranked table of trials for each sweep in an experiment,
showing varying parameters and key metrics. Useful for quickly
checking results without opening the MLflow UI.
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display sweep results from MLflow.",
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="MLflow experiment name (e.g. 'streaming-ipc-sweeps').",
    )
    parser.add_argument(
        "-s", "--sweep",
        type=str,
        default=None,
        help="Filter to a specific sweep_name (supports substring match).",
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        default=None,
        help="Sort by this metric (default: first accuracy-like metric found, or first metric).",
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=None,
        help="Show only the top N trials per sweep.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Defaults to $MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        default=False,
        help="Sort metric in ascending order (default: descending).",
    )
    return parser.parse_args()


def _try_numeric(value: str) -> int | float | str:
    """Try to convert a string to int, then float, else return as-is."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _pick_sort_metric(df: pd.DataFrame, metric_hint: str | None) -> str | None:
    """Pick the best metric column to sort by."""
    metric_cols = [c for c in df.columns if not c.startswith("_")]
    # Filter to only actual metric columns (non-param)
    metric_cols = [c for c in metric_cols if c not in df.attrs.get("param_cols", [])]

    if metric_hint:
        # Exact match first
        if metric_hint in metric_cols:
            return metric_hint
        # Substring match
        matches = [c for c in metric_cols if metric_hint in c]
        if matches:
            return matches[0]
        return None

    # Prefer accuracy-like metrics
    for col in metric_cols:
        if "accuracy" in col.lower() and "asymptotic" not in col.lower():
            return col
    for col in metric_cols:
        if "accuracy" in col.lower():
            return col
    return metric_cols[0] if metric_cols else None


def get_sweep_results(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
) -> pd.DataFrame:
    """Build a DataFrame of trial results for one sweep."""
    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f'tags.mlflow.parentRunId = "{parent_run_id}"'
            ' and attributes.status = "FINISHED"'
        ),
    )

    if not children:
        return pd.DataFrame()

    rows = []
    for run in children:
        row: dict[str, object] = {}
        for k, v in run.data.params.items():
            if k == "full_command":
                continue
            row[k] = _try_numeric(v)
        for k, v in run.data.metrics.items():
            row[k] = round(v, 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Track which columns are params vs metrics.
    param_cols = [c for c in df.columns if c in children[0].data.params and c != "full_command"]
    metric_cols = [c for c in df.columns if c not in param_cols]
    df.attrs["param_cols"] = param_cols
    df.attrs["metric_cols"] = metric_cols

    # Drop constant param columns.
    for col in list(param_cols):
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])
            param_cols = [c for c in param_cols if c != col]
            df.attrs["param_cols"] = param_cols

    return df


def main() -> None:
    from mlflow_sweeper.sweep import configure_logging
    configure_logging()

    args = parse_args()

    tracking_uri = args.tracking_uri
    if tracking_uri is None:
        import os
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            logger.error("No tracking URI provided. Set $MLFLOW_TRACKING_URI or use --tracking-uri.")
            sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        logger.error("Experiment '%s' not found.", args.experiment)
        sys.exit(1)

    # Find parent (sweep) runs.
    parent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string='tags.optuna_study_name LIKE "%"',
    )

    if not parent_runs:
        logger.info("No sweeps found in experiment '%s'.", args.experiment)
        return

    # Filter by sweep name if requested.
    if args.sweep:
        parent_runs = [
            r for r in parent_runs
            if args.sweep in r.data.tags.get("sweep_name", "")
        ]
        if not parent_runs:
            logger.info("No sweeps matching '%s'.", args.sweep)
            return

    parent_runs.sort(key=lambda r: r.data.tags.get("sweep_name", ""))

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)

    for parent in parent_runs:
        sweep_name = parent.data.tags.get("sweep_name", "?")
        df = get_sweep_results(client, experiment.experiment_id, parent.info.run_id)

        if df.empty:
            print(f"=== {sweep_name} === (no finished runs)")
            print()
            continue

        sort_col = _pick_sort_metric(df, args.metric)
        if sort_col:
            df = df.sort_values(sort_col, ascending=args.ascending)

        if args.top_n:
            df = df.head(args.top_n)

        n_runs = len(df)
        print(f"=== {sweep_name} ({n_runs} runs) ===")
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
