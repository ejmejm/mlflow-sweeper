"""Plot generation for completed mlflow-sweeper studies.

Generates interactive Plotly visualizations from sweep data and logs them
as MLflow artifacts on the parent run.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
if TYPE_CHECKING:
    import optuna
    from mlflow_sweeper.config import (
        PlotsConfig,
        SweepConfig,
    )


logger = logging.getLogger(__name__)

PLOTS_ARTIFACT_DIR = "plots"


def generate_plots(study: optuna.Study, config: SweepConfig) -> None:
    """Generate and log all default plots for a completed study."""
    if config.spec.metric is None:
        logger.info("No metric configured; skipping plot generation.")
        return

    plots_config = config.plots

    if "best_hyperparameters" in plots_config.enabled_plots:
        try:
            plot_best_hyperparameters(study, config, plots_config)
        except Exception:
            logger.warning("Failed to generate best_hyperparameters plot.", exc_info=True)
    else:
        logger.info("best_hyperparameters plot disabled; skipping.")

    if "sensitivity" in plots_config.enabled_plots:
        if config.algorithm != "grid":
            logger.warning(
                "Sensitivity plot is only supported for grid sweeps "
                "(current algorithm: '%s'); skipping.",
                config.algorithm,
            )
        else:
            try:
                plot_sensitivity(config, plots_config)
            except Exception:
                logger.warning("Failed to generate sensitivity plot.", exc_info=True)
    else:
        logger.info("sensitivity plot disabled; skipping.")


def _get_varying_param_names(config: SweepConfig) -> list[str]:
    """Return param names that vary (not AtomicParam with a single fixed value)."""
    from mlflow_sweeper.config import AtomicParam
    return [
        name for name, spec in config.param_specs.items()
        if not isinstance(spec, AtomicParam)
    ]


def _try_numeric(value: str) -> int | float | str:
    """Try to convert a string param value to a numeric type."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


_SUPERSCRIPT_MAP = str.maketrans(
    "-0123456789",
    "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
)


def _is_power_of_2(value: object) -> bool:
    """Check if a value is a positive power of 2 (including e.g. 0.25 = 2**-2)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return False
    if fval <= 0:
        return False
    exp = math.log2(fval)
    return abs(exp - round(exp)) < 1e-9


def _format_power_of_2(value: object) -> str:
    """Format a power-of-2 value as '2x' with Unicode superscripts."""
    exp = round(math.log2(float(value)))
    return "2" + str(exp).translate(_SUPERSCRIPT_MAP)


def _detect_pow2_params(df: pd.DataFrame, param_names: list[str]) -> set[str]:
    """Return param names where ALL non-null values are powers of 2."""
    pow2_params: set[str] = set()
    for name in param_names:
        if name not in df.columns:
            continue
        values = df[name].dropna().unique()
        if len(values) > 0 and all(_is_power_of_2(v) for v in values):
            pow2_params.add(name)
    return pow2_params


def _fmt_val(value: object, param: str, pow2_params: set[str]) -> str:
    """Format a parameter value, using power-of-2 notation if applicable."""
    if param in pow2_params and _is_power_of_2(value):
        return _format_power_of_2(value)
    return str(value)


def _trials_to_dataframe(
    config: SweepConfig,
    metric_names: list[str],
) -> pd.DataFrame:
    """Build a DataFrame from MLflow child runs with params + metric columns.

    Queries direct children of the active parent run (via mlflow.parentRunId tag)
    for params and metrics. Only FINISHED runs are included.
    """
    parent_run = mlflow.active_run()
    assert parent_run is not None
    parent_run_id = parent_run.info.run_id

    client = MlflowClient(tracking_uri=config.mlflow_storage)
    experiment = client.get_experiment_by_name(config.experiment)
    if experiment is None:
        return pd.DataFrame()

    child_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f'tags.mlflow.parentRunId = "{parent_run_id}"'
            ' and attributes.status = "FINISHED"'
        ),
    )

    param_names = list(config.param_specs.keys())
    rows = []
    for run in child_runs:
        row: dict[str, object] = {}
        for name in param_names:
            if name in run.data.params:
                row[name] = _try_numeric(run.data.params[name])
        for metric in metric_names:
            if metric in run.data.metrics:
                row[metric] = run.data.metrics[metric]
        rows.append(row)

    return pd.DataFrame(rows)


def _log_figure(fig: go.Figure, name: str) -> None:
    """Log a Plotly figure as an MLflow artifact."""
    mlflow.log_figure(fig, f"{PLOTS_ARTIFACT_DIR}/{name}")
    logger.info("Logged plot: %s", name)


def _log_html(html: str, name: str) -> None:
    """Log raw HTML as an MLflow artifact."""
    mlflow.log_text(html, f"{PLOTS_ARTIFACT_DIR}/{name}")
    logger.info("Logged plot: %s", name)


def _build_sensitivity_figure(
    df: pd.DataFrame,
    x_param: str,
    metric_name: str,
    hue_params: list[str],
    pow2_params: set[str] | None = None,
) -> go.Figure:
    """Build a single sensitivity figure for one (metric, split, tab) combination."""
    pow2_params = pow2_params or set()
    group_cols = [x_param] + hue_params
    grouped = df.groupby(group_cols, as_index=False)[metric_name].mean()

    if hue_params:
        if len(hue_params) == 1:
            hp = hue_params[0]
            grouped["_hue_label"] = (
                hp + "=" + grouped[hp].apply(
                    lambda v, _p=hp: _fmt_val(v, _p, pow2_params)
                )
            )
        else:
            labels = []
            for _, row in grouped[hue_params].iterrows():
                parts = [
                    f"{p}={_fmt_val(row[p], p, pow2_params)}" for p in hue_params
                ]
                labels.append(", ".join(parts))
            grouped["_hue_label"] = labels
        # Sort hue labels by underlying numeric values, not formatted strings.
        hue_rows = (
            grouped[hue_params + ["_hue_label"]]
            .drop_duplicates(subset=hue_params)
            .to_dict("records")
        )
        hue_rows.sort(
            key=lambda r: tuple(
                (isinstance(r[p], str), r[p]) for p in hue_params
            ),
        )
        hue_values = [r["_hue_label"] for r in hue_rows]
    else:
        grouped["_hue_label"] = ""
        hue_values = [""]

    x_values = sorted(
        grouped[x_param].unique(), key=lambda v: (isinstance(v, str), v),
    )
    x_positions = list(range(len(x_values)))
    x_map = dict(zip(x_values, x_positions))

    fig = go.Figure()

    for hue_val in hue_values:
        subset = grouped[grouped["_hue_label"] == hue_val] if hue_val else grouped

        xs = [x_map[v] for v in subset[x_param]]
        ys = subset[metric_name].tolist()

        sorted_pairs = sorted(zip(xs, ys))
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=hue_val if hue_val else x_param,
        ))

    fig.update_layout(
        xaxis=dict(
            title=x_param,
            tickvals=x_positions,
            ticktext=[_fmt_val(v, x_param, pow2_params) for v in x_values],
        ),
        yaxis=dict(title=metric_name),
        margin=dict(l=60, r=20, t=20, b=60),
    )

    return fig


def _build_interactive_html(
    figures: dict[str, go.Figure],
    dims: list[tuple[str, list[str]]],
    title: str,
) -> str:
    """Build an HTML page with dropdowns + optional tabs for interactive plots.

    dims: list of (label, values) tuples. The last dim is rendered as tabs
          (unless it has only 1 value, in which case the tab bar is hidden).
          Other dims are rendered as dropdowns (hidden when only 1 value).
    figures: keyed by dash-separated indices matching dims order.
    """
    fig_entries = []
    for key, fig in figures.items():
        fig_entries.append(f"{json.dumps(key)}:{fig.to_json()}")
    fig_data_js = "{" + ",".join(fig_entries) + "}"

    dropdown_html_parts = []
    dims_js_parts = []

    for i, (label, values) in enumerate(dims[:-1]):
        if len(values) > 1:
            options = "".join(
                f'<option value="{j}">{v}</option>' for j, v in enumerate(values)
            )
            dropdown_html_parts.append(
                f'<div class="dropdown-row">'
                f'<label for="dd-{i}">{label}:</label> '
                f'<select id="dd-{i}" onchange="update()">{options}</select>'
                f"</div>"
            )
            dims_js_parts.append(f'{{type:"dropdown",id:"dd-{i}"}}')
        else:
            dims_js_parts.append("{type:\"fixed\",value:0}")

    tab_label, tab_values = dims[-1]
    show_tabs = len(tab_values) > 1

    if show_tabs:
        dims_js_parts.append('{type:"tab"}')
        tab_buttons = []
        for i, name in enumerate(tab_values):
            active = " active" if i == 0 else ""
            tab_buttons.append(
                f'<div class="tab{active}" onclick="showTab({i})">{name}</div>'
            )
        tab_buttons_html = "".join(tab_buttons)
    else:
        dims_js_parts.append("{type:\"fixed\",value:0}")

    dropdown_html = "\n    ".join(dropdown_html_parts)
    dims_js = "[" + ",".join(dims_js_parts) + "]"

    tab_bar_html = ""
    plot_area_style = ""
    if show_tabs:
        tab_bar_html = f"""<div class="tab-bar">
    {tab_buttons_html}
  </div>"""
        plot_area_style = ' class="plot-area"'

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 20px;
    }}
    h3 {{ margin-bottom: 16px; color: #333; }}
    .dropdown-row {{
      margin-bottom: 8px;
    }}
    .dropdown-row label {{
      font-weight: 600;
      margin-right: 4px;
    }}
    .dropdown-row select {{
      padding: 4px 8px;
      font-size: 14px;
    }}
    .tab-bar {{
      display: flex;
      border-bottom: 2px solid #ddd;
      margin-bottom: 0;
      margin-top: 12px;
    }}
    .tab {{
      padding: 10px 20px;
      cursor: pointer;
      border: 1px solid transparent;
      border-bottom: none;
      background: #f5f5f5;
      color: #666;
      margin-right: 2px;
      border-radius: 6px 6px 0 0;
      font-size: 14px;
      transition: background 0.15s, color 0.15s;
    }}
    .tab:hover {{ background: #e8e8e8; }}
    .tab.active {{
      background: white;
      color: #333;
      font-weight: 600;
      border-color: #ddd;
      border-bottom: 2px solid white;
      margin-bottom: -2px;
    }}
    .plot-area {{
      border: 1px solid #ddd;
      border-top: none;
      padding: 16px;
    }}
  </style>
</head>
<body>
  <h3>{title}</h3>
  {dropdown_html}
  {tab_bar_html}
  <div{plot_area_style}>
    <div id="plot" style="width:100%;height:500px;"></div>
  </div>
  <script>
    var figData = {fig_data_js};
    var dims = {dims_js};
    var currentTab = 0;
    var lastKey = null;

    function getKey() {{
      return dims.map(function(d) {{
        if (d.type === "tab") return currentTab;
        if (d.type === "dropdown") return document.getElementById(d.id).selectedIndex;
        return d.value;
      }}).join("-");
    }}

    function update() {{
      var key = getKey();
      if (key === lastKey) return;
      lastKey = key;
      var spec = figData[key];
      Plotly.react("plot", spec.data, spec.layout, {{responsive: true}});
    }}

    function showTab(idx) {{
      currentTab = idx;
      document.querySelectorAll(".tab").forEach(function(t, i) {{
        t.classList.toggle("active", i === idx);
      }});
      update();
    }}

    update();
  </script>
</body>
</html>"""


def _build_table_figure(
    df: pd.DataFrame,
    metric_name: str,
    ascending: bool,
) -> go.Figure:
    """Build a Plotly table figure from a prepared DataFrame."""
    direction_label = "lowest" if ascending else "highest"
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in df.columns],
            fill_color="rgb(55, 83, 109)",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color="lavender",
            align="center",
        ),
    )])
    fig.update_layout(
        title=f"Hyperparameter Ranking ({direction_label} {metric_name} is best)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_best_hyperparameters(
    study: optuna.Study,
    config: SweepConfig,
    plots_config: PlotsConfig,
) -> None:
    """Generate an interactive table of trials ranked by metric value."""
    metric_name = config.spec.metric
    assert metric_name is not None

    plot_config = plots_config.best_hyperparameters
    metric_names = plots_config.metrics or [metric_name]
    split_by = plots_config.split_by or []

    df = _trials_to_dataframe(config, metric_names)
    if df.empty:
        return

    from optuna.study import StudyDirection
    ascending = config.spec.direction != StudyDirection.MAXIMIZE
    pow2_params = _detect_pow2_params(df, list(config.param_specs.keys()))
    metric_set = set(metric_names)

    def prepare_table(sub_df: pd.DataFrame, sort_metric: str) -> pd.DataFrame:
        sub_df = sub_df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)
        if plot_config.top_n is not None:
            sub_df = sub_df.head(plot_config.top_n)
        # Drop split_by columns (shown in dropdown) and constant params.
        sub_df = sub_df.drop(columns=[c for c in split_by if c in sub_df.columns])
        param_cols = [c for c in sub_df.columns if c not in metric_set]
        constant = [c for c in param_cols if sub_df[c].nunique(dropna=False) <= 1]
        sub_df = sub_df.drop(columns=constant)
        for col in pow2_params:
            if col in sub_df.columns:
                sub_df[col] = sub_df[col].apply(
                    lambda v: _format_power_of_2(v) if _is_power_of_2(v) else v
                )
        sub_df.insert(0, "rank", range(1, len(sub_df) + 1))
        return sub_df

    # Simple case: single metric, no split_by → plain Plotly figure.
    if len(metric_names) == 1 and not split_by:
        table_df = prepare_table(df, metric_name)
        fig = _build_table_figure(table_df, metric_name, ascending)
        _log_figure(fig, "best_hyperparameters.html")
        return

    # Interactive case: metric dropdown and/or split_by dropdowns.
    split_values: dict[str, list] = {}
    for param in split_by:
        split_values[param] = sorted(
            df[param].unique(), key=lambda v: (isinstance(v, str), v),
        )

    figures: dict[str, go.Figure] = {}
    split_ranges = [range(len(split_values[p])) for p in split_by]
    split_combos = list(itertools.product(*split_ranges)) if split_by else [()]

    for m_idx, metric in enumerate(metric_names):
        if metric not in df.columns:
            continue
        for split_combo in split_combos:
            sub_df = df
            for i, param in enumerate(split_by):
                sub_df = sub_df[sub_df[param] == split_values[param][split_combo[i]]]

            key_parts = [str(m_idx)]
            for s_idx in split_combo:
                key_parts.append(str(s_idx))
            key_parts.append("0")  # dummy tab
            key = "-".join(key_parts)

            table_df = prepare_table(sub_df, metric)
            figures[key] = _build_table_figure(table_df, metric, ascending)

    dims: list[tuple[str, list[str]]] = []
    dims.append(("Metric", metric_names))
    for param in split_by:
        dims.append((
            param,
            [_fmt_val(v, param, pow2_params) for v in split_values[param]],
        ))
    dims.append(("_", [""]))  # dummy tab (hidden)

    html = _build_interactive_html(
        figures, dims,
        title="Hyperparameter Ranking",
    )
    _log_html(html, "best_hyperparameters.html")


def plot_sensitivity(
    config: SweepConfig,
    plots_config: PlotsConfig,
) -> None:
    """Generate a sensitivity plot with dropdowns for metrics/splits and tabs per param."""
    metric_name = config.spec.metric
    assert metric_name is not None

    plot_config = plots_config.sensitivity
    metric_names = plots_config.metrics or [metric_name]
    split_by = plots_config.split_by or []

    df = _trials_to_dataframe(config, metric_names)
    if df.empty:
        return

    varying_params = _get_varying_param_names(config)
    if not varying_params:
        logger.info("No varying parameters; skipping sensitivity plot.")
        return

    average_over = plot_config.average_over
    if average_over is None:
        average_over = [p for p in varying_params if p.lower() == "seed"]

    hue_params = plot_config.hue or []

    tab_params = plot_config.params
    if tab_params is None:
        excluded = set(average_over) | set(hue_params) | set(split_by)
        tab_params = [p for p in varying_params if p not in excluded]

    if not tab_params:
        logger.info("No parameters to plot tabs for; skipping sensitivity plot.")
        return

    pow2_params = _detect_pow2_params(df, varying_params)

    split_values: dict[str, list] = {}
    for param in split_by:
        split_values[param] = sorted(
            df[param].unique(), key=lambda v: (isinstance(v, str), v),
        )

    figures: dict[str, go.Figure] = {}
    split_ranges = [range(len(split_values[p])) for p in split_by]
    split_combos = list(itertools.product(*split_ranges)) if split_by else [()]

    for m_idx, metric in enumerate(metric_names):
        if metric not in df.columns:
            continue
        for split_combo in split_combos:
            sub_df = df
            for i, param in enumerate(split_by):
                sub_df = sub_df[sub_df[param] == split_values[param][split_combo[i]]]

            for t_idx, tab_param in enumerate(tab_params):
                key_parts = [str(m_idx)]
                for s_idx in split_combo:
                    key_parts.append(str(s_idx))
                key_parts.append(str(t_idx))
                key = "-".join(key_parts)

                figures[key] = _build_sensitivity_figure(
                    sub_df, tab_param, metric, hue_params, pow2_params,
                )

    if not figures:
        return

    dims: list[tuple[str, list[str]]] = []
    dims.append(("Metric", metric_names))
    for param in split_by:
        dims.append((
            param,
            [_fmt_val(v, param, pow2_params) for v in split_values[param]],
        ))
    dims.append(("Parameter", tab_params))

    html = _build_interactive_html(
        figures, dims,
        title="Hyperparameter Sensitivity",
    )
    _log_html(html, "sensitivity.html")
