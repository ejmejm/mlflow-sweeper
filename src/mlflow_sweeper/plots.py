"""Plot generation for completed mlflow-sweeper studies.

Generates interactive Plotly visualizations from sweep data and logs them
as MLflow artifacts on the parent run.
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
if TYPE_CHECKING:
    import optuna
    from mlflow_sweeper.config import (
        BestHyperparametersPlotConfig,
        SensitivityPlotConfig,
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
    plot_best_hyperparameters(study, config, plots_config.best_hyperparameters)

    # Sensitivity plots require controlled experiments (all other params held constant
    # while one varies), which is only meaningful for grid sweeps.
    if config.algorithm == "grid":
        plot_sensitivity(study, config, plots_config.sensitivity)


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
    best_trial: optuna.trial.FrozenTrial,
) -> go.Figure:
    """Build a single sensitivity figure for one (metric, split, tab) combination."""
    group_cols = [x_param] + hue_params
    grouped = df.groupby(group_cols, as_index=False)[metric_name].mean()

    if hue_params:
        if len(hue_params) == 1:
            grouped["_hue_label"] = (
                hue_params[0] + "=" + grouped[hue_params[0]].astype(str)
            )
        else:
            labels = []
            for _, row in grouped[hue_params].iterrows():
                parts = [f"{p}={row[p]}" for p in hue_params]
                labels.append(", ".join(parts))
            grouped["_hue_label"] = labels
        hue_values = sorted(grouped["_hue_label"].unique())
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
            ticktext=[str(v) for v in x_values],
        ),
        yaxis=dict(title=metric_name),
        margin=dict(l=60, r=20, t=20, b=60),
    )

    best_x_value = best_trial.params.get(x_param)
    if best_x_value is not None and best_x_value in x_map:
        fig.add_vline(
            x=x_map[best_x_value],
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"best ({best_x_value})",
            annotation_font_color="green",
        )

    return fig


def _build_sensitivity_html(
    figures: dict[str, go.Figure],
    dims: list[tuple[str, list[str]]],
    title: str,
) -> str:
    """Build an HTML page with dropdowns + tabs for multi-dimensional sensitivity plots.

    dims: list of (label, values) tuples. Last dim is always tabs, others are dropdowns
          (hidden when only 1 value).
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
    dims_js_parts.append('{type:"tab"}')

    tab_buttons = []
    for i, name in enumerate(tab_values):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<div class="tab{active}" onclick="showTab({i})">{name}</div>'
        )

    dropdown_html = "\n    ".join(dropdown_html_parts)
    tab_buttons_html = "".join(tab_buttons)
    dims_js = "[" + ",".join(dims_js_parts) + "]"

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
  <div class="tab-bar">
    {tab_buttons_html}
  </div>
  <div class="plot-area">
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


def plot_best_hyperparameters(
    study: optuna.Study,
    config: SweepConfig,
    plot_config: BestHyperparametersPlotConfig,
) -> None:
    """Generate an interactive table of trials ranked by metric value."""
    metric_name = config.spec.metric
    assert metric_name is not None

    df = _trials_to_dataframe(config, [metric_name])
    if df.empty:
        return

    from optuna.study import StudyDirection
    ascending = config.spec.direction != StudyDirection.MAXIMIZE
    df = df.sort_values(metric_name, ascending=ascending).reset_index(drop=True)

    if plot_config.top_n is not None:
        df = df.head(plot_config.top_n)

    # Drop parameters that are constant across all runs.
    param_cols = [c for c in df.columns if c != metric_name]
    constant = [c for c in param_cols if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant)

    df.insert(0, "rank", range(1, len(df) + 1))

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

    direction_label = "lowest" if ascending else "highest"
    fig.update_layout(
        title=f"Hyperparameter Ranking ({direction_label} {metric_name} is best)",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    _log_figure(fig, "best_hyperparameters.html")


def plot_sensitivity(
    study: optuna.Study,
    config: SweepConfig,
    plot_config: SensitivityPlotConfig,
) -> None:
    """Generate a sensitivity plot with dropdowns for metrics/splits and tabs per param."""
    metric_name = config.spec.metric
    assert metric_name is not None

    metric_names = plot_config.metrics or [metric_name]
    split_by = plot_config.split_by or []

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

    best_trial = study.best_trial

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
                    sub_df, tab_param, metric, hue_params, best_trial,
                )

    if not figures:
        return

    dims: list[tuple[str, list[str]]] = []
    dims.append(("Metric", metric_names))
    for param in split_by:
        dims.append((param, [str(v) for v in split_values[param]]))
    dims.append(("Parameter", tab_params))

    html = _build_sensitivity_html(
        figures, dims,
        title="Hyperparameter Sensitivity",
    )
    _log_html(html, "sensitivity.html")
