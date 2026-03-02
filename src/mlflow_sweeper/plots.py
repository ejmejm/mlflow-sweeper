"""Plot generation for completed mlflow-sweeper studies.

Generates interactive Plotly visualizations from Optuna study data after a sweep
completes and logs them as MLflow artifacts on the parent run.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
import plotly.graph_objects as go
from optuna.trial import TrialState

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

    completed = study.get_trials(states=[TrialState.COMPLETE])
    if len(completed) == 0:
        logger.warning("No completed trials; skipping plot generation.")
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


def _trials_to_dataframe(
    study: optuna.Study,
    metric_name: str,
) -> pd.DataFrame:
    """Build a DataFrame from completed trials with params + metric columns."""
    completed = study.get_trials(states=[TrialState.COMPLETE])
    rows = []
    for trial in completed:
        row = dict(trial.params)
        row[metric_name] = trial.value
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


def _build_tabbed_html(
    tab_figures: dict[str, go.Figure],
    title: str,
) -> str:
    """Build an HTML page with tab navigation between multiple Plotly figures."""
    tab_names = list(tab_figures.keys())

    tab_buttons = []
    tab_contents = []
    plot_specs = []

    for i, (name, fig) in enumerate(tab_figures.items()):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<div class="tab{active}" onclick="showTab({i})">{name}</div>',
        )
        tab_contents.append(
            f'<div id="tab-{i}" class="tab-content{active}">'
            f'<div id="plot-{i}" style="width:100%;height:500px;"></div>'
            f"</div>",
        )
        plot_specs.append(fig.to_json())

    plots_js = ",\n      ".join(plot_specs)

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
    .tab-bar {{
      display: flex;
      border-bottom: 2px solid #ddd;
      margin-bottom: 0;
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
    .tab-content {{
      display: none;
      border: 1px solid #ddd;
      border-top: none;
      padding: 16px;
    }}
    .tab-content.active {{ display: block; }}
  </style>
</head>
<body>
  <h3>{title}</h3>
  <div class="tab-bar">
    {"".join(tab_buttons)}
  </div>
  {"".join(tab_contents)}
  <script>
    var plots = [
      {plots_js}
    ];
    plots.forEach(function(p, i) {{
      Plotly.newPlot("plot-" + i, p.data, p.layout, {{responsive: true}});
    }});
    function showTab(idx) {{
      document.querySelectorAll(".tab").forEach(function(t, i) {{
        t.classList.toggle("active", i === idx);
      }});
      document.querySelectorAll(".tab-content").forEach(function(c, i) {{
        c.classList.toggle("active", i === idx);
      }});
      Plotly.Plots.resize("plot-" + idx);
    }}
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

    df = _trials_to_dataframe(study, metric_name)
    if df.empty:
        return

    from optuna.study import StudyDirection
    ascending = config.spec.direction != StudyDirection.MAXIMIZE
    df = df.sort_values(metric_name, ascending=ascending).reset_index(drop=True)

    if plot_config.top_n is not None:
        df = df.head(plot_config.top_n)

    # Add rank column
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
    """Generate a tabbed sensitivity plot with one tab per hyperparameter."""
    metric_name = config.spec.metric
    assert metric_name is not None

    df = _trials_to_dataframe(study, metric_name)
    if df.empty:
        return

    varying_params = _get_varying_param_names(config)
    if not varying_params:
        logger.info("No varying parameters; skipping sensitivity plot.")
        return

    # Determine which params to average over (default: auto-detect "seed")
    average_over = plot_config.average_over
    if average_over is None:
        average_over = [p for p in varying_params if p.lower() == "seed"]

    # Hue params: shown as separate lines, consistent across all tabs (default: none)
    hue_params = plot_config.hue or []

    # Determine which params get tabs (exclude average_over and hue)
    tab_params = plot_config.params
    if tab_params is None:
        excluded = set(average_over) | set(hue_params)
        tab_params = [p for p in varying_params if p not in excluded]

    if not tab_params:
        logger.info("No parameters to plot tabs for; skipping sensitivity plot.")
        return

    best_trial = study.best_trial
    tab_figures: dict[str, go.Figure] = {}

    for x_param in tab_params:
        # Group by x_param + hue_params; everything else is averaged over
        group_cols = [x_param] + hue_params
        grouped = df.groupby(group_cols, as_index=False)[metric_name].mean()

        if hue_params:
            # Create a combined hue label for each unique combo
            grouped["_hue_label"] = grouped[hue_params].astype(str).agg(
                ", ".join, axis=1,
            )
            if len(hue_params) == 1:
                grouped["_hue_label"] = hue_params[0] + "=" + grouped["_hue_label"]
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

        # Sort x values for consistent ordering
        x_values = sorted(grouped[x_param].unique(), key=lambda v: (isinstance(v, str), v))
        x_positions = list(range(len(x_values)))
        x_map = dict(zip(x_values, x_positions))

        fig = go.Figure()

        for hue_val in hue_values:
            if hue_val:
                subset = grouped[grouped["_hue_label"] == hue_val]
            else:
                subset = grouped

            # Map x values to evenly spaced positions
            xs = [x_map[v] for v in subset[x_param]]
            ys = subset[metric_name].tolist()

            # Sort by x position for line continuity
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

        # Highlight the best trial's value for this param
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

        tab_figures[x_param] = fig

    if not tab_figures:
        return

    html = _build_tabbed_html(
        tab_figures,
        title=f"Hyperparameter Sensitivity ({metric_name})",
    )
    _log_html(html, "sensitivity.html")
