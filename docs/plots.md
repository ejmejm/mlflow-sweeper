# Plot Configuration Reference

Plots are generated automatically after a sweep completes when `spec.metric` is set. They are logged as HTML artifacts under `plots/` on the parent MLflow run.

## Selecting plots

The `plots` config key is a list of plot names to generate. Per-plot options are provided separately under `plot_params`.

```yaml
# All plots with defaults (same as omitting the key entirely)
plots: [best_hyperparameters, sensitivity]

# Only the table
plots: [best_hyperparameters]

# Select plots and configure them separately
plots: [best_hyperparameters, sensitivity]
plot_params:
  best_hyperparameters:
    top_n: 5
  sensitivity:
    average_over: [seed]
```

When `plots` is omitted, all plots are enabled with default settings. `plot_params` can be provided on its own to configure plots without changing which ones are enabled. If a plot is incompatible with the current sweep (e.g. `sensitivity` on a random sweep), a warning is logged and the plot is skipped without error.

---

## `best_hyperparameters`

An interactive table of trials ranked by metric value. Works with any sweep algorithm.

Parameters that are constant across all trials are automatically hidden. Values that are powers of 2 are displayed using exponent notation (e.g. 2^3 for 8).

**Artifact**: `plots/best_hyperparameters.html`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | `int` | `null` (show all) | Limit the table to the top N trials. When `null`, all trials are shown. |

### Example

```yaml
plot_params:
  best_hyperparameters:
    top_n: 20
```

---

## `sensitivity`

An interactive line chart showing how each parameter affects the metric, with one tab per parameter. Only supported for **grid** sweeps, since sensitivity analysis requires controlled experiments where all other parameters are held constant while one varies.

The plot includes dropdowns for selecting the metric and split dimensions, and tabs for switching between parameters. A green dashed line marks the best trial's value on each axis. Powers-of-2 values use exponent notation on tick labels.

**Artifact**: `plots/sensitivity.html`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `average_over` | `list[str]` | Auto-detects params named `seed` | Parameters to average over rather than plot individually. Useful for seed/replicate dimensions where you want to see the mean performance. |
| `params` | `list[str]` | All varying params not in `average_over`, `hue`, or `split_by` | Which parameters get their own tab in the plot. By default this is computed automatically by excluding `average_over`, `hue`, and `split_by` params from the full list of varying parameters. |
| `hue` | `list[str]` | `[]` | Parameters that create separate colored lines within each tab rather than getting their own tab. Useful for comparing across a categorical dimension (e.g. model size) within each parameter's plot. |
| `metrics` | `list[str]` | `[spec.metric]` | Metrics to include in the plot. When multiple metrics are provided, a dropdown is added to switch between them. Each metric must be logged by the sweep command. |
| `split_by` | `list[str]` | `[]` | Parameters that create separate sub-plots selectable via dropdown. Unlike `hue` (which overlays lines), `split_by` fully partitions the data so each dropdown value shows an independent plot. |

### Example

```yaml
plot_params:
  sensitivity:
    average_over: [seed, replicate]
    hue: [model_size]
    metrics: [loss, accuracy]
    split_by: [dataset]
```

In this example:
- Results are averaged across `seed` and `replicate` values
- Each tab shows a different parameter (all varying params except the ones listed above)
- Within each tab, separate lines are drawn for each `model_size`
- A dropdown switches between the `loss` and `accuracy` metrics
- Another dropdown switches between `dataset` values, showing independent plots for each
