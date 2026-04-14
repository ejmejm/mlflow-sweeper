# MLFlow Sweeper

This module runs parameter sweeps (currently grid search) by executing a user-provided command for each trial and logging results to MLflow. It uses Optuna to enumerate/coordinate trials and supports both single-agent runs and multi-agent/distributed execution against shared storages.

- **Sweep identity / config ID**: each sweep is keyed by `experiment/sweep_name` (used as the Optuna study name) plus the configured storages. This lets you safely resume and/or distribute the *same* sweep across multiple agents, and also run *multiple different* sweeps at once (as long as their `experiment/sweep_name` differs).
- **Single-agent vs multi-agent**:
  - **Single-agent**: one process runs the sweep locally (optionally with local parallelism via `-j/--n_jobs`).
  - **Multi-agent**: start the same command on multiple machines/processes pointing at the same Optuna + MLflow storage, so trials are shared across agents.

## How to run

### Install (so you can run from anywhere)

From the repo root:

```bash
pip install -e .
```

This installs a CLI named `mlflow-sweep`.

### Run one sweep config

```bash
mlflow-sweep path/to/sweep.yaml
```

### Use local parallelism

```bash
mlflow-sweep path/to/sweep.yaml -j 4
```

### Run distributed (multi-agent)

Run the same command on multiple agents that share the same storages:

```bash
# Agent 1
mlflow-sweep path/to/sweep.yaml

# Agent 2 (same config + same storages)
mlflow-sweep path/to/sweep.yaml
```

Note: coordination relies on file locks in `output_dir`, so for true multi-host coordination, `output_dir` should be on a shared filesystem.

### Delete a sweep

```bash
mlflow-sweep path/to/sweep.yaml --delete
```

### View sweep results

After sweeps finish, quickly inspect results from the terminal without opening the MLflow UI:

```bash
# Show all sweeps in an experiment (sorted by accuracy descending)
mlflow-sweep-results my-experiment

# Filter to a specific sweep
mlflow-sweep-results my-experiment -s lr_sweep

# Sort by a specific metric, show top 5
mlflow-sweep-results my-experiment -m loss --ascending -n 5
```

## Programmatic sweeps (Python API)

Instead of defining a `command` that spawns a subprocess for each trial, you can pass a Python function directly. This lets you run a full sweep entirely within a single script.

### Basic example

```python
from omegaconf import OmegaConf
from mlflow_sweeper import SweepConfig, run_sweep

config = SweepConfig.from_dict_config(OmegaConf.create({
    "experiment": "my-experiment",
    "sweep_name": "lr_grid",
    "algorithm": "grid",
    "optuna_storage": "sqlite:///optuna.db",
    "mlflow_storage": "sqlite:///mlruns.db",
    "output_dir": "./sweeps",
    "parameters": {
        "lr": [1e-3, 1e-4],
        "batch_size": [32, 64],
    },
    "spec": {
        "direction": "minimize",
        "metric": "loss",
    },
}))

def train(**params):
    import mlflow
    lr = float(params["lr"])
    batch_size = int(params["batch_size"])
    # ... your training code here ...
    loss = (lr - 1e-4) ** 2 + (batch_size - 64) ** 2
    mlflow.log_metric("loss", loss)

run_sweep(config, train, n_jobs=4)
```

### Function contract

Your function is called with `**param_values` (keyword arguments matching the parameter names in your config). An MLflow run is **already active** when your function runs, so you can call `mlflow.log_metric()`, `mlflow.log_param()`, etc. directly. Do **not** call `mlflow.start_run()` or `mlflow.end_run()` — the runner manages the trial's MLflow run for you.

**Return value** determines how the optimization metric is resolved:

| Return type | Behavior |
|---|---|
| `float` | Used directly as the optimization metric |
| `dict[str, float]` | All entries logged via `mlflow.log_metrics()`; the entry matching `spec.metric` is used for optimization |
| `None` | Metric is read from the MLflow run after the function returns (same as the subprocess path) |

If your function raises an exception, the trial is marked as `FAILED`. With `abort_on_fail=True`, the first failure terminates the entire sweep.

### Execution modes

Control parallelism with `n_jobs` and `executor`:

```python
# Serial (inline, no pool)
run_sweep(config, train, n_jobs=1)

# Thread pool — works with closures/lambdas, limited by the GIL
run_sweep(config, train, n_jobs=4, executor="thread")

# Process pool (default) — real parallelism, requires an importable function
run_sweep(config, train, n_jobs=4, executor="process")
```

When `n_jobs=1`, the `executor` parameter is ignored and the function runs inline.

**Process pool requirements**: the function must be a top-level function in an importable module (not a closure, lambda, or nested function). If you need closures, use `executor="thread"`. Passing a non-importable function with `executor="process"` raises a clear error before the sweep starts.

### Full API reference

```python
run_sweep(
    config: SweepConfig,
    fn: Callable[..., float | dict[str, float] | None],
    *,
    n_trials: int | None = None,       # Max trials; None = run until exhausted
    n_jobs: int = 1,                    # Worker count
    executor: "thread" | "process" = "process",
    abort_on_fail: bool = False,        # Stop sweep on first failure
    log_params: bool = False,           # Log param values on each trial run
    no_plots: bool = False,             # Skip plot generation
    allow_param_change: bool = False,   # Permit config changes (migrates trials)
)
```

### Updating parameters

If you change your sweep's parameters between runs, `run_sweep` will detect the change and abort with an error by default. Pass `allow_param_change=True` to migrate valid trials from the old config and continue running:

```python
# First run
run_sweep(config_v1, train, n_jobs=4)

# Later, with expanded parameters
run_sweep(config_v2, train, n_jobs=4, allow_param_change=True)
```

Trials whose parameters are still valid under the new config are kept; incompatible trials are dropped from the Optuna study (but remain visible in MLflow). If the config hasn't actually changed, a warning is logged and the sweep proceeds normally.

## Config format (minimal example)

```yaml
experiment: "my-experiment"
sweep_name: "lr_bs_grid"
optuna_storage: "sqlite:///optuna.db"
mlflow_storage: "sqlite:///mlruns.db"
output_dir: "./sweeps"

command: "python train.py"
algorithm: "grid"

spec:
  direction: "maximize"   # or "minimize"
  metric: "val/accuracy"

parameters:
  lr: [1e-3, 1e-4]
  batch_size: [32, 64]
  seed: 0

# Optional optimization spec (used to return a metric to Optuna)
```

The sweep runner appends parameters to `command` as `name=value` tokens and executes the resulting command for each trial, nesting each trial as a child MLflow run under a single parent run for the sweep.

## Parameter value types

In `parameters:`, each entry can be specified in a few ways:

- **Atomic (single fixed value)**:

```yaml
parameters:
  seed: 0
  model_name: "resnet18"
```

- **Categorical (explicit set of choices)**:

```yaml
parameters:
  batch_size: [32, 64, 128]
  optimizer: ["adam", "sgd"]
```

- **Typed dictionary forms** (equivalent to the above, and adds ranges):

```yaml
parameters:
  # categorical (typed)
  activation:
    type: categorical
    values: ["relu", "gelu"]

  # int range
  depth:
    type: int_range
    low: 2
    high: 8
    step: 1        # optional (default 1)
    log: false     # optional (default false)

  # float range
  dropout:
    type: float_range
    low: 0.0
    high: 0.5
    step: 0.1      # optional; required if converting to a grid internally
    log: false     # optional (default false)
```

## Plots

When a sweep completes and a `spec.metric` is configured, mlflow-sweeper automatically generates interactive plots and logs them as MLflow artifacts on the parent run. Use `--no-plots` to skip plot generation entirely. Plots require parameter logging to be generated. If your experiment script does not log parameters (i.e. `mlflow.log_params(...)`), you can use the `--log-params` flag to have the sweeper log the parameters for you.

There are two built-in plots:

- **`best_hyperparameters`** -- a ranked table of trials sorted by metric value. Works with any algorithm.
- **`sensitivity`** -- an interactive line chart showing how each parameter affects the metric. Only supported for grid sweeps.

By default, all compatible plots are generated. If a plot is not compatible with the current sweep (e.g. sensitivity on a random sweep), a warning is logged and the plot is skipped.

### Selecting which plots to generate

Use the `plots` key to choose which plots to generate, and `plot_params` to configure them. Global options like `metrics` and `split_by` apply to all plots; per-plot options are nested under the plot name:

```yaml
# Only generate the listed plots (default settings)
plots: [best_hyperparameters]

# Select plots and configure them separately
plots: [best_hyperparameters, sensitivity]
plot_params:
  metrics: [loss, accuracy]
  split_by: [dataset]
  best_hyperparameters:
    top_n: 10
  sensitivity:
    average_over: [seed]
```

If `plots` is omitted entirely, all plots are enabled with default settings. See [docs/plots.md](docs/plots.md) for the full reference of available options for each plot.

## TODO (moved from code)

- [x] Implement grid sweep from config with run command and parameters
- [x] Only have a single parent MLFlow run for all trials even when distributed
- [x] Make a sampler that uses a lock to avoid duplicate runs (still a potential race condition)
- [x] Change sweep run # that is printed to be the # run in this sweep, not in the whole experiment
- [x] Don't mark parent MLFlow run as complete until all trials are done
- [x] Add an option to delete a sweep, and remove it from both MLFlow and Optuna storages
- [x] Fix bug with deleted MLFlow runs leading to duplicate Optuna trials
- [x] Better handle failed runs, ideally retrying or just overwritting with the same param set
- [x] Add test for testing retrying failed runs
- [ ] Delete failed runs when being replaced (do I really want to do this?)
- [ ] Implement random sweep from config with run command and parameters
- [ ] Implement hyperparameter sensitivity from config with run command and parameters
- [x] Fix bug where using the `n_jobs` option causes mlflow runs to not be parented properly
- [ ] Move locks to the MLFlow storage
- [x] Pass actual arg values to sweep command so that the function can easily be used externally
- [x] Allow sweeps to contain a command or a function to run