# mlflow-sweeper

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

### Run multiple sweeps (sequentially) from one command

```bash
mlflow-sweep path/to/sweep_a.yaml path/to/sweep_b.yaml
```

## Config format (minimal example)

```yaml
experiment: "my-experiment"
sweep_name: "lr_bs_grid"
command: "python train.py"
algorithm: "grid"

parameters:
  lr: [1e-3, 1e-4]
  batch_size: [32, 64]
  seed: 0

optuna_storage: "sqlite:///optuna.db"
mlflow_storage: "sqlite:///mlruns.db"
output_dir: "./sweeps"

# Optional optimization spec (used to return a metric to Optuna)
spec:
  direction: "maximize"   # or "minimize"
  metric: "val/accuracy"
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

## TODO (moved from code)

- [x] Implement grid sweep from config with run command and parameters
- [x] Only have a single parent MLFlow run for all trials even when distributed
- [x] Make a sampler that uses a lock to avoid duplicate runs (still a potential race condition)
- [x] Change sweep run # that is printed to be the # run in this sweep, not in the whole experiment
- [x] Don't mark parent MLFlow run as complete until all trials are done
- [x] Add an option to delete a sweep, and remove it from both MLFlow and Optuna storages
- [x] Fix bug with deleted MLFlow runs leading to duplicate Optuna trials
- [ ] Better handle failed runs, ideally retrying or just overwritting with the same param set
- [ ] Delete failed runs when being replaced
- [ ] Implement random sweep from config with run command and parameters
- [ ] Implement hyperparameter sensitivity from config with run command and parameters
- [ ] Fix bug where using the `n_jobs` option causes mlflow runs to not be parented properly
