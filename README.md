# mlflow-sweeper

Run Optuna-driven parameter sweeps while logging trials to MLflow.

- **Sweep identity / config ID**: each sweep is keyed by `experiment/sweep_name` (used as the Optuna study name) plus the configured storages. This lets you safely resume and/or distribute the *same* sweep across multiple agents, and also run *multiple different* sweeps at once (as long as their `experiment/sweep_name` differs).
- **Single-agent vs multi-agent**:
  - **Single-agent**: one process runs the sweep locally (optionally with local parallelism via `-j/--n_jobs`).
  - **Multi-agent**: start the same command on multiple machines/processes pointing at the same Optuna + MLflow storage, so trials are shared across agents.

## How to run

### Run one sweep config

```bash
python sweep.py path/to/sweep.yaml
```

### Run multiple sweeps (sequentially) from one command

```bash
python sweep.py path/to/sweep_a.yaml path/to/sweep_b.yaml
```

### Use local parallelism

```bash
python sweep.py path/to/sweep.yaml -j 4
```

### Run distributed (multi-agent)

Run the same command on multiple agents that share the same storages:

```bash
# Agent 1
python sweep.py path/to/sweep.yaml -j 1

# Agent 2 (same config + same storages)
python sweep.py path/to/sweep.yaml -j 1
```

Note: coordination relies on file locks in `output_dir`, so for true multi-host coordination, `output_dir` should be on a shared filesystem.

### Delete a sweep

```bash
python sweep.py path/to/sweep.yaml --delete
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

## TODO (moved from code)

- [x] Implement grid sweep from config with run command and parameters
- [x] Only have a single parent MLFlow run for all trials even when distributed
- [ ] Make a sampler that uses a lock to avoid duplicate runs
- [ ] Change sweep run # that is printed to be the # run in this sweep, not in the whole experiment
- [ ] Don't mark parent MLFlow run as complete until all trials are done
- [ ] Make sure filelocks are deleted after use
- [x] Add an option to delete a sweep, and remove it from both MLFlow and Optuna storages
- [ ] Better handle failed runs, ideally retrying or just overwritting with the same param set
- [ ] Delete failed runs when being replaced
- [ ] Implement random sweep from config with run command and parameters
- [ ] Implement hyperparameter sensitivity from config with run command and parameters
