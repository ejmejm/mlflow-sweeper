# Sweep Internals

## How a Sweep Runs

This applies to all samplers. The sampler controls two things: which
parameters to try next (`before_trial`) and when to stop (`after_trial` /
preemption). Everything else is shared.

```
run_sweep                                         [runner.py]
  |
  |-- Create/load Optuna study + MLflow parent run
  |-- Sync: void any Optuna trials whose MLflow runs were deleted
  |-- Early exit if study already complete (sampler.is_exhausted)
  |
  |-- OPTIMIZATION LOOP (up to n_trials iterations)
  |     |
  |     |-- study.ask() creates a new trial (RUNNING state)
  |     |     \__ calls sampler.before_trial(study, trial)
  |     |           Sampler decides what params this trial should use,
  |     |           or marks it PREEMPTED if no work remains.
  |     |
  |     |-- Trial preempted?
  |     |     YES --> set trial state to PRUNED, call study.stop(), continue
  |     |     NO  --> run the subprocess (run_experiment)
  |     |
  |     |-- Subprocess exits
  |     |     exit 0 --> Optuna state = COMPLETE, MLflow status = FINISHED
  |     |     exit 1 --> raise TrialRunError (caught by loop, not fatal)
  |     |                Optuna state = FAIL, MLflow status = FAILED
  |     |                (--abort-on-fail raises TrialRunAbortError instead,
  |     |                 which IS fatal and kills the sweep)
  |     |
  |     |-- sampler.after_trial(study, trial, state, values)
  |     |     Sampler may call study.stop() here to end the loop early.
  |     |
  |     v (next iteration, or break if stop_flag / n_trials / timeout)
  |
  |-- Set parent MLflow run status: FINISHED if complete, else RUNNING
```

## Grid Sampler

```mermaid
flowchart TD
    Start([run_sweep]) --> Init[Create/load Optuna study<br>+ MLflow parent run]
    Init --> Sync[Sync: void Optuna trials<br>whose MLflow runs were deleted]
    Sync --> Complete{Study already<br>complete?}
    Complete -- YES --> DoneEarly([Return early])
    Complete -- NO --> Loop

    subgraph Loop [Optimization Loop]
        Ask[study.ask<br>Creates trial in<br>RUNNING state] --> BeforeTrial

        subgraph BeforeTrial [GridSampler.before_trial]
            Lock[Acquire FileLock] --> Pending[Compute pending grid IDs]
            Pending --> HasPending{Pending IDs<br>exist?}
            HasPending -- YES --> Assign[Assign random<br>pending grid ID<br>to trial]
            HasPending -- NO --> Preempt[Mark trial as<br>PREEMPTED]
        end

        Assign --> RunSub[Run subprocess<br>with assigned params]
        Preempt --> PruneStop[Set trial state = PRUNED<br>study.stop]
        PruneStop --> NextIter

        RunSub --> ExitCode{Subprocess<br>exit code}

        ExitCode -- "0" --> Success[Optuna: COMPLETE<br>MLflow: FINISHED]
        ExitCode -- "!= 0" --> AbortCheck{--abort-on-fail?}
        AbortCheck -- YES --> Abort([TrialRunAbortError<br>Sweep killed])
        AbortCheck -- NO --> Fail[Optuna: FAIL<br>MLflow: FAILED<br>TrialRunError caught]

        Success --> NextIter{n_trials reached<br>or timed out?}
        Fail --> NextIter

        NextIter -- NO --> Ask
        NextIter -- YES --> ExitLoop
    end

    ExitLoop --> ParentStatus{Study<br>complete?}
    ParentStatus -- YES --> Finished([Parent run = FINISHED])
    ParentStatus -- NO --> Running([Parent run = RUNNING])

    style BeforeTrial fill:#1a1a2e,stroke:#4a4a6a
    style Loop fill:#0d0d1a,stroke:#3a3a5a
```

## Sensitivity Sampler

A sensitivity sweep holds every parameter at a baseline (default) value and
varies **one parameter at a time**, instead of running the full cartesian
product. It answers "how sensitive is the metric to each parameter
individually?" at a fraction of a grid's cost.

`SensitivitySampler` subclasses `GridSampler` and overrides **only** how the
trial list (`_all_grids`) is built. Everything that makes the grid sampler work
— `before_trial` (FileLock + grid-ID assignment + preemption),
`_get_pending_grid_ids` (the retry accounting described in
[How retries work](#how-retries-work)), the no-op `after_trial`, and
`is_exhausted` — is **inherited unchanged**. So the runtime flow is identical to
the Grid Sampler diagram above; the only difference is the contents of the grid.

### Building the trial list

The config has two sections: `parameters` gives a single base value per
parameter, and a sibling `sensitivity` block lists the candidate values to try
for the parameters you want to perturb. Only the varied (multi-valued)
parameters enter the grid; parameters held fixed are atomic and bypass the
sampler entirely.

```
parameters:  a: 1, b: 10, c: 5      sensitivity:  a: [1, 2, 3], b: [10, 20]

  grid 0 (baseline):  a=1, b=10      <- every varied param at its default
  grid 1:             a=2, b=10      <- vary a (b at default)
  grid 2:             a=3, b=10      <- vary a
  grid 3:             a=1, b=20      <- vary b (a at default)

  (c is not in `sensitivity`, so it stays fixed at 5 on every trial)
```

So `_all_grids` holds **1 baseline + one point per non-default candidate value**:
`1 + Σ (candidates_i − 1)` trials, versus `Π candidates_i` for a grid. A
candidate equal to the default is skipped (the baseline already covers it). From
there, grid-ID assignment, retries, preemption, and `is_exhausted` behave
exactly as for the Grid Sampler.

```mermaid
flowchart TD
    Start([run_sweep]) --> Init[Create/load Optuna study<br>+ MLflow parent run]
    Init --> Sync[Sync: void Optuna trials<br>whose MLflow runs were deleted]
    Sync --> Complete{Study already<br>complete?}
    Complete -- YES --> DoneEarly([Return early])
    Complete -- NO --> Loop

    subgraph Loop [Optimization Loop]
        Ask[study.ask<br>Creates trial in<br>RUNNING state] --> BeforeTrial

        subgraph BeforeTrial [SensitivitySampler.before_trial inherited from GridSampler]
            Lock[Acquire FileLock] --> Pending[Compute pending grid IDs<br>over the one-at-a-time list]
            Pending --> HasPending{Pending IDs<br>exist?}
            HasPending -- YES --> Assign[Assign random<br>pending grid ID<br>to trial]
            HasPending -- NO --> Preempt[Mark trial as<br>PREEMPTED]
        end

        Assign --> RunSub[Run subprocess<br>with assigned params]
        Preempt --> PruneStop[Set trial state = PRUNED<br>study.stop]
        PruneStop --> NextIter

        RunSub --> ExitCode{Subprocess<br>exit code}

        ExitCode -- "0" --> Success[Optuna: COMPLETE<br>MLflow: FINISHED]
        ExitCode -- "!= 0" --> AbortCheck{--abort-on-fail?}
        AbortCheck -- YES --> Abort([TrialRunAbortError<br>Sweep killed])
        AbortCheck -- NO --> Fail[Optuna: FAIL<br>MLflow: FAILED<br>TrialRunError caught]

        Success --> NextIter{n_trials reached<br>or timed out?}
        Fail --> NextIter

        NextIter -- NO --> Ask
        NextIter -- YES --> ExitLoop
    end

    ExitLoop --> ParentStatus{Study<br>complete?}
    ParentStatus -- YES --> Finished([Parent run = FINISHED])
    ParentStatus -- NO --> Running([Parent run = RUNNING])

    style BeforeTrial fill:#1a1a2e,stroke:#4a4a6a
    style Loop fill:#0d0d1a,stroke:#3a3a5a
```

## Random Sampler

```mermaid
flowchart TD
    Start([run_sweep]) --> Init[Create/load Optuna study<br>+ MLflow parent run]
    Init --> Sync[Sync: void Optuna trials<br>whose MLflow runs were deleted]
    Sync --> Complete{Study already<br>complete?}
    Complete -- YES --> DoneEarly([Return early])
    Complete -- NO --> Loop

    subgraph Loop [Optimization Loop]
        Ask[study.ask<br>Creates trial in<br>RUNNING state] --> BeforeTrial

        subgraph BeforeTrial [RandomSampler.before_trial]
            Lock[Acquire FileLock] --> Count[Count random_group trials]
            Count --> CapHit{count >= n_runs?}
            CapHit -- YES --> Preempt[Mark trial as<br>PREEMPTED]
            CapHit -- NO --> MarkGroup[Mark trial as<br>random_group]
        end

        MarkGroup --> RunSub[Run subprocess<br>with sampled params]
        Preempt --> PruneStop[Set trial state = PRUNED<br>study.stop]
        PruneStop --> NextIter

        RunSub --> ExitCode{Subprocess<br>exit code}

        ExitCode -- "0" --> Success[Optuna: COMPLETE<br>MLflow: FINISHED]
        ExitCode -- "!= 0" --> AbortCheck{--abort-on-fail?}
        AbortCheck -- YES --> Abort([TrialRunAbortError<br>Sweep killed])
        AbortCheck -- NO --> Fail[Optuna: FAIL<br>MLflow: FAILED<br>TrialRunError caught]

        BeforeTrial ~~~ AfterTrial
        Success --> AfterTrial
        Fail --> AfterTrial

        subgraph AfterTrial [RandomSampler.after_trial]
            RetryCheck{FAIL and<br>retries left?}
            RetryCheck -- YES --> EnqueueRetry[Enqueue retry<br>with same params]
            RetryCheck -- NO --> GridCheck
            EnqueueRetry --> GridCheck{grid_search_space<br>set?}
            GridCheck -- YES --> Expand[Enqueue remaining<br>grid combos with<br>same random params]
            GridCheck -- NO --> Done[ ]
            Expand --> Done
        end

        Done --> NextIter{n_trials reached<br>or timed out?}

        NextIter -- NO --> Ask
        NextIter -- YES --> ExitLoop
    end

    ExitLoop --> ParentStatus{Study<br>complete?}
    ParentStatus -- YES --> Finished([Parent run = FINISHED])
    ParentStatus -- NO --> Running([Parent run = RUNNING])

    style BeforeTrial fill:#1a1a2e,stroke:#4a4a6a
    style AfterTrial fill:#1a1a2e,stroke:#4a4a6a
    style Loop fill:#0d0d1a,stroke:#3a3a5a
```

### How retries work

Each parameter combination maps to a **grid ID** (0 to N-1). The key
function is `_get_pending_grid_ids`, called in `before_trial` to decide
what to sample next:

```
For each non-voided trial with a matching search space:
  - If RUNNING or COMPLETE --> that grid ID is RESOLVED (done, skip it)
  - If finished (COMPLETE/FAIL/PRUNED) --> increment visitation_count for that grid ID

Pending = all grid IDs that are NOT resolved
          AND have visitation_count <= max_retry_count
```

A grid ID that succeeds (COMPLETE) is immediately resolved and never
retried. A grid ID that keeps failing stays pending until its visitation
count exceeds `max_retry_count`. Since the count is checked *before* the
next attempt (`<=` not `<`), a failing combo gets **max_retry_count + 1
total attempts** (1 original + N retries).

Example with `max_retry_count = 2`:

```
Attempt 1: count 0 -> 1, 1 <= 2 --> still pending (retry)
Attempt 2: count 1 -> 2, 2 <= 2 --> still pending (retry)
Attempt 3: count 2 -> 3, 3 >  2 --> no longer pending (give up)
```
