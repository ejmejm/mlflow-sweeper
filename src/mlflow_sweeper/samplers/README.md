# Grid Sweep Internals

## How a Sweep Runs

```
run_sweep                                         [runner.py]
  |
  |-- Create/load Optuna study + MLflow parent run
  |-- Sync: void any Optuna trials whose MLflow runs were deleted
  |-- Early exit if study already complete (is_exhausted)
  |
  |-- OPTIMIZATION LOOP (up to n_trials iterations)
  |     |
  |     |-- study.ask() creates a new trial (RUNNING state)
  |     |     \__ calls GridSampler.before_trial:
  |     |           Acquires FileLock, computes pending grid IDs.
  |     |           If pending IDs exist --> assign one randomly to this trial.
  |     |           If none remain ------> mark trial as PREEMPTED.
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
  |     |-- GridSampler.after_trial: no-op
  |     |     (Overrides base class to prevent premature stopping.
  |     |      Optuna calls after_trial BEFORE writing the trial's
  |     |      final state to storage, so the trial still looks RUNNING
  |     |      and _get_pending_grid_ids would incorrectly treat it as
  |     |      resolved.  The stop decision is handled by before_trial's
  |     |      preemption on the next iteration, when storage is current.)
  |     |
  |     v (next iteration, or break if stop_flag / n_trials / timeout)
  |
  |-- Set parent MLflow run status: FINISHED if complete, else RUNNING
```

## How Retries Work

Each parameter combination maps to a **grid ID** (0 to N-1). After each
trial finishes, `_get_pending_grid_ids` decides which grid IDs still need
work:

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

## Mermaid Flowchart

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

        Success --> AfterTrial[GridSampler.after_trial:<br>no-op]
        Fail --> AfterTrial

        AfterTrial --> NextIter{n_trials reached<br>or timed out?}
        NextIter -- NO --> Ask
        NextIter -- YES --> ExitLoop
    end

    ExitLoop --> ParentStatus{Study<br>complete?}
    ParentStatus -- YES --> Finished([Parent run = FINISHED])
    ParentStatus -- NO --> Running([Parent run = RUNNING])

    style BeforeTrial fill:#1a1a2e,stroke:#4a4a6a
    style Loop fill:#0d0d1a,stroke:#3a3a5a
```
