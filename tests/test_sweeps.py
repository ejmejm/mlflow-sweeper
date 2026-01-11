"""Basic integration tests for mlflow-sweeper."""

from __future__ import annotations

import heapq
import itertools
import queue
import subprocess
import sys
import threading

import mlflow
import optuna

from tests.conftest import SweepHarness


def _trial_param(run: mlflow.entities.Run, name: str) -> str | None:
    value = run.data.params.get(name)
    if value is None:
        return None
    return str(value)


def _expected_combinations(parameters: dict[str, list[str]], keys: list[str]) -> set[tuple[str, ...]]:
    values = [parameters[k] for k in keys]
    return set(itertools.product(*values))


def _seen_combinations(
    trial_runs: list[mlflow.entities.Run], keys: list[str]
) -> set[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    for run in trial_runs:
        combo: list[str] = []
        for k in keys:
            value = _trial_param(run, k)
            if value is None:
                break
            combo.append(value)
        if len(combo) == len(keys):
            seen.add(tuple(combo))
    return seen


def _assert_all_combinations_seen(
    *, trial_runs: list[mlflow.entities.Run], keys: list[str], parameters: dict[str, list[str]]
) -> None:
    expected = _expected_combinations(parameters, keys)
    seen = _seen_combinations(trial_runs, keys)
    missing = expected.difference(seen)
    assert not missing, f"Missing combinations for {keys}: {sorted(missing)}"


def _assert_all_runs_finished(trial_runs: list[mlflow.entities.Run]) -> None:
    for run in trial_runs:
        assert run.info.status == "FINISHED", (
            f"Run {run.info.run_id} has status {run.info.status}, expected FINISHED"
        )


def _assert_parent_and_get_trial_runs(*, harness: SweepHarness) -> list[mlflow.entities.Run]:
    runs = harness.list_mlflow_runs()
    parents = [r for r in runs if "optuna_study_name" in r.data.tags]
    assert len(parents) == 1, f"Expected exactly 1 parent run, found {len(parents)}."
    return [r for r in runs if "optuna_study_name" not in r.data.tags]


def _read_stream_with_timestamp(
    stream, output_queue: queue.Queue[tuple[float, str, str]], stream_name: str
) -> None:
    """Read from a stream and put lines into a queue with timestamps for ordering."""
    import time
    for line in stream:
        output_queue.put((time.time(), stream_name, line))
    output_queue.put((time.time(), stream_name, None))  # Sentinel to mark end


def _capture_and_print_interleaved(
    proc: subprocess.Popen[str], timeout: float | None = None
) -> tuple[str, str]:
    """Capture stdout and stderr while printing them interleaved in original order.
    
    Args:
        proc: The subprocess to read from.
        timeout: Maximum time to wait for process (None for no timeout).
    
    Returns:
        Tuple of (captured_stdout, captured_stderr) strings.
    """
    output_queue: queue.Queue[tuple[float, str, str]] = queue.Queue()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    priority_buffer: list[tuple[float, str, str]] = []  # Heap for sorting by timestamp
    
    # Start threads to read from both streams
    threads = []
    if proc.stdout is not None:
        t = threading.Thread(
            target=_read_stream_with_timestamp,
            args=(proc.stdout, output_queue, "stdout"),
            daemon=True,
        )
        t.start()
        threads.append(t)
    
    if proc.stderr is not None:
        t = threading.Thread(
            target=_read_stream_with_timestamp,
            args=(proc.stderr, output_queue, "stderr"),
            daemon=True,
        )
        t.start()
        threads.append(t)
    
    # Wait for process to finish (with timeout if specified)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    
    # Read from queue, buffer, and output in timestamp order
    streams_done = {"stdout": False, "stderr": False}
    import time as time_module
    last_output_time = time_module.time()
    buffer_window = 0.01  # Small window to allow items to arrive in order
    
    while not all(streams_done.values()) or not output_queue.empty() or priority_buffer:
        # Get items from queue and add to priority buffer
        try:
            while True:
                item = output_queue.get_nowait()
                heapq.heappush(priority_buffer, item)
        except queue.Empty:
            pass
        
        # Output items from buffer in timestamp order (oldest first)
        current_time = time_module.time()
        # Output if we have items and either:
        # - The buffer window has passed since last output, or
        # - Both streams are done (no more items coming)
        should_output = (
            priority_buffer and
            (all(streams_done.values()) or 
             current_time - last_output_time >= buffer_window or
             len(priority_buffer) > 10)  # Or if buffer is getting large
        )
        
        while priority_buffer and should_output:
            timestamp, stream_name, line = priority_buffer[0]
            
            # If this is a sentinel, mark stream as done
            if line is None:
                heapq.heappop(priority_buffer)
                streams_done[stream_name] = True
                continue
            
            # Output the oldest item
            heapq.heappop(priority_buffer)
            print(line, file=sys.__stdout__, end="")
            sys.__stdout__.flush()
            last_output_time = current_time
            
            # Also capture for return value
            if stream_name == "stdout":
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
        
        # Small sleep to avoid busy-waiting
        if not should_output:
            time_module.sleep(0.001)
    
    # Wait for threads to finish
    for t in threads:
        t.join(timeout=1.0)
    
    return "".join(stdout_lines), "".join(stderr_lines)


def test_grid_runs_all_4_combinations(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "color": ["red", "blue"],
        "shape": ["square", "circle"],
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            # Passed through to the subprocess script to satisfy the required MLflow init.
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path)

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)
    assert len(trial_runs) == 4

    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["color", "shape"], parameters=grid_params
    )


def test_delete_sweep_removes_optuna_and_mlflow_runs(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={
            "a": ["0", "1"],
            "b": ["x", "y"],
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "1")

    # Sanity: Optuna study exists and MLflow has runs.
    _ = sweep_harness.load_optuna_study()
    assert len(sweep_harness.list_mlflow_runs()) > 0

    sweep_harness.run_cli(config_path, "--delete")

    # Optuna study deleted.
    try:
        optuna.load_study(
            study_name=sweep_harness.optuna_study_name(),
            storage=sweep_harness.optuna_storage,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("Expected Optuna study to be deleted, but it still exists.")

    # MLflow runs deleted (ACTIVE view should be empty).
    assert sweep_harness.list_mlflow_runs() == []


def test_parallel_inprocess_jobs_does_not_double_runs(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "x": ["0", "1"],
        "y": ["0", "1"],
        "z": ["0", "1"],  # 2*2*2 = 8 combos
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    sweep_harness.run_cli(config_path, "-n", "100", "-j", "4")

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    # Grid search should produce exactly 8 combinations (2*2*2).
    assert len(trial_runs) == 8
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )

    _assert_all_runs_finished(trial_runs)


def test_parallel_two_processes_does_not_double_runs(sweep_harness: SweepHarness) -> None:
    grid_params = {
        "x": ["0", "1"],
        "y": ["0", "1"],
        "z": ["0", "1"],  # 2*2*2 = 8 combos
    }
    config_path = sweep_harness.write_config(
        parameters={
            **grid_params,
            "mlflow_storage": sweep_harness.mlflow_storage,
            "project": sweep_harness.experiment,
        }
    )

    cmd = [
        sys.executable,
        f"{sweep_harness.repo_root}/sweep.py",
        config_path,
        "-n",
        "100",
        "-j",
        "1",
    ]

    proc1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Capture and print output interleaved in original order
    print("=== Runner 1 Output ===", file=sys.__stdout__)
    sys.__stdout__.flush()
    out1, err1 = _capture_and_print_interleaved(proc1, timeout=300)
    
    print("=== Runner 2 Output ===", file=sys.__stdout__)
    sys.__stdout__.flush()
    out2, err2 = _capture_and_print_interleaved(proc2, timeout=300)
    
    print(f"Runner 1 exit code: {proc1.returncode}", file=sys.__stdout__)
    print(f"Runner 2 exit code: {proc2.returncode}", file=sys.__stdout__)
    sys.__stdout__.flush()

    if proc1.returncode != 0:
        raise RuntimeError(
            f"Runner 1 failed with exit code {proc1.returncode}.\n\n"
            f"Stdout:\n{out1}\n\nStderr:\n{err1}"
        )
    if proc2.returncode != 0:
        raise RuntimeError(
            f"Runner 2 failed with exit code {proc2.returncode}.\n\n"
            f"Stdout:\n{out2}\n\nStderr:\n{err2}"
        )

    trial_runs = _assert_parent_and_get_trial_runs(harness=sweep_harness)

    # Grid search should produce exactly 8 combinations (2*2*2).
    assert len(trial_runs) == 8
    _assert_all_combinations_seen(
        trial_runs=trial_runs, keys=["x", "y", "z"], parameters=grid_params
    )
    _assert_all_runs_finished(trial_runs)
