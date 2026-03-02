"""Integration tests for post-sweep plot generation."""

from __future__ import annotations

import os
import sys

from tests.conftest import SweepHarness


def _log_metric_command(harness: SweepHarness) -> str:
    """Return the command string for the log_metric.py test script."""
    return f"{sys.executable} {os.path.join(harness.assets_dir, 'log_metric.py')}"


def _list_plot_artifacts(harness: SweepHarness) -> list[str]:
    """List plot artifact paths on the parent run."""
    client = harness.mlflow_client()
    parent_run_id = harness.get_parent_run_id()
    assert parent_run_id is not None, "Expected a parent run to exist"
    artifacts = client.list_artifacts(parent_run_id, "plots")
    return [a.path for a in artifacts]


def test_grid_sweep_with_metric_generates_both_plots(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts
    assert "plots/sensitivity.html" in artifacts


def test_sweep_without_metric_skips_plots(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
    )

    sweep_harness.run_cli(config_path)

    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    artifacts = client.list_artifacts(parent_run_id, "plots")
    assert len(artifacts) == 0


def test_no_plots_flag_suppresses_generation(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
    )

    sweep_harness.run_cli(config_path, "--no-plots")

    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    artifacts = client.list_artifacts(parent_run_id, "plots")
    assert len(artifacts) == 0


def test_rerun_completed_sweep_generates_plots(sweep_harness: SweepHarness) -> None:
    """First run with --no-plots, then re-run to get plots."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
    )

    # First run: no plots (but log params so they're in MLflow for the re-run)
    sweep_harness.run_cli(config_path, "--no-plots", "--log-params")

    client = sweep_harness.mlflow_client()
    parent_run_id = sweep_harness.get_parent_run_id()
    assert parent_run_id is not None
    artifacts = client.list_artifacts(parent_run_id, "plots")
    assert len(artifacts) == 0

    # Re-run: study is already complete, should generate plots
    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts
    assert "plots/sensitivity.html" in artifacts


def test_sensitivity_with_seed_averages_over_seed(sweep_harness: SweepHarness) -> None:
    """When a 'seed' param exists, it should be averaged over in the sensitivity plot."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "seed": [0, 1]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    # Verify plots are generated (seed should be auto-detected and averaged)
    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/sensitivity.html" in artifacts


def test_best_hyperparameters_top_n(sweep_harness: SweepHarness) -> None:
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2, 3, 4]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"best_hyperparameters": {"top_n": 2}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts


def test_sensitivity_average_over_config(sweep_harness: SweepHarness) -> None:
    """Custom average_over param specified in config."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "replicate": [0, 1]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"sensitivity": {"average_over": ["replicate"]}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/sensitivity.html" in artifacts


def test_sensitivity_hue_config(sweep_harness: SweepHarness) -> None:
    """Hue params produce separate lines and don't get their own tabs."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20], "z": [100, 200]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"sensitivity": {"hue": ["y"]}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/sensitivity.html" in artifacts


def test_sensitivity_multiple_metrics(sweep_harness: SweepHarness) -> None:
    """Multiple metrics produces a dropdown in the sensitivity HTML."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"sensitivity": {"metrics": ["loss", "accuracy"]}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/sensitivity.html" in artifacts


def test_sensitivity_split_by(sweep_harness: SweepHarness) -> None:
    """split_by param produces separate plots per value with a dropdown."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "model_size": ["small", "large"]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"sensitivity": {"split_by": ["model_size"]}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/sensitivity.html" in artifacts


def test_random_sweep_with_metric_generates_table_only(sweep_harness: SweepHarness) -> None:
    """Random sweeps get the best hyperparameters table but not the sensitivity plot."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2, 3], "y": [10, 20, 30]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize", "n_runs": 4},
        algorithm="random",
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts
    assert "plots/sensitivity.html" not in artifacts


def test_plots_disabled_via_list(sweep_harness: SweepHarness) -> None:
    """Only plots listed in the config are generated."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots=["best_hyperparameters"],
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts
    assert "plots/sensitivity.html" not in artifacts


def test_plots_disabled_via_dict(sweep_harness: SweepHarness) -> None:
    """Only plots keyed in the dict are generated."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2], "y": [10, 20]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize"},
        plots={"sensitivity": {}},
    )

    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" not in artifacts
    assert "plots/sensitivity.html" in artifacts


def test_sensitivity_on_random_sweep_warns_but_succeeds(sweep_harness: SweepHarness) -> None:
    """Enabling sensitivity on a random sweep warns but does not error."""
    config_path = sweep_harness.write_config(
        parameters={"x": [1, 2, 3], "y": [10, 20, 30]},
        command=_log_metric_command(sweep_harness),
        spec={"metric": "loss", "direction": "minimize", "n_runs": 4},
        algorithm="random",
        plots=["best_hyperparameters", "sensitivity"],
    )

    # Should succeed (no error), sensitivity just gets skipped with a warning.
    sweep_harness.run_cli(config_path, "--log-params")

    artifacts = _list_plot_artifacts(sweep_harness)
    assert "plots/best_hyperparameters.html" in artifacts
    assert "plots/sensitivity.html" not in artifacts
