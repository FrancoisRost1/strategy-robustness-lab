"""Tests for src/parameter_stability.py — grid, plateau, sensitivity, heatmaps."""

import numpy as np
import pandas as pd
import pytest

from src.parameter_stability import (
    build_metric_grid,
    pairwise_heatmap_data,
    plateau_detection,
    sensitivity_curves,
)


@pytest.fixture
def config():
    return {
        "parameter_stability": {
            "plateau_tolerance": 0.10,
            "stable_threshold": 0.30,
            "moderate_threshold": 0.10,
        },
    }


# ---------------------------------------------------------------------------
# Build metric grid
# ---------------------------------------------------------------------------

class TestBuildMetricGrid:
    def test_correct_shape(self):
        grid = [
            {"trial_id": 0, "a": 1, "b": "x"},
            {"trial_id": 1, "a": 2, "b": "y"},
        ]
        metrics = {"0": 1.5, "1": 2.0}
        df = build_metric_grid(grid, metrics)
        assert len(df) == 2
        assert "metric" in df.columns
        assert "a" in df.columns

    def test_int_key_lookup(self):
        """trial_metrics keyed by int should still work."""
        grid = [{"trial_id": 0, "a": 1}]
        metrics = {0: 0.75}
        df = build_metric_grid(grid, metrics)
        assert df.iloc[0]["metric"] == 0.75

    def test_missing_metric_is_nan(self):
        grid = [{"trial_id": 99, "a": 1}]
        metrics = {"0": 1.0}
        df = build_metric_grid(grid, metrics)
        assert np.isnan(df.iloc[0]["metric"])


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

class TestPlateauDetection:
    def test_all_equal_is_full_plateau(self, config):
        """All metrics equal → plateau_fraction = 1.0 → STABLE."""
        df = pd.DataFrame({"metric": [1.0] * 20, "trial_id": range(20)})
        result = plateau_detection(df, config)
        assert result["plateau_fraction"] == pytest.approx(1.0)
        assert result["classification"] == "STABLE"

    def test_single_spike_is_fragile(self, config):
        """One high value, rest much lower → FRAGILE."""
        metrics = [0.1] * 99 + [10.0]
        df = pd.DataFrame({"metric": metrics, "trial_id": range(100)})
        result = plateau_detection(df, config)
        # Only 1/100 ≥ 10.0 * 0.9 = 9.0
        assert result["plateau_fraction"] < 0.10
        assert result["classification"] == "FRAGILE"

    def test_moderate_plateau(self, config):
        """~20% in plateau → MODERATE."""
        metrics = [0.5] * 80 + [1.0] * 20
        df = pd.DataFrame({"metric": metrics, "trial_id": range(100)})
        result = plateau_detection(df, config)
        # 20/100 cells ≥ 1.0 * 0.9 = 0.9 — exactly the 20 cells with 1.0
        assert result["classification"] == "MODERATE"

    def test_empty_grid(self, config):
        df = pd.DataFrame({"metric": pd.Series(dtype=float), "trial_id": pd.Series(dtype=int)})
        result = plateau_detection(df, config)
        assert result["classification"] == "UNKNOWN"

    def test_negative_metrics(self, config):
        """All negative and equal: best = -0.1, threshold = -0.1 - 0.01 = -0.11.
        All values -0.1 >= -0.11 → True. fraction = 1.0 → STABLE."""
        metrics = [-0.1] * 10
        df = pd.DataFrame({"metric": metrics, "trial_id": range(10)})
        result = plateau_detection(df, config)
        # All equal but negative: threshold = -0.1 - abs(-0.1)*0.10 = -0.11
        # -0.1 >= -0.11 → True. So plateau = 1.0 → STABLE
        assert result["plateau_fraction"] == pytest.approx(1.0)
        assert result["classification"] == "STABLE"

    def test_output_keys(self, config):
        df = pd.DataFrame({"metric": [1.0, 2.0], "trial_id": [0, 1]})
        result = plateau_detection(df, config)
        expected_keys = {
            "plateau_fraction", "classification", "best_metric",
            "threshold_metric", "n_plateau_cells", "n_total_cells",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Sensitivity curves
# ---------------------------------------------------------------------------

class TestSensitivityCurves:
    def test_output_per_param(self):
        df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": ["x", "y", "x", "y"],
            "metric": [0.5, 0.6, 0.7, 0.8],
            "trial_id": range(4),
        })
        result = sensitivity_curves(df, ["a", "b"])
        assert "a" in result
        assert "b" in result

    def test_columns_present(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "metric": [0.5, 0.6, 0.7],
            "trial_id": range(3),
        })
        result = sensitivity_curves(df, ["a"])
        assert list(result["a"].columns) == ["param_value", "metric_mean", "metric_std"]


# ---------------------------------------------------------------------------
# Pairwise heatmaps
# ---------------------------------------------------------------------------

class TestPairwiseHeatmapData:
    def test_correct_pairs(self):
        df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": [10, 20, 10, 20],
            "metric": [0.5, 0.6, 0.7, 0.8],
            "trial_id": range(4),
        })
        result = pairwise_heatmap_data(df, ["a", "b"])
        assert ("a", "b") in result
        assert isinstance(result[("a", "b")], pd.DataFrame)

    def test_three_params_gives_three_pairs(self):
        df = pd.DataFrame({
            "a": [1] * 8,
            "b": [1, 1, 2, 2] * 2,
            "c": [1, 2] * 4,
            "metric": np.random.rand(8),
            "trial_id": range(8),
        })
        result = pairwise_heatmap_data(df, ["a", "b", "c"])
        # C(3,2) = 3 pairs
        assert len(result) == 3
