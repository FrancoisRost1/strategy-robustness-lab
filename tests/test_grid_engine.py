"""Tests for src/grid_engine.py — Cartesian product and parameter utilities."""

import pytest

from src.grid_engine import generate_param_grid, get_param_names, grid_summary


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

class TestGenerateParamGrid:
    def test_cartesian_product_count(self):
        """2 × 3 = 6 combinations."""
        grid = generate_param_grid({"a": [1, 2], "b": ["x", "y", "z"]})
        assert len(grid) == 6

    def test_single_param(self):
        grid = generate_param_grid({"a": [1, 2, 3]})
        assert len(grid) == 3

    def test_trial_ids_sequential(self):
        grid = generate_param_grid({"a": [1, 2], "b": [10, 20]})
        ids = [g["trial_id"] for g in grid]
        assert ids == list(range(4))

    def test_parameter_names_preserved(self):
        grid = generate_param_grid({"lookback": [6, 12], "weight": ["eq", "cap"]})
        for g in grid:
            assert "lookback" in g
            assert "weight" in g
            assert "trial_id" in g

    def test_all_values_appear(self):
        grid = generate_param_grid({"a": [1, 2], "b": ["x", "y"]})
        a_vals = {g["a"] for g in grid}
        b_vals = {g["b"] for g in grid}
        assert a_vals == {1, 2}
        assert b_vals == {"x", "y"}

    def test_large_grid(self):
        """4 × 3 × 2 × 2 × 2 = 96 (factor engine default)."""
        grid = generate_param_grid({
            "factor_weights": ["equal", "value_tilt", "momentum_tilt", "quality_tilt"],
            "lookback_months": [6, 9, 12],
            "rebalance_freq": ["monthly", "quarterly"],
            "weighting": ["equal_weight", "cap_weight"],
            "n_quantiles": [5, 10],
        })
        assert len(grid) == 96

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="empty"):
            generate_param_grid({})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestGetParamNames:
    def test_sorted_names(self):
        names = get_param_names({"z_param": [1], "a_param": [2], "m_param": [3]})
        assert names == ["a_param", "m_param", "z_param"]


class TestGridSummary:
    def test_unique_values(self):
        grid = generate_param_grid({"a": [1, 2], "b": ["x", "y"]})
        summary = grid_summary(grid)
        assert set(summary["a"]) == {1, 2}
        assert set(summary["b"]) == {"x", "y"}

    def test_empty_grid(self):
        assert grid_summary([]) == {}

    def test_excludes_trial_id(self):
        grid = generate_param_grid({"a": [1]})
        summary = grid_summary(grid)
        assert "trial_id" not in summary
