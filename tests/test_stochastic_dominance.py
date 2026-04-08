"""Tests for src/stochastic_dominance.py — KS test and 2nd-order SD."""

import numpy as np
import pytest

from src.stochastic_dominance import (
    first_order_dominance,
    run_stochastic_dominance,
    second_order_dominance,
)


@pytest.fixture
def config():
    return {
        "stochastic_dominance": {
            "significance_level": 0.05,
            "test_second_order": True,
        },
    }


# ---------------------------------------------------------------------------
# First-order dominance
# ---------------------------------------------------------------------------

class TestFirstOrderDominance:
    def test_identical_distributions_not_rejected(self, config):
        """Same distribution → KS should NOT reject H0."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 0.01, 1000)
        result = first_order_dominance(a, a.copy(), config)
        assert result["rejects_null"] == False
        assert result["ks_statistic"] == pytest.approx(0.0, abs=1e-10)

    def test_clearly_different_distributions(self, config):
        """Very different distributions → KS should reject."""
        rng = np.random.RandomState(42)
        strategy = rng.normal(0.05, 0.01, 500)
        benchmark = rng.normal(-0.05, 0.01, 500)
        result = first_order_dominance(strategy, benchmark, config)
        assert result["rejects_null"] == True
        assert result["p_value"] < 0.01

    def test_dominant_strategy(self, config):
        """Strategy always > benchmark → strategy_dominates = True."""
        strategy = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
        benchmark = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = first_order_dominance(strategy, benchmark, config)
        assert result["strategy_dominates"] is True

    def test_not_dominant(self, config):
        """Overlapping distributions → no dominance."""
        rng = np.random.RandomState(42)
        a = rng.normal(0.001, 0.01, 500)
        b = rng.normal(0.0, 0.01, 500)
        result = first_order_dominance(a, b, config)
        assert result["strategy_dominates"] is False

    def test_too_few_observations(self, config):
        """< 2 obs → NaN results, no error."""
        result = first_order_dominance(np.array([0.01]), np.array([0.02, 0.03]), config)
        assert np.isnan(result["ks_statistic"])
        assert result["rejects_null"] is False

    def test_nan_handling(self, config):
        """NaN values should be filtered out."""
        strategy = np.array([0.01, np.nan, 0.02, 0.03, np.nan, 0.04])
        benchmark = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        result = first_order_dominance(strategy, benchmark, config)
        assert not np.isnan(result["ks_statistic"])


# ---------------------------------------------------------------------------
# Second-order dominance
# ---------------------------------------------------------------------------

class TestSecondOrderDominance:
    def test_dominant_strategy(self):
        """Clearly better strategy → SSD dominates."""
        strategy = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
        benchmark = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = second_order_dominance(strategy, benchmark)
        assert result["dominates"] is True

    def test_identical_dominates(self):
        """Same distribution → trivially dominates (integrated diff = 0)."""
        a = np.linspace(-0.05, 0.05, 100)
        result = second_order_dominance(a, a.copy())
        assert result["dominates"] is True
        assert result["min_integrated_diff"] >= -1e-10

    def test_too_few_returns_nan(self):
        result = second_order_dominance(np.array([0.01]), np.array([0.02]))
        assert result["dominates"] is False
        assert np.isnan(result["min_integrated_diff"])


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

class TestRunStochasticDominance:
    def test_both_orders_returned(self, config):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 0.01, 200)
        b = rng.normal(0, 0.01, 200)
        result = run_stochastic_dominance(a, b, config)
        assert "first_order" in result
        assert "second_order" in result

    def test_second_order_disabled(self, config):
        config["stochastic_dominance"]["test_second_order"] = False
        rng = np.random.RandomState(42)
        a = rng.normal(0, 0.01, 200)
        b = rng.normal(0, 0.01, 200)
        result = run_stochastic_dominance(a, b, config)
        assert "first_order" in result
        assert "second_order" not in result
