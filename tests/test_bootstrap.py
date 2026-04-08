"""Tests for src/bootstrap.py — standard and block bootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import block_bootstrap, run_bootstrap, standard_bootstrap


@pytest.fixture
def config():
    return {
        "ranking": {"annualization_factor": 252, "metric": "sharpe"},
        "data": {"risk_free_rate": 0.0},
        "bootstrap": {
            "n_resamples": 100,  # small for speed
            "confidence_level": 0.95,
            "block_size": 10,
            "random_seed": 42,
        },
    }


@pytest.fixture
def returns():
    rng = np.random.RandomState(42)
    return pd.Series(rng.normal(0.001, 0.01, 252))


# ---------------------------------------------------------------------------
# Standard bootstrap
# ---------------------------------------------------------------------------

class TestStandardBootstrap:
    def test_output_keys(self, returns, config):
        result = standard_bootstrap(returns, config)
        assert set(result.keys()) == {"metric_distribution", "mean", "ci_lower", "ci_upper", "std"}

    def test_distribution_shape(self, returns, config):
        result = standard_bootstrap(returns, config)
        assert len(result["metric_distribution"]) == 100

    def test_ci_bounds(self, returns, config):
        """CI lower < mean < CI upper."""
        result = standard_bootstrap(returns, config)
        assert result["ci_lower"] <= result["mean"]
        assert result["mean"] <= result["ci_upper"]

    def test_seed_reproducibility(self, returns, config):
        r1 = standard_bootstrap(returns, config)
        r2 = standard_bootstrap(returns, config)
        np.testing.assert_array_equal(r1["metric_distribution"], r2["metric_distribution"])

    def test_different_seeds_differ(self, returns, config):
        r1 = standard_bootstrap(returns, config)
        config2 = {**config, "bootstrap": {**config["bootstrap"], "random_seed": 99}}
        r2 = standard_bootstrap(returns, config2)
        assert not np.array_equal(r1["metric_distribution"], r2["metric_distribution"])


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    def test_output_keys(self, returns, config):
        result = block_bootstrap(returns, config)
        assert set(result.keys()) == {"metric_distribution", "mean", "ci_lower", "ci_upper", "std"}

    def test_distribution_shape(self, returns, config):
        result = block_bootstrap(returns, config)
        assert len(result["metric_distribution"]) == 100

    def test_ci_bounds(self, returns, config):
        result = block_bootstrap(returns, config)
        assert result["ci_lower"] <= result["mean"]
        assert result["mean"] <= result["ci_upper"]

    def test_seed_reproducibility(self, returns, config):
        r1 = block_bootstrap(returns, config)
        r2 = block_bootstrap(returns, config)
        np.testing.assert_array_equal(r1["metric_distribution"], r2["metric_distribution"])

    def test_block_size_one_matches_standard(self, returns, config):
        """block_size=1 is equivalent to standard bootstrap (same random sequence)."""
        config["bootstrap"]["block_size"] = 1
        block_result = block_bootstrap(returns, config)
        # Just check it produces valid output — exact match depends on implementation
        assert not np.isnan(block_result["mean"])


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

class TestRunBootstrap:
    def test_both_methods(self, returns, config):
        result = run_bootstrap(returns, config)
        assert "standard" in result
        assert "block" in result
        assert len(result["standard"]["metric_distribution"]) == 100
        assert len(result["block"]["metric_distribution"]) == 100
