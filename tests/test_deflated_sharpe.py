"""Tests for src/deflated_sharpe.py — DSR formula validation."""

import numpy as np
import pytest

from src.deflated_sharpe import (
    _expected_max_sr,
    _variance_of_sr,
    deflated_sharpe_ratio,
)


@pytest.fixture
def config():
    return {
        "deflated_sharpe": {"significance_level": 0.95},
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestVarianceOfSR:
    def test_normal_returns(self):
        """Normal returns (skew=0, kurt=3) → V(SR) ≈ (1 + 0.5*SR^2) / (T-1)."""
        var = _variance_of_sr(sr=1.0, skew=0.0, kurt=3.0, t=253)
        expected = (1 + 0.5 * 1.0) / 252
        assert var == pytest.approx(expected, rel=1e-6)

    def test_t_one_returns_nan(self):
        assert np.isnan(_variance_of_sr(1.0, 0.0, 3.0, t=1))

    def test_high_kurtosis_increases_variance(self):
        """Fat tails (kurt > 3) increase SR uncertainty."""
        v_normal = _variance_of_sr(1.0, 0.0, 3.0, 252)
        v_fat = _variance_of_sr(1.0, 0.0, 6.0, 252)
        assert v_fat > v_normal


class TestExpectedMaxSR:
    def test_single_trial_returns_zero(self):
        assert _expected_max_sr(1, 0.01) == 0.0

    def test_more_trials_higher_expected(self):
        """More trials → higher expected max SR (data snooping bias)."""
        sr5 = _expected_max_sr(5, 0.01)
        sr100 = _expected_max_sr(100, 0.01)
        assert sr100 > sr5

    def test_negative_variance_returns_zero(self):
        assert _expected_max_sr(10, -0.01) == 0.0


# ---------------------------------------------------------------------------
# Full DSR
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatio:
    def test_strong_strategy_single_trial(self, config):
        """Single trial with strong signal → DSR should be high."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 1000)
        observed_sr = 1.5
        result = deflated_sharpe_ratio(observed_sr, returns, n_trials=1, config=config)
        assert result["dsr"] > 0.9

    def test_many_trials_reduces_dsr(self, config):
        """Same SR with more trials → lower DSR (data snooping penalty)."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0, 0.01, 500)
        observed_sr = 0.3  # weak signal so DSR doesn't saturate at 1.0
        dsr_few = deflated_sharpe_ratio(observed_sr, returns, n_trials=2, config=config)
        dsr_many = deflated_sharpe_ratio(observed_sr, returns, n_trials=500, config=config)
        assert dsr_many["dsr"] < dsr_few["dsr"]

    def test_zero_sr_low_dsr(self, config):
        """SR ≈ 0 → DSR should be low."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.01, 1000)
        result = deflated_sharpe_ratio(0.0, returns, n_trials=100, config=config)
        assert result["dsr"] < 0.5

    def test_output_keys(self, config):
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.01, 100)
        result = deflated_sharpe_ratio(0.5, returns, n_trials=10, config=config)
        assert set(result.keys()) == {"dsr", "sr_zero", "var_sr", "sr_star", "is_significant"}

    def test_too_few_observations(self, config):
        """< 3 observations → NaN."""
        result = deflated_sharpe_ratio(1.0, np.array([0.01, 0.02]), n_trials=5, config=config)
        assert np.isnan(result["dsr"])
        assert result["is_significant"] is False

    def test_significance_flag(self, config):
        """DSR > 0.95 → is_significant = True."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.002, 0.01, 2000)
        result = deflated_sharpe_ratio(3.0, returns, n_trials=1, config=config)
        assert result["is_significant"] is True

    def test_nan_in_returns(self, config):
        """NaN returns should be filtered out."""
        returns = np.array([0.01, np.nan, 0.02, np.nan, -0.01, 0.005] * 20)
        result = deflated_sharpe_ratio(0.5, returns, n_trials=10, config=config)
        assert not np.isnan(result["dsr"])
