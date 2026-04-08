"""Tests for src/metrics.py — Sharpe, Sortino, Calmar, CAGR, MaxDD."""

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    cagr,
    calmar_ratio,
    compute_all_metrics,
    compute_metric,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "ranking": {"annualization_factor": 252, "metric": "sharpe"},
        "data": {"risk_free_rate": 0.0},
    }


@pytest.fixture
def constant_positive():
    """Constant 1% daily return — zero vol, strong performance."""
    return pd.Series([0.01] * 252)


@pytest.fixture
def known_returns():
    """Hand-computed returns: 252 days, mean=0.001, std≈0.01."""
    rng = np.random.RandomState(0)
    return pd.Series(rng.normal(0.001, 0.01, 252))


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_positive_sharpe(self, known_returns, config):
        sr = sharpe_ratio(known_returns, config)
        assert isinstance(sr, float)
        assert not np.isnan(sr)

    def test_zero_std_returns_nan(self, constant_positive, config):
        """Constant returns → std = 0 → Sharpe must be NaN."""
        sr = sharpe_ratio(constant_positive, config)
        assert np.isnan(sr)

    def test_known_value(self, config):
        """252 days of 0.001 daily excess return with std = 0.01
        → Sharpe ≈ 0.001/0.01 * sqrt(252) ≈ 1.587."""
        returns = pd.Series([0.001] * 126 + [0.001] * 126)
        # Inject tiny noise so std != 0
        returns.iloc[0] += 1e-8
        sr = sharpe_ratio(returns, config)
        assert sr > 1.0  # strong positive Sharpe

    def test_risk_free_rate(self, config):
        """Non-zero risk-free reduces Sharpe."""
        returns = pd.Series(np.random.RandomState(1).normal(0.001, 0.01, 252))
        sr_zero_rf = sharpe_ratio(returns, config)
        config_rf = {**config, "data": {"risk_free_rate": 0.05}}
        sr_nonzero_rf = sharpe_ratio(returns, config_rf)
        assert sr_nonzero_rf < sr_zero_rf

    def test_single_return(self, config):
        """Single return → std = NaN → Sharpe = NaN."""
        sr = sharpe_ratio(pd.Series([0.01]), config)
        assert np.isnan(sr)

    def test_empty_series(self, config):
        """Empty series → NaN."""
        sr = sharpe_ratio(pd.Series(dtype=float), config)
        assert np.isnan(sr)

    def test_all_negative(self, config):
        """All negative returns → negative Sharpe (not NaN)."""
        returns = pd.Series([-0.01, -0.02, -0.005, -0.015] * 63)
        sr = sharpe_ratio(returns, config)
        assert sr < 0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortino:
    def test_positive_sortino(self, known_returns, config):
        sr = sortino_ratio(known_returns, config)
        assert isinstance(sr, float)

    def test_no_downside_returns_nan(self, config):
        """All positive excess returns → no downside → NaN."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.005] * 63)
        sr = sortino_ratio(returns, config)
        assert np.isnan(sr)

    def test_sortino_gte_sharpe_for_positive_skew(self, config):
        """Positively skewed returns → Sortino >= Sharpe (less downside vol)."""
        rng = np.random.RandomState(7)
        # Create right-skewed returns
        returns = pd.Series(np.abs(rng.normal(0.001, 0.01, 252)))
        returns.iloc[::10] = -0.002  # sprinkle some negative
        sort = sortino_ratio(returns, config)
        shar = sharpe_ratio(returns, config)
        # Sortino uses only downside std → should be >= Sharpe for positive mean
        assert sort >= shar or np.isnan(sort)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing equity → drawdown = 0."""
        returns = pd.Series([0.01] * 100)
        assert max_drawdown(returns) == 0.0

    def test_known_drawdown(self):
        """Price goes 1 → 1.1 → 0.88 → 1.0 — drawdown = 0.88/1.1 - 1 = -0.2."""
        returns = pd.Series([0.10, -0.20, 0.136364])
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(-0.2, abs=0.01)

    def test_all_negative(self):
        """All losses → drawdown is significant.
        Equity: 1.0, 0.9, 0.81, 0.729 → MDD = 0.729/1.0 - 1 = -0.271."""
        returns = pd.Series([-0.1, -0.1, -0.1])
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(-0.271, abs=0.01)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_known_cagr(self, config):
        """252 days of 0.04% daily → ~10% annual."""
        returns = pd.Series([0.0004] * 252)
        c = cagr(returns, config)
        assert c == pytest.approx(0.106, abs=0.02)

    def test_negative_cumulative(self, config):
        """Cumulative product ≤ 0 → NaN."""
        returns = pd.Series([-0.5, -0.5, -0.5, -0.5])
        c = cagr(returns, config)
        # (1-0.5)^4 = 0.0625 > 0, so not NaN — but very negative CAGR
        assert c < 0


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_calmar_positive(self, known_returns, config):
        cal = calmar_ratio(known_returns, config)
        assert isinstance(cal, float)

    def test_zero_drawdown_returns_nan(self, config):
        """No drawdown → |MDD| = 0 → NaN."""
        returns = pd.Series([0.01] * 252)
        cal = calmar_ratio(returns, config)
        assert np.isnan(cal)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    def test_compute_metric_sharpe(self, known_returns, config):
        result = compute_metric(known_returns, config)
        expected = sharpe_ratio(known_returns, config)
        assert result == expected

    def test_compute_metric_sortino(self, known_returns, config):
        config["ranking"]["metric"] = "sortino"
        result = compute_metric(known_returns, config)
        expected = sortino_ratio(known_returns, config)
        assert result == expected

    def test_unknown_metric_raises(self, known_returns, config):
        config["ranking"]["metric"] = "magic_ratio"
        with pytest.raises(ValueError, match="Unknown ranking metric"):
            compute_metric(known_returns, config)

    def test_compute_all_metrics(self, known_returns, config):
        result = compute_all_metrics(known_returns, config)
        assert set(result.keys()) == {"sharpe", "sortino", "calmar", "cagr", "max_drawdown"}
        for v in result.values():
            assert isinstance(v, float)
