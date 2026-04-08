"""Tests for src/pbo.py — PBO computation and convergence."""

import numpy as np
import pandas as pd
import pytest

from src.pbo import compute_pbo, pbo_convergence


# ---------------------------------------------------------------------------
# Fixtures — synthetic CSCV results with known PBO
# ---------------------------------------------------------------------------

@pytest.fixture
def cscv_all_worst():
    """IS-best always ranks last OOS → PBO = 1.0 (always underperforms OOS median).

    rank = N/N → relative_rank = 1.0 → logit > 0 → counted as underperformance.
    """
    n_combos = 50
    n_trials = 10
    return pd.DataFrame({
        "oos_rank": [n_trials] * n_combos,  # rank N/N = worst
        "n_trials": [n_trials] * n_combos,
    })


@pytest.fixture
def cscv_all_best():
    """IS-best always ranks 1st OOS → PBO = 0.0 (never underperforms OOS median).

    rank = 1/N → relative_rank ≈ 0.1 → logit < 0 → not counted as underperformance.
    """
    n_combos = 50
    n_trials = 10
    return pd.DataFrame({
        "oos_rank": [1] * n_combos,  # rank 1/N = best
        "n_trials": [n_trials] * n_combos,
    })


@pytest.fixture
def cscv_mixed():
    """Half best, half worst → PBO ≈ 0.5."""
    n_trials = 10
    ranks = [1] * 25 + [n_trials] * 25
    return pd.DataFrame({
        "oos_rank": ranks,
        "n_trials": [n_trials] * 50,
    })


# ---------------------------------------------------------------------------
# PBO computation
# ---------------------------------------------------------------------------

class TestComputePBO:
    def test_always_worst_oos_pbo_one(self, cscv_all_worst):
        """IS-best always ranks worst OOS → relative_rank ≈ 1.0 → logit > 0 → PBO = 1.0.

        PBO = fraction(logit > 0). logit > 0 ↔ relative_rank > 0.5 ↔ bottom-half OOS.
        All-worst means every combination has the IS-best underperforming → PBO = 1.0.
        """
        result = compute_pbo(cscv_all_worst)
        assert result["pbo"] == pytest.approx(1.0)
        assert result["n_combinations"] == 50
        assert len(result["logits"]) == 50

    def test_always_best_oos_pbo_zero(self, cscv_all_best):
        """IS-best always ranks 1st OOS → relative_rank = 0.1 → logit < 0 → PBO = 0.0."""
        result = compute_pbo(cscv_all_best)
        assert result["pbo"] == pytest.approx(0.0)

    def test_always_worst_oos_zero(self, cscv_all_worst):
        """Verify logits are all positive when IS-best always ranks last."""
        result = compute_pbo(cscv_all_worst)
        assert np.all(result["logits"] > 0)

    def test_pbo_mixed(self, cscv_mixed):
        """Half best, half worst → PBO ≈ 0.5."""
        result = compute_pbo(cscv_mixed)
        assert result["pbo"] == pytest.approx(0.5)

    def test_logit_shape(self, cscv_all_best):
        result = compute_pbo(cscv_all_best)
        assert result["logits"].shape == (50,)

    def test_relative_ranks_range(self, cscv_mixed):
        result = compute_pbo(cscv_mixed)
        assert np.all(result["relative_ranks"] > 0)
        assert np.all(result["relative_ranks"] <= 1)

    def test_pbo_in_unit_interval(self, cscv_mixed):
        result = compute_pbo(cscv_mixed)
        assert 0.0 <= result["pbo"] <= 1.0


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestPBOConvergence:
    def test_convergence_shape(self, cscv_mixed):
        conv = pbo_convergence(cscv_mixed, step=10)
        assert "n_evaluated" in conv.columns
        assert "pbo_estimate" in conv.columns
        assert len(conv) > 0

    def test_final_row_matches_pbo(self, cscv_mixed):
        conv = pbo_convergence(cscv_mixed, step=10)
        full_pbo = compute_pbo(cscv_mixed)["pbo"]
        assert conv.iloc[-1]["pbo_estimate"] == pytest.approx(full_pbo)

    def test_n_evaluated_increasing(self, cscv_mixed):
        conv = pbo_convergence(cscv_mixed, step=10)
        assert conv["n_evaluated"].is_monotonic_increasing
