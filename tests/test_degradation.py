"""Tests for src/degradation.py — degradation ratios and haircut."""

import numpy as np
import pandas as pd
import pytest

from src.degradation import compute_degradation, haircut_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_cscv():
    """OOS = IS → degradation ratio = 1.0 everywhere."""
    return pd.DataFrame({
        "is_best_metric": [1.0, 2.0, 0.5, 1.5],
        "oos_metric_of_is_best": [1.0, 2.0, 0.5, 1.5],
    })


@pytest.fixture
def half_decay_cscv():
    """OOS = IS / 2 → degradation ratio = 0.5 everywhere."""
    return pd.DataFrame({
        "is_best_metric": [2.0, 4.0, 1.0, 3.0],
        "oos_metric_of_is_best": [1.0, 2.0, 0.5, 1.5],
    })


@pytest.fixture
def sign_flip_cscv():
    """Half positive OOS, half negative → sign flip rate = 0.5."""
    return pd.DataFrame({
        "is_best_metric": [1.0, 1.0, 1.0, 1.0],
        "oos_metric_of_is_best": [0.5, 0.8, -0.2, -0.5],
    })


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

class TestComputeDegradation:
    def test_perfect_preservation(self, perfect_cscv):
        result = compute_degradation(perfect_cscv)
        assert result["mean_degradation"] == pytest.approx(1.0)
        assert result["median_degradation"] == pytest.approx(1.0)
        assert result["sign_flip_rate"] == pytest.approx(0.0)

    def test_half_decay(self, half_decay_cscv):
        result = compute_degradation(half_decay_cscv)
        assert result["mean_degradation"] == pytest.approx(0.5)
        assert result["median_degradation"] == pytest.approx(0.5)

    def test_sign_flip_rate(self, sign_flip_cscv):
        result = compute_degradation(sign_flip_cscv)
        assert result["sign_flip_rate"] == pytest.approx(0.5)

    def test_division_by_zero_handled(self):
        """IS metric = 0 → degradation ratio = NaN (not error)."""
        cscv = pd.DataFrame({
            "is_best_metric": [0.0, 1.0],
            "oos_metric_of_is_best": [0.5, 0.5],
        })
        result = compute_degradation(cscv)
        assert np.isnan(result["degradation_ratios"][0])
        assert not np.isnan(result["degradation_ratios"][1])

    def test_output_arrays_correct_length(self, sign_flip_cscv):
        result = compute_degradation(sign_flip_cscv)
        assert len(result["degradation_ratios"]) == 4
        assert len(result["is_metrics"]) == 4
        assert len(result["oos_metrics"]) == 4

    def test_negative_is_ratios_nan(self):
        """IS ≤ 0 → degradation ratios all NaN, mean NaN."""
        cscv = pd.DataFrame({
            "is_best_metric": [-1.0, -0.5, -2.0],
            "oos_metric_of_is_best": [-0.8, -0.3, -1.5],
        })
        result = compute_degradation(cscv)
        assert np.all(np.isnan(result["degradation_ratios"]))
        assert np.isnan(result["mean_degradation"])
        assert result["n_unprofitable_is"] == 3

    def test_n_unprofitable_is_key(self, sign_flip_cscv):
        """n_unprofitable_is is present and correct (all IS > 0 → 0)."""
        result = compute_degradation(sign_flip_cscv)
        assert result["n_unprofitable_is"] == 0


# ---------------------------------------------------------------------------
# Haircut summary
# ---------------------------------------------------------------------------

class TestHaircutSummary:
    def test_no_degradation_message(self, perfect_cscv):
        result = compute_degradation(perfect_cscv)
        msg = haircut_summary(result)
        assert "No degradation" in msg

    def test_decay_message(self, half_decay_cscv):
        result = compute_degradation(half_decay_cscv)
        msg = haircut_summary(result)
        assert "50%" in msg
        assert "decay" in msg.lower()

    def test_nan_message(self):
        result = {"median_degradation": np.nan}
        msg = haircut_summary(result)
        assert "Insufficient" in msg
