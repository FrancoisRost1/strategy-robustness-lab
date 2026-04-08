"""Tests for src/verdict.py — traffic light classification with boundary values."""

import pytest

from src.verdict import classify


@pytest.fixture
def config():
    return {
        "verdict": {
            "robust_pbo_max": 0.25,
            "robust_dsr_min": 0.95,
            "robust_plateau_min": 0.30,
            "overfit_pbo_min": 0.50,
            "overfit_dsr_max": 0.95,
            "overfit_plateau_max": 0.10,
        },
        "pbo": {
            "green_threshold": 0.25,
            "yellow_threshold": 0.50,
        },
    }


# ---------------------------------------------------------------------------
# All 5 verdicts
# ---------------------------------------------------------------------------

class TestRobust:
    def test_all_green(self, config):
        """PBO < 0.25, DSR > 0.95, plateau > 0.30 → ROBUST."""
        v = classify(pbo=0.10, dsr=0.98, plateau_fraction=0.50, config=config)
        assert v["verdict"] == "ROBUST"
        assert v["color"] == "GREEN"

    def test_boundary_pbo(self, config):
        """PBO exactly at boundary: 0.25 is NOT < 0.25 → falls through to BORDERLINE."""
        v = classify(pbo=0.25, dsr=0.99, plateau_fraction=0.50, config=config)
        assert v["verdict"] != "ROBUST"


class TestLikelyRobust:
    def test_low_pbo_low_plateau(self, config):
        """PBO < 0.25 but plateau < 0.30 → LIKELY ROBUST (not ROBUST)."""
        v = classify(pbo=0.10, dsr=0.99, plateau_fraction=0.15, config=config)
        assert v["verdict"] == "LIKELY ROBUST"
        assert v["color"] == "GREEN"

    def test_low_pbo_low_dsr(self, config):
        """PBO < 0.25 but DSR < 0.95 → LIKELY ROBUST."""
        v = classify(pbo=0.10, dsr=0.80, plateau_fraction=0.50, config=config)
        assert v["verdict"] == "LIKELY ROBUST"


class TestBorderline:
    def test_mid_pbo(self, config):
        """0.25 ≤ PBO ≤ 0.50 → BORDERLINE."""
        v = classify(pbo=0.35, dsr=0.90, plateau_fraction=0.20, config=config)
        assert v["verdict"] == "BORDERLINE"
        assert v["color"] == "YELLOW"

    def test_boundary_at_0_25(self, config):
        """PBO = 0.25 → BORDERLINE (0.25 is not < 0.25)."""
        v = classify(pbo=0.25, dsr=0.90, plateau_fraction=0.20, config=config)
        assert v["verdict"] == "BORDERLINE"

    def test_boundary_at_0_50(self, config):
        """PBO = 0.50 → BORDERLINE (0.50 is <= 0.50)."""
        v = classify(pbo=0.50, dsr=0.96, plateau_fraction=0.20, config=config)
        assert v["verdict"] == "BORDERLINE"


class TestLikelyOverfit:
    def test_high_pbo_but_not_all_bad(self, config):
        """PBO > 0.50, but DSR or plateau don't qualify for full OVERFIT."""
        v = classify(pbo=0.70, dsr=0.96, plateau_fraction=0.05, config=config)
        assert v["verdict"] == "LIKELY OVERFIT"
        assert v["color"] == "RED"


class TestOverfit:
    def test_all_red(self, config):
        """PBO > 0.50, DSR < 0.95, plateau < 0.10 → OVERFIT."""
        v = classify(pbo=0.80, dsr=0.60, plateau_fraction=0.05, config=config)
        assert v["verdict"] == "OVERFIT"
        assert v["color"] == "RED"

    def test_boundary_overfit_pbo(self, config):
        """PBO exactly 0.50 is NOT > 0.50 → not OVERFIT."""
        v = classify(pbo=0.50, dsr=0.60, plateau_fraction=0.05, config=config)
        assert v["verdict"] != "OVERFIT"


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------

class TestSpecialCases:
    def test_none_plateau_csv_mode(self, config):
        """plateau_fraction=None (CSV mode) → uses neutral 0.20, doesn't crash."""
        v = classify(pbo=0.10, dsr=0.98, plateau_fraction=None, config=config)
        assert v["verdict"] == "LIKELY ROBUST"  # plateau 0.20 < 0.30 → not full ROBUST

    def test_output_keys(self, config):
        v = classify(pbo=0.30, dsr=0.90, plateau_fraction=0.20, config=config)
        assert set(v.keys()) == {"verdict", "color", "details", "scores"}
        assert set(v["scores"].keys()) == {"pbo", "dsr", "plateau_fraction"}

    def test_nan_pbo_insufficient_data(self, config):
        """PBO = NaN (single trial, no CSCV) → INSUFFICIENT DATA verdict."""
        import numpy as np
        v = classify(pbo=np.nan, dsr=0.90, plateau_fraction=0.20, config=config)
        assert v["verdict"] == "INSUFFICIENT DATA"
        assert v["color"] == "GRAY"
        assert "PBO cannot be computed" in v["details"]

    def test_none_pbo_insufficient_data(self, config):
        """PBO = None → INSUFFICIENT DATA verdict."""
        v = classify(pbo=None, dsr=0.90, plateau_fraction=0.20, config=config)
        assert v["verdict"] == "INSUFFICIENT DATA"
        assert v["color"] == "GRAY"

    def test_custom_thresholds(self):
        """Custom config thresholds are respected."""
        custom_config = {
            "verdict": {
                "robust_pbo_max": 0.50,  # very lenient
                "robust_dsr_min": 0.50,
                "robust_plateau_min": 0.10,
                "overfit_pbo_min": 0.90,
                "overfit_dsr_max": 0.50,
                "overfit_plateau_max": 0.05,
            },
            "pbo": {"green_threshold": 0.50, "yellow_threshold": 0.90},
        }
        v = classify(pbo=0.40, dsr=0.60, plateau_fraction=0.20, config=custom_config)
        assert v["verdict"] == "ROBUST"
