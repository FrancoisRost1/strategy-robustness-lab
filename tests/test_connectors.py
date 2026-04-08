"""Tests for connectors — CSV, factor, and TSMOM."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.connectors.csv_connector import load_trial_matrix


@pytest.fixture
def config():
    return {
        "ranking": {"annualization_factor": 252, "metric": "sharpe"},
        "data": {"risk_free_rate": 0.0},
    }


# ---------------------------------------------------------------------------
# CSV connector
# ---------------------------------------------------------------------------

class TestCSVConnector:
    def test_valid_csv(self, config, tmp_path):
        """Load a well-formed trial matrix CSV."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            rng.normal(0, 0.01, (100, 5)),
            index=dates,
            columns=["s0", "s1", "s2", "s3", "s4"],
        )
        path = str(tmp_path / "trial_matrix.csv")
        df.to_csv(path)

        result = load_trial_matrix(path, config)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 5)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_column_names_are_strings(self, config, tmp_path):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame(
            np.random.rand(10, 3),
            index=dates,
            columns=[0, 1, 2],  # int columns
        )
        path = str(tmp_path / "int_cols.csv")
        df.to_csv(path)

        result = load_trial_matrix(path, config)
        for col in result.columns:
            assert isinstance(col, str)

    def test_nan_handling(self, config, tmp_path):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame(
            {"a": [0.01] * 10, "b": [0.02] * 10},
            index=dates,
        )
        df.iloc[3, 0] = np.nan
        df.iloc[3, 1] = np.nan
        path = str(tmp_path / "nan.csv")
        df.to_csv(path)

        result = load_trial_matrix(path, config)
        assert result.isna().sum().sum() == 0  # NaN rows dropped
        assert len(result) == 9  # row 3 dropped

    def test_file_not_found(self, config):
        with pytest.raises(Exception):
            load_trial_matrix("/nonexistent/path.csv", config)

    def test_empty_csv_raises(self, config, tmp_path):
        """CSV with only an index column → 0 trial columns → ValueError."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(index=dates)
        path = str(tmp_path / "empty.csv")
        df.to_csv(path)

        with pytest.raises(ValueError, match="at least 1"):
            load_trial_matrix(path, config)

    def test_price_warning(self, config, tmp_path, caplog):
        """Values > 1 should trigger a warning about price data."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({"a": [100, 101, 102, 103, 104]}, index=dates)
        path = str(tmp_path / "prices.csv")
        df.to_csv(path)

        import logging
        with caplog.at_level(logging.WARNING):
            load_trial_matrix(path, config)
        assert any("price data" in m.lower() for m in caplog.messages)
