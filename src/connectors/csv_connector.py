"""
CSV connector — loads an external pre-computed trial matrix.

Financial rationale: allows users to test any strategy for overfitting
without needing to integrate a full connector. The CSV must have rows = dates
(DatetimeIndex) and columns = trial/strategy variations, values = daily returns.

This is the simplest input path — no parameter grid, so Tab 5 (parameter
stability) is unavailable in CSV mode.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_trial_matrix(csv_path: str, config: dict) -> pd.DataFrame:
    """Load a trial matrix from CSV.

    Parameters
    ----------
    csv_path : str
        Path to CSV file. First column must be a date index.
        Remaining columns are trial IDs with daily return values.
    config : dict
        Full config (currently unused, reserved for future options).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex rows, columns = trial IDs, values = daily returns.

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    ValueError
        If the CSV has fewer than 2 columns (need at least 1 trial).
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if df.shape[1] < 1:
        raise ValueError(
            f"Trial matrix CSV must have at least 1 trial column, "
            f"got {df.shape[1]} columns."
        )

    # Handle NaN: drop rows with any NaN (no forward-fill for returns data)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        logger.warning(
            "Trial matrix has %d NaN values. Dropping affected rows.", n_nan
        )
        df = df.dropna()

    # Ensure column names are strings (trial IDs)
    df.columns = [str(c) for c in df.columns]

    # Validate: warn if values look like prices instead of returns
    if (df.abs() > 1).any().any():
        logger.warning(
            "Some values exceed ±1 — this looks like price data, not returns. "
            "Ensure the CSV contains daily returns, not prices."
        )

    logger.info(
        "Loaded trial matrix: %d days × %d trials from %s",
        df.shape[0], df.shape[1], csv_path,
    )
    return df
