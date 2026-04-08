"""
Factor engine connector — generates a trial matrix by sweeping factor strategy parameters.

Financial rationale: the factor-backtest-engine (Project 3) computes cross-
sectional factor scores (Value, Momentum, Quality, Size, LowVol), ranks
stocks into quantiles, and measures long-short returns. By varying the
parameter grid (weights, lookback, rebalance frequency, weighting scheme,
quantile count), we produce many strategy variations to test for overfitting.

Each parameter combination produces one column of daily returns in the
trial matrix. Signal timing follows the original project: scores at t,
returns measured t → t+1 (no lookahead).
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.data_loader import compute_returns, load_prices
from src.grid_engine import generate_param_grid

logger = logging.getLogger(__name__)

# Weight schemes: maps name to factor weight vector
# Order: [value, momentum, quality, size, low_vol]
_WEIGHT_SCHEMES = {
    "equal": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
    "value_tilt": np.array([0.35, 0.15, 0.20, 0.15, 0.15]),
    "momentum_tilt": np.array([0.15, 0.35, 0.20, 0.15, 0.15]),
    "quality_tilt": np.array([0.15, 0.15, 0.35, 0.20, 0.15]),
}


def generate_trial_matrix(config: dict) -> tuple:
    """Run the factor engine across all parameter combinations.

    Produces a simplified factor model: each trial uses a composite
    score = weighted sum of 5 factor z-scores, built from SPY returns
    as a single-asset proxy. Real multi-stock factor models would import
    from factor-backtest-engine; this standalone version generates
    meaningful variation for PBO analysis.

    Parameters
    ----------
    config : dict
        Full config with factor_connector.grid.

    Returns
    -------
    tuple of (pd.DataFrame, list of dict)
        - trial_matrix: DatetimeIndex × trial IDs, values = daily returns
        - param_grid: list of dicts with trial_id + parameter values
    """
    grid_config = config["factor_connector"]["grid"]
    param_grid = generate_param_grid(grid_config)
    logger.info("Factor connector: %d parameter combinations.", len(param_grid))

    prices = load_prices(config)
    returns = compute_returns(prices)

    # Use the first column as the base return series
    base_col = returns.columns[0]
    base_returns = returns[base_col].dropna()

    trial_matrix = pd.DataFrame(index=base_returns.index)

    for params in param_grid:
        trial_id = str(params["trial_id"])
        trial_returns = _run_single_trial(base_returns, params, config)
        trial_matrix[trial_id] = trial_returns

    # Align and drop any NaN rows
    trial_matrix = trial_matrix.dropna()
    logger.info("Factor trial matrix: %d days × %d trials.",
                trial_matrix.shape[0], trial_matrix.shape[1])
    return trial_matrix, param_grid


def _run_single_trial(
    base_returns: pd.Series,
    params: dict,
    config: dict,
) -> pd.Series:
    """Generate daily returns for one parameter combination.

    Applies a simplified factor model: composite signal from rolling
    statistics, weighted by the chosen scheme, producing a long/short
    return series with parameter-dependent characteristics.
    """
    # Read connector defaults from config (fall back to hardcoded for backwards compat)
    cd = config.get("connector_defaults", {})
    trading_days_monthly = cd.get("trading_days_per_month", 21)
    trading_days_quarterly = cd.get("trading_days_per_quarter", 63)
    quantile_scale = cd.get("quantile_scale_factor", 0.05)
    cw_dampening = cd.get("cap_weight_dampening", 0.95)

    ann = config["ranking"]["annualization_factor"]
    lookback = params["lookback_months"] * trading_days_monthly  # approximate trading days
    weights_name = params["factor_weights"]
    weights = _WEIGHT_SCHEMES.get(weights_name, _WEIGHT_SCHEMES["equal"])

    n = len(base_returns)
    if lookback >= n:
        return pd.Series(np.nan, index=base_returns.index)

    # Simplified factor signals from rolling statistics of base returns
    rolling_mean = base_returns.rolling(lookback).mean()  # value proxy
    rolling_mom = base_returns.rolling(trading_days_quarterly).mean()  # momentum proxy
    rolling_vol = base_returns.rolling(lookback).std()  # low-vol proxy (inverted)
    rolling_skew = base_returns.rolling(lookback).skew()  # quality proxy
    rolling_size = base_returns.rolling(lookback).sum()  # size proxy (inverted)

    factors = pd.DataFrame({
        "value": rolling_mean,
        "momentum": rolling_mom,
        "quality": rolling_skew,
        "size": -rolling_size,  # inverted: smaller = better
        "low_vol": -rolling_vol,  # inverted: lower vol = better
    }).dropna()

    # Z-score normalisation (cross-sectional rank in full model)
    z_scores = (factors - factors.mean()) / factors.std().replace(0, np.nan)
    z_scores = z_scores.fillna(0)

    # Composite signal
    composite = z_scores @ weights

    # Rebalance frequency filter
    rebal = params["rebalance_freq"]
    rebal_days = trading_days_monthly if rebal == "monthly" else trading_days_quarterly
    signal = composite.copy()
    for i in range(len(signal)):
        if i % rebal_days != 0:
            signal.iloc[i] = np.nan
    signal = signal.ffill().fillna(0)

    # Position: sign of composite (long if positive, short if negative)
    position = np.sign(signal)

    # Shift signal by 1 day to avoid lookahead: signal at t, trade at t+1
    position = position.shift(1).fillna(0)

    # Trial returns = position * base returns
    trial_returns = position * base_returns.reindex(position.index)

    # Quantile scaling: more quantiles = more concentrated positions
    n_quantiles = params.get("n_quantiles", 5)
    scale = 1.0 + (n_quantiles - 5) * quantile_scale  # slight scaling with granularity

    # Weighting scheme adjustment
    if params["weighting"] == "cap_weight":
        scale *= cw_dampening  # cap-weight slightly dampens extreme positions

    trial_returns = trial_returns * scale
    return trial_returns.reindex(base_returns.index)
