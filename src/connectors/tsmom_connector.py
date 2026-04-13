"""
TSMOM connector, generates a trial matrix by sweeping time-series momentum parameters.

Financial rationale: the tsmom-engine (Project 6) implements Moskowitz et al. (2012)
cross-asset time-series momentum. Signal = sign(past return), position sized
by inverse vol targeting. By varying lookback, vol target, position/gross caps,
and rebalance frequency, we produce diverse strategy variations to stress-test
for overfitting via CSCV/PBO.

Signal timing: signal at close(t), trade at close(t+1), strict no-lookahead.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.data_loader import compute_returns, load_prices
from src.grid_engine import generate_param_grid

logger = logging.getLogger(__name__)


def generate_trial_matrix(config: dict) -> tuple:
    """Run TSMOM strategy across all parameter combinations.

    Parameters
    ----------
    config : dict
        Full config with tsmom_connector.grid and tsmom_connector.universe.

    Returns
    -------
    tuple of (pd.DataFrame, list of dict)
        - trial_matrix: DatetimeIndex × trial IDs, values = daily portfolio returns
        - param_grid: list of dicts with trial_id + parameter values
    """
    grid_config = config["tsmom_connector"]["grid"]
    param_grid = generate_param_grid(grid_config)
    logger.info("TSMOM connector: %d parameter combinations.", len(param_grid))

    prices = load_prices(config)
    returns = compute_returns(prices)

    trial_matrix = pd.DataFrame(index=returns.index)

    for params in param_grid:
        trial_id = str(params["trial_id"])
        trial_returns = _run_single_trial(returns, params, config)
        trial_matrix[trial_id] = trial_returns

    trial_matrix = trial_matrix.dropna()
    logger.info("TSMOM trial matrix: %d days × %d trials.",
                trial_matrix.shape[0], trial_matrix.shape[1])
    return trial_matrix, param_grid


def _run_single_trial(
    returns: pd.DataFrame,
    params: dict,
    config: dict,
) -> pd.Series:
    """Generate daily portfolio returns for one TSMOM parameter combination.

    Implements the core TSMOM logic: sign(past return) × vol-targeted weight.
    """
    lookback = params["momentum_lookback"]
    vol_target = params["vol_target"]
    pos_cap = params["position_cap"]
    gross_cap = params["gross_cap"]
    rebal_freq = params["rebalance_freq"]
    ann = config["ranking"]["annualization_factor"]

    # Read connector defaults from config (fall back to hardcoded for backwards compat)
    cd = config.get("connector_defaults", {})
    trading_days_monthly = cd.get("trading_days_per_month", 21)
    biweekly_days = cd.get("biweekly_trading_days", 10)
    ewma_hl = cd.get("ewma_halflife", 60)

    n_assets = returns.shape[1]
    portfolio_returns = pd.Series(0.0, index=returns.index)

    # Rebalance schedule
    rebal_days = trading_days_monthly if rebal_freq == "monthly" else biweekly_days

    # Pre-compute signals and vol estimates for all assets
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    for col in returns.columns:
        asset_ret = returns[col]

        # TSMOM signal: sign of cumulative return over lookback period
        cum_ret = asset_ret.rolling(lookback).sum()
        signal = np.sign(cum_ret)

        # Realised vol: EWMA with configurable halflife, annualised
        realised_vol = asset_ret.ewm(halflife=ewma_hl).std() * np.sqrt(ann)
        realised_vol = realised_vol.replace(0, np.nan)

        # Vol-targeted weight: signal × (target_vol / realised_vol) / n_assets
        raw_weight = signal * (vol_target / realised_vol) / n_assets

        # Per-asset position cap
        raw_weight = raw_weight.clip(-pos_cap, pos_cap)

        weights[col] = raw_weight

    # Apply rebalance frequency: only update weights on rebal dates
    rebal_mask = pd.Series(False, index=returns.index)
    rebal_indices = list(range(lookback, len(returns), rebal_days))
    for idx in rebal_indices:
        if idx < len(rebal_mask):
            rebal_mask.iloc[idx] = True

    # Forward-fill weights between rebalance dates
    for col in weights.columns:
        held = weights[col].copy()
        held[~rebal_mask] = np.nan
        held = held.ffill().fillna(0)
        weights[col] = held

    # Gross exposure cap: scale down if total |weights| exceeds gross_cap
    gross_exposure = weights.abs().sum(axis=1)
    scale = (gross_cap / gross_exposure).clip(upper=1.0)
    weights = weights.mul(scale, axis=0)

    # Shift weights by 1 day: signal at t, trade at t+1 (no lookahead)
    weights = weights.shift(1).fillna(0)

    # Portfolio return = sum of (weight × asset return) across assets
    portfolio_returns = (weights * returns).sum(axis=1)

    return portfolio_returns
