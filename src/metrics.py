"""
Performance metrics, Sharpe, Sortino, Calmar, CAGR, MaxDD.

Financial rationale: these are the standard risk-adjusted return measures
used by allocators to rank strategies. All computations use excess returns
(net of risk-free rate) and annualise via a configurable factor (default 252
trading days). Edge cases (zero volatility, all-negative returns) return
np.nan to avoid misleading results.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, config: dict) -> float:
    """Annualised Sharpe ratio = mean(excess) / std(excess) * sqrt(ann).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    config : dict
        Must contain ranking.annualization_factor and data.risk_free_rate.

    Returns
    -------
    float
        Annualised Sharpe ratio, or np.nan if std == 0.
    """
    ann = config["ranking"]["annualization_factor"]
    rf_daily = config["data"]["risk_free_rate"] / ann
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(ann))


def sortino_ratio(returns: pd.Series, config: dict) -> float:
    """Annualised Sortino ratio, penalises only downside volatility.

    Uses downside deviation = sqrt(mean(min(0, excess)^2)) over ALL
    observations, not just the negative subset. This is the correct
    formula per Sortino & van der Meer (1991).
    """
    ann = config["ranking"]["annualization_factor"]
    rf_daily = config["data"]["risk_free_rate"] / ann
    excess = returns - rf_daily
    # Downside deviation: sqrt(mean(min(0, excess)^2)) over all observations
    downside_diff = np.minimum(excess, 0)
    downside_dev = np.sqrt((downside_diff ** 2).mean())
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return float(excess.mean() / downside_dev * np.sqrt(ann))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown, largest peak-to-trough decline.

    Measures the worst cumulative loss an investor would have experienced.
    Returns a negative number (e.g. -0.25 for a 25% drawdown).
    Prepends 1.0 as the starting equity value so that initial losses
    are correctly measured from the true starting point.
    """
    cumulative = (1 + returns).cumprod()
    # Prepend 1.0 as the starting equity value
    cumulative = pd.concat([pd.Series([1.0]), cumulative]).reset_index(drop=True)
    rolling_max = cumulative.cummax()
    drawdowns = cumulative / rolling_max - 1
    mdd = drawdowns.min()
    return float(mdd) if not np.isnan(mdd) else np.nan


def cagr(returns: pd.Series, config: dict) -> float:
    """Compound annual growth rate.

    Converts a daily returns series into an annualised geometric return,
    the standard measure for comparing strategy absolute performance.
    """
    ann = config["ranking"]["annualization_factor"]
    cumulative = (1 + returns).prod()
    n_years = len(returns) / ann
    if n_years <= 0 or cumulative <= 0:
        return np.nan
    return float(cumulative ** (1 / n_years) - 1)


def calmar_ratio(returns: pd.Series, config: dict) -> float:
    """Calmar ratio = CAGR / |MaxDD|.

    Measures return per unit of tail risk. Preferred by CTAs and macro
    funds where drawdown control is the binding constraint.
    """
    annual_return = cagr(returns, config)
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd) or np.isnan(annual_return):
        return np.nan
    return float(annual_return / abs(mdd))


# --- Metric dispatcher ---

_METRIC_REGISTRY = {
    "sharpe": sharpe_ratio,
    "sortino": sortino_ratio,
    "calmar": calmar_ratio,
}


def compute_metric(returns: pd.Series, config: dict) -> float:
    """Compute the ranking metric specified in config.ranking.metric.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    config : dict
        Full config dict.

    Returns
    -------
    float
        Metric value, or np.nan on error.
    """
    name = config["ranking"]["metric"]
    fn = _METRIC_REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown ranking metric '{name}'. "
            f"Available: {list(_METRIC_REGISTRY.keys())}"
        )
    return fn(returns, config)


def compute_all_metrics(returns: pd.Series, config: dict) -> dict:
    """Compute all standard metrics for a single returns series.

    Returns dict with keys: sharpe, sortino, calmar, cagr, max_drawdown.
    """
    return {
        "sharpe": sharpe_ratio(returns, config),
        "sortino": sortino_ratio(returns, config),
        "calmar": calmar_ratio(returns, config),
        "cagr": cagr(returns, config),
        "max_drawdown": max_drawdown(returns),
    }
