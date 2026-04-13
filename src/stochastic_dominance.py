"""
Stochastic dominance tests.

Financial rationale: a strategy that first-order stochastically dominates
a benchmark is unambiguously preferred by all investors who prefer more
to less. Second-order dominance (integrated CDF) adds the constraint of
risk aversion, relevant for institutional allocators.

Uses the Kolmogorov-Smirnov test for first-order dominance and cumulative
CDF integration for second-order dominance.
"""

import numpy as np
from scipy import stats


def first_order_dominance(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: dict,
) -> dict:
    """Test first-order stochastic dominance via KS test.

    H0: strategy and benchmark returns are drawn from the same distribution.
    Rejection suggests the CDFs differ; direction checked separately.

    Parameters
    ----------
    strategy_returns : np.ndarray
        OOS daily returns of the IS-best strategy.
    benchmark_returns : np.ndarray
        OOS daily returns of the benchmark (e.g. equal-weight of all trials).
    config : dict
        Uses stochastic_dominance.significance_level.

    Returns
    -------
    dict
        - ks_statistic: float
        - p_value: float
        - rejects_null: bool (at configured alpha)
        - strategy_dominates: bool (strategy CDF is always below benchmark CDF)
    """
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)

    # Remove NaNs
    strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
    benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]

    alpha = config.get("stochastic_dominance", {}).get("significance_level", 0.05)

    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return {
            "ks_statistic": np.nan,
            "p_value": np.nan,
            "rejects_null": False,
            "strategy_dominates": False,
        }

    ks_stat, p_val = stats.ks_2samp(strategy_returns, benchmark_returns)

    # Check direction: strategy FSD benchmark iff F_s(x) <= F_b(x) for all x
    # (CDF of strategy is never above CDF of benchmark → higher returns)
    all_vals = np.sort(np.concatenate([strategy_returns, benchmark_returns]))
    s_cdf = np.searchsorted(np.sort(strategy_returns), all_vals, side="right") / len(strategy_returns)
    b_cdf = np.searchsorted(np.sort(benchmark_returns), all_vals, side="right") / len(benchmark_returns)
    dominates = bool(np.all(s_cdf <= b_cdf + 1e-10))

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_val),
        "rejects_null": p_val < alpha,
        "strategy_dominates": dominates,
    }


def second_order_dominance(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict:
    """Test second-order stochastic dominance via integrated CDF comparison.

    Strategy SSD-dominates benchmark iff the integral of (F_b - F_s) >= 0
    at every point, i.e., the cumulative area under the benchmark CDF is
    always at least as large as under the strategy CDF.

    Relevant for risk-averse investors: SSD implies higher expected utility
    for all concave utility functions.

    Parameters
    ----------
    strategy_returns, benchmark_returns : np.ndarray
        Daily return arrays.

    Returns
    -------
    dict
        - dominates: bool
        - min_integrated_diff: float (most negative gap; >=0 means dominance)
        - mean_integrated_diff: float
    """
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
    benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]

    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return {
            "dominates": False,
            "min_integrated_diff": np.nan,
            "mean_integrated_diff": np.nan,
        }

    # Evaluate CDFs on a common grid
    all_vals = np.sort(np.unique(np.concatenate([strategy_returns, benchmark_returns])))
    s_sorted = np.sort(strategy_returns)
    b_sorted = np.sort(benchmark_returns)

    s_cdf = np.searchsorted(s_sorted, all_vals, side="right") / len(s_sorted)
    b_cdf = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)

    # Integrated CDF difference: cumulative sum of (F_b - F_s) * dx
    # Use trapezoidal integration over the sorted grid
    diffs = b_cdf - s_cdf
    if len(all_vals) > 1:
        dx = np.diff(all_vals)
        avg_diff = (diffs[:-1] + diffs[1:]) / 2
        integrated = np.cumsum(avg_diff * dx)
    else:
        integrated = np.array([0.0])

    min_int = float(np.min(integrated)) if len(integrated) > 0 else 0.0
    mean_int = float(np.mean(integrated)) if len(integrated) > 0 else 0.0

    return {
        "dominates": min_int >= -1e-10,
        "min_integrated_diff": min_int,
        "mean_integrated_diff": mean_int,
    }


def run_stochastic_dominance(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    config: dict,
) -> dict:
    """Run both first-order and (optionally) second-order SD tests.

    Parameters
    ----------
    strategy_returns, benchmark_returns : np.ndarray
        Daily return arrays.
    config : dict
        Full config dict.

    Returns
    -------
    dict with keys 'first_order' and optionally 'second_order'.
    """
    result = {
        "first_order": first_order_dominance(strategy_returns, benchmark_returns, config),
    }

    if config.get("stochastic_dominance", {}).get("test_second_order", True):
        result["second_order"] = second_order_dominance(strategy_returns, benchmark_returns)

    return result
