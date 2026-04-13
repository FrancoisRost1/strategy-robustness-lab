"""
Out-of-sample degradation analysis.

Financial rationale: even if a strategy's IS-best consistently ranks high
OOS (low PBO), the *magnitude* of performance may decay dramatically.
Degradation analysis quantifies how much Sharpe ratio (or other metric)
investors should expect to lose when moving from backtest to live trading.

The degradation ratio = OOS metric / IS metric. Values near 1.0 indicate
minimal decay; values near 0 or negative indicate the backtest is
unreliable. The "sign flip rate", fraction of combinations where the
IS-best trial has negative OOS metric, is particularly damning.
"""

import numpy as np
import pandas as pd


def compute_degradation(cscv_results: pd.DataFrame) -> dict:
    """Compute IS-to-OOS degradation statistics.

    Parameters
    ----------
    cscv_results : pd.DataFrame
        Output of cscv.run_cscv(). Must contain columns:
        is_best_metric, oos_metric_of_is_best.

    Returns
    -------
    dict
        - degradation_ratios: np.ndarray (one per combination)
        - mean_degradation: float
        - median_degradation: float
        - std_degradation: float
        - sign_flip_rate: float (fraction where OOS metric < 0)
        - is_metrics: np.ndarray (IS metric of IS-best, per combo)
        - oos_metrics: np.ndarray (OOS metric of IS-best, per combo)
    """
    is_metrics = cscv_results["is_best_metric"].values.astype(float)
    oos_metrics = cscv_results["oos_metric_of_is_best"].values.astype(float)

    # Degradation ratio: OOS / IS. Only meaningful when IS > 0.
    # Negative/negative = positive which is misleading, so exclude IS <= 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            (is_metrics <= 0) | np.isnan(is_metrics),
            np.nan,
            oos_metrics / is_metrics,
        )

    n_unprofitable_is = int(np.sum(is_metrics <= 0))

    valid = ratios[~np.isnan(ratios)]

    mean_deg = float(np.nanmean(valid)) if len(valid) > 0 else np.nan
    median_deg = float(np.nanmedian(valid)) if len(valid) > 0 else np.nan
    std_deg = float(np.nanstd(valid)) if len(valid) > 0 else np.nan

    # True sign flip: IS metric positive but OOS metric negative
    n_positive_is = np.sum(is_metrics > 0)
    if n_positive_is > 0:
        n_flips = np.sum((is_metrics > 0) & (oos_metrics < 0))
        sign_flip = float(n_flips / n_positive_is)
    else:
        sign_flip = np.nan

    return {
        "degradation_ratios": ratios,
        "mean_degradation": mean_deg,
        "median_degradation": median_deg,
        "std_degradation": std_deg,
        "sign_flip_rate": sign_flip,
        "n_unprofitable_is": n_unprofitable_is,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
    }


def haircut_summary(degradation_result: dict) -> str:
    """Produce a human-readable haircut statement.

    E.g. 'Expect ~35% performance decay out-of-sample (median degradation 0.65).'
    """
    median = degradation_result["median_degradation"]
    if np.isnan(median):
        return "Insufficient data for haircut estimate."

    if median < 0 or median > 2.0:
        return (
            "Degradation ratio not meaningful "
            "(many strategies have negative IS performance)."
        )

    if median >= 1.0:
        return (
            f"No degradation detected (median ratio {median:.2f}). "
            "OOS performance matches or exceeds IS, unusual, verify data."
        )

    haircut_pct = (1 - median) * 100
    return (
        f"Expect ~{haircut_pct:.0f}% performance decay out-of-sample "
        f"(median degradation ratio {median:.2f})."
    )
