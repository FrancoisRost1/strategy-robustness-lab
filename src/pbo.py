"""
Probability of Backtest Overfitting (PBO).

Financial rationale: when researchers try many strategy variations and
report only the best, the selected strategy's in-sample performance is
biased upward. PBO measures the probability that the IS-best strategy
underperforms the median OOS, a PBO near 1 means the top-ranked backtest
is almost certainly overfit.

The logit transformation maps relative OOS rank to (-inf, +inf), making
the distribution easier to analyse. PBO = P(logit > 0), i.e. the fraction
of combinations where the IS-best trial ranks in the bottom half OOS
(relative_rank > 0.5 → logit > 0 → underperformance).

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2014).
"""

import numpy as np
import pandas as pd


def compute_pbo(cscv_results: pd.DataFrame) -> dict:
    """Compute PBO from CSCV combination-level results.

    Parameters
    ----------
    cscv_results : pd.DataFrame
        Output of cscv.run_cscv(). Must contain columns:
        oos_rank, n_trials.

    Returns
    -------
    dict
        - pbo: float in [0, 1]
        - logits: np.ndarray of logit-transformed relative ranks
        - relative_ranks: np.ndarray in (0, 1)
        - n_combinations: int
    """
    n_trials = cscv_results["n_trials"].iloc[0]
    oos_ranks = cscv_results["oos_rank"].values.astype(float)

    # Relative rank: 1/N (best) to N/N (worst)
    relative_ranks = oos_ranks / n_trials

    # Clamp to avoid log(0) or log(inf) at boundaries
    eps = 1e-10
    clamped = np.clip(relative_ranks, eps, 1 - eps)

    # Logit: ln(r / (1 - r)). Positive logit = relative_rank > 0.5 = IS-best
    # ranks in the bottom half OOS (underperforms OOS median).
    logits = np.log(clamped / (1 - clamped))

    # PBO = fraction of combinations where IS-best underperforms OOS median
    # logit > 0 ↔ relative_rank > 0.5 ↔ bottom-half OOS
    pbo = float(np.mean(logits > 0))

    return {
        "pbo": pbo,
        "logits": logits,
        "relative_ranks": relative_ranks,
        "n_combinations": len(cscv_results),
    }


def pbo_convergence(cscv_results: pd.DataFrame, step: int = 100) -> pd.DataFrame:
    """Compute PBO as a function of number of combinations evaluated.

    Useful for checking whether PBO has stabilised, if it hasn't, more
    partitions (larger S) may be needed.

    Parameters
    ----------
    cscv_results : pd.DataFrame
        Output of cscv.run_cscv().
    step : int
        Evaluate PBO every `step` combinations.

    Returns
    -------
    pd.DataFrame
        Columns: n_evaluated, pbo_estimate.
    """
    n_trials = cscv_results["n_trials"].iloc[0]
    oos_ranks = cscv_results["oos_rank"].values.astype(float)
    relative_ranks = oos_ranks / n_trials

    eps = 1e-10
    clamped = np.clip(relative_ranks, eps, 1 - eps)
    logits = np.log(clamped / (1 - clamped))

    rows = []
    for n in range(step, len(logits) + 1, step):
        pbo_est = float(np.mean(logits[:n] > 0))
        rows.append({"n_evaluated": n, "pbo_estimate": pbo_est})

    # Always include the final count
    if len(logits) % step != 0:
        pbo_est = float(np.mean(logits > 0))
        rows.append({"n_evaluated": len(logits), "pbo_estimate": pbo_est})

    return pd.DataFrame(rows)
