"""
Bootstrap inference — standard and block bootstrap.

Financial rationale: the Sharpe ratio is a point estimate. Bootstrap
resampling generates a distribution of Sharpe ratios, yielding confidence
intervals that account for sampling uncertainty. Block bootstrap preserves
the autocorrelation structure in daily returns (momentum, mean-reversion),
which standard i.i.d. bootstrap destroys.

Block size default = 21 trading days (~1 month), preserving monthly
seasonality and short-term serial dependence.
"""

import numpy as np
import pandas as pd

from src.metrics import compute_metric


def standard_bootstrap(
    returns: pd.Series,
    config: dict,
) -> dict:
    """Standard i.i.d. bootstrap of the ranking metric.

    Resamples individual daily returns with replacement, destroying
    temporal structure but providing a baseline distributional estimate.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of a single strategy.
    config : dict
        Uses bootstrap.n_resamples, bootstrap.confidence_level,
        bootstrap.random_seed.

    Returns
    -------
    dict
        - metric_distribution: np.ndarray of bootstrapped metric values
        - mean: float
        - ci_lower: float
        - ci_upper: float
        - std: float
    """
    boot_cfg = config.get("bootstrap", {})
    n_resamples = boot_cfg.get("n_resamples", 1000)
    conf = boot_cfg.get("confidence_level", 0.95)
    seed = boot_cfg.get("random_seed", 42)

    rng = np.random.RandomState(seed)
    n = len(returns)
    values = returns.values

    metrics = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        resampled = pd.Series(values[idx])
        metrics[i] = compute_metric(resampled, config)

    valid = metrics[~np.isnan(metrics)]
    alpha = (1 - conf) / 2

    return {
        "metric_distribution": metrics,
        "mean": float(np.nanmean(valid)) if len(valid) > 0 else np.nan,
        "ci_lower": float(np.nanpercentile(valid, alpha * 100)) if len(valid) > 0 else np.nan,
        "ci_upper": float(np.nanpercentile(valid, (1 - alpha) * 100)) if len(valid) > 0 else np.nan,
        "std": float(np.nanstd(valid)) if len(valid) > 0 else np.nan,
    }


def block_bootstrap(
    returns: pd.Series,
    config: dict,
) -> dict:
    """Block bootstrap — resamples contiguous blocks to preserve autocorrelation.

    Randomly selects blocks of consecutive daily returns (with replacement)
    and concatenates them to form a pseudo-series of the same length.
    This preserves short-term serial dependence (momentum, mean-reversion)
    that i.i.d. bootstrap would destroy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of a single strategy.
    config : dict
        Uses bootstrap.n_resamples, bootstrap.confidence_level,
        bootstrap.block_size, bootstrap.random_seed.

    Returns
    -------
    dict
        Same structure as standard_bootstrap output.
    """
    boot_cfg = config.get("bootstrap", {})
    n_resamples = boot_cfg.get("n_resamples", 1000)
    conf = boot_cfg.get("confidence_level", 0.95)
    block_size = boot_cfg.get("block_size", 21)
    seed = boot_cfg.get("random_seed", 42)

    rng = np.random.RandomState(seed)
    values = returns.values
    n = len(values)

    # Number of blocks needed to fill a series of length n
    n_blocks = int(np.ceil(n / block_size))

    metrics = np.empty(n_resamples)
    for i in range(n_resamples):
        # Randomly pick block start indices (circular if needed)
        starts = rng.randint(0, n, size=n_blocks)
        blocks = []
        for s in starts:
            end = s + block_size
            if end <= n:
                blocks.append(values[s:end])
            else:
                # Wrap around for blocks near the end
                blocks.append(np.concatenate([values[s:], values[:end - n]]))
        resampled = pd.Series(np.concatenate(blocks)[:n])
        metrics[i] = compute_metric(resampled, config)

    valid = metrics[~np.isnan(metrics)]
    alpha = (1 - conf) / 2

    return {
        "metric_distribution": metrics,
        "mean": float(np.nanmean(valid)) if len(valid) > 0 else np.nan,
        "ci_lower": float(np.nanpercentile(valid, alpha * 100)) if len(valid) > 0 else np.nan,
        "ci_upper": float(np.nanpercentile(valid, (1 - alpha) * 100)) if len(valid) > 0 else np.nan,
        "std": float(np.nanstd(valid)) if len(valid) > 0 else np.nan,
    }


def run_bootstrap(returns: pd.Series, config: dict) -> dict:
    """Run both standard and block bootstrap.

    Returns
    -------
    dict with keys 'standard' and 'block', each containing bootstrap results.
    """
    return {
        "standard": standard_bootstrap(returns, config),
        "block": block_bootstrap(returns, config),
    }
