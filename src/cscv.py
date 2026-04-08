"""
Combinatorial Symmetric Cross-Validation (CSCV).

Financial rationale: standard train/test splits give a single point estimate
of out-of-sample quality. CSCV generates C(S, S/2) symmetric IS/OOS splits
from S contiguous time blocks, producing a full distribution of OOS
performance for the in-sample best strategy. This exposes whether strong
backtests are robust or merely artefacts of a specific data partition.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2014).
"""

import logging
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.metrics import compute_metric

logger = logging.getLogger(__name__)


def partition_blocks(trial_matrix: pd.DataFrame, n_partitions: int) -> List[pd.DataFrame]:
    """Split trial matrix into S contiguous, equal-length time blocks.

    If total days is not evenly divisible by S, leading rows are truncated
    (preserving the most recent data at the end).

    Parameters
    ----------
    trial_matrix : pd.DataFrame
        DatetimeIndex rows (dates), columns = trial IDs, values = daily returns.
    n_partitions : int
        Number of blocks S (must be even, validated upstream).

    Returns
    -------
    list of pd.DataFrame
        S blocks, each with the same number of rows.
    """
    n_rows = len(trial_matrix)
    block_size = n_rows // n_partitions
    if block_size < 1:
        raise ValueError(
            f"Not enough data: {n_rows} rows for {n_partitions} partitions "
            f"(need at least {n_partitions} rows)."
        )

    # Warn if blocks are very short
    if block_size < 60:
        logger.warning(
            "Block size = %d trading days (< 60). "
            "Results may be unreliable with very short blocks.", block_size
        )

    # Truncate leading rows so total is divisible by S
    remainder = n_rows % n_partitions
    if remainder > 0:
        logger.info("Truncating %d leading rows for even partition.", remainder)
    data = trial_matrix.iloc[remainder:]

    blocks = []
    for i in range(n_partitions):
        start = i * block_size
        end = start + block_size
        blocks.append(data.iloc[start:end])
    return blocks


def generate_combinations(
    n_partitions: int, max_combinations: int = None, random_seed: int = 42
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate all C(S, S/2) IS/OOS partition index combinations.

    Parameters
    ----------
    n_partitions : int
        Number of blocks S.
    max_combinations : int, optional
        If set, randomly sample this many combinations (memory safety).
    random_seed : int
        Seed for reproducibility when sampling.

    Returns
    -------
    list of (is_indices, oos_indices)
        Each element is a pair of tuples with block indices.
    """
    half = n_partitions // 2
    all_indices = list(range(n_partitions))
    all_combos = list(combinations(all_indices, half))

    pairs = []
    for is_idx in all_combos:
        oos_idx = tuple(i for i in all_indices if i not in is_idx)
        pairs.append((is_idx, oos_idx))

    if max_combinations and max_combinations < len(pairs):
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(len(pairs), size=max_combinations, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        logger.info(
            "Sampled %d of %d combinations.", max_combinations, len(all_combos)
        )

    return pairs


def run_cscv(trial_matrix: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Execute full CSCV analysis.

    For each combination, ranks all trials on IS data, identifies the
    IS-best trial, and records that trial's OOS rank and metric values.

    Parameters
    ----------
    trial_matrix : pd.DataFrame
        Daily returns matrix (rows = dates, cols = trial IDs).
    config : dict
        Full configuration.

    Returns
    -------
    pd.DataFrame
        One row per combination with columns:
        - combo_id, is_indices, oos_indices
        - is_best_trial, is_best_metric, oos_metric_of_is_best
        - oos_rank, n_trials
    """
    cscv_cfg = config["cscv"]
    n_partitions = cscv_cfg["n_partitions"]
    max_combos = cscv_cfg.get("max_combinations")
    seed = cscv_cfg.get("random_seed", 42)

    blocks = partition_blocks(trial_matrix, n_partitions)
    combos = generate_combinations(n_partitions, max_combos, seed)

    n_trials = trial_matrix.shape[1]
    results = []

    for combo_id, (is_idx, oos_idx) in enumerate(combos):
        # Assemble IS and OOS data by concatenating selected blocks
        is_data = pd.concat([blocks[i] for i in is_idx])
        oos_data = pd.concat([blocks[i] for i in oos_idx])

        # Compute ranking metric for every trial on IS and OOS
        is_scores = {col: compute_metric(is_data[col], config) for col in trial_matrix.columns}
        oos_scores = {col: compute_metric(oos_data[col], config) for col in trial_matrix.columns}

        # Rank trials by IS metric (highest = rank 1)
        is_ranked = sorted(is_scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        is_best_trial = is_ranked[0][0]
        is_best_metric = is_ranked[0][1]

        # OOS rank of the IS-best trial (1 = best, N = worst)
        oos_ranked = sorted(oos_scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        oos_rank = next(i + 1 for i, (col, _) in enumerate(oos_ranked) if col == is_best_trial)

        results.append({
            "combo_id": combo_id,
            "is_indices": is_idx,
            "oos_indices": oos_idx,
            "is_best_trial": is_best_trial,
            "is_best_metric": is_best_metric,
            "oos_metric_of_is_best": oos_scores[is_best_trial],
            "oos_rank": oos_rank,
            "n_trials": n_trials,
        })

    return pd.DataFrame(results)
