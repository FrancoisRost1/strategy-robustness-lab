"""
Parameter stability analysis, heatmaps, sensitivity curves, plateau detection.

Financial rationale: a robust strategy should perform well across a wide
range of parameter values, not just at a single optimised point. Plateau
detection measures what fraction of the parameter grid produces near-optimal
performance. A narrow peak suggests the strategy is tuned to noise; a broad
plateau suggests genuine economic signal.

Only available for built-in connectors (factor engine, TSMOM) where the
parameter grid is known. CSV uploads lack parameter metadata.
"""

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_metric_grid(
    param_grid: List[dict],
    trial_metrics: Dict[str, float],
) -> pd.DataFrame:
    """Map each trial's parameters to its metric value.

    Parameters
    ----------
    param_grid : list of dict
        One dict per trial with parameter names as keys + 'trial_id'.
    trial_metrics : dict
        {trial_id: metric_value} for every trial.

    Returns
    -------
    pd.DataFrame
        Columns = parameter names + 'metric'. One row per trial.
    """
    rows = []
    for params in param_grid:
        trial_id = params["trial_id"]
        row = {k: v for k, v in params.items() if k != "trial_id"}
        # Look up metric by trial_id, trying both str and int keys
        row["metric"] = trial_metrics.get(str(trial_id),
                         trial_metrics.get(trial_id, np.nan))
        row["trial_id"] = trial_id
        rows.append(row)
    return pd.DataFrame(rows)


def plateau_detection(metric_grid: pd.DataFrame, config: dict) -> dict:
    """Detect whether the parameter space has a stable performance plateau.

    plateau_fraction = (cells within tolerance of best) / total cells.
    A high fraction means performance is insensitive to parameter choice
   , the signal is real, not an artefact of specific tuning.

    Parameters
    ----------
    metric_grid : pd.DataFrame
        Output of build_metric_grid.
    config : dict
        Uses parameter_stability.plateau_tolerance,
        parameter_stability.stable_threshold,
        parameter_stability.moderate_threshold.

    Returns
    -------
    dict
        - plateau_fraction: float
        - classification: str (STABLE / MODERATE / FRAGILE)
        - best_metric: float
        - threshold_metric: float (best * (1 - tolerance))
        - n_plateau_cells: int
        - n_total_cells: int
    """
    stab_cfg = config.get("parameter_stability", {})
    tolerance = stab_cfg.get("plateau_tolerance", 0.10)
    stable_thresh = stab_cfg.get("stable_threshold", 0.30)
    moderate_thresh = stab_cfg.get("moderate_threshold", 0.10)

    metrics = metric_grid["metric"].dropna()
    if len(metrics) == 0:
        return {
            "plateau_fraction": np.nan,
            "classification": "UNKNOWN",
            "best_metric": np.nan,
            "threshold_metric": np.nan,
            "n_plateau_cells": 0,
            "n_total_cells": 0,
        }

    best = metrics.max()

    # Handle negative best metric: tolerance widens the acceptable range
    if best > 0:
        threshold = best * (1 - tolerance)
    else:
        # For negative metrics, "within tolerance" means going more negative
        # (wider band), so threshold is LOWER than best
        threshold = best - abs(best) * tolerance

    n_plateau = int((metrics >= threshold).sum())
    n_total = len(metrics)
    fraction = n_plateau / n_total

    if fraction > stable_thresh:
        classification = "STABLE"
    elif fraction > moderate_thresh:
        classification = "MODERATE"
    else:
        classification = "FRAGILE"

    return {
        "plateau_fraction": float(fraction),
        "classification": classification,
        "best_metric": float(best),
        "threshold_metric": float(threshold),
        "n_plateau_cells": n_plateau,
        "n_total_cells": n_total,
    }


def sensitivity_curves(
    metric_grid: pd.DataFrame,
    param_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """Compute sensitivity curves: vary one parameter, hold others at grid-median.

    For each parameter, groups trials by that parameter's value (with all
    other parameters at their median/mode), and returns the metric mean ± std.

    Parameters
    ----------
    metric_grid : pd.DataFrame
        Output of build_metric_grid.
    param_names : list of str
        Parameter column names to analyse.

    Returns
    -------
    dict of {param_name: pd.DataFrame}
        Each DataFrame has columns: param_value, metric_mean, metric_std.
    """
    result = {}
    for param in param_names:
        other_params = [p for p in param_names if p != param]

        # Find median/mode for other parameters to hold constant
        baseline = {}
        for op in other_params:
            col = metric_grid[op]
            if col.dtype in ("float64", "int64"):
                median_val = col.median()
                # Snap to nearest actual grid value (median may not exist in grid)
                unique_vals = col.unique()
                baseline[op] = unique_vals[np.argmin(np.abs(unique_vals - median_val))]
            else:
                baseline[op] = col.mode().iloc[0] if len(col.mode()) > 0 else col.iloc[0]

        # Filter to rows matching baseline for other params
        mask = pd.Series(True, index=metric_grid.index)
        for op, val in baseline.items():
            mask &= (metric_grid[op] == val)

        subset = metric_grid[mask]
        if len(subset) == 0:
            # Fallback: use all data grouped by this parameter
            subset = metric_grid

        grouped = subset.groupby(param)["metric"].agg(["mean", "std"]).reset_index()
        grouped.columns = ["param_value", "metric_mean", "metric_std"]
        grouped["metric_std"] = grouped["metric_std"].fillna(0)
        result[param] = grouped

    return result


def pairwise_heatmap_data(
    metric_grid: pd.DataFrame,
    param_names: List[str],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Generate 2D heatmap data for all pairwise parameter combinations.

    For each pair, other parameters are held at their median/mode.
    Returns a pivot table suitable for seaborn/plotly heatmap rendering.

    Parameters
    ----------
    metric_grid : pd.DataFrame
        Output of build_metric_grid.
    param_names : list of str
        Parameter column names.

    Returns
    -------
    dict of {(param_a, param_b): pd.DataFrame}
        Each value is a pivot table (rows=param_a, cols=param_b, values=metric).
    """
    result = {}
    for pa, pb in combinations(param_names, 2):
        others = [p for p in param_names if p not in (pa, pb)]

        # Baseline for other params
        baseline = {}
        for op in others:
            col = metric_grid[op]
            if col.dtype in ("float64", "int64"):
                median_val = col.median()
                # Snap to nearest actual grid value (median may not exist in grid)
                unique_vals = col.unique()
                baseline[op] = unique_vals[np.argmin(np.abs(unique_vals - median_val))]
            else:
                baseline[op] = col.mode().iloc[0] if len(col.mode()) > 0 else col.iloc[0]

        mask = pd.Series(True, index=metric_grid.index)
        for op, val in baseline.items():
            mask &= (metric_grid[op] == val)

        subset = metric_grid[mask] if mask.any() else metric_grid
        pivot = subset.pivot_table(index=pa, columns=pb, values="metric", aggfunc="mean")
        result[(pa, pb)] = pivot

    return result
