"""
Parameter grid engine — generates all parameter combinations and orchestrates sweeps.

Financial rationale: to test for overfitting we need many strategy variations
evaluated on the same data. The grid engine takes a parameter grid from
config.yaml and produces every combination (Cartesian product), assigning
each a unique trial ID. Connectors consume these combinations to build
the trial matrix.
"""

import logging
from itertools import product
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_param_grid(grid_config: dict) -> List[dict]:
    """Generate Cartesian product of all parameter values.

    Parameters
    ----------
    grid_config : dict
        Keys = parameter names, values = lists of values to try.
        Example: {'lookback': [6, 12], 'weight': ['equal', 'cap']}

    Returns
    -------
    list of dict
        One dict per combination, with a 'trial_id' key (0-indexed).
        Example: [{'trial_id': 0, 'lookback': 6, 'weight': 'equal'}, ...]
    """
    if not grid_config:
        raise ValueError("Parameter grid is empty — nothing to sweep.")

    param_names = sorted(grid_config.keys())
    param_values = [grid_config[name] for name in param_names]

    combos = list(product(*param_values))
    logger.info("Generated %d parameter combinations from %d parameters.",
                len(combos), len(param_names))

    grid = []
    for trial_id, values in enumerate(combos):
        params = {"trial_id": trial_id}
        for name, val in zip(param_names, values):
            params[name] = val
        grid.append(params)

    return grid


def get_param_names(grid_config: dict) -> List[str]:
    """Return sorted list of parameter names from a grid config.

    Parameters
    ----------
    grid_config : dict
        Same format as generate_param_grid input.

    Returns
    -------
    list of str
    """
    return sorted(grid_config.keys())


def grid_summary(param_grid: List[dict]) -> Dict[str, list]:
    """Summarise the unique values for each parameter in the grid.

    Useful for UI display and validation.

    Parameters
    ----------
    param_grid : list of dict
        Output of generate_param_grid.

    Returns
    -------
    dict
        {param_name: [unique_values]} for each non-trial_id key.
    """
    if not param_grid:
        return {}

    summary = {}
    keys = [k for k in param_grid[0].keys() if k != "trial_id"]
    for key in keys:
        vals = sorted(set(p[key] for p in param_grid), key=lambda x: (isinstance(x, str), x))
        summary[key] = vals
    return summary
