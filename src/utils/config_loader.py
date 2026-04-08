"""
Config loader — loads YAML configuration once and returns as dict.

Financial rationale: centralised configuration ensures all thresholds
(PBO cutoffs, plateau tolerances, bootstrap parameters) are auditable
in one place and never hardcoded in analytics code.
"""

import os
import yaml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config.yaml"
)


def load_config(path: str = None) -> dict:
    """Load config.yaml and return as a plain dict.

    Parameters
    ----------
    path : str, optional
        Absolute or relative path to YAML file. Falls back to the
        project-root config.yaml.

    Returns
    -------
    dict
        Full configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    ValueError
        If the YAML is empty or unparseable.
    """
    resolved = path or _DEFAULT_CONFIG_PATH
    resolved = os.path.abspath(resolved)

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg)}")

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Run basic sanity checks on the loaded config.

    Raises ValueError for invalid settings that would cause downstream
    errors (e.g. odd partition count).
    """
    cscv = cfg.get("cscv", {})
    n_partitions = cscv.get("n_partitions", 16)
    if n_partitions % 2 != 0:
        raise ValueError(
            f"cscv.n_partitions must be even (got {n_partitions}). "
            "CSCV requires symmetric splits."
        )
    if n_partitions < 4:
        raise ValueError(
            f"cscv.n_partitions must be >= 4 (got {n_partitions})."
        )

    ann = cfg.get("ranking", {}).get("annualization_factor", 252)
    if ann <= 0:
        raise ValueError(f"ranking.annualization_factor must be > 0 (got {ann}).")
