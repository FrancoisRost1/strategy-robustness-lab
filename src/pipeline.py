"""
Pipeline orchestration, runs the full PBO/CSCV robustness analysis.

Contains the core pipeline logic extracted from main.py so that main.py
remains a thin CLI orchestrator (argparse + dispatch only).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_pipeline(config: dict, mode: str = "tsmom") -> dict:
    """Execute the full robustness analysis pipeline.

    Parameters
    ----------
    config : dict
        Full config dict.
    mode : str
        Connector mode: 'tsmom', 'factor', or 'csv'.

    Returns a dict with all diagnostic results for downstream
    consumption (dashboard, export, etc.).
    """
    from src.bootstrap import run_bootstrap
    from src.cscv import run_cscv
    from src.deflated_sharpe import deflated_sharpe_ratio
    from src.degradation import compute_degradation, haircut_summary
    from src.metrics import compute_all_metrics, compute_metric, sharpe_ratio
    from src.pbo import compute_pbo, pbo_convergence
    from src.stochastic_dominance import run_stochastic_dominance
    from src.verdict import classify

    # --- Step 1: Generate trial matrix ---
    trial_matrix, param_grid = _load_trial_matrix(config, mode=mode)
    n_trials = trial_matrix.shape[1]
    logger.info("Trial matrix: %d days × %d trials.", len(trial_matrix), n_trials)

    if n_trials < 2:
        logger.warning("Only %d trial(s). PBO requires ≥ 2. Running bootstrap only.", n_trials)

    # --- Step 2: CSCV ---
    cscv_results = run_cscv(trial_matrix, config) if n_trials >= 2 else None

    # --- Step 3: PBO ---
    pbo_result = compute_pbo(cscv_results) if cscv_results is not None else None
    pbo_conv = pbo_convergence(cscv_results) if cscv_results is not None else None

    # --- Step 4: Degradation ---
    degrad = compute_degradation(cscv_results) if cscv_results is not None else None
    haircut = haircut_summary(degrad) if degrad else "N/A"

    # --- Step 5: Identify IS-best overall ---
    full_metrics = {col: compute_metric(trial_matrix[col], config) for col in trial_matrix.columns}
    is_best_trial = max(full_metrics, key=lambda k: full_metrics[k] if not np.isnan(full_metrics[k]) else -np.inf)
    is_best_returns = trial_matrix[is_best_trial]

    # --- Step 6: Deflated Sharpe ---
    observed_sr = sharpe_ratio(is_best_returns, config)
    dsr_result = deflated_sharpe_ratio(observed_sr, is_best_returns.values, n_trials, config)

    # --- Step 7: Bootstrap ---
    boot_result = run_bootstrap(is_best_returns, config)

    # --- Step 8: Stochastic dominance (OOS returns only) ---
    # Split trial matrix in half: first half = IS, second half = OOS.
    # Use OOS portion to avoid inflating dominance with in-sample fit.
    midpoint = len(trial_matrix) // 2
    oos_slice = trial_matrix.iloc[midpoint:]
    oos_best_returns = oos_slice[is_best_trial].values
    # Benchmark type from config (currently only equal_weight implemented)
    benchmark_type = config.get("stochastic_dominance", {}).get(
        "benchmark", "equal_weight"
    )
    if benchmark_type == "equal_weight":
        oos_benchmark_returns = oos_slice.mean(axis=1).values
    else:
        # Default fallback to equal-weight average of all trials
        oos_benchmark_returns = oos_slice.mean(axis=1).values
    sd_result = run_stochastic_dominance(
        oos_best_returns, oos_benchmark_returns, config
    )

    # --- Step 9: Parameter stability ---
    plateau_result = _run_stability(param_grid, full_metrics, config)

    # --- Step 10: Verdict ---
    pbo_val = pbo_result["pbo"] if pbo_result else np.nan
    dsr_val = dsr_result["dsr"]
    plateau_frac = plateau_result["plateau_fraction"] if plateau_result else None
    verdict_result = classify(pbo_val, dsr_val, plateau_frac, config)

    # --- Compile results ---
    results = {
        "trial_matrix": trial_matrix,
        "param_grid": param_grid,
        "cscv_results": cscv_results,
        "pbo": pbo_result,
        "pbo_convergence": pbo_conv,
        "degradation": degrad,
        "haircut": haircut,
        "deflated_sharpe": dsr_result,
        "bootstrap": boot_result,
        "stochastic_dominance": sd_result,
        "parameter_stability": plateau_result,
        "verdict": verdict_result,
        "trial_metrics": full_metrics,
        "is_best_trial": is_best_trial,
        "all_trial_metrics": {col: compute_all_metrics(trial_matrix[col], config) for col in trial_matrix.columns},
    }

    _print_summary(results)
    return results


def _load_trial_matrix(config: dict, mode: str = "tsmom") -> tuple:
    """Route to the correct connector based on config and mode.

    Parameters
    ----------
    config : dict
        Full config dict.
    mode : str
        One of 'tsmom', 'factor', 'csv'. Determines which connector to use.
    """
    # Check for CSV mode first (explicit mode or config-based)
    if mode == "csv" or (config["data"].get("source") == "csv" and config["data"].get("csv_path")):
        from src.connectors.csv_connector import load_trial_matrix
        csv_path = config["data"]["csv_path"]
        return load_trial_matrix(csv_path, config), []

    if mode == "factor":
        from src.connectors.factor_connector import generate_trial_matrix
        return generate_trial_matrix(config)

    # Default to TSMOM connector
    from src.connectors.tsmom_connector import generate_trial_matrix
    return generate_trial_matrix(config)


def _run_stability(param_grid, trial_metrics, config):
    """Run parameter stability if a param grid is available."""
    if not param_grid:
        return None
    from src.parameter_stability import build_metric_grid, plateau_detection
    metric_grid = build_metric_grid(param_grid, {str(p["trial_id"]): trial_metrics.get(str(p["trial_id"]), np.nan) for p in param_grid})
    return plateau_detection(metric_grid, config)


def _print_summary(results: dict) -> None:
    """Print a concise summary to the console."""
    v = results["verdict"]
    logger.info("=" * 60)
    logger.info("VERDICT: %s (%s)", v["verdict"], v["color"])
    logger.info("=" * 60)
    if results["pbo"]:
        logger.info("PBO: %.3f", results["pbo"]["pbo"])
    logger.info("DSR: %.3f (significant: %s)",
                results["deflated_sharpe"]["dsr"],
                results["deflated_sharpe"]["is_significant"])
    if results["degradation"]:
        logger.info("Mean degradation: %.2f | Sign flip rate: %.1f%%",
                    results["degradation"]["mean_degradation"],
                    results["degradation"]["sign_flip_rate"] * 100)
    logger.info("Haircut: %s", results["haircut"])
    logger.info(v["details"])


def generate_synthetic(config: dict) -> pd.DataFrame:
    """Generate a synthetic trial matrix for testing.

    Creates n_trials strategies where one has a planted alpha signal
    and the rest are pure noise, useful for validating the PBO pipeline.
    """
    syn = config.get("synthetic", {})
    n_trials = syn.get("n_trials", 50)
    n_days = syn.get("n_days", 2520)
    signal = syn.get("signal_strength", 0.02)
    noise = syn.get("noise_std", 0.01)
    seed = syn.get("random_seed", 42)

    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    data = {}
    for i in range(n_trials):
        daily_alpha = signal / 252 if i == 0 else 0  # only trial 0 has signal
        data[str(i)] = rng.normal(daily_alpha, noise, n_days)

    return pd.DataFrame(data, index=dates)
