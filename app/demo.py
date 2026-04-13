"""
Demo data generator, produces synthetic results for first-load dashboard.

Separated from app.py to avoid circular imports (app → tab_input → app).
"""

import numpy as np
import streamlit as st

from src.utils.config_loader import load_config


def get_config():
    """Load config once and cache in session_state for reuse across tabs."""
    if "config" not in st.session_state:
        st.session_state["config"] = load_config()
    return st.session_state["config"]


def generate_demo_results():
    """Run full pipeline on synthetic data so the dashboard has content on first load."""
    from src.pipeline import generate_synthetic
    from src.cscv import run_cscv
    from src.pbo import compute_pbo, pbo_convergence
    from src.degradation import compute_degradation, haircut_summary
    from src.metrics import compute_metric, compute_all_metrics, sharpe_ratio
    from src.deflated_sharpe import deflated_sharpe_ratio
    from src.bootstrap import run_bootstrap
    from src.stochastic_dominance import run_stochastic_dominance
    from src.parameter_stability import (
        build_metric_grid, plateau_detection, sensitivity_curves,
        pairwise_heatmap_data,
    )
    from src.verdict import classify

    config = get_config()
    # Use smaller params for fast demo
    config["cscv"]["n_partitions"] = 8
    config["cscv"]["max_combinations"] = 200
    config["bootstrap"]["n_resamples"] = 200
    config["synthetic"]["n_trials"] = 30
    config["synthetic"]["n_days"] = 1260  # ~5 years

    trial_matrix = generate_synthetic(config)

    # Build a fake param grid so stability tab works
    n_trials = trial_matrix.shape[1]
    param_grid = []
    p1_vals = [1, 2, 3, 5, 6]
    p2_vals = [10, 20, 30]
    p3_vals = ["A", "B"]
    idx = 0
    for p1 in p1_vals:
        for p2 in p2_vals:
            for p3 in p3_vals:
                if idx >= n_trials:
                    break
                param_grid.append({
                    "trial_id": idx,
                    "lookback": p1,
                    "vol_target": p2,
                    "scheme": p3,
                })
                idx += 1

    cscv_results = run_cscv(trial_matrix, config)
    pbo_result = compute_pbo(cscv_results)
    pbo_conv = pbo_convergence(cscv_results, step=10)
    degrad = compute_degradation(cscv_results)
    haircut = haircut_summary(degrad)

    full_metrics = {
        col: compute_metric(trial_matrix[col], config)
        for col in trial_matrix.columns
    }
    is_best = max(
        full_metrics,
        key=lambda k: full_metrics[k] if not np.isnan(full_metrics[k]) else -np.inf,
    )
    is_best_returns = trial_matrix[is_best]

    observed_sr = sharpe_ratio(is_best_returns, config)
    dsr_result = deflated_sharpe_ratio(
        observed_sr, is_best_returns.values, n_trials, config,
    )
    boot_result = run_bootstrap(is_best_returns, config)

    benchmark_type = config.get("stochastic_dominance", {}).get("benchmark", "equal_weight")
    benchmark_returns = trial_matrix.mean(axis=1).values
    sd_result = run_stochastic_dominance(
        is_best_returns.values, benchmark_returns, config,
    )

    param_names = ["lookback", "vol_target", "scheme"]
    mg = build_metric_grid(param_grid, full_metrics)
    plateau = plateau_detection(mg, config)
    sens = sensitivity_curves(mg, param_names)
    heatmaps = pairwise_heatmap_data(mg, param_names)

    pbo_val = pbo_result["pbo"]
    dsr_val = dsr_result["dsr"]
    plateau_frac = plateau["plateau_fraction"]
    verdict_result = classify(pbo_val, dsr_val, plateau_frac, config)

    return {
        "trial_matrix": trial_matrix,
        "param_grid": param_grid,
        "param_names": param_names,
        "cscv_results": cscv_results,
        "pbo": pbo_result,
        "pbo_convergence": pbo_conv,
        "degradation": degrad,
        "haircut": haircut,
        "deflated_sharpe": dsr_result,
        "bootstrap": boot_result,
        "stochastic_dominance": sd_result,
        "parameter_stability": plateau,
        "sensitivity": sens,
        "heatmaps": heatmaps,
        "metric_grid": mg,
        "verdict": verdict_result,
        "trial_metrics": full_metrics,
        "is_best_trial": is_best,
        "all_trial_metrics": {
            col: compute_all_metrics(trial_matrix[col], config)
            for col in trial_matrix.columns
        },
        "config": config,
    }
