"""
Strategy Robustness Lab — Streamlit dashboard entry point.

Bloomberg dark mode. 6 tabs. No business logic here — delegates to tab modules.
Generates synthetic demo data on first load so dashboard is never empty.
"""

import sys
import os

# Ensure project root is on path so src imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="Strategy Robustness Lab",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.style_inject import (
    inject_styles,
    styled_header,
    styled_divider,
    styled_section_label,
    TOKENS,
)

inject_styles()

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config


def _get_config():
    """Load config once and cache in session_state for reuse across tabs."""
    if "config" not in st.session_state:
        st.session_state["config"] = load_config()
    return st.session_state["config"]


def _generate_demo_results():
    """Run full pipeline on synthetic data so the dashboard has content on first load."""
    from src.pipeline import generate_synthetic

    config = _get_config()
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

    # Run the full pipeline using the trial matrix directly
    from src.cscv import run_cscv
    from src.pbo import compute_pbo, pbo_convergence
    from src.degradation import compute_degradation, haircut_summary
    from src.metrics import compute_metric, compute_all_metrics, sharpe_ratio
    from src.deflated_sharpe import deflated_sharpe_ratio
    from src.bootstrap import run_bootstrap
    from src.stochastic_dominance import run_stochastic_dominance
    from src.parameter_stability import build_metric_grid, plateau_detection, sensitivity_curves, pairwise_heatmap_data
    from src.verdict import classify

    cscv_results = run_cscv(trial_matrix, config)
    pbo_result = compute_pbo(cscv_results)
    pbo_conv = pbo_convergence(cscv_results, step=10)
    degrad = compute_degradation(cscv_results)
    haircut = haircut_summary(degrad)

    full_metrics = {col: compute_metric(trial_matrix[col], config) for col in trial_matrix.columns}
    is_best = max(full_metrics, key=lambda k: full_metrics[k] if not np.isnan(full_metrics[k]) else -np.inf)
    is_best_returns = trial_matrix[is_best]

    observed_sr = sharpe_ratio(is_best_returns, config)
    dsr_result = deflated_sharpe_ratio(observed_sr, is_best_returns.values, n_trials, config)
    boot_result = run_bootstrap(is_best_returns, config)
    # Benchmark type from config (currently only equal_weight implemented)
    benchmark_type = config.get("stochastic_dominance", {}).get("benchmark", "equal_weight")
    if benchmark_type == "equal_weight":
        benchmark_returns = trial_matrix.mean(axis=1).values
    else:
        benchmark_returns = trial_matrix.mean(axis=1).values
    sd_result = run_stochastic_dominance(is_best_returns.values, benchmark_returns, config)

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


def main():
    """Dashboard entry point."""
    # Sidebar
    with st.sidebar:
        st.markdown(
            f"<h2 style='color: {TOKENS['accent_primary']}; "
            f"font-family: {TOKENS['font_display']}; font-weight: 700;'>"
            f"◆ Robustness Lab</h2>",
            unsafe_allow_html=True,
        )
        styled_section_label("About")
        st.markdown(
            f"<span style='font-size: 0.85rem; color: {TOKENS['text_secondary']};'>"
            "Strategy overfitting detection via CSCV, PBO, and deflated Sharpe ratio. "
            "Bailey, Borwein, López de Prado & Zhu (2014)."
            "</span>",
            unsafe_allow_html=True,
        )
        styled_divider()
        styled_section_label("Status")
        if "results" in st.session_state:
            v = st.session_state["results"]["verdict"]
            color_map = {"GREEN": TOKENS["accent_success"], "YELLOW": TOKENS["accent_warning"], "RED": TOKENS["accent_danger"]}
            c = color_map.get(v["color"], TOKENS["text_secondary"])
            st.markdown(
                f"<span style='color: {c}; font-weight: 700; font-size: 1.1rem;'>"
                f"● {v['verdict']}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span style='color: {TOKENS['text_muted']};'>No analysis run yet</span>",
                unsafe_allow_html=True,
            )

    # Generate demo data on first load
    if "results" not in st.session_state:
        with st.spinner("Generating demo data..."):
            st.session_state["results"] = _generate_demo_results()

    # Header
    styled_header(
        "Strategy Robustness Lab",
        "Backtest overfitting detection · CSCV · PBO · Deflated Sharpe",
    )

    # Tabs
    from app.tab_input import render as render_input
    from app.tab_overview import render as render_overview
    from app.tab_cscv import render as render_cscv
    from app.tab_degradation import render as render_degradation
    from app.tab_stability import render as render_stability
    from app.tab_explorer import render as render_explorer

    tabs = st.tabs([
        "INPUT", "OVERVIEW", "CSCV ANALYSIS",
        "DEGRADATION", "STABILITY", "TRIAL EXPLORER",
    ])

    with tabs[0]:
        render_input()
    with tabs[1]:
        render_overview()
    with tabs[2]:
        render_cscv()
    with tabs[3]:
        render_degradation()
    with tabs[4]:
        render_stability()
    with tabs[5]:
        render_explorer()

    # Footer
    st.markdown(
        f"<div style='text-align: center; color: {TOKENS['text_muted']}; "
        f"font-size: 0.75rem; margin-top: 3rem;'>"
        "Strategy Robustness Lab · Bailey, Borwein, López de Prado & Zhu (2014) · "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
