"""
Tab 2 — Overview & Verdict.

Traffic light verdict, KPI cards, and summary diagnostic table.
"""

import streamlit as st
import numpy as np
import pandas as pd

from app.style_inject import (
    styled_kpi,
    styled_section_label,
    styled_divider,
    styled_card,
    TOKENS,
)


def render():
    """Render the Overview & Verdict tab."""
    if "results" not in st.session_state:
        styled_card(
            "Run analysis first — go to the INPUT tab and click 'Run Analysis'.",
            accent_color=TOKENS["accent_warning"],
        )
        return

    r = st.session_state["results"]
    v = r["verdict"]

    # --- Traffic light verdict ---
    color_map = {
        "GREEN": "#00CC66",
        "YELLOW": "#FFAA00",
        "RED": "#FF4444",
        "GRAY": TOKENS["text_muted"],
    }
    emoji_map = {"GREEN": "●", "YELLOW": "●", "RED": "●"}
    verdict_color = color_map.get(v["color"], TOKENS["text_muted"])

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 2rem 1rem;
            background: {TOKENS['bg_surface']};
            border: 1px solid {TOKENS['border_default']};
            border-radius: {TOKENS['radius_lg']};
            margin-bottom: 1.5rem;
            box-shadow: {TOKENS['shadow_md']};
        ">
            <div style="
                font-size: 3.5rem;
                color: {verdict_color};
                line-height: 1;
                margin-bottom: 0.5rem;
            ">{emoji_map.get(v['color'], '●')}</div>
            <div style="
                font-family: {TOKENS['font_display']};
                font-size: 1.75rem;
                font-weight: 700;
                color: {verdict_color};
                letter-spacing: 0.05em;
            ">{v['verdict']}</div>
            <div style="
                font-size: 0.9rem;
                color: {TOKENS['text_secondary']};
                margin-top: 0.5rem;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            ">{v['details'].replace(chr(10), '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- All-negative warning ---
    trial_metrics = r.get("trial_metrics", {})
    all_negative = all(v < 0 for v in trial_metrics.values() if not np.isnan(v))
    if all_negative and trial_metrics:
        styled_card(
            "All strategies are unprofitable. Robustness analysis is limited to relative "
            "ranking and should not be interpreted as evidence of a viable strategy.",
            accent_color=TOKENS["accent_danger"],
        )

    # --- KPI row ---
    pbo_val = r["pbo"]["pbo"] if r.get("pbo") else np.nan
    degrad_mean = r["degradation"]["mean_degradation"] if r.get("degradation") else np.nan
    dsr_val = r["deflated_sharpe"]["dsr"]
    sd_result = r.get("stochastic_dominance", {}).get("first_order", {})
    sd_dominates = sd_result.get("strategy_dominates", False)
    sd_rejects = sd_result.get("rejects_null", False)
    if sd_dominates and sd_rejects:
        sd_text = "IS-best dominates"
        sd_delta = "Consistent OOS edge"
    elif sd_rejects:
        sd_text = "Differs"
        sd_delta = "No stochastic dominance — no reliable OOS advantage"
    else:
        sd_text = "No difference"
        sd_delta = "Cannot distinguish IS-best from average strategy"
    plateau_frac = r["parameter_stability"]["plateau_fraction"] if r.get("parameter_stability") else np.nan

    cols = st.columns(5)
    with cols[0]:
        pbo_color = _pbo_color(pbo_val)
        styled_kpi("PBO Score", f"{pbo_val:.3f}" if not np.isnan(pbo_val) else "N/A",
                    delta=_pbo_label(pbo_val), delta_color=pbo_color)
        st.caption("Probability the IS-best strategy underperforms OOS")
    with cols[1]:
        styled_kpi("Mean Degradation", f"{degrad_mean:.2f}" if not np.isnan(degrad_mean) else "N/A")
        st.caption("Average OOS/IS performance ratio across CSCV splits")
    with cols[2]:
        dsr_color = TOKENS["accent_success"] if dsr_val >= 0.95 else TOKENS["accent_danger"]
        n_trials = r.get("trial_matrix", pd.DataFrame()).shape[1]
        dsr_delta = "Significant" if dsr_val >= 0.95 else f"Not significant (tested {n_trials} strategies)"
        styled_kpi("Deflated Sharpe", f"{dsr_val:.3f}" if not np.isnan(dsr_val) else "N/A",
                    delta=dsr_delta,
                    delta_color=dsr_color)
        st.caption("Sharpe significance after multiple-testing correction")
    with cols[3]:
        styled_kpi("Stochastic Dominance", sd_text,
                    delta=sd_delta,
                    delta_color=TOKENS["accent_success"] if sd_dominates and sd_rejects else TOKENS["text_secondary"])
        st.caption("Whether IS-best OOS returns dominate the benchmark")
    with cols[4]:
        plat_color = TOKENS["accent_success"] if not np.isnan(plateau_frac) and plateau_frac > 0.30 else TOKENS["accent_warning"]
        styled_kpi("Plateau Fraction", f"{plateau_frac:.2f}" if not np.isnan(plateau_frac) else "N/A",
                    delta=r["parameter_stability"]["classification"] if r.get("parameter_stability") else "",
                    delta_color=plat_color)
        st.caption("Fraction of parameter grid near optimal performance")

    st.html("<div style='height: 1.5rem'></div>")

    # --- Summary table ---
    styled_section_label("Diagnostic Summary")

    sign_flip = r["degradation"]["sign_flip_rate"] if r.get("degradation") else np.nan
    summary_data = {
        "Diagnostic": [
            "Probability of Backtest Overfitting (PBO)",
            "Deflated Sharpe Ratio (DSR)",
            "Mean IS→OOS Degradation Ratio",
            "OOS Sign Flip Rate",
            "KS Test (Stochastic Dominance)",
            "Parameter Plateau Fraction",
            "Bootstrap Sharpe 95% CI (Standard)",
            "Bootstrap Sharpe 95% CI (Block)",
        ],
        "Value": [
            f"{pbo_val:.3f}" if not np.isnan(pbo_val) else "N/A",
            f"{dsr_val:.3f}" if not np.isnan(dsr_val) else "N/A",
            f"{degrad_mean:.2f}" if not np.isnan(degrad_mean) else "N/A",
            f"{sign_flip:.1%}" if not np.isnan(sign_flip) else "N/A",
            f"KS={sd_result.get('ks_statistic', np.nan):.3f}, p={sd_result.get('p_value', np.nan):.4f}",
            f"{plateau_frac:.2f}" if not np.isnan(plateau_frac) else "N/A (CSV mode)",
            _format_ci(r.get("bootstrap", {}).get("standard", {})),
            _format_ci(r.get("bootstrap", {}).get("block", {})),
        ],
        "Interpretation": [
            _pbo_label(pbo_val),
            _dsr_interpretation(dsr_val, r),
            f"{(1-degrad_mean)*100:.0f}% expected haircut" if not np.isnan(degrad_mean) and degrad_mean < 1 else "No decay",
            "High risk" if not np.isnan(sign_flip) and sign_flip > 0.3 else "Acceptable",
            _sd_interpretation(sd_result),
            r["parameter_stability"]["classification"] if r.get("parameter_stability") else "N/A",
            "",
            "",
        ],
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
    )

    # --- Haircut callout ---
    if r.get("haircut"):
        styled_card(r["haircut"], accent_color=TOKENS["accent_warning"])


def _dsr_interpretation(dsr_val, results):
    """Human-readable DSR interpretation including number of trials tested."""
    n_trials = results.get("trial_matrix", pd.DataFrame()).shape[1]
    if np.isnan(dsr_val):
        return "N/A"
    if dsr_val >= 0.95:
        return "Significant"
    return f"Not significant after multiple-testing correction (performance likely due to luck after testing {n_trials} strategies)"


def _pbo_color(pbo):
    if np.isnan(pbo):
        return TOKENS["text_muted"]
    if pbo < 0.25:
        return TOKENS["accent_success"]
    if pbo <= 0.50:
        return TOKENS["accent_warning"]
    return TOKENS["accent_danger"]


def _pbo_label(pbo):
    if np.isnan(pbo):
        return "N/A"
    if pbo < 0.25:
        return "Low overfitting risk"
    if pbo <= 0.50:
        return "Borderline"
    if pbo > 0.90:
        return "Catastrophic — likely driven by noise rather than persistent signal"
    return "High overfitting risk"


def _sd_interpretation(sd_result):
    """Human-readable stochastic dominance interpretation."""
    if sd_result.get("strategy_dominates") and sd_result.get("rejects_null"):
        return "IS-best shows consistent OOS edge"
    elif sd_result.get("rejects_null"):
        return "No stochastic dominance — IS-best has no reliable OOS advantage"
    return "Cannot distinguish IS-best from average strategy"


def _format_ci(boot_result):
    if not boot_result:
        return "N/A"
    lo = boot_result.get("ci_lower", np.nan)
    hi = boot_result.get("ci_upper", np.nan)
    if np.isnan(lo) or np.isnan(hi):
        return "N/A"
    return f"[{lo:.3f}, {hi:.3f}]"
