"""
Tab 4 — Degradation Analysis.

IS vs OOS scatter, degradation histogram, sign flip rate, bootstrap distributions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.style_inject import (
    apply_plotly_theme,
    styled_section_label,
    styled_divider,
    styled_card,
    styled_kpi,
    TOKENS,
)


def render():
    """Render the Degradation Analysis tab."""
    if "results" not in st.session_state:
        styled_card("Run analysis first.", accent_color=TOKENS["accent_warning"])
        return

    r = st.session_state["results"]
    degrad = r.get("degradation")
    boot = r.get("bootstrap")

    if degrad is None:
        styled_card("Degradation data not available.", accent_color=TOKENS["accent_warning"])
        return

    # KPI row
    cols = st.columns(4)
    with cols[0]:
        styled_kpi("Mean Degradation", f"{degrad['mean_degradation']:.3f}")
    with cols[1]:
        styled_kpi("Median Degradation", f"{degrad['median_degradation']:.3f}")
    with cols[2]:
        flip_color = TOKENS["accent_danger"] if degrad["sign_flip_rate"] > 0.3 else TOKENS["accent_success"]
        styled_kpi("Sign Flip Rate", f"{degrad['sign_flip_rate']:.1%}",
                    delta="High risk" if degrad["sign_flip_rate"] > 0.3 else "Acceptable",
                    delta_color=flip_color)
    with cols[3]:
        styled_kpi("Std Degradation", f"{degrad['std_degradation']:.3f}")

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # Row 1: IS vs OOS scatter + degradation histogram
    col1, col2 = st.columns(2)

    with col1:
        styled_section_label("IS Sharpe vs OOS Sharpe")
        is_m = degrad["is_metrics"]
        oos_m = degrad["oos_metrics"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=is_m, y=oos_m,
            mode="markers",
            marker=dict(color=TOKENS["accent_primary"], size=5, opacity=0.5),
            name="Combinations",
        ))

        # 45-degree line
        valid = np.concatenate([is_m, oos_m])
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            lo, hi = valid.min(), valid.max()
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                line=dict(color=TOKENS["accent_warning"], dash="dash", width=1),
                name="No degradation (45°)",
            ))

            # Regression line
            mask = ~np.isnan(is_m) & ~np.isnan(oos_m)
            if mask.sum() > 2:
                from numpy.polynomial.polynomial import polyfit
                coeffs = polyfit(is_m[mask], oos_m[mask], 1)
                x_fit = np.linspace(is_m[mask].min(), is_m[mask].max(), 50)
                y_fit = coeffs[0] + coeffs[1] * x_fit
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode="lines",
                    line=dict(color=TOKENS["accent_danger"], width=2),
                    name="Regression fit",
                ))

        fig.update_layout(xaxis_title="IS Metric", yaxis_title="OOS Metric", height=380)
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        styled_section_label("Degradation Ratio Distribution")
        ratios = degrad["degradation_ratios"]
        valid_ratios = ratios[~np.isnan(ratios)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_ratios,
            nbinsx=30,
            marker_color=TOKENS["accent_secondary"],
            opacity=0.85,
        ))
        fig.add_vline(
            x=1.0, line_dash="dash", line_color=TOKENS["accent_success"],
            annotation_text="No degradation",
        )
        fig.add_vline(
            x=0.0, line_dash="dot", line_color=TOKENS["accent_danger"],
            annotation_text="Sign flip",
        )
        fig.update_layout(
            xaxis_title="Degradation Ratio (OOS/IS)",
            yaxis_title="Count",
            height=380,
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # Row 2: Bootstrap distributions
    if boot:
        styled_section_label("Bootstrapped Metric Distributions")
        col3, col4 = st.columns(2)

        with col3:
            _plot_bootstrap(boot["standard"], "Standard Bootstrap")
        with col4:
            _plot_bootstrap(boot["block"], "Block Bootstrap")

    # Haircut text
    if r.get("haircut"):
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        styled_card(r["haircut"], accent_color=TOKENS["accent_info"])


def _plot_bootstrap(boot_result, title):
    """Plot a single bootstrap distribution with CI bands."""
    dist = boot_result.get("metric_distribution", np.array([]))
    valid = dist[~np.isnan(dist)]
    if len(valid) == 0:
        st.info(f"{title}: no valid bootstrap samples.")
        return

    ci_lo = boot_result["ci_lower"]
    ci_hi = boot_result["ci_upper"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid,
        nbinsx=30,
        marker_color=TOKENS["accent_info"],
        opacity=0.7,
        name="Distribution",
    ))

    # CI bands
    fig.add_vline(x=ci_lo, line_dash="dash", line_color=TOKENS["accent_warning"],
                  annotation_text=f"2.5% ({ci_lo:.3f})")
    fig.add_vline(x=ci_hi, line_dash="dash", line_color=TOKENS["accent_warning"],
                  annotation_text=f"97.5% ({ci_hi:.3f})")
    fig.add_vline(x=boot_result["mean"], line_dash="solid",
                  line_color=TOKENS["accent_primary"],
                  annotation_text=f"Mean ({boot_result['mean']:.3f})")

    fig.update_layout(
        title=title,
        xaxis_title="Metric Value",
        yaxis_title="Count",
        height=360,
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
