"""
Tab 3 — CSCV Analysis.

Histogram of OOS ranks, logit distribution, PBO convergence, IS vs OOS scatter.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.style_inject import (
    apply_plotly_theme,
    styled_section_label,
    styled_card,
    TOKENS,
)


def render():
    """Render the CSCV Analysis tab."""
    if "results" not in st.session_state:
        styled_card(
            "Run analysis first: go to the INPUT tab.",
            accent_color=TOKENS["accent_warning"],
        )
        return

    r = st.session_state["results"]
    cscv = r.get("cscv_results")
    pbo = r.get("pbo")
    pbo_conv = r.get("pbo_convergence")

    if cscv is None or pbo is None:
        styled_card("CSCV results not available (need ≥ 2 trials).", accent_color=TOKENS["accent_warning"])
        return

    n_trials = cscv["n_trials"].iloc[0]

    # Row 1: OOS rank histogram + logit distribution
    col1, col2 = st.columns(2)

    with col1:
        styled_section_label("OOS Rank Distribution (IS-Best Trial)")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cscv["oos_rank"],
            nbinsx=min(n_trials, 20),
            marker_color=TOKENS["accent_primary"],
            opacity=0.85,
        ))
        fig.update_layout(
            title="OOS Rank Distribution of IS-Best Trial",
            xaxis_title="OOS Rank (1=best)",
            yaxis_title="Count",
            height=380,
        )
        # Add median line
        fig.add_vline(
            x=n_trials / 2,
            line_dash="dash",
            line_color=TOKENS["accent_warning"],
            annotation_text="Median",
            annotation_position="top",
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        styled_section_label("Logit-Transformed Rank Distribution")
        logits = pbo["logits"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=logits,
            nbinsx=30,
            marker_color=TOKENS["accent_secondary"],
            opacity=0.85,
        ))
        # PBO threshold line at 0
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=TOKENS["accent_danger"],
            annotation_text=f"PBO threshold (PBO={pbo['pbo']:.3f})",
            annotation_position="top left",
        )
        fig.update_layout(
            title=f"Logit Distribution (PBO = {pbo['pbo']:.3f})",
            xaxis_title="Logit(relative rank)",
            yaxis_title="Count",
            height=380,
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: PBO convergence + IS vs OOS scatter
    col3, col4 = st.columns(2)

    with col3:
        styled_section_label("PBO Convergence")
        if pbo_conv is not None and len(pbo_conv) > 0:
            # Ensure first point is included for a proper curve
            if pbo_conv.iloc[0]["n_evaluated"] > 1:
                first_row = pd.DataFrame([{"n_evaluated": 1, "pbo_estimate": pbo_conv.iloc[0]["pbo_estimate"]}])
                pbo_conv = pd.concat([first_row, pbo_conv], ignore_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pbo_conv["n_evaluated"],
                y=pbo_conv["pbo_estimate"],
                mode="lines+markers",
                line=dict(color=TOKENS["accent_primary"], width=2),
                marker=dict(size=4),
            ))
            fig.add_hline(
                y=0.5,
                line_dash="dot",
                line_color=TOKENS["accent_warning"],
                annotation_text="50% threshold",
            )
            fig.update_layout(
                title="PBO Convergence (stabilisation check)",
                xaxis_title="Combinations Evaluated",
                yaxis_title="PBO Estimate",
                yaxis_range=[0, 1],
                height=380,
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Convergence data not available.")

    with col4:
        styled_section_label("IS Metric vs OOS Metric (IS-Best)")
        is_vals = cscv["is_best_metric"].values
        oos_vals = cscv["oos_metric_of_is_best"].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=is_vals,
            y=oos_vals,
            mode="markers",
            marker=dict(
                color=TOKENS["accent_info"],
                size=5,
                opacity=0.6,
            ),
            name="Combinations",
        ))
        # 45-degree line (no degradation)
        all_vals = np.concatenate([is_vals, oos_vals])
        valid = all_vals[~np.isnan(all_vals)]
        if len(valid) > 0:
            lo, hi = valid.min(), valid.max()
            fig.add_trace(go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color=TOKENS["text_muted"], dash="dash", width=1),
                name="No degradation",
                showlegend=True,
            ))

        fig.update_layout(
            title="IS Metric vs OOS Metric (CSCV Combinations)",
            xaxis_title="IS Metric (best trial)",
            yaxis_title="OOS Metric (same trial)",
            height=380,
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Combination detail table (collapsed)
    # Reset rendering context after st.html()-based components inside columns
    st.markdown("")
    with st.expander("Combination-Level Detail"):
        display_df = cscv[["combo_id", "is_best_trial", "is_best_metric",
                           "oos_metric_of_is_best", "oos_rank", "n_trials"]].copy()
        display_df["is_best_metric"] = display_df["is_best_metric"].round(4)
        display_df["oos_metric_of_is_best"] = display_df["oos_metric_of_is_best"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
