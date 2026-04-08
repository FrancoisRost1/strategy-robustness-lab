"""
Tab 5 — Parameter Stability.

2D heatmaps, sensitivity curves, plateau detection overlay.
Only available for built-in connectors (not CSV upload).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.style_inject import (
    apply_plotly_theme,
    styled_section_label,
    styled_divider,
    styled_card,
    styled_kpi,
    TOKENS,
)


def render():
    """Render the Parameter Stability tab."""
    if "results" not in st.session_state:
        styled_card("Run analysis first.", accent_color=TOKENS["accent_warning"])
        return

    r = st.session_state["results"]
    plateau = r.get("parameter_stability")
    param_grid = r.get("param_grid")

    if not param_grid:
        styled_card(
            "Parameter stability analysis is not available for CSV upload mode. "
            "This tab requires a known parameter grid (Factor Engine or TSMOM connector).",
            accent_color=TOKENS["accent_info"],
        )
        return

    if plateau is None:
        styled_card("Stability results not available.", accent_color=TOKENS["accent_warning"])
        return

    # --- Stability classification badge ---
    classification = plateau.get("classification", "UNKNOWN")
    class_colors = {
        "STABLE": TOKENS["accent_success"],
        "MODERATE": TOKENS["accent_warning"],
        "FRAGILE": TOKENS["accent_danger"],
        "UNKNOWN": TOKENS["text_muted"],
    }
    badge_color = class_colors.get(classification, TOKENS["text_muted"])

    cols = st.columns(4)
    with cols[0]:
        styled_kpi("Classification", classification,
                    delta=f"Plateau: {plateau['plateau_fraction']:.1%}",
                    delta_color=badge_color)
    with cols[1]:
        styled_kpi("Plateau Fraction", f"{plateau['plateau_fraction']:.3f}")
    with cols[2]:
        styled_kpi("Best Metric", f"{plateau['best_metric']:.4f}")
    with cols[3]:
        styled_kpi("Cells in Plateau", f"{plateau['n_plateau_cells']} / {plateau['n_total_cells']}")

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # --- Heatmaps ---
    heatmaps = r.get("heatmaps", {})
    if heatmaps:
        styled_section_label("Parameter Heatmaps (Metric across grid slices)")
        n_heatmaps = len(heatmaps)
        if n_heatmaps <= 3:
            hm_cols = st.columns(n_heatmaps)
        else:
            hm_cols = st.columns(3)

        for idx, ((pa, pb), pivot) in enumerate(heatmaps.items()):
            col_idx = idx % len(hm_cols)
            with hm_cols[col_idx]:
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(r) for r in pivot.index],
                    colorscale=[
                        [0, TOKENS["accent_danger"]],
                        [0.5, TOKENS["bg_elevated"]],
                        [1, TOKENS["accent_success"]],
                    ],
                    text=np.round(pivot.values, 3),
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    hoverongaps=False,
                ))
                fig.update_layout(
                    title=f"{pa} vs {pb}",
                    xaxis_title=pb,
                    yaxis_title=pa,
                    height=340,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # --- Sensitivity curves ---
    sens = r.get("sensitivity", {})
    if sens:
        styled_section_label("Sensitivity Curves (one param varied, others at baseline)")
        n_params = len(sens)
        sens_cols = st.columns(min(n_params, 3))

        for idx, (param, df) in enumerate(sens.items()):
            col_idx = idx % len(sens_cols)
            with sens_cols[col_idx]:
                fig = go.Figure()

                x_vals = df["param_value"]
                y_vals = df["metric_mean"]
                y_err = df["metric_std"]

                # Convert non-numeric x to string for plotting
                x_plot = [str(v) for v in x_vals]

                fig.add_trace(go.Scatter(
                    x=x_plot, y=y_vals + y_err,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot, y=y_vals - y_err,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(99, 102, 241, 0.15)",
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot, y=y_vals,
                    mode="lines+markers",
                    line=dict(color=TOKENS["accent_primary"], width=2),
                    marker=dict(size=6),
                    name=param,
                ))

                # Plateau threshold line
                threshold = plateau.get("threshold_metric", 0)
                if not np.isnan(threshold):
                    fig.add_hline(
                        y=threshold,
                        line_dash="dot",
                        line_color=TOKENS["accent_success"],
                        annotation_text="Plateau threshold",
                        annotation_position="bottom right",
                    )

                fig.update_layout(
                    title=f"Sensitivity: {param}",
                    xaxis_title=param,
                    yaxis_title="Metric ± 1σ",
                    height=320,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
