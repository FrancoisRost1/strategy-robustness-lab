"""
Tab 6, Trial Explorer.

Full performance matrix, equity curves, parameter values, export.
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
    TOKENS,
)


def render():
    """Render the Trial Explorer tab."""
    if "results" not in st.session_state:
        styled_card("Run analysis first.", accent_color=TOKENS["accent_warning"])
        return

    r = st.session_state["results"]
    trial_matrix = r["trial_matrix"]
    all_metrics = r.get("all_trial_metrics", {})
    param_grid = r.get("param_grid", [])
    is_best = r.get("is_best_trial", "")
    config = r.get("config", {})
    max_overlay = config.get("dashboard", {}).get("max_equity_curves_overlay", 5)

    # --- Build performance table ---
    styled_section_label("Performance Matrix: All Trials")

    rows = []
    param_lookup = {str(p["trial_id"]): p for p in param_grid} if param_grid else {}

    for trial_id in trial_matrix.columns:
        m = all_metrics.get(trial_id, {})
        row = {"Trial": trial_id}

        # Add parameter values if available
        params = param_lookup.get(trial_id, {})
        for k, v in params.items():
            if k != "trial_id":
                row[k] = v

        row["Sharpe"] = round(m.get("sharpe", np.nan), 2)
        row["Sortino"] = round(m.get("sortino", np.nan), 2)
        row["Calmar"] = round(m.get("calmar", np.nan), 2)
        row["CAGR"] = f"{m.get('cagr', np.nan):.1%}" if not np.isnan(m.get("cagr", np.nan)) else "N/A"
        row["Max DD"] = f"{m.get('max_drawdown', np.nan):.1%}" if not np.isnan(m.get("max_drawdown", np.nan)) else "N/A"
        row["IS-Best"] = "★" if trial_id == is_best else ""
        rows.append(row)

    df = pd.DataFrame(rows)

    # Highlight IS-best row
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Export
    col_exp, col_space = st.columns([1, 3])
    with col_exp:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="trial_performance.csv",
            mime="text/csv",
        )

    styled_divider()

    # --- Equity curves ---
    styled_section_label("Equity Curves (select up to 5 trials)")

    available_trials = list(trial_matrix.columns)
    # Default: IS-best + a few others
    defaults = [is_best] if is_best in available_trials else []
    for t in available_trials[:4]:
        if t not in defaults:
            defaults.append(t)
        if len(defaults) >= max_overlay:
            break

    selected = st.multiselect(
        "Select trials to overlay",
        options=available_trials,
        default=defaults[:max_overlay],
        max_selections=max_overlay,
    )

    if selected:
        fig = go.Figure()

        colorway = [
            TOKENS["accent_primary"],
            TOKENS["accent_secondary"],
            TOKENS["accent_info"],
            TOKENS["accent_success"],
            TOKENS["accent_warning"],
        ]

        for i, trial_id in enumerate(selected):
            equity = (1 + trial_matrix[trial_id]).cumprod()
            color = colorway[i % len(colorway)]
            opacity = 1.0 if trial_id == is_best else 0.2
            width = 2.5 if trial_id == is_best else 1.5
            name = f"Trial {trial_id}" + (" (IS-Best)" if trial_id == is_best else "")

            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                name=name,
            ))

        fig.update_layout(
            title="Equity Curves: Selected Trials",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (growth of $1)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # --- IS-best vs OOS-best discrepancy ---
    styled_section_label("IS-Best vs OOS-Best Discrepancy")

    cscv = r.get("cscv_results")
    if cscv is not None and len(cscv) > 0:
        # Find the trial that ranks best OOS most often
        oos_best_counts = {}
        for _, row in cscv.iterrows():
            # IS-best trial for this combo
            t = row["is_best_trial"]
            if t not in oos_best_counts:
                oos_best_counts[t] = 0

        # For display: IS-best (full-sample) vs most common IS-best across combos
        is_best_counts = cscv["is_best_trial"].value_counts()

        disc_df = pd.DataFrame({
            "Trial": is_best_counts.index[:10],
            "Times Selected as IS-Best": is_best_counts.values[:10],
            "Median OOS Rank": [
                cscv[cscv["is_best_trial"] == t]["oos_rank"].median()
                for t in is_best_counts.index[:10]
            ],
        })
        disc_df["Median OOS Rank"] = disc_df["Median OOS Rank"].round(1)

        st.dataframe(disc_df, use_container_width=True, hide_index=True)

        styled_card(
            f"Full-sample IS-best: Trial {is_best} | "
            f"Most frequently IS-best across CSCV combos: Trial {is_best_counts.index[0]} "
            f"({is_best_counts.values[0]} times)",
            accent_color=TOKENS["accent_primary"],
        )
    else:
        styled_card("CSCV data not available for discrepancy analysis.", accent_color=TOKENS["accent_info"])
