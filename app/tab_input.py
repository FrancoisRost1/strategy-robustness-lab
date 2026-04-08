"""
Tab 1 — Strategy Input.

Mode selector, parameter grid configuration, ranking metric, CSCV settings,
and the Run Analysis button that triggers the full pipeline.
"""

import streamlit as st
import numpy as np
import pandas as pd

from app.style_inject import (
    styled_section_label,
    styled_divider,
    styled_card,
    TOKENS,
)
def render():
    """Render the Strategy Input tab."""
    # Read config from session_state (loaded once in app.py) instead of calling load_config() again
    config = st.session_state.get("config")
    if config is None:
        from src.utils.config_loader import load_config
        config = load_config()
        st.session_state["config"] = config

    col_left, col_right = st.columns([2, 1])

    with col_left:
        styled_section_label("Analysis Mode")
        mode = st.selectbox(
            "Data source",
            ["Synthetic (Demo)", "TSMOM Engine", "Factor Engine", "CSV Upload"],
            index=0,
            help="Select how to generate the trial matrix",
        )

        if mode == "CSV Upload":
            csv_file = st.file_uploader(
                "Upload trial matrix CSV",
                type=["csv"],
                help="Rows = dates, columns = strategy variations, values = daily returns",
            )
            if csv_file is not None:
                st.session_state["_csv_upload_file"] = csv_file

        elif mode in ("TSMOM Engine", "Factor Engine"):
            connector_key = "tsmom_connector" if mode == "TSMOM Engine" else "factor_connector"
            grid_config = config.get(connector_key, {}).get("grid", {})

            styled_section_label("Parameter Grid")
            st.markdown(
                f"<span style='color: {TOKENS['text_muted']}; font-size: 0.85rem;'>"
                f"Grid produces <b>{_grid_count(grid_config)}</b> trial combinations"
                "</span>",
                unsafe_allow_html=True,
            )

            # Display grid parameters as an editable reference
            grid_rows = []
            for param, values in grid_config.items():
                grid_rows.append({
                    "Parameter": param,
                    "Values": ", ".join(str(v) for v in values),
                    "Count": len(values),
                })
            if grid_rows:
                st.dataframe(
                    pd.DataFrame(grid_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Values": st.column_config.TextColumn(width="large")},
                )

        else:
            styled_card(
                "Synthetic mode generates noise strategies with one planted-signal trial. "
                "Useful for validating the pipeline and taking screenshots.",
                accent_color=TOKENS["accent_info"],
            )

        styled_divider()

    with col_right:
        styled_section_label("Ranking Metric")
        metric = st.selectbox(
            "Metric for CSCV ranking",
            ["sharpe", "sortino", "calmar"],
            index=0,
        )

        styled_section_label("CSCV Partitions (S)")
        n_partitions = st.slider(
            "Number of time blocks",
            min_value=4,
            max_value=32,
            value=config["cscv"]["n_partitions"],
            step=2,
            help="Must be even. C(S, S/2) combinations generated. Default 16 → 12,870.",
        )

        if n_partitions < 12:
            st.warning("S < 12 produces unstable PBO estimates. Recommend S ≥ 16 for reliable results.")

        styled_section_label("Bootstrap Resamples")
        n_resamples = st.slider(
            "Number of bootstrap iterations",
            min_value=100,
            max_value=5000,
            value=config["bootstrap"]["n_resamples"],
            step=100,
        )

        if n_resamples < 500:
            st.warning("Fewer than 500 resamples may produce unreliable confidence intervals.")

        styled_section_label("Lookback")
        lookback = st.number_input(
            "Years of data",
            min_value=1,
            max_value=20,
            value=config["data"]["lookback_years"],
        )

    # Run button
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        _run_analysis(mode, config, metric, n_partitions, n_resamples, lookback)


def _run_analysis(mode, config, metric, n_partitions, n_resamples, lookback):
    """Execute the pipeline with user-selected parameters."""
    # Update config with user selections
    config["ranking"]["metric"] = metric
    config["cscv"]["n_partitions"] = n_partitions
    config["bootstrap"]["n_resamples"] = n_resamples
    config["data"]["lookback_years"] = lookback

    with st.spinner("Running robustness analysis..."):
        if mode == "Synthetic (Demo)":
            from app.demo import generate_demo_results
            st.session_state["results"] = generate_demo_results()
        elif mode == "TSMOM Engine":
            results = _run_pipeline_on_connector("tsmom", config)
            st.session_state["results"] = results
        elif mode == "Factor Engine":
            results = _run_pipeline_on_connector("factor", config)
            st.session_state["results"] = results
        elif mode == "CSV Upload":
            csv_file = st.session_state.get("_csv_upload_file")
            if csv_file is None:
                st.error("Please upload a CSV file first.")
                return
            results = _run_pipeline_on_csv(csv_file, config)
            st.session_state["results"] = results

    st.success("Analysis complete. Switch to other tabs to view results.")
    st.rerun()


def _run_pipeline_on_connector(connector_mode: str, config: dict) -> dict:
    """Run the full pipeline using a built-in connector (tsmom or factor).

    Imports the main.run_pipeline function which handles the full
    CSCV -> PBO -> degradation -> DSR -> bootstrap -> SD -> stability -> verdict chain.
    """
    from src.pipeline import run_pipeline
    return run_pipeline(config, mode=connector_mode)


def _run_pipeline_on_csv(csv_file, config: dict) -> dict:
    """Run the full pipeline on an uploaded CSV file.

    Saves the uploaded file to a temp location, then runs the pipeline
    in CSV mode.
    """
    import tempfile
    import os

    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
        tmp.write(csv_file.getvalue())
        tmp_path = tmp.name

    try:
        config["data"]["source"] = "csv"
        config["data"]["csv_path"] = tmp_path
        from src.pipeline import run_pipeline
        return run_pipeline(config, mode="csv")
    finally:
        os.unlink(tmp_path)


def _grid_count(grid_config: dict) -> int:
    """Count total combinations in a parameter grid."""
    count = 1
    for values in grid_config.values():
        count *= len(values)
    return count
