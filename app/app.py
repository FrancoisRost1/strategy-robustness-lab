"""
Strategy Robustness Lab — Streamlit dashboard entry point.

Bloomberg dark mode. 6 tabs. No business logic here — delegates to tab modules.
Generates synthetic demo data on first load so dashboard is never empty.
"""

import sys
import os
from pathlib import Path

# Resolve project root and make it both import-visible AND the working directory,
# so relative paths in config.yaml (data/cache/...) resolve correctly on
# Streamlit Cloud regardless of how the app is launched. Without os.chdir, a
# shifted CWD makes the cache lookup miss, the loader falls through to live
# yfinance (blocked on Cloud), and downstream CSCV crashes with
# "Not enough data: N rows for 16 partitions."
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

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

from app.demo import generate_demo_results, get_config


def main():
    """Dashboard entry point."""
    # Ensure config is loaded once
    get_config()

    # Sidebar
    with st.sidebar:
        st.html(
            f'<h2 style="color: {TOKENS["accent_primary"]}; '
            f'font-family: {TOKENS["font_display"]}; font-weight: 700;">'
            f'◆ Robustness Lab</h2>'
        )
        styled_section_label("About")
        st.html(
            f'<span style="font-size: 0.85rem; color: {TOKENS["text_secondary"]};">'
            "Strategy overfitting detection via CSCV, PBO, and deflated Sharpe ratio. "
            "Bailey, Borwein, L\u00f3pez de Prado & Zhu (2014)."
            "</span>"
        )
        styled_divider()
        styled_section_label("Status")
        if "results" in st.session_state:
            v = st.session_state["results"]["verdict"]
            color_map = {
                "GREEN": TOKENS["accent_success"],
                "YELLOW": TOKENS["accent_warning"],
                "RED": TOKENS["accent_danger"],
                "GRAY": TOKENS["text_muted"],
            }
            c = color_map.get(v["color"], TOKENS["text_secondary"])
            st.html(
                f'<span style="color: {c}; font-weight: 700; font-size: 1.1rem;">'
                f"● {v['verdict']}</span>"
            )
        else:
            st.html(
                f'<span style="color: {TOKENS["text_muted"]};">No analysis run yet</span>'
            )

    # Generate demo data on first load
    if "results" not in st.session_state:
        with st.spinner("Generating demo data..."):
            st.session_state["results"] = generate_demo_results()

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
    st.html(
        f'<div style="text-align: center; color: {TOKENS["text_muted"]}; '
        f'font-size: 0.75rem; margin-top: 3rem;">'
        "Strategy Robustness Lab \u00b7 Bailey, Borwein, L\u00f3pez de Prado & Zhu (2014) \u00b7 "
        "Built with Streamlit"
        "</div>"
    )


if __name__ == "__main__":
    main()
