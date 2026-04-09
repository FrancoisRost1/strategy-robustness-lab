"""
Data loader — yfinance fetch, CSV override, and caching.

Financial rationale: consistent data sourcing is critical for reproducibility.
This module provides a single entry point for price data, with local caching
to avoid redundant API calls and a CSV override for user-supplied datasets.
"""

import glob
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_prices(config: dict) -> pd.DataFrame:
    """Load adjusted close prices for the configured universe.

    Checks cache first, falls back to yfinance, with CSV override option.
    Returns a DataFrame with DatetimeIndex and one column per ticker.

    Parameters
    ----------
    config : dict
        Full config. Uses data.source, data.csv_path, data.start_date,
        data.end_date, data.cache_dir, and connector-specific universe.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices, DatetimeIndex, columns = tickers.
    """
    source = config["data"].get("source", "yfinance")

    if source == "csv":
        return _load_csv_prices(config)
    return _load_yfinance(config)


def _load_yfinance(config: dict) -> pd.DataFrame:
    """Fetch price data from yfinance with local parquet caching."""
    import yfinance as yf

    from datetime import timedelta

    tickers = _resolve_tickers(config)
    end = config["data"].get("end_date") or datetime.now().strftime("%Y-%m-%d")
    start = config["data"].get("start_date")
    if not start:
        # Fall back to lookback_years if start_date not explicitly set
        lookback_years = config["data"].get("lookback_years", 10)
        end_dt = datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end
        start = (end_dt - timedelta(days=int(lookback_years * 365.25))).strftime("%Y-%m-%d")

    cache_dir = config["data"].get("cache_dir", "data/cache")

    os.makedirs(cache_dir, exist_ok=True)
    ticker_key = "_".join(sorted(tickers))
    cache_file = os.path.join(cache_dir, f"prices_{ticker_key}_{start}_{end}.parquet")

    if os.path.isfile(cache_file):
        logger.info("Loading cached prices from %s", cache_file)
        return pd.read_parquet(cache_file)

    # Fallback: the exact filename encodes today's date in `end`, so it drifts
    # daily and misses shipped snapshots. Glob for any cached parquet covering
    # this ticker set — critical on Streamlit Cloud where yfinance is usually
    # blocked and the committed snapshot is the only data source.
    glob_pattern = os.path.join(cache_dir, f"prices_{ticker_key}_*.parquet")
    matches = sorted(glob.glob(glob_pattern))
    if matches:
        fallback = matches[-1]  # most recent by filename sort (dates are ISO)
        logger.info(
            "Exact cache miss; using committed snapshot %s", fallback
        )
        return pd.read_parquet(fallback)

    logger.info("Downloading prices for %s from yfinance (%s to %s).",
                tickers, start, end)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance on Streamlit Cloud is rate-limited/blocked: it often returns an
    # empty or tiny DataFrame instead of raising. Fail loud and clear here so
    # the user gets an actionable error instead of a cryptic downstream crash
    # (e.g. "Not enough data: 4 rows for 16 partitions" from CSCV).
    if data is None or data.empty:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} ({start} → {end}). "
            "On Streamlit Cloud, Yahoo typically rate-limits cloud provider IPs — "
            f"ship a pre-cached parquet in {cache_dir}/ and the loader will pick it up."
        )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Clean: forward-fill gaps, drop rows still missing
    n_before = prices.isna().sum().sum()
    prices = prices.ffill().dropna()
    n_dropped = n_before - prices.isna().sum().sum()
    if n_dropped > 0:
        logger.info("Forward-filled %d NaN price entries.", n_dropped)

    # Second sanity check: even if the frame isn't empty, a handful of rows
    # can't support a 10-year backtest or 16-partition CSCV.
    min_rows = config["cscv"].get("n_partitions", 16) * 4
    if len(prices) < min_rows:
        raise RuntimeError(
            f"yfinance returned only {len(prices)} usable rows for {tickers} "
            f"(need ≥ {min_rows}). Likely rate-limited — ship a pre-cached "
            f"parquet in {cache_dir}/."
        )

    prices.to_parquet(cache_file)
    return prices


def _load_csv_prices(config: dict) -> pd.DataFrame:
    """Load price data from a user-supplied CSV."""
    csv_path = config["data"].get("csv_path")
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV price file not found: {csv_path}. "
            "Set data.source='yfinance' or provide a valid csv_path."
        )

    logger.info("Loading prices from CSV: %s", csv_path)
    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    prices = prices.ffill().dropna()
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price DataFrame to daily simple returns.

    Simple returns = P(t)/P(t-1) - 1. First row is dropped (NaN).
    """
    returns = prices.pct_change().iloc[1:]
    return returns


def _resolve_tickers(config: dict) -> list:
    """Extract ticker list from the active connector config."""
    # TSMOM universe is explicit
    tsmom = config.get("tsmom_connector", {})
    if tsmom.get("universe"):
        return tsmom["universe"]

    # Factor engine uses a universe identifier — default to SPY for now
    factor = config.get("factor_connector", {})
    universe = factor.get("universe", "sp500")
    if universe == "sp500":
        # Representative subset for the factor engine
        return ["SPY"]

    return ["SPY"]
