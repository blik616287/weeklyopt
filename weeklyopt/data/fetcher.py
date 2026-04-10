"""Fetch and cache historical equity data from yfinance."""

import hashlib
from pathlib import Path
from datetime import date

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".weeklyopt_cache"


def _cache_path(ticker: str, start: date, end: date) -> Path:
    key = f"{ticker}_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return CACHE_DIR / f"{ticker}_{h}.parquet"


def fetch_equity_data(
    ticker: str,
    start: date = date(2020, 1, 1),
    end: date = date(2025, 12, 31),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single ticker.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = _cache_path(ticker, start, end)

    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return multi-level columns for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    if use_cache:
        df.to_parquet(cache)

    return df


def fetch_all_equities(
    tickers: list[str],
    start: date = date(2020, 1, 1),
    end: date = date(2025, 12, 31),
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch data for all tickers. Returns dict of ticker -> DataFrame."""
    data = {}
    for t in tickers:
        try:
            data[t] = fetch_equity_data(t, start, end, use_cache)
        except ValueError as e:
            print(f"WARNING: {e}")
    return data


def load_cached(ticker: str) -> pd.DataFrame | None:
    """Load cached data if available, otherwise return None."""
    matches = list(CACHE_DIR.glob(f"{ticker}_*.parquet"))
    if matches:
        return pd.read_parquet(matches[0])
    return None
