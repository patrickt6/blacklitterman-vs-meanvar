"""Adjusted-close pulls from Yahoo with a parquet cache.

The cache exists so re-running the backtest does not re-hit Yahoo for the
same window. Cache files are keyed by ticker + start + end and live under
data/cache/. They are gitignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}_{start}_{end}.parquet"


def _fetch_one(ticker: str, start: str, end: str) -> pd.Series:
    cache = _cache_path(ticker, start, end)
    if cache.exists():
        return pd.read_parquet(cache).iloc[:, 0]

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance returns a multi-index column when downloading >1 ticker; for a
    # single ticker the column is just "Close" (with auto_adjust=True the close
    # is already split/dividend adjusted).
    series = raw["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series.name = ticker

    series.to_frame().to_parquet(cache)
    return series


def get_adjusted_close(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Adjusted close prices indexed by trading date with one column per ticker.

    Missing days are forward-filled because sector ETFs occasionally skip a
    print on partial holidays; small gaps would otherwise corrupt the daily
    return computation.
    """
    frames = [_fetch_one(t, start, end) for t in tickers]
    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index().ffill().dropna(how="any")
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; first row drops out."""
    return prices.pct_change().dropna(how="any")


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month total returns from daily price levels."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna(how="any")
