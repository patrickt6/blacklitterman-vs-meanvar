"""Single source of truth for tickers, weights, and backtest parameters.

The asset universe is the eleven SPDR sector ETFs covering the GICS sectors
of the S&P 500. SPY itself is the benchmark (and the source of the implicit
market-cap weights used to derive Black-Litterman implied returns).
"""

from __future__ import annotations

# --- universe ---------------------------------------------------------------

# We keep the ten SPDR sector ETFs that have a full price history across
# the test window. XLC was carved out of the GICS Communication Services
# sector in 2018 and has insufficient pre-launch history; its weight is
# folded back into XLK (tech) and XLY (consumer discretionary), which is
# where its constituents lived before the reclassification.
SECTOR_TICKERS: list[str] = [
    "XLK",   # information technology (incl. former-XLC tech components)
    "XLF",   # financials
    "XLV",   # health care
    "XLY",   # consumer discretionary (incl. former-XLC media components)
    "XLI",   # industrials
    "XLP",   # consumer staples
    "XLE",   # energy
    "XLU",   # utilities
    "XLRE",  # real estate
    "XLB",   # materials
]

BENCHMARK_TICKER: str = "SPY"

# Approximate SPY GICS sector weights as of late 2025, with XLC's ~9% rolled
# into XLK (6 pp) and XLY (3 pp). Treating these as fixed market-cap weights
# avoids re-fetching iShares fact sheets every rebalance; the BL prior is
# fairly insensitive to small drifts in w_mkt.
MARKET_CAP_WEIGHTS: dict[str, float] = {
    "XLK":  0.37,
    "XLF":  0.13,
    "XLV":  0.10,
    "XLY":  0.14,
    "XLI":  0.08,
    "XLP":  0.06,
    "XLE":  0.04,
    "XLU":  0.03,
    "XLRE": 0.03,
    "XLB":  0.02,
}

# --- date range -------------------------------------------------------------

# XLRE's earliest print is October 2015. Starting then gives a clean panel.
START_DATE: str = "2015-10-01"
END_DATE:   str = "2025-12-31"

# --- backtest parameters ----------------------------------------------------

TRAINING_WINDOW_YEARS: int = 3
TRADING_DAYS_PER_YEAR: int = 252
MONTHS_PER_YEAR:       int = 12

# annualised risk-free rate used in Sharpe and reverse-optimisation. 2% is a
# rough average of 3-month T-bill yields across the test window.
RISK_FREE_RATE: float = 0.02

# Risk aversion lambda. 2.5 is the midpoint of the 2-3 range that Black &
# Litterman cite from market historical Sharpe ratios.
RISK_AVERSION: float = 2.5

# Tau scales the prior covariance. 0.05 is a common practitioner default;
# He & Litterman (1999) use values in the same neighbourhood.
TAU: float = 0.05

# --- view generation --------------------------------------------------------

MOMENTUM_LOOKBACK_MONTHS: int = 6

# Annualised expected outperformance of top-momentum sectors over bottom-
# momentum sectors expressed in the relative view Q.
MOMENTUM_VIEW_SPREAD: float = 0.02

# How many sectors to include on each side of the long/short relative view.
MOMENTUM_TOP_N: int = 3
MOMENTUM_BOTTOM_N: int = 3
