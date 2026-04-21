"""Systematic Black-Litterman views from cross-sectional momentum.

The canonical BL "view" is qualitative: a portfolio manager states an
opinion such as "tech outperforms financials by 2%". To keep the comparison
honest and reproducible we drop a quantitative signal in instead: at each
rebalance date we rank sectors by their trailing 6-month return, take the
top three and bottom three, and assert the relative view

    mean(top 3 sectors) - mean(bottom 3 sectors) = MOMENTUM_VIEW_SPREAD

This becomes one row in the pick matrix P:

    P = [ +1/3  ...  -1/3  ...  0  ... ]      (1, N)
    Q = [ MOMENTUM_VIEW_SPREAD ]               (1,)
    Omega = [ tau * (P Sigma P') ]             (1, 1)

He & Litterman (1999) use this proportional Omega, so the view is taken with
confidence on the same order as the prior's view variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_BOTTOM_N,
    MOMENTUM_TOP_N,
    MOMENTUM_VIEW_SPREAD,
    TAU,
)


def momentum_signal(prices: pd.DataFrame, lookback_months: int) -> pd.Series:
    """Trailing total return over `lookback_months` months ending today.

    Uses month-end prices on either side of the window to avoid intra-month
    sampling noise. Index is the ticker.
    """
    monthly = prices.resample("ME").last()
    if len(monthly) <= lookback_months:
        raise ValueError(
            f"need at least {lookback_months + 1} months of prices, got {len(monthly)}"
        )
    end   = monthly.iloc[-1]
    start = monthly.iloc[-1 - lookback_months]
    return (end / start - 1.0).rename("momentum")


def build_momentum_view(
    prices: pd.DataFrame,
    sigma: pd.DataFrame,
    lookback_months: int,
    top_n: int = MOMENTUM_TOP_N,
    bottom_n: int = MOMENTUM_BOTTOM_N,
    spread: float = MOMENTUM_VIEW_SPREAD,
    tau: float = TAU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (P, Q, Omega) for a single long/short momentum view."""
    if top_n + bottom_n > len(prices.columns):
        raise ValueError("top_n + bottom_n exceeds asset count")

    signal = momentum_signal(prices[sigma.columns], lookback_months)
    ranked = signal.sort_values(ascending=False)
    longs  = ranked.iloc[:top_n].index
    shorts = ranked.iloc[-bottom_n:].index

    n = len(sigma.columns)
    p = np.zeros((1, n))
    cols = list(sigma.columns)
    for t in longs:
        p[0, cols.index(t)] =  1.0 / top_n
    for t in shorts:
        p[0, cols.index(t)] = -1.0 / bottom_n

    q = np.array([spread])

    view_var = float(p @ sigma.values @ p.T)
    omega = np.array([[tau * view_var]])

    return p, q, omega
