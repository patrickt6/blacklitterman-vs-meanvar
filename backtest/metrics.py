"""Out-of-sample performance metrics for monthly portfolio returns.

All metrics consume a (T,) series of *period* returns (monthly here) and a
(T, N) matrix of weights chosen at the start of each period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MONTHS_PER_YEAR, RISK_FREE_RATE


def annualised_return(returns: pd.Series) -> float:
    return float(returns.mean() * MONTHS_PER_YEAR)


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    vol = annualised_volatility(returns)
    if vol == 0:
        return float("nan")
    return (annualised_return(returns) - rf) / vol


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    monthly_target = rf / MONTHS_PER_YEAR
    downside = returns[returns < monthly_target] - monthly_target
    if len(downside) == 0:
        return float("nan")
    downside_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(MONTHS_PER_YEAR))
    if downside_vol == 0:
        return float("nan")
    return (annualised_return(returns) - rf) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Max peak-to-trough drawdown of the cumulative return path. Negative number."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def turnover(weights: pd.DataFrame) -> float:
    """Average one-way portfolio turnover per rebalance.

    Each row of `weights` is the target portfolio at the start of period t.
    Turnover_t = 0.5 * sum_i |w_t - w_{t-1}|. Mean across rows >= 1.
    """
    if len(weights) < 2:
        return float("nan")
    diffs = weights.diff().dropna(how="all").abs()
    return float((0.5 * diffs.sum(axis=1)).mean())


def avg_concentration(weights: pd.DataFrame) -> float:
    """Average Herfindahl index of weights across rebalances.

    HHI = sum w_i^2. Equals 1/N for uniform allocation, 1 for a single-asset
    portfolio. A useful one-number summary of how concentrated the strategy is.
    MVO with sample-mean inputs typically corners on one or two assets.
    """
    if weights.empty:
        return float("nan")
    return float((weights ** 2).sum(axis=1).mean())


def summary(returns: pd.Series, weights: pd.DataFrame | None = None) -> dict[str, float]:
    out = {
        "annual_return":  annualised_return(returns),
        "annual_vol":     annualised_volatility(returns),
        "sharpe":         sharpe_ratio(returns),
        "sortino":        sortino_ratio(returns),
        "max_drawdown":   max_drawdown(returns),
    }
    if weights is not None and not weights.empty:
        out["turnover"] = turnover(weights)
        out["concentration_hhi"] = avg_concentration(weights)
    else:
        out["turnover"] = float("nan")
        out["concentration_hhi"] = float("nan")
    return out
