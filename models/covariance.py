"""Ledoit-Wolf shrinkage covariance.

Sample covariance is asymptotically unbiased but with N similar in size to T
the off-diagonals are dominated by sampling noise. Ledoit & Wolf (2004) shrink
the sample matrix toward a structured target (here a constant-correlation
identity-like matrix) by an analytically chosen weight. The result is always
positive definite and well-conditioned, so the inverse used in mean-variance
and Black-Litterman behaves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import TRADING_DAYS_PER_YEAR


def shrunk_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf shrinkage covariance.

    Args:
        daily_returns: (T, N) matrix of simple daily returns, columns = tickers.

    Returns:
        (N, N) annualised covariance as a DataFrame indexed and columned by ticker.
    """
    if daily_returns.isna().any().any():
        raise ValueError("daily_returns contains NaNs; clean upstream")

    estimator = LedoitWolf().fit(daily_returns.values)
    daily = estimator.covariance_
    annual = daily * TRADING_DAYS_PER_YEAR

    return pd.DataFrame(
        annual,
        index=daily_returns.columns,
        columns=daily_returns.columns,
    )
