"""Walk-forward backtest comparing four strategies.

Procedure for each monthly rebalance date t (starting after the training
window has filled):

    1. Take daily prices in [t - training_window, t].
    2. Compute Ledoit-Wolf annualised covariance Sigma.
    3. Choose target weights w_t under the strategy.
    4. Apply w_t to realised returns over [t, t + 1 month].

The output is a (T, N) DataFrame of target weights and a (T,) Series of
realised monthly portfolio returns for each strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    MOMENTUM_LOOKBACK_MONTHS,
    RISK_AVERSION,
    TAU,
    TRADING_DAYS_PER_YEAR,
    TRAINING_WINDOW_YEARS,
)
from data.fetch import daily_returns
from models.black_litterman import (
    implied_equilibrium_returns,
    posterior_returns_and_cov,
)
from models.covariance import shrunk_covariance
from models.mean_variance import mean_variance_weights
from views.momentum import build_momentum_view


@dataclass
class StrategyResult:
    name: str
    weights: pd.DataFrame      # (T, N), target weight at start of period
    returns: pd.Series         # (T,), realised return of period


def _rebalance_dates(prices: pd.DataFrame, training_years: int) -> pd.DatetimeIndex:
    """Month-end dates from the first one with a full training window onward."""
    months = prices.resample("ME").last().index
    first_idx = training_years * 12
    if first_idx >= len(months) - 1:
        raise ValueError("not enough history to form a single out-of-sample month")
    return months[first_idx:-1]   # last one would have no realised next month


def _next_month_return(prices: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Per-asset simple return from month-end t to the next month-end."""
    monthly = prices.resample("ME").last()
    if t not in monthly.index:
        t = monthly.index[monthly.index.get_indexer([t], method="nearest")[0]]
    pos = monthly.index.get_loc(t)
    if pos + 1 >= len(monthly):
        raise IndexError("no next month available for date " + str(t))
    return monthly.iloc[pos + 1] / monthly.iloc[pos] - 1.0


def _training_slice(prices: pd.DataFrame, t: pd.Timestamp, years: int) -> pd.DataFrame:
    start = t - pd.DateOffset(years=years)
    return prices.loc[(prices.index > start) & (prices.index <= t)]


def _equal_weight(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    return pd.Series(np.full(n, 1.0 / n), index=tickers)


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    market_weights: pd.Series,
    training_years: int = TRAINING_WINDOW_YEARS,
    risk_aversion: float = RISK_AVERSION,
    tau: float = TAU,
    momentum_lookback: int = MOMENTUM_LOOKBACK_MONTHS,
) -> dict[str, StrategyResult]:
    """Run the four strategies and return their results.

    Args:
        prices:           (T, N) daily adjusted close per asset.
        benchmark_prices: (T,)   daily adjusted close of the benchmark (SPY).
        market_weights:   (N,)   approximate cap weights, summing to 1.

    Returns:
        Dict keyed by "equal_weight" / "spy" / "mvo" / "black_litterman".
    """
    if not market_weights.index.equals(prices.columns):
        raise ValueError("market_weights index must match prices columns")

    tickers = list(prices.columns)
    rebal_dates = _rebalance_dates(prices, training_years)

    rows_ew, rows_mvo, rows_bl = [], [], []
    rets_ew, rets_mvo, rets_bl, rets_spy = [], [], [], []

    monthly_bench = benchmark_prices.resample("ME").last()
    monthly_bench_ret = monthly_bench.pct_change().dropna()

    for t in rebal_dates:
        train = _training_slice(prices, t, training_years)
        if len(train) < TRADING_DAYS_PER_YEAR:
            continue

        d_ret = daily_returns(train)
        sigma = shrunk_covariance(d_ret)

        # ---- mean-variance: sample-mean expected returns ----
        mu_sample = (d_ret.mean() * TRADING_DAYS_PER_YEAR).reindex(tickers)

        # ---- Black-Litterman: prior + momentum view ----
        pi = implied_equilibrium_returns(sigma, market_weights, risk_aversion)
        P, Q, Omega = build_momentum_view(
            train,
            sigma,
            momentum_lookback,
            tau=tau,
        )
        mu_bl, sigma_bl = posterior_returns_and_cov(pi, sigma, P, Q, Omega, tau)

        w_ew  = _equal_weight(tickers)
        w_mvo = mean_variance_weights(mu_sample, sigma, risk_aversion)
        w_bl  = mean_variance_weights(mu_bl,    sigma_bl, risk_aversion)

        nxt = _next_month_return(prices, t).reindex(tickers)

        rows_ew.append(w_ew.rename(t))
        rows_mvo.append(w_mvo.rename(t))
        rows_bl.append(w_bl.rename(t))

        rets_ew.append((float((w_ew  * nxt).sum()), t))
        rets_mvo.append((float((w_mvo * nxt).sum()), t))
        rets_bl.append((float((w_bl  * nxt).sum()), t))

        # benchmark: the realised SPY monthly return aligned with t->t+1
        next_t = monthly_bench.index[monthly_bench.index.get_indexer([t], method="nearest")[0] + 1]
        rets_spy.append((float(monthly_bench_ret.loc[next_t]), t))

    def _series(items: list[tuple[float, pd.Timestamp]]) -> pd.Series:
        vals, idx = zip(*items)
        return pd.Series(vals, index=pd.DatetimeIndex(idx, name="rebalance_date"))

    weights_ew  = pd.DataFrame(rows_ew)
    weights_mvo = pd.DataFrame(rows_mvo)
    weights_bl  = pd.DataFrame(rows_bl)
    weights_spy = pd.DataFrame(index=weights_ew.index)  # benchmark holds no sectors

    return {
        "equal_weight":     StrategyResult("equal_weight",     weights_ew,  _series(rets_ew)),
        "spy":              StrategyResult("spy",              weights_spy, _series(rets_spy)),
        "mvo":              StrategyResult("mvo",              weights_mvo, _series(rets_mvo)),
        "black_litterman":  StrategyResult("black_litterman",  weights_bl,  _series(rets_bl)),
    }
