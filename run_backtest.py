"""End-to-end runner: fetch -> backtest -> metrics -> save results and figures.

Usage:
    python run_backtest.py

Outputs land in:
    results/   weights and returns CSVs, summary JSON
    figures/   PNGs used in the README
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summary
from config import (
    BENCHMARK_TICKER,
    END_DATE,
    MARKET_CAP_WEIGHTS,
    SECTOR_TICKERS,
    START_DATE,
)
from data.fetch import get_adjusted_close

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# strategies in display order
DISPLAY_NAMES = {
    "equal_weight":    "Equal Weight",
    "spy":             "S&P 500 (SPY)",
    "mvo":             "Mean-Variance",
    "black_litterman": "Black-Litterman",
}

# matplotlib settings applied once at module load
plt.rcParams.update({
    "figure.figsize": (10, 5.5),
    "figure.dpi":     120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.size":         10,
})


def fetch_universe() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    sector_prices = get_adjusted_close(SECTOR_TICKERS, START_DATE, END_DATE)
    benchmark = get_adjusted_close([BENCHMARK_TICKER], START_DATE, END_DATE)[BENCHMARK_TICKER]

    sector_prices = sector_prices.loc[:, SECTOR_TICKERS]
    market_w = pd.Series(MARKET_CAP_WEIGHTS, index=SECTOR_TICKERS)
    market_w = market_w / market_w.sum()
    return sector_prices, benchmark, market_w


def _save_strategy_artifacts(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    for key, res in results.items():
        res.weights.to_csv(RESULTS / f"weights_{key}.csv")
        res.returns.to_csv(RESULTS / f"returns_{key}.csv", header=["return"])


def _build_summary(results: dict) -> pd.DataFrame:
    rows = []
    for key in ["equal_weight", "spy", "mvo", "black_litterman"]:
        res = results[key]
        s = summary(res.returns, res.weights if not res.weights.empty else None)
        s["strategy"] = DISPLAY_NAMES[key]
        rows.append(s)
    df = pd.DataFrame(rows).set_index("strategy")
    return df[[
        "annual_return", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "turnover", "concentration_hhi",
    ]]


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def plot_stacked_weights(results: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    for key in ["mvo", "black_litterman"]:
        w = results[key].weights
        fig, ax = plt.subplots()
        ax.stackplot(w.index, w.T.values, labels=w.columns, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Portfolio weight")
        ax.set_title(f"{DISPLAY_NAMES[key]} weights over time")
        ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(FIGURES / f"weights_{key}.png")
        plt.close(fig)


def plot_equity_curves(results: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    for key, label in DISPLAY_NAMES.items():
        eq = _equity_curve(results[key].returns)
        ax.plot(eq.index, eq.values, label=label, linewidth=1.5)
    ax.set_ylabel("Growth of $1 (out of sample)")
    ax.set_title("Out-of-sample equity curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "equity_curves.png")
    plt.close(fig)


def plot_drawdown(results: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    for key, label in DISPLAY_NAMES.items():
        eq = _equity_curve(results[key].returns)
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=label, linewidth=1.5)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown paths")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "drawdown.png")
    plt.close(fig)


def main() -> None:
    print(f"fetching prices {START_DATE} -> {END_DATE} ...")
    prices, benchmark, market_w = fetch_universe()
    print(f"  {prices.shape[1]} sectors, {len(prices)} trading days")

    print("running walk-forward backtest ...")
    results = run_backtest(prices, benchmark, market_w)
    n_periods = len(next(iter(results.values())).returns)
    print(f"  {n_periods} out-of-sample monthly periods")

    print("writing artefacts ...")
    _save_strategy_artifacts(results)

    summary_df = _build_summary(results)
    summary_df.to_csv(RESULTS / "summary.csv")
    summary_df.round(4).to_json(RESULTS / "summary.json", orient="index", indent=2)

    plot_stacked_weights(results)
    plot_equity_curves(results)
    plot_drawdown(results)

    print()
    print(summary_df.round(4).to_string())
    print()
    print("done. results/ and figures/ updated.")


if __name__ == "__main__":
    main()
