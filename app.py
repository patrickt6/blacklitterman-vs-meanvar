"""Streamlit dashboard for the Black-Litterman vs. Mean-Variance backtest.

Loads pre-computed artefacts from `results/` and renders headline metrics,
weight paths, equity curves, drawdowns, and a monthly snapshot picker. No
yfinance calls happen on launch.

Run:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

STRATEGIES = [
    ("equal_weight",    "Equal Weight"),
    ("spy",             "S&P 500 (SPY)"),
    ("mvo",             "Mean-Variance"),
    ("black_litterman", "Black-Litterman"),
]

st.set_page_config(
    page_title="Black-Litterman vs. Mean-Variance",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "summary.csv", index_col="strategy")


@st.cache_data
def load_returns(key: str) -> pd.Series:
    df = pd.read_csv(RESULTS / f"returns_{key}.csv", index_col=0, parse_dates=True)
    return df["return"]


@st.cache_data
def load_weights(key: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"weights_{key}.csv", index_col=0, parse_dates=True)


@st.cache_data
def equity_curves() -> pd.DataFrame:
    return pd.DataFrame({
        name: (1.0 + load_returns(key)).cumprod()
        for key, name in STRATEGIES
    })


summary = load_summary()

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("Black-Litterman vs. Mean-Variance")
st.markdown(
    "A walk-forward comparison of Markowitz mean-variance optimisation "
    "against the Black-Litterman model on 10 S&P 500 sector ETFs. "
    "October 2018 through December 2025, 86 monthly out-of-sample rebalances."
)

# ---------------------------------------------------------------------------
# headline KPIs: BL vs MVO
# ---------------------------------------------------------------------------

st.subheader("Headline: Black-Litterman vs. Mean-Variance")
bl = summary.loc["Black-Litterman"]
mv = summary.loc["Mean-Variance"]

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Sharpe ratio",
    f"{bl['sharpe']:.2f}",
    f"{bl['sharpe'] - mv['sharpe']:+.2f} vs MVO",
)
k2.metric(
    "Max drawdown",
    f"{bl['max_drawdown'] * 100:.1f}%",
    f"{(bl['max_drawdown'] - mv['max_drawdown']) * 100:+.1f} pp",
)
k3.metric(
    "Concentration (HHI)",
    f"{bl['concentration_hhi']:.2f}",
    f"{bl['concentration_hhi'] - mv['concentration_hhi']:+.2f}",
    delta_color="inverse",
)
k4.metric(
    "Annualised volatility",
    f"{bl['annual_vol'] * 100:.1f}%",
    f"{(bl['annual_vol'] - mv['annual_vol']) * 100:+.1f} pp",
    delta_color="inverse",
)

# ---------------------------------------------------------------------------
# results table
# ---------------------------------------------------------------------------

st.subheader("All strategies, out-of-sample")

display = summary.copy()
for c in ["annual_return", "annual_vol", "max_drawdown"]:
    display[c] = display[c].map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")
for c in ["sharpe", "sortino", "turnover", "concentration_hhi"]:
    display[c] = display[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")

display = display.rename(columns={
    "annual_return":     "Return",
    "annual_vol":        "Vol",
    "sharpe":            "Sharpe",
    "sortino":           "Sortino",
    "max_drawdown":      "Max DD",
    "turnover":          "Turnover",
    "concentration_hhi": "HHI",
})
st.dataframe(display, use_container_width=True)

# ---------------------------------------------------------------------------
# visual tabs
# ---------------------------------------------------------------------------

t_wts, t_eq, t_dd, t_snap = st.tabs([
    "Weights over time",
    "Equity curves",
    "Drawdown",
    "Snapshot by month",
])

with t_wts:
    choice = st.radio(
        "Strategy",
        ["Mean-Variance", "Black-Litterman"],
        horizontal=True,
        key="weights_radio",
    )
    wkey = "mvo" if choice == "Mean-Variance" else "black_litterman"
    st.area_chart(load_weights(wkey))
    st.caption(
        "Mean-Variance concentrates on whichever sector had the highest "
        "in-sample mean return, then jumps elsewhere as the leader changes. "
        "Black-Litterman stays diversified because the equilibrium prior "
        "anchors the posterior."
    )

with t_eq:
    st.line_chart(equity_curves())
    st.caption("Growth of $1 invested at the first out-of-sample rebalance.")

with t_dd:
    eq = equity_curves()
    dd = eq / eq.cummax() - 1.0
    st.area_chart(dd)
    st.caption("Peak-to-trough drawdown of each strategy.")

with t_snap:
    wts_mvo = load_weights("mvo")
    wts_bl  = load_weights("black_litterman")
    dates   = wts_mvo.index.strftime("%Y-%m").tolist()

    idx = st.slider(
        "Pick a rebalance month",
        min_value=0,
        max_value=len(dates) - 1,
        value=len(dates) - 1,
        format="%d",
    )
    st.caption(f"Allocations chosen at the start of {dates[idx]}.")

    side = pd.concat([
        wts_mvo.iloc[idx].rename("Mean-Variance"),
        wts_bl.iloc[idx].rename("Black-Litterman"),
    ], axis=1)
    st.bar_chart(side)
    st.caption(
        "MVO corners on one or two sectors; BL keeps a diversified sleeve "
        "close to the cap-weighted prior."
    )

# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

with st.expander("The math"):
    st.markdown("**Ledoit-Wolf shrinkage covariance.**")
    st.latex(r"\hat{\Sigma} = \delta F + (1 - \delta) S")

    st.markdown("**Implied equilibrium returns.** Reverse-optimise the consensus portfolio:")
    st.latex(r"\Pi = \lambda \, \Sigma \, w_{mkt}")

    st.markdown(
        "**Posterior.** The Black-Litterman master formula is the Gaussian "
        "product of prior and view:"
    )
    st.latex(
        r"\hat{\mu} = \left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}"
        r"\left[(\tau \Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]"
    )
    st.latex(
        r"\hat{\Sigma} = \Sigma + \left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}"
    )

st.divider()
st.caption(
    "Source: github.com/patrickt6/blacklitterman-vs-meanvar"
    " · Patrick Taylor, Queen's University"
)
