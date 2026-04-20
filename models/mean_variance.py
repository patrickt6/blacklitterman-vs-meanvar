"""Markowitz mean-variance optimisation.

We maximise the quadratic utility

    U(w) = w' mu - (lambda / 2) w' Sigma w

subject to long-only fully-invested constraints (0 <= w_i, sum w = 1). With
no short-sales the problem is a convex QP with a unique solution; SLSQP
handles it in a few iterations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def mean_variance_weights(
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float,
) -> pd.Series:
    """Long-only utility-maximising weights.

    Args:
        mu: (N,) expected excess returns indexed by ticker.
        sigma: (N, N) covariance matrix (same units as mu).
        risk_aversion: lambda; larger -> less risky portfolio.

    Returns:
        (N,) weights indexed by ticker, summing to 1.
    """
    if not mu.index.equals(sigma.index):
        raise ValueError("mu and sigma must share the same ticker order")

    n = len(mu)
    mu_v = mu.values.astype(float)
    s_v  = sigma.values.astype(float)

    def neg_utility(w: np.ndarray) -> float:
        return -(w @ mu_v - 0.5 * risk_aversion * w @ s_v @ w)

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        return -(mu_v - risk_aversion * s_v @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_utility,
        w0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"MVO solver failed: {result.message}")

    w = np.clip(result.x, 0.0, None)
    w = w / w.sum()
    return pd.Series(w, index=mu.index)
