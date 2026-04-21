"""Black-Litterman expected returns and posterior covariance.

Reverse-optimisation gives an implied prior on returns from the assumption
that observed market-cap weights are themselves the outcome of mean-variance
optimisation by a representative investor. Investor views (P, Q, Omega) are
then blended with that prior in a Gaussian Bayesian update. The result is a
posterior mean and covariance fed straight into the same MVO solver.

Notation follows He & Litterman (1999):

    Pi          (N,)    implied equilibrium excess returns
    Sigma       (N, N)  asset return covariance
    P           (K, N)  pick matrix; each row encodes one view
    Q           (K,)    view return vector
    Omega       (K, K)  diagonal view uncertainty matrix
    tau         scalar  scaling on the prior covariance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    sigma: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> pd.Series:
    """Pi = lambda * Sigma * w_mkt via reverse optimisation."""
    if not sigma.index.equals(market_weights.index):
        raise ValueError("sigma and market_weights must share ticker order")
    pi = risk_aversion * sigma.values @ market_weights.values
    return pd.Series(pi, index=sigma.index, name="implied_returns")


def posterior_returns_and_cov(
    pi: pd.Series,
    sigma: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Bayesian posterior mean and covariance.

    Posterior mean (the BL master formula):

        mu_post = [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1
                  * [ (tau Sigma)^-1 Pi + P' Omega^-1 Q ]

    Posterior covariance of the parameter estimate is M; total uncertainty
    seen by the optimiser is Sigma + M (the parameter uncertainty M is added
    to the asset return covariance Sigma).
    """
    n = len(pi)
    if sigma.shape != (n, n):
        raise ValueError("sigma must be (N, N) matching pi")
    if P.shape[1] != n:
        raise ValueError("P must have N columns")
    if Q.shape[0] != P.shape[0]:
        raise ValueError("Q rows must match P rows")
    if Omega.shape != (P.shape[0], P.shape[0]):
        raise ValueError("Omega must be (K, K)")

    pi_v   = pi.values.astype(float)
    sig_v  = sigma.values.astype(float)
    tau_s  = tau * sig_v

    tau_s_inv = np.linalg.inv(tau_s)
    omega_inv = np.linalg.inv(Omega)

    precision_post = tau_s_inv + P.T @ omega_inv @ P
    M = np.linalg.inv(precision_post)

    mu_post = M @ (tau_s_inv @ pi_v + P.T @ omega_inv @ Q)

    sigma_post = sig_v + M

    return (
        pd.Series(mu_post, index=pi.index, name="bl_posterior_returns"),
        pd.DataFrame(sigma_post, index=sigma.index, columns=sigma.columns),
    )
