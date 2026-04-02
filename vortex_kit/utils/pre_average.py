"""
vortex_kit.utils.pre_average
==============================
Pre-averaging utilities for noise-robust estimators.

Functions used in modulated realized covariance (MRC) estimation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .stats import gfunction


@njit(cache=True)
def pre_average_returns(returns: np.ndarray, kn: int) -> np.ndarray:
    """Compute pre-averaged returns using the g-function weight.

    Parameters
    ----------
    returns : ndarray, shape (N,)
        Returns series.
    kn : int
        Pre-averaging window (number of returns to average).

    Returns
    -------
    ndarray, shape (N-kn+1,)
        Pre-averaged returns.
    """
    n = len(returns)
    m = n - kn + 1
    out = np.empty(m, dtype=np.float64)

    # g(j/kn) weights
    weights = gfunction(np.arange(kn, dtype=np.float64) / kn)
    g_sum = np.sum(weights)

    for i in range(m):
        s = 0.0
        for j in range(kn):
            s += returns[i + j] * weights[j]
        out[i] = s / g_sum

    return out


@njit(cache=True)
def compute_psi_constants(kn: int) -> tuple[float, float]:
    """Compute psi₁ and psi₂ constants for pre-averaging.

    ψ₁ = kn × Σⱼ(g(j/kn) - g((j-1)/kn))²
    ψ₂ = (1/kn) × Σⱼ g(j/kn)²

    For g(x) = min(x, 1-x), psi₂ → 1/12 as kn → ∞.

    Parameters
    ----------
    kn : int
        Pre-averaging window.

    Returns
    -------
    tuple[float, float]
        (psi1, psi2)
    """
    # g(j/kn) for j = 0, ..., kn
    j_over_kn = np.arange(kn + 1, dtype=np.float64) / kn
    g_vals = gfunction(j_over_kn)

    # psi1
    diff = g_vals[1:] - g_vals[:-1]
    psi1 = kn * np.sum(diff * diff)

    # psi2 (use j = 0, ..., kn-1 for averaging)
    psi2 = np.sum(g_vals[:-1] * g_vals[:-1]) / kn

    return psi1, psi2
