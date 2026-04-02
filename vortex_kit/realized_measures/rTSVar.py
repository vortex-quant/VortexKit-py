"""
vortex_kit.realized_measures.rTSVar
======================================
Two-Scale Realized Variance (noise-robust).

Zhang, Mykland, Aït-Sahalia (2005) estimator that uses subsampling
at a slow scale K and a fast scale J (typically 1), then bias-corrects.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _tsrv_core(log_prices: np.ndarray, K: int, J: int) -> float:
    """Two-Scale Realized Variance (TSRV) on log-prices.

    Zhang, Mykland, Aït-Sahalia (2005).
    """
    n = len(log_prices)
    nbar_K = (n - K) / K       # (n - K + 1 - 1) / K  →  average # of K-returns
    nbar_J = (n - J) / J

    # K-grid subsampled RV
    sum_K = 0.0
    for k in range(K):
        i = k
        while i + K < n:
            diff = log_prices[i + K] - log_prices[i]
            sum_K += diff * diff
            i += K

    # J-grid subsampled RV
    sum_J = 0.0
    for j in range(J):
        i = j
        while i + J < n:
            diff = log_prices[i + J] - log_prices[i]
            sum_J += diff * diff
            i += J

    adj = 1.0 / (1.0 - nbar_K / nbar_J)
    return adj * ((1.0 / K) * sum_K - (nbar_K / nbar_J) * (1.0 / J) * sum_J)


def rTSVar(prices: np.ndarray, *, K: int = 300, J: int = 1) -> float:
    """Two-Scale Realized Variance (univariate).

    Noise-robust estimator that uses subsampling at a slow scale K
    and a fast scale J (typically 1), then bias-corrects.

    Parameters
    ----------
    prices : ndarray
        Tick-level log-prices (or raw prices — log is taken internally).
    K : int
        Slow time-scale (default 300).
    J : int
        Fast time-scale (default 1).

    Returns
    -------
    float
        Two-scale realized variance.
    """
    lp = np.log(np.asarray(prices, dtype=np.float64))
    return _tsrv_core(lp, K, J)
