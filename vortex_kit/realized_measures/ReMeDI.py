"""
vortex_kit.realized_measures.ReMeDI
======================================
Realized Method-of-moments Estimator of noise auto-covariance.

Estimates the auto-covariance of market-microstructure noise at
specified lags.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _remedi_core(prices: np.ndarray, kn: int, lag: int) -> float:
    """Core ReMeDI estimator for a single lag. Uses raw prices (not log)."""
    n = len(prices)
    s = 0.0
    count = 0
    for i in range(2 * kn, n - kn - lag):
        s += ((prices[i + lag] - prices[i + lag + kn])
              * (prices[i] - prices[i - 2 * kn]))
        count += 1
    if count == 0:
        return 0.0
    return s / n


def ReMeDI(prices: np.ndarray,
           *,
           kn: int = 1,
           lags: list[int] | int = 1) -> dict[int, float]:
    """Realized Method-of-moments Estimator of noise auto-covariance.

    Estimates the auto-covariance of market-microstructure noise at
    specified lags.

    Parameters
    ----------
    prices : ndarray
        Tick-level prices.
    kn : int
        Tuning parameter (default 1).
    lags : int or list of int
        Lag(s) at which to estimate noise auto-covariance.

    Returns
    -------
    dict
        Mapping from lag to estimated noise auto-covariance.
    """
    lp = np.asarray(prices, dtype=np.float64)
    if isinstance(lags, int):
        lags = [lags]
    result = {}
    for lag in lags:
        result[lag] = _remedi_core(lp, kn, lag)
    return result
