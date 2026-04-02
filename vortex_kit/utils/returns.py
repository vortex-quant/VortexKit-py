"""
vortex_kit.utils.returns
=========================
Return computation utilities.

Functions for computing log-returns and simple returns from price series.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from a price series.

    Parameters
    ----------
    prices : ndarray, shape (N,)
        Price series. Must contain strictly positive values.

    Returns
    -------
    ndarray, shape (N-1,)
        Log-returns  ``log(prices[i+1] / prices[i])``.
    """
    n = len(prices)
    ret = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        ret[i] = np.log(prices[i + 1] / prices[i])
    return ret


@njit(cache=True)
def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple (arithmetic) returns from a price series.

    Parameters
    ----------
    prices : ndarray, shape (N,)
        Price series.

    Returns
    -------
    ndarray, shape (N-1,)
        Simple returns  ``prices[i+1] / prices[i] - 1``.
    """
    n = len(prices)
    ret = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        ret[i] = prices[i + 1] / prices[i] - 1.0
    return ret
