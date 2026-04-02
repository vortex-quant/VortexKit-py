"""
vortex_kit.utils.rolling
=========================
Rolling statistics utilities.

Numba-accelerated helpers for computing rolling min, median, and product.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def roll_min2(x: np.ndarray) -> np.ndarray:
    """Rolling minimum over window of 2.

    Parameters
    ----------
    x : ndarray, shape (N,)

    Returns
    -------
    ndarray, shape (N-1,)
        ``min(x[i], x[i+1])`` for i = 0 .. N-2.
    """
    n = len(x)
    out = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        out[i] = min(x[i], x[i + 1])
    return out


@njit(cache=True)
def roll_median3(x: np.ndarray) -> np.ndarray:
    """Rolling median over window of 3.

    Parameters
    ----------
    x : ndarray, shape (N,)

    Returns
    -------
    ndarray, shape (N-2,)
        ``median(x[i], x[i+1], x[i+2])`` for i = 0 .. N-3.
    """
    n = len(x)
    out = np.empty(n - 2, dtype=np.float64)
    for i in range(n - 2):
        a, b, c = x[i], x[i + 1], x[i + 2]
        # Median of three values
        if a <= b:
            if b <= c:
                out[i] = b
            elif a <= c:
                out[i] = c
            else:
                out[i] = a
        else:
            if a <= c:
                out[i] = a
            elif b <= c:
                out[i] = c
            else:
                out[i] = b
    return out


@njit(cache=True)
def roll_prod(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling product over specified window.

    Parameters
    ----------
    x : ndarray, shape (N,)
    window : int
        Window size for product.

    Returns
    -------
    ndarray, shape (N-window+1,)
        Rolling product of window consecutive elements.
    """
    n = len(x)
    m = n - window + 1
    out = np.empty(m, dtype=np.float64)
    for i in range(m):
        prod = 1.0
        for j in range(window):
            prod *= x[i + j]
        out[i] = prod
    return out
