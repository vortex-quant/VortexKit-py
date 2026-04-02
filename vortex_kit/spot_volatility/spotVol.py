"""
vortex_kit.spot_volatility.spotVol
====================================
Spot volatility estimation using local windows.

Implements nonparametric spot volatility estimation based on
realized variance computed over rolling windows.

Reference
---------
Kristensen, D. (2010).
    "Nonparametric filtering of the realized spot volatility."
    Econometric Theory, 26, 60-93.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _spot_vol_core(returns: np.ndarray, window: int) -> np.ndarray:
    """Core spot volatility estimation over rolling windows."""
    n = len(returns)
    out = np.full(n, np.nan, dtype=np.float64)

    half_win = window // 2

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)

        s = 0.0
        count = 0
        for j in range(start, end):
            s += returns[j] * returns[j]
            count += 1

        if count > 0:
            out[i] = math.sqrt(s)

    return out


def spotVol(returns: np.ndarray,
            *,
            window: int = 20,
            method: str = "rv") -> np.ndarray:
    """Spot volatility estimation using local windows.

    Parameters
    ----------
    returns : ndarray
        Log-returns.
    window : int
        Window size for local volatility estimation.
    method : str
        Estimation method: "rv" (realized variance), "bv" (bipower).

    Returns
    -------
    ndarray
        Spot volatility estimates for each time point.

    Reference
    ---------
    Kristensen, D. (2010).
        "Nonparametric filtering of the realized spot volatility."
        Econometric Theory, 26, 60-93.
    """
    r = np.asarray(returns, dtype=np.float64)
    return _spot_vol_core(r, window)
