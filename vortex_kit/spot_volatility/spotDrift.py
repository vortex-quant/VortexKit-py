"""
vortex_kit.spot_volatility.spotDrift
=======================================
Spot drift estimation using local windows.

Estimates the instantaneous drift (mean return) using rolling windows.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _spot_drift_core(returns: np.ndarray, window: int) -> np.ndarray:
    """Core spot drift estimation over rolling windows."""
    n = len(returns)
    out = np.full(n, np.nan, dtype=np.float64)

    half_win = window // 2

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)

        s = 0.0
        count = 0
        for j in range(start, end):
            s += returns[j]
            count += 1

        if count > 0:
            out[i] = s / count

    return out


def spotDrift(data: np.ndarray, *, window: int = 20, make_returns: bool = False) -> np.ndarray:
    """Spot drift estimation using local windows.

    Parameters
    ----------
    data : ndarray
        Log-returns or prices.
    window : int
        Window size for local drift estimation.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    ndarray
        Spot drift estimates for each time point.
    """
    from vortex_kit.utils.returns import log_returns

    if make_returns:
        r = log_returns(data)
    elif data.ndim == 1 and len(data) > 0 and data[0] > 0:
        r = log_returns(data)
    else:
        r = np.asarray(data, dtype=np.float64)

    return _spot_drift_core(r, window)
