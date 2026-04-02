"""
vortex_kit.covariance.rThresholdCov
====================================
Threshold (jump-truncated) realized covariance.

Truncates returns that exceed a threshold based on local volatility
estimate to remove jumps.

Reference
---------
Boudt, K., Croux, C. and Laurent, S. (2011).
    "Robust estimation of intraweek periodicity in volatility and jump
    detection." Journal of Empirical Finance, 18, 353-367.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.matrices import make_psd


@njit(cache=True)
def _threshold_cov_pair(r1: np.ndarray, r2: np.ndarray,
                        threshold: float) -> float:
    """Threshold covariance for a pair of return series."""
    n = len(r1)
    s = 0.0
    for i in range(n):
        if abs(r1[i]) <= threshold and abs(r2[i]) <= threshold:
            s += r1[i] * r2[i]
    return s


def rThresholdCov(returns_list: list[np.ndarray],
                  *,
                  threshold: float | None = None,
                  make_psd_flag: bool = False) -> np.ndarray:
    """Threshold (jump-truncated) realized covariance.

    Parameters
    ----------
    returns_list : list of ndarray
        K aligned return series (same length N).
    threshold : float or None
        Jump threshold. If None, computed from data as 3 × sqrt(RV).
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray, shape (K, K)
        Threshold covariance matrix.
    """
    from vortex_kit.realized_measures.rRVar import rRVar  # Lazy import

    k = len(returns_list)
    mat = np.zeros((k, k), dtype=np.float64)

    # Compute RV for diagonal and threshold
    rv_vals = [rRVar(r) for r in returns_list]

    for i in range(k):
        mat[i, i] = rv_vals[i]

    # Use adaptive threshold if not provided
    if threshold is None:
        # Use average local volatility estimate
        avg_rv = sum(rv_vals) / len(rv_vals)
        threshold = 3.0 * math.sqrt(avg_rv)

    for i in range(k):
        for j in range(i + 1, k):
            c = _threshold_cov_pair(returns_list[i], returns_list[j], threshold)
            mat[i, j] = c
            mat[j, i] = c

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
