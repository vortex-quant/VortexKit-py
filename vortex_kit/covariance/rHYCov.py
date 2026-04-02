"""
vortex_kit.covariance.rHYCov
=============================
Hayashi-Yoshida non-synchronous realized covariance.

Handles asynchronous trading by summing products of returns whose
sampling intervals overlap. The diagonal uses standard realized
variance (rRVar) on each asset's own tick data.

Reference
---------
Hayashi, T. and Yoshida, N. (2005).
    "On covariance estimation of non-synchronously observed diffusion
    processes." Bernoulli, 11, 359-379.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns
from vortex_kit.utils.matrices import make_psd


@njit(cache=True)
def _hy_pair_core(ts1: np.ndarray, r1: np.ndarray,
                  ts2: np.ndarray, r2: np.ndarray) -> float:
    """Hayashi-Yoshida cross-covariance for one pair of asynchronous series.

    Sums r1[i] * r2[j] whenever return intervals [s_{i-1}, s_i] and
    [t_{j-1}, t_j] overlap (i.e. s_{i-1} < t_j and t_{j-1} < s_i).

    Parameters
    ----------
    ts1, ts2 : ndarray
        Sorted tick timestamps for asset 1 and 2 (prices at each tick).
    r1, r2 : ndarray
        Log-returns for asset 1 and 2, length len(ts)-1.

    Returns
    -------
    float
    """
    n1 = len(r1)
    n2 = len(r2)
    total = 0.0
    for i in range(n1):
        # interval for return i in series 1: [ts1[i], ts1[i+1])
        a_lo = ts1[i]
        a_hi = ts1[i + 1]
        for j in range(n2):
            b_lo = ts2[j]
            b_hi = ts2[j + 1]
            # overlap condition: a_lo < b_hi and b_lo < a_hi
            if a_lo < b_hi and b_lo < a_hi:
                total += r1[i] * r2[j]
    return total


def rHYCov(prices_list: list[np.ndarray],
           timestamps_list: list[np.ndarray],
           *,
           make_psd_flag: bool = True) -> np.ndarray | float:
    """Hayashi-Yoshida non-synchronous realized covariance.

    Handles asynchronous trading by summing products of returns whose
    sampling intervals overlap. The diagonal uses standard realized
    variance (rRVar) on each asset's own tick data.

    Parameters
    ----------
    prices_list : list of ndarray
        Tick-level prices for each asset (possibly different lengths).
    timestamps_list : list of ndarray
        Sorted numeric timestamps corresponding to each price series.
    make_psd_flag : bool
        If True (default), project result to nearest PSD matrix.

    Returns
    -------
    ndarray, shape (K, K)  or  float  (univariate)
        Hayashi-Yoshida covariance matrix.

    Reference
    ---------
    Hayashi, T. and Yoshida, N. (2005).
    """
    from vortex_kit.realized_measures.rRVar import rRVar  # Lazy import

    k = len(prices_list)
    if k == 1:
        r = log_returns(np.asarray(prices_list[0], dtype=np.float64))
        return rRVar(r)

    mat = np.zeros((k, k), dtype=np.float64)

    # Diagonal: standard RV per asset
    returns_list = []
    for i in range(k):
        r = log_returns(np.asarray(prices_list[i], dtype=np.float64))
        returns_list.append(r)
        mat[i, i] = rRVar(r)

    # Off-diagonal: Hayashi-Yoshida
    for i in range(k):
        ts_i = np.asarray(timestamps_list[i], dtype=np.float64)
        r_i = returns_list[i]
        for j in range(i + 1, k):
            ts_j = np.asarray(timestamps_list[j], dtype=np.float64)
            r_j = returns_list[j]
            c = _hy_pair_core(ts_i, r_i, ts_j, r_j)
            mat[i, j] = c
            mat[j, i] = c

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
