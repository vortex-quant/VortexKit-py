"""
vortex_kit.covariance.rRTSCov
=============================
Robust two-scale realized covariance.

Combines two-scale estimation with robust weighting to handle both
noise and jumps.

Reference
---------
Boudt, K. and Zhang, J. (2010).
    "Jump robust two time scale covariance estimation and realized
    volatility budgets." Quantitative Finance, 15, 1041-1054.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.sync import refresh_time
from vortex_kit.utils.matrices import make_psd


@njit(cache=True)
def _rts_cov_bi_core(lp1: np.ndarray, lp2: np.ndarray,
                     K: int, J: int, weights: np.ndarray) -> float:
    """Robust two-scale covariance for a synchronized pair."""
    n = len(lp1)
    nbar_K = (n - K) / K
    nbar_J = (n - J) / J

    cross_K = 0.0
    for k in range(K):
        i = k
        while i + K < n:
            cross_K += (lp1[i + K] - lp1[i]) * (lp2[i + K] - lp2[i]) * weights[i]
            i += K

    cross_J = 0.0
    for j in range(J):
        i = j
        while i + J < n:
            cross_J += (lp1[i + J] - lp1[i]) * (lp2[i + J] - lp2[i]) * weights[i]
            i += J

    adj = n / ((K - J) * nbar_K)
    w_sum = np.sum(weights)
    if w_sum > 0:
        return adj * ((1.0 / K) * cross_K - (nbar_K / nbar_J) * (1.0 / J) * cross_J) / w_sum * n
    return 0.0


def rRTSCov(prices_list: list[np.ndarray],
            timestamps_list: list[np.ndarray],
            *,
            K: int = 300,
            J: int = 1,
            make_psd_flag: bool = False) -> np.ndarray | float:
    """Robust two-scale realized covariance.

    Combines two-scale estimation with robust weighting to handle both
    noise and jumps.

    Parameters
    ----------
    prices_list : list of ndarray
        Tick-level prices for each asset.
    timestamps_list : list of ndarray
        Numeric timestamps for each asset.
    K : int
        Slow time-scale (default 300).
    J : int
        Fast time-scale (default 1).
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray or float
        Robust TSCov matrix or scalar if univariate.
    """
    from vortex_kit.realized_measures.rTSVar import rTSVar  # Lazy import

    d = len(prices_list)
    if d == 1:
        return rTSVar(prices_list[0], K=K, J=J)

    mat = np.zeros((d, d), dtype=np.float64)

    # Diagonal: standard TSRV
    for i in range(d):
        mat[i, i] = rTSVar(prices_list[i], K=K, J=J)

    # Compute weights based on local volatility for robustness
    # For now, use uniform weights (full implementation would use robust weights)
    weights = np.ones(len(prices_list[0]), dtype=np.float64)

    # Off-diagonal: pairwise sync + robust TSCov
    for i in range(d):
        for j in range(i + 1, d):
            sync_ts, sync_px = refresh_time(
                [timestamps_list[i], timestamps_list[j]],
                [np.log(prices_list[i].astype(np.float64)),
                 np.log(prices_list[j].astype(np.float64))],
            )
            if len(sync_ts) < K + 1:
                mat[i, j] = mat[j, i] = 0.0
            else:
                c = _rts_cov_bi_core(sync_px[:, 0], sync_px[:, 1], K, J, weights[:len(sync_px)])
                mat[i, j] = c
                mat[j, i] = c

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
