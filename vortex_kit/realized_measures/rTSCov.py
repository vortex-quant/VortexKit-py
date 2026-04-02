"""
vortex_kit.realized_measures.rTSCov
======================================
Two-Scale Realized Covariance (noise-robust, multivariate).

Diagonal: TSRV per asset on raw ticks.
Off-diagonal: pairwise TSCov after refresh-time synchronisation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.sync import refresh_time
from vortex_kit.utils.matrices import make_psd
from .rTSVar import _tsrv_core


@njit(cache=True)
def _tscov_bi_core(lp1: np.ndarray, lp2: np.ndarray,
                   K: int, J: int) -> float:
    """Two-Scale Covariance for a synchronized pair."""
    n = len(lp1)
    nbar_K = (n - K) / K
    nbar_J = (n - J) / J

    cross_K = 0.0
    for k in range(K):
        i = k
        while i + K < n:
            cross_K += (lp1[i + K] - lp1[i]) * (lp2[i + K] - lp2[i])
            i += K

    cross_J = 0.0
    for j in range(J):
        i = j
        while i + J < n:
            cross_J += (lp1[i + J] - lp1[i]) * (lp2[i + J] - lp2[i])
            i += J

    adj = n / ((K - J) * nbar_K)
    return adj * ((1.0 / K) * cross_K - (nbar_K / nbar_J) * (1.0 / J) * cross_J)


def rTSCov(prices_list: list[np.ndarray],
           timestamps_list: list[np.ndarray] | None = None,
           *,
           K: int = 300,
           J: int = 1,
           make_psd_flag: bool = False) -> np.ndarray | float:
    """Two-Scale Realized Covariance.

    Diagonal: TSRV per asset (on raw ticks).
    Off-diagonal: pairwise TSCov after refresh-time synchronisation.

    Parameters
    ----------
    prices_list : list of ndarray
        Tick-level prices for each asset.
    timestamps_list : list of ndarray or None
        Numeric timestamps for each asset (needed for synchronisation).
        Required if ``len(prices_list) > 1``.
    K : int
        Slow time-scale.
    J : int
        Fast time-scale.
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray or float
        Scalar if single asset, else K × K matrix.
    """
    from .rTSVar import rTSVar  # Lazy import to avoid circular deps

    d = len(prices_list)
    if d == 1:
        return rTSVar(prices_list[0], K=K, J=J)

    mat = np.zeros((d, d), dtype=np.float64)

    # Diagonal: TSRV on full tick data
    for i in range(d):
        mat[i, i] = rTSVar(prices_list[i], K=K, J=J)

    # Off-diagonal: pairwise sync + TSCov
    if timestamps_list is None:
        raise ValueError("timestamps_list required for multivariate rTSCov")
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
                c = _tscov_bi_core(sync_px[:, 0], sync_px[:, 1], K, J)
                mat[i, j] = c
                mat[j, i] = c

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
