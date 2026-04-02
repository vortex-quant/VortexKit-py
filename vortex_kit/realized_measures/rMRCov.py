"""
vortex_kit.realized_measures.rMRCov
=====================================
Modulated Realized Covariance (pre-averaging approach, multivariate).

Diagonal: pre-averaged realized variance per asset.
Off-diagonal: pairwise pre-averaged covariance after refresh-time sync.
"""

from __future__ import annotations

import numpy as np

from vortex_kit.utils.sync import refresh_time
from vortex_kit.utils.pre_average import pre_average_returns
from vortex_kit.utils.matrices import make_psd
from .rMRVar import rMRVar


def rMRCov(prices_list: list[np.ndarray],
           timestamps_list: list[np.ndarray] | None = None,
           *,
           theta: float = 0.8,
           make_psd_flag: bool = False) -> np.ndarray | float:
    """Modulated Realized Covariance (pre-averaging approach).

    Diagonal: pre-averaged realized variance per asset.
    Off-diagonal: pairwise pre-averaged covariance after refresh-time sync.

    Parameters
    ----------
    prices_list : list of ndarray
        Tick-level prices for each asset.
    timestamps_list : list of ndarray or None
        Required for multivariate case.
    theta : float
        Bandwidth parameter.
    make_psd_flag : bool
        Project to PSD if True.

    Returns
    -------
    ndarray or float
    """
    d = len(prices_list)
    if d == 1:
        return rMRVar(prices_list[0], theta=theta)

    mat = np.zeros((d, d), dtype=np.float64)

    # Diagonal
    for i in range(d):
        mat[i, i] = rMRVar(prices_list[i], theta=theta)

    # Off-diagonal (pairwise)
    if timestamps_list is None:
        raise ValueError("timestamps_list required for multivariate rMRCov")

    PSI2 = 1.0 / 12.0  # integral of g(x)^2 for g(x)=min(x,1-x)

    for i in range(d):
        for j in range(i + 1, d):
            lp_i = np.log(np.asarray(prices_list[i], dtype=np.float64))
            lp_j = np.log(np.asarray(prices_list[j], dtype=np.float64))

            sync_ts, sync_px = refresh_time(
                [timestamps_list[i], timestamps_list[j]],
                [lp_i, lp_j],
            )
            if len(sync_ts) < 4:
                mat[i, j] = mat[j, i] = 0.0
                continue

            N = len(sync_ts)
            kn = int(np.floor(theta * np.sqrt(N)))
            if kn < 2:
                kn = 2

            r1 = np.diff(sync_px[:, 0])
            r2 = np.diff(sync_px[:, 1])
            pa1 = pre_average_returns(r1, kn)
            pa2 = pre_average_returns(r2, kn)

            mrc = (N / (N - kn + 2)) * (1.0 / (PSI2 * kn)) * np.nansum(pa1 * pa2)
            mat[i, j] = mrc
            mat[j, i] = mrc

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
