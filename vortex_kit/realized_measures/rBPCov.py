"""
vortex_kit.realized_measures.rBPCov
=====================================
Realized bipower covariance matrix.

Diagonal elements use standard bipower variation (rBPVar).
Off-diagonal elements use the polarisation identity from
Barndorff-Nielsen and Shephard (2004).

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2004).
    "Econometric analysis of realized covariation: High frequency based
    covariance, regression, and correlation in financial economics."
    Econometrica, 72, 885-925.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.matrices import make_psd
from .rBPVar import _bpv_core


@njit(cache=True)
def _bpcov_bi_core(r1: np.ndarray, r2: np.ndarray) -> float:
    """Off-diagonal bipower covariance for two aligned return series."""
    n = len(r1)
    s = 0.0
    for i in range(1, n):
        ap = abs(r1[i] + r2[i])
        am = abs(r1[i] - r2[i])
        ap1 = abs(r1[i - 1] + r2[i - 1])
        am1 = abs(r1[i - 1] - r2[i - 1])
        s += ap * ap1 - am * am1
    return (math.pi / 8.0) * s


def rBPCov(returns_list: list[np.ndarray],
           *,
           make_psd_flag: bool = False) -> np.ndarray:
    """Realized bipower covariance matrix.

    Diagonal elements use :func:`rBPVar`.
    Off-diagonal elements use the polarisation identity
    from Barndorff-Nielsen and Shephard (2004).

    Parameters
    ----------
    returns_list : list of ndarray
        List of K aligned return series (same length N).
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray, shape (K, K)
        Bipower covariance matrix.
    """
    k = len(returns_list)
    mat = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        mat[i, i] = _bpv_core(returns_list[i])
    for i in range(k):
        for j in range(i + 1, k):
            c = _bpcov_bi_core(returns_list[i], returns_list[j])
            mat[i, j] = c
            mat[j, i] = c
    if make_psd_flag:
        mat = make_psd(mat)
    return mat
