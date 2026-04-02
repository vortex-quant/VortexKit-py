"""
vortex_kit.realized_measures.rKernelVar
=========================================
Realized kernel variance (noise-robust, univariate).

Noise-robust covariance estimator using weighted autocovariances
(Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns
from vortex_kit.kernels.core import kernel_weight, kernel_name_to_int


@njit(cache=True)
def _kernel_estimator_core(a: np.ndarray, b: np.ndarray,
                           lags: int, dof_adj: bool,
                           kernel_type: int) -> float:
    """Core realized kernel estimator — direct port of C++ kernelEstimator.

    Parameters
    ----------
    a, b : ndarray, shape (N,)
        Return series (already computed).
    lags : int
        Bandwidth (number of lags H).
    dof_adj : bool
        If True, apply finite-sample degrees-of-freedom adjustment.
    kernel_type : int
        Integer kernel selector.

    Returns
    -------
    float
    """
    nab = len(a)  # number of returns

    ans = 0.0
    for lag in range(lags + 1):
        # Forward autocovariance
        ab_fwd = 0.0
        for i in range(nab - lag):
            ab_fwd += a[i] * b[i + lag]

        # Backward autocovariance
        ab_bwd = 0.0
        for i in range(lag, nab):
            ab_bwd += a[i] * b[i - lag]

        # Kernel weight
        if lag == 0:
            w = 1.0
        else:
            w = kernel_weight((lag - 1.0) / lags, kernel_type)

        # DOF adjustment
        if dof_adj:
            theadj = (nab + 1.0) / (nab + 1.0 - lag)
        else:
            theadj = 1.0

        if lag == 0:
            ans += w * theadj * ab_fwd
        else:
            ans += w * (theadj * ab_fwd + theadj * ab_bwd)

    return ans


def rKernelVar(data: np.ndarray,
               *,
               kernel_type: str = "parzen",
               kernel_param: int = 1,
               dof_adj: bool = True,
               make_returns: bool = False) -> float:
    """Realized kernel variance (univariate).

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    kernel_type : str
        Kernel name (default ``"parzen"``).
    kernel_param : int
        Bandwidth H (number of autocovariance lags).
    dof_adj : bool
        Finite-sample DOF adjustment.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized kernel variance.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    kt = kernel_name_to_int(kernel_type)
    return _kernel_estimator_core(r, r, kernel_param, dof_adj, kt)
