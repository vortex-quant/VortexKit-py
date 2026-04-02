"""
vortex_kit.realized_measures.rKernelCov
=========================================
Realized kernel covariance (noise-robust, multivariate).

Noise-robust covariance estimator using weighted autocovariances
(Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).
"""

from __future__ import annotations

import numpy as np

from vortex_kit.utils.returns import log_returns
from vortex_kit.utils.matrices import make_psd
from vortex_kit.kernels.core import kernel_name_to_int
from .rKernelVar import _kernel_estimator_core


def rKernelCov(data_list: list[np.ndarray] | np.ndarray,
               *,
               kernel_type: str = "parzen",
               kernel_param: int = 1,
               dof_adj: bool = True,
               make_returns: bool = False,
               make_psd_flag: bool = True) -> np.ndarray | float:
    """Realized kernel covariance.

    Noise-robust covariance estimator using weighted autocovariances
    (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008).

    Parameters
    ----------
    data_list : list of ndarray or ndarray
        If a list, each element is a return (or price) series for one asset.
        If a 2-D array, shape (N, K).
    kernel_type : str
        Kernel name.
    kernel_param : int
        Bandwidth H.
    dof_adj : bool
        Finite-sample DOF adjustment.
    make_returns : bool
        If True, compute log-returns from prices.
    make_psd_flag : bool
        If True (default), project to nearest PSD matrix.

    Returns
    -------
    ndarray or float
    """
    from .rKernelVar import rKernelVar  # Lazy import

    # Normalise input
    if isinstance(data_list, np.ndarray) and data_list.ndim == 2:
        if make_returns:
            cols = [np.diff(np.log(data_list[:, j])) for j in range(data_list.shape[1])]
        else:
            cols = [data_list[:, j] for j in range(data_list.shape[1])]
    elif isinstance(data_list, list):
        if make_returns:
            cols = [log_returns(np.asarray(s, dtype=np.float64)) for s in data_list]
        else:
            cols = [np.asarray(s, dtype=np.float64) for s in data_list]
    else:
        # Scalar / 1-D
        return rKernelVar(data_list, kernel_type=kernel_type,
                          kernel_param=kernel_param, dof_adj=dof_adj,
                          make_returns=make_returns)

    k = len(cols)
    if k == 1:
        return rKernelVar(cols[0], kernel_type=kernel_type,
                          kernel_param=kernel_param, dof_adj=dof_adj)

    kt = kernel_name_to_int(kernel_type)
    mat = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            val = _kernel_estimator_core(cols[i], cols[j],
                                         kernel_param, dof_adj, kt)
            mat[i, j] = val
            mat[j, i] = val

    if make_psd_flag:
        mat = make_psd(mat)
    return mat
