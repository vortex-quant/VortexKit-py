"""
vortex_kit.covariance.rAVGCov
================================
Subsampled/averaged realized covariance (rAVGCov).

Averages realized covariance over multiple sub-grids to reduce
discretization error.

Reference
---------
Zhang, L. (2011). "Estimating covariation: Epps effect, microstructure noise."
    Journal of Econometrics, 160, 33-47.

Liu, L. Y., Patton, A. J. and Sheppard, K. (2015).
    "Does anything beat 5-minute RV?" Journal of Econometrics, 187, 293-311.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.matrices import make_psd


@njit(cache=True)
def _avg_cov_subsample(returns: np.ndarray, delta: int, offset: int) -> np.ndarray:
    """Compute covariance on a sub-grid with given offset."""
    n = len(returns)
    k = returns.shape[1]

    # Collect returns at offset, offset+delta, offset+2*delta, ...
    indices = np.arange(offset, n, delta)
    m = len(indices)

    if m < 2:
        return np.zeros((k, k), dtype=np.float64)

    # Subsampled returns
    sub_returns = np.zeros((m - 1, k), dtype=np.float64)
    for i in range(m - 1):
        for j in range(k):
            sub_returns[i, j] = returns[indices[i + 1], j] - returns[indices[i], j]

    # Covariance
    cov = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            s = 0.0
            for t in range(m - 1):
                s += sub_returns[t, i] * sub_returns[t, j]
            cov[i, j] = s

    return cov


def rAVGCov(prices_or_returns: np.ndarray,
            *,
            delta: int = 1,
            is_returns: bool = False,
            make_psd_flag: bool = False) -> np.ndarray:
    """Subsampled/averaged realized covariance.

    Parameters
    ----------
    prices_or_returns : ndarray, shape (N, K)
        Price or return matrix (N observations × K assets).
    delta : int
        Subsampling frequency (default 1 for all grids).
    is_returns : bool
        If True, input is returns; if False, input is prices.
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray, shape (K, K)
        Averaged realized covariance matrix.
    """
    data = np.asarray(prices_or_returns, dtype=np.float64)
    n, k = data.shape

    if not is_returns:
        # Convert prices to log-prices, then compute returns
        data = np.diff(np.log(data), axis=0)
        n = len(data)

    if delta == 1:
        # Standard realized covariance
        cov = data.T @ data
    else:
        # Average over all sub-grids
        total_cov = np.zeros((k, k), dtype=np.float64)

        # Primary grid (offset = delta - 1, like R)
        n_bins_0 = (n - (delta - 1)) // delta
        cov_0 = _avg_cov_subsample(data, delta, delta - 1)
        total_cov += cov_0

        # Offset grids 1 to delta-1
        for offset in range(1, delta):
            n_bins_sub = (n - offset) // delta
            if n_bins_sub > 0:
                cov_sub = _avg_cov_subsample(data, delta, offset)
                # R applies bias correction for sub-grids
                bias_corr = n_bins_0 / n_bins_sub if n_bins_sub > 0 else 1.0
                total_cov += cov_sub * bias_corr

        cov = total_cov / delta

    if make_psd_flag:
        cov = make_psd(cov)
    return cov
