"""
vortex_kit.covariance.rOWCov
=============================
Outlyingness-weighted realized covariance.

Uses weights based on Mahalanobis distance to down-weight outlying
observations, providing robustness to jumps.

Reference
---------\nBoudt, K. and Zhang, J. (2010).
    "Jump robust two time scale covariance estimation and realized
    volatility budgets." Quantitative Finance, 15, 1041-1054.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.matrices import make_psd


@njit(cache=True)
def _compute_weights(returns: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Compute outlyingness weights based on Mahalanobis distance."""
    n, k = returns.shape
    weights = np.ones(n, dtype=np.float64)

    for t in range(n):
        # Robust Mahalanobis-like distance
        dist = 0.0
        for i in range(k):
            d = abs(returns[t, i] - med[i]) / (mad[i] + 1e-10)
            dist += d * d

        # Weight: down-weight extreme observations
        if dist > 9.0:  # 3-squared threshold
            weights[t] = 9.0 / dist

    return weights


def rOWCov(returns_list: list[np.ndarray],
           *,
           make_psd_flag: bool = False) -> np.ndarray:
    """Outlyingness-weighted realized covariance.

    Uses weights based on Mahalanobis distance to down-weight outlying
    observations, providing robustness to jumps.

    Parameters
    ----------
    returns_list : list of ndarray
        K aligned return series (same length N).
    make_psd_flag : bool
        If True, project result to nearest PSD matrix.

    Returns
    -------
    ndarray, shape (K, K)
        Outlyingness-weighted covariance matrix.
    """
    k = len(returns_list)
    n = len(returns_list[0])

    # Stack returns into matrix
    R = np.column_stack([np.asarray(r, dtype=np.float64) for r in returns_list])

    # Compute robust center (median) and scale (MAD)
    med = np.median(R, axis=0)
    mad = np.median(np.abs(R - med), axis=0) * 1.4826  # Convert to std-like scale

    # Compute weights
    weights = _compute_weights(R, med, mad)

    # Weighted covariance
    W_sqrt = np.sqrt(weights)
    R_weighted = R * W_sqrt[:, np.newaxis]

    cov = (R_weighted.T @ R_weighted) / np.sum(weights) * n

    if make_psd_flag:
        cov = make_psd(cov)
    return cov
