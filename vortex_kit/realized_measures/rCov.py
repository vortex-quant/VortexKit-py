"""
vortex_kit.realized_measures.rCov
====================================
Realized covariance (outer product of returns).

For univariate input, returns scalar (= rRVar).
For multivariate input (N × K), returns K × K matrix.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _rv_core(returns: np.ndarray) -> float:
    """Core: sum of squared returns."""
    s = 0.0
    for i in range(len(returns)):
        s += returns[i] * returns[i]
    return s


def rCov(data: np.ndarray, *, make_returns: bool = False) -> np.ndarray | float:
    """Realized covariance (outer product of returns).

    For univariate input, returns scalar (= rRVar).
    For multivariate input (N × K), returns K × K matrix.

    Parameters
    ----------
    data : ndarray, shape (N,) or (N, K)
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices along axis 0.

    Returns
    -------
    ndarray or float
        Realized covariance.
    """
    d = np.asarray(data, dtype=np.float64)
    if d.ndim == 1:
        r = log_returns(d) if make_returns else d
        return _rv_core(r)
    else:
        if make_returns:
            r = np.diff(np.log(d), axis=0)
        else:
            r = d
        return r.T @ r
