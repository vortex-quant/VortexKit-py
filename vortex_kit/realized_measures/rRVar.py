"""
vortex_kit.realized_measures.rRVar
====================================
Realized variance — sum of squared log-returns.

This module provides the core realized variance estimator (rRVar),
which is the building block for all realized measures.

Formula
-------
rRVar = Σᵢ rᵢ²

where rᵢ = log(Pᵢ₊₁ / Pᵢ) are log-returns.

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2004).
    "Power and bipower variation with stochastic volatility and jumps."
    Journal of Financial Econometrics, 2, 1-37.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _rv_core(returns: np.ndarray) -> float:
    """Core kernel: sum of squared returns."""
    s = 0.0
    for i in range(len(returns)):
        s += returns[i] * returns[i]
    return s


def rRVar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized variance (sum of squared log-returns).

    Parameters
    ----------
    data : ndarray, shape (N,)
        Log-returns, **or** prices if *make_returns=True*.
    make_returns : bool
        If True, *data* is treated as a price series and log-returns
        are computed internally.

    Returns
    -------
    float
        Realized variance for the period.

    Examples
    --------
    >>> import numpy as np
    >>> from vortex_kit.realized_measures.rRVar import rRVar
    >>> prices = np.array([100.0, 101.0, 100.5, 102.0])
    >>> rv = rRVar(prices, make_returns=True)
    >>> rv > 0
    True
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _rv_core(r)
