"""
vortex_kit.realized_measures.rQuar
====================================
Realized quarticity.

rQuar = (N/3) × Σᵢ rᵢ⁴

Used in asymptotic variance formulas for realized variance.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _rquar_core(returns: np.ndarray) -> float:
    """rQuar = (N/3) × Σᵢ rᵢ⁴, where N = n_returns + 1 (R convention)."""
    n = len(returns)
    N = n + 1  # R uses N = nrow(q) + 1, i.e. number of price observations
    s = 0.0
    for i in range(n):
        r2 = returns[i] * returns[i]
        s += r2 * r2
    return (N / 3.0) * s


def rQuar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized quarticity.

    rQuar = (N/3) × Σᵢ rᵢ⁴

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized quarticity.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _rquar_core(r)
