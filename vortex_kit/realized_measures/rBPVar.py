"""
vortex_kit.realized_measures.rBPVar
=====================================
Realized bipower variation — jump-robust variance estimator.

BPV_t = (π/2) × Σᵢ |rᵢ| × |rᵢ₋₁|

This estimator is robust to occasional price jumps while remaining
consistent for integrated variance.

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2004).
    "Power and bipower variation with stochastic volatility and jumps."
    Journal of Financial Econometrics, 2, 1-37.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _bpv_core(returns: np.ndarray) -> float:
    """Bipower variation: (pi/2) * sum |r_i| * |r_{i-1}|."""
    n = len(returns)
    s = 0.0
    for i in range(1, n):
        s += abs(returns[i]) * abs(returns[i - 1])
    return (math.pi / 2.0) * s


def rBPVar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized bipower variation — jump-robust variance estimator.

    BPV_t = (π/2) × Σᵢ |rᵢ| × |rᵢ₋₁|

    Under no jumps, BPV is a consistent estimator of integrated variance.
    In the presence of jumps, BPV still consistently estimates IV.

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized bipower variation.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _bpv_core(r)
