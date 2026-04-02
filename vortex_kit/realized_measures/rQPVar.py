"""
vortex_kit.realized_measures.rQPVar
=====================================
Realized quad-power variation.

rQPVar = N × N/(N-3) × (π/2)² × Σᵢ |rᵢ| |rᵢ₋₁| |rᵢ₋₂| |rᵢ₋₃|
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _qpvar_core(returns: np.ndarray) -> float:
    """Quad-power variation: N × N/(N-3) × (π/2)² × prod of 4 |r|.

    R convention: N = nrow(q) + 3 for rQPVar.
    """
    n = len(returns)
    if n < 4:
        return 0.0
    N = n  # R: nrow(q_after_rolling) + 3 = (n-3) + 3 = n
    scale = N * (N / (N - 3.0)) * (math.pi / 2.0) ** 2
    s = 0.0
    for i in range(3, n):
        s += (abs(returns[i]) * abs(returns[i - 1])
              * abs(returns[i - 2]) * abs(returns[i - 3]))
    return scale * s


def rQPVar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized quad-power variation.

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized quad-power variation.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _qpvar_core(r)
