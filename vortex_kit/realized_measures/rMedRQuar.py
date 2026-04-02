"""
vortex_kit.realized_measures.rMedRQuar
========================================
Median realized quarticity — jump-robust.

MedRQuar = 3πN / (9π + 72 - 52√3) × N/(N-2)
           × Σᵢ median(|rᵢ₋₁|, |rᵢ|, |rᵢ₊₁|)⁴
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _medrquar_core(returns: np.ndarray) -> float:
    """Median realized quarticity.

    R convention: N = nrow(q) + 2 for rMedRQuar.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    N = n  # R: nrow(q_after_rolling) + 2 = (n-2) + 2 = n
    scale = 3.0 * math.pi * N / (9.0 * math.pi + 72.0 - 52.0 * math.sqrt(3.0))
    finite_corr = N / (N - 2.0)
    s = 0.0
    for i in range(1, n - 1):
        a = abs(returns[i - 1])
        b = abs(returns[i])
        c = abs(returns[i + 1])
        if a <= b:
            if b <= c:
                med = b
            elif a <= c:
                med = c
            else:
                med = a
        else:
            if a <= c:
                med = a
            elif b <= c:
                med = c
            else:
                med = b
        s += med ** 4
    return scale * finite_corr * s


def rMedRQuar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Median realized quarticity — jump-robust.

    MedRQuar = 3πN / (9π + 72 - 52√3) × N/(N-2)
               × Σᵢ median(|rᵢ₋₁|, |rᵢ|, |rᵢ₊₁|)⁴

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Median realized quarticity.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _medrquar_core(r)
