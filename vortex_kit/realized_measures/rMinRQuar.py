"""
vortex_kit.realized_measures.rMinRQuar
========================================
Minimum realized quarticity — jump-robust.

MinRQuar = πN / (3π - 8) × N/(N-1) × Σᵢ min(|rᵢ|, |rᵢ₊₁|)⁴
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _minrquar_core(returns: np.ndarray) -> float:
    """Minimum realized quarticity.

    R convention: N = nrow(q) + 1 for rMinRQuar.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    N = n  # R: nrow(q_after_rolling) + 1 = (n-1) + 1 = n
    scale = math.pi * N / (3.0 * math.pi - 8.0)
    finite_corr = N / (N - 1.0)
    s = 0.0
    for i in range(n - 1):
        m = min(abs(returns[i]), abs(returns[i + 1]))
        s += m ** 4
    return scale * finite_corr * s


def rMinRQuar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Minimum realized quarticity — jump-robust.

    MinRQuar = πN / (3π - 8) × N/(N-1) × Σᵢ min(|rᵢ|, |rᵢ₊₁|)⁴

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Minimum realized quarticity.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _minrquar_core(r)
