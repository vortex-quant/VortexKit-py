"""
vortex_kit.realized_measures.rMinRVar
=======================================
Minimum realized variance — jump-robust variance estimator.

MinRV = π/(π-2) × M/(M-1) × Σᵢ min(|rᵢ|, |rᵢ₊₁|)²

Reference
---------
Andersen, T.G., Dobrev, D. and Schaumburg, E. (2012).
    "Jump-robust volatility estimation using nearest neighbor truncation."
    Journal of Econometrics, 169, 75-93.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _minrv_core(returns: np.ndarray) -> float:
    """MinRV = π/(π-2) × M/(M-1) × Σᵢ min(|rᵢ|, |rᵢ₊₁|)²."""
    n = len(returns)
    if n < 2:
        return 0.0
    scale = math.pi / (math.pi - 2.0)
    finite_corr = n / (n - 1.0)
    s = 0.0
    for i in range(n - 1):
        a = abs(returns[i])
        b = abs(returns[i + 1])
        m = min(a, b)
        s += m * m
    return scale * finite_corr * s


def rMinRVar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Minimum realized variance.

    MinRV = π/(π-2) × M/(M-1) × Σᵢ min(|rᵢ|, |rᵢ₊₁|)²

    Reference: Andersen, Dobrev, Schaumburg (2012).

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Minimum realized variance.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _minrv_core(r)
