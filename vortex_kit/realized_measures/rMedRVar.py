"""
vortex_kit.realized_measures.rMedRVar
=======================================
Median realized variance — highly jump-robust variance estimator.

MedRV = π / (6 - 4√3 + π) × M/(M-2) × Σᵢ median(|rᵢ₋₁|, |rᵢ|, |rᵢ₊₁|)²

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
def _medrv_core(returns: np.ndarray) -> float:
    """MedRV = pi / (6 - 4*sqrt(3) + pi) * M/(M-2) * sum med(|r|)^2."""
    n = len(returns)
    if n < 3:
        return 0.0
    scale = math.pi / (6.0 - 4.0 * math.sqrt(3.0) + math.pi)
    finite_corr = n / (n - 2.0)
    s = 0.0
    for i in range(1, n - 1):
        a = abs(returns[i - 1])
        b = abs(returns[i])
        c = abs(returns[i + 1])
        # median of three
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
        s += med * med
    return scale * finite_corr * s


def rMedRVar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Median realized variance — highly jump-robust.

    MedRV = π / (6 - 4√3 + π) × M/(M-2) ×
            Σᵢ median(|rᵢ₋₁|, |rᵢ|, |rᵢ₊₁|)²

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
        Median realized variance.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _medrv_core(r)
