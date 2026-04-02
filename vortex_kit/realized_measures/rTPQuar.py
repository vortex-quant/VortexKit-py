"""
vortex_kit.realized_measures.rTPQuar
======================================
Realized tri-power quarticity — jump-robust quarticity estimator.

rTPQuar = N × N/(N-2) × μ₄/₃⁻³ × Σᵢ |rᵢ|⁴/³ |rᵢ₋₁|⁴/³ |rᵢ₋₂|⁴/³

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
def _tpquar_core(returns: np.ndarray) -> float:
    """Tri-power quarticity (jump-robust, Barndorff-Nielsen & Shephard).

    R convention: N = nrow(q) + 2 for rTPQuar.
    Formula: N × (N/(N-2)) × μ₄/₃⁻³ × sum.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    # μ₄/₃ = 2^(2/3) × Gamma(7/6) / sqrt(pi)
    mu43 = 0.8308609
    N = n  # R: nrow(q_after_rolling) + 2 = (n-2) + 2 = n
    scale = N * (N / (N - 2.0)) * (mu43 ** (-3.0))
    s = 0.0
    for i in range(2, n):
        s += (abs(returns[i]) ** (4.0 / 3.0)
              * abs(returns[i - 1]) ** (4.0 / 3.0)
              * abs(returns[i - 2]) ** (4.0 / 3.0))
    return scale * s


def rTPQuar(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized tri-power quarticity — jump-robust quarticity estimator.

    rTPQuar = N × N/(N-2) × μ₄/₃⁻³ ×
              Σᵢ |rᵢ|⁴/³ |rᵢ₋₁|⁴/³ |rᵢ₋₂|⁴/³

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized tri-power quarticity.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    return _tpquar_core(r)
