"""
vortex_kit.realized_measures.rSVar
====================================
Realized semivariance — decompose RV into downside + upside.

rSVar_down = Σᵢ rᵢ² × I(rᵢ < 0)
rSVar_up   = Σᵢ rᵢ² × I(rᵢ > 0)

Separates volatility into components driven by negative and positive returns.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from vortex_kit.utils.returns import log_returns


@njit(cache=True)
def _svar_core(returns: np.ndarray) -> tuple[float, float]:
    """Core: sum of squared negative and positive returns."""
    neg = 0.0
    pos = 0.0
    for i in range(len(returns)):
        r2 = returns[i] * returns[i]
        if returns[i] < 0.0:
            neg += r2
        elif returns[i] > 0.0:
            pos += r2
    return neg, pos


def rSVar(data: np.ndarray,
          *,
          make_returns: bool = False) -> dict[str, float]:
    """Realized semivariance — decompose RV into downside + upside.

    rSVar_down = Σᵢ rᵢ² × I(rᵢ < 0)
    rSVar_up   = Σᵢ rᵢ² × I(rᵢ > 0)

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    dict
        ``{"downside": float, "upside": float}``
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    neg, pos = _svar_core(r)
    return {"downside": neg, "upside": pos}
