"""
vortex_kit.utils.stats
=======================
Statistical utility functions.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def mu_p(p: float) -> float:
    """μₚ = E[|Z|ᵖ] for Z ~ N(0,1) = 2^(p/2) × Γ((p+1)/2) / √π.

    Parameters
    ----------
    p : float
        Power parameter.

    Returns
    -------
    float
        The p-th absolute moment of standard normal.
    """
    return (2.0 ** (p / 2.0)) * math.gamma((p + 1.0) / 2.0) / math.sqrt(math.pi)


@njit(cache=True)
def gfunction(x: np.ndarray) -> np.ndarray:
    """Elementwise  g(x) = min(x, 1 - x).

    Used as the pre-averaging weight function (Jacod et al., 2009).

    Parameters
    ----------
    x : ndarray
        Values typically in [0, 1].

    Returns
    -------
    ndarray
        Same shape as *x*.
    """
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = min(x[i], 1.0 - x[i])
    return out
