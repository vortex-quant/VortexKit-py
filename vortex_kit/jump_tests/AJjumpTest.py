"""
vortex_kit.jump_tests.AJjumpTest
=================================
Aït-Sahalia & Jacod (2009) power-variation ratio jump test.

Tests for jumps using the ratio of high-frequency power variations
with different powers.

Reference
---------
Aït-Sahalia, Y. and Jacod, J. (2009).
    "Testing for jumps in a discretely observed process."
    Annals of Statistics, 37, 184-222.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy import stats


@njit(cache=True)
def _mu_p(p: float) -> float:
    """μₚ = E[|Z|ᵖ] for Z ~ N(0,1)."""
    return (2.0 ** (p / 2.0)) * math.gamma((p + 1.0) / 2.0) / math.sqrt(math.pi)


def AJjumpTest(data: np.ndarray,
               *,
               p: float = 4.0,
               alpha: float = 0.975,
               make_returns: bool = False) -> dict:
    """Aït-Sahalia & Jacod (2009) power-variation ratio jump test.

    Tests H₀: No jumps using the ratio of p-power variation to
    (p/2)-power variation.

    Parameters
    ----------
    data : ndarray
        Log-returns or prices.
    p : float
        Power for test (default 4.0 for quarticity-based test).
    alpha : float
        Confidence level.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    dict
        Contains ztest, pvalue, critical_value, and test components.
    """
    from vortex_kit.utils.returns import log_returns

    if make_returns:
        r = log_returns(data)
    elif data.ndim == 1 and len(data) > 0 and data[0] > 0:
        r = log_returns(data)
    else:
        r = np.asarray(data, dtype=np.float64)

    n = len(r)

    # Power variations
    abs_r = np.abs(r)
    PV_p = np.sum(abs_r ** p)
    PV_p2 = np.sum(abs_r ** (p / 2.0))

    if PV_p2 <= 1e-15:
        return {
            "ztest": 0.0,
            "pvalue": 1.0,
            "critical_value": stats.norm.ppf(alpha),
            "PV_p": PV_p,
            "PV_p2": PV_p2,
        }

    # Ratio statistic
    ratio = PV_p / (PV_p2 ** 2)

    # Asymptotic variance under null
    mu_p = _mu_p(p)
    mu_p2 = _mu_p(p / 2.0)
    sigma2 = (mu_p ** 2) / (mu_p2 ** 4) - 1.0

    # Test statistic
    z = (ratio - (mu_p / (mu_p2 ** 2))) / math.sqrt(sigma2 / n)

    pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    critical_value = stats.norm.ppf(alpha)

    return {
        "ztest": z,
        "pvalue": pvalue,
        "critical_value": critical_value,
        "PV_p": PV_p,
        "PV_p2": PV_p2,
        "ratio": ratio,
    }
