"""
vortex_kit.covariance.rSkew
============================
Realized skewness.

Higher moment estimator based on high-frequency returns.

Reference
---------
Amaya, D. et al. (2015).
    "Does realized skewness and kurtosis predict the cross-section of
    equity returns?" Journal of Financial Economics, 118, 135-167.
"""

from __future__ import annotations

import numpy as np

from vortex_kit.utils.returns import log_returns


def rSkew(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized skewness.

    RS = (1/N) × Σᵢ rᵢ³ / (RV)^(3/2)

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized skewness.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    n = len(r)

    rv = np.sum(r * r)
    cubed_sum = np.sum(r ** 3)

    if rv < 1e-15:
        return 0.0

    return (cubed_sum / n) / (rv / n) ** 1.5
