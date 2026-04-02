"""
vortex_kit.covariance.rKurt
============================
Realized kurtosis.

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


def rKurt(data: np.ndarray, *, make_returns: bool = False) -> float:
    """Realized kurtosis.

    RK = (1/N) × Σᵢ rᵢ⁴ / (RV)²

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized kurtosis.
    """
    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    n = len(r)

    rv = np.sum(r * r)
    fourth_sum = np.sum(r ** 4)

    if rv < 1e-15:
        return 0.0

    return (fourth_sum / n) / (rv / n) ** 2
