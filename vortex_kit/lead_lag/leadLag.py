"""
vortex_kit.lead_lag.leadLag
============================
Lead-lag estimation via Hayashi-Yoshida shifted contrast function.

Estimates the lead-lag relationship between two assets by maximizing
the Hayashi-Yoshida covariance over different time shifts.

Reference
---------
Hayashi, T. and Yoshida, N. (2005).
    "On covariance estimation of non-synchronously observed diffusion
    processes." Bernoulli, 11, 359-379.

Hoffmann, M., Rosenbaum, M. and Yoshida, N. (2013).
    "Estimation of the lead-lag parameter from non-synchronous data."
    Bernoulli, 19, 596-620.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _hy_shifted_cov(ts1: np.ndarray, r1: np.ndarray,
                    ts2: np.ndarray, r2: np.ndarray,
                    shift: int) -> float:
    """Hayashi-Yoshida covariance with time shift.

    Positive shift means series 1 leads series 2.
    """
    n1 = len(r1)
    n2 = len(r2)
    total = 0.0

    for i in range(n1):
        a_lo = ts1[i]
        a_hi = ts1[i + 1]
        for j in range(n2):
            # Apply shift to series 2
            b_lo = ts2[j] + shift
            b_hi = ts2[j + 1] + shift

            if a_lo < b_hi and b_lo < a_hi:
                total += r1[i] * r2[j]

    return total


def leadLag(prices1: np.ndarray,
            timestamps1: np.ndarray,
            prices2: np.ndarray,
            timestamps2: np.ndarray,
            *,
            max_lag: int = 10) -> dict:
    """Lead-lag estimation via Hayashi-Yoshida contrast function.

    Estimates the time shift that maximizes the covariance between
    two asynchronous series.

    Parameters
    ----------
    prices1, prices2 : ndarray
        Tick-level prices for both assets.
    timestamps1, timestamps2 : ndarray
        Timestamps for each asset.
    max_lag : int
        Maximum lag to test (in timestamp units).

    Returns
    -------
    dict
        Estimated lag, maximum covariance, and all tested lags.

    Reference
    ---------
    Hoffmann, M., Rosenbaum, M. and Yoshida, N. (2013).
        "Estimation of the lead-lag parameter from non-synchronous data."
        Bernoulli, 19, 596-620.
    """
    from vortex_kit.utils.returns import log_returns

    # Compute returns
    r1 = log_returns(np.asarray(prices1, dtype=np.float64))
    r2 = log_returns(np.asarray(prices2, dtype=np.float64))

    ts1 = np.asarray(timestamps1, dtype=np.float64)
    ts2 = np.asarray(timestamps2, dtype=np.float64)

    # Test different lags
    lags = np.arange(-max_lag, max_lag + 1)
    covs = np.zeros(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        covs[i] = _hy_shifted_cov(ts1, r1, ts2, r2, lag)

    # Find best lag
    best_idx = np.argmax(np.abs(covs))
    best_lag = lags[best_idx]
    best_cov = covs[best_idx]

    return {
        "lag": best_lag,
        "covariance": best_cov,
        "lags_tested": lags,
        "covariances": covs,
        "interpretation": f"Series 1 leads series 2 by {best_lag}" if best_lag > 0 else
                         f"Series 2 leads series 1 by {abs(best_lag)}" if best_lag < 0 else
                         "No lead-lag relationship",
    }
