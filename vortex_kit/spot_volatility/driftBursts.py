"""
vortex_kit.spot_volatility.driftBursts
=======================================
Drift burst detection test.

Detects periods where the drift (mean return) exhibits a sharp
increase or decrease, which may indicate informed trading or
other market microstructure effects.

Reference
---------
Christensen, K., Oomen, R. and Reno, R. (2018).
    "The drift burst hypothesis." Journal of Econometrics, 208, 25-48.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy import stats


@njit(cache=True)
def _drift_burst_stat(returns: np.ndarray,
                      vol_window: int,
                      drift_window: int) -> np.ndarray:
    """Compute drift burst test statistic."""
    n = len(returns)
    t_stats = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        # Local volatility estimate
        vol_start = max(0, i - vol_window // 2)
        vol_end = min(n, i + vol_window // 2 + 1)

        vol_sum = 0.0
        for j in range(vol_start, vol_end):
            vol_sum += returns[j] * returns[j]
        vol = math.sqrt(vol_sum / (vol_end - vol_start))

        # Local drift estimate
        drift_start = max(0, i - drift_window // 2)
        drift_end = min(n, i + drift_window // 2 + 1)

        drift_sum = 0.0
        for j in range(drift_start, drift_end):
            drift_sum += returns[j]
        drift = drift_sum / (drift_end - drift_start)

        # t-statistic
        if vol > 1e-10:
            t_stats[i] = drift / (vol / math.sqrt(drift_end - drift_start))

    return t_stats


def driftBursts(returns: np.ndarray,
                *,
                vol_window: int = 20,
                drift_window: int = 5,
                alpha: float = 0.99) -> dict:
    """Drift burst detection test.

    Parameters
    ----------
    returns : ndarray
        Log-returns.
    vol_window : int
        Window for local volatility estimation.
    drift_window : int
        Window for local drift estimation.
    alpha : float
        Confidence level for critical value.

    Returns
    -------
    dict
        Test statistics, critical values, burst indicators, and timing.

    Reference
    ---------
    Christensen, K., Oomen, R. and Reno, R. (2018).
        "The drift burst hypothesis." Journal of Econometrics, 208, 25-48.
    """
    r = np.asarray(returns, dtype=np.float64)

    # Test statistics
    t_stats = _drift_burst_stat(r, vol_window, drift_window)

    # Critical value
    critical_value = stats.norm.ppf(alpha)

    # Burst indicators
    burst_indicators = ~np.isnan(t_stats) & (np.abs(t_stats) > critical_value)
    n_bursts = np.sum(burst_indicators)

    return {
        "t_statistics": t_stats,
        "critical_value": critical_value,
        "burst_indicators": burst_indicators,
        "n_bursts": n_bursts,
        "max_t_stat": np.nanmax(np.abs(t_stats)),
    }
