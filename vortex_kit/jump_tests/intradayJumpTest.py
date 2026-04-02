"""
vortex_kit.jump_tests.intradayJumpTest
=========================================
Lee & Mykland (2008) intraday jump detection.

Tests for jumps at each observation using local volatility estimates.
Identifies the timing and magnitude of individual jumps.

Reference
---------
Lee, S.S. and Mykland, P.A. (2008).
    "Jumps in financial markets: A new nonparametric test and jump dynamics."
    Review of Financial Studies, 21, 2535-2563.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy import stats


@njit(cache=True)
def _local_bpv(returns: np.ndarray, window: int, i: int) -> float:
    """Compute bipower variation in a local window around index i."""
    half_win = window // 2
    start = max(1, i - half_win)
    end = min(len(returns), i + half_win + 1)

    s = 0.0
    count = 0
    for j in range(start, end):
        s += abs(returns[j]) * abs(returns[j - 1])
        count += 1

    if count > 0:
        return (math.pi / 2.0) * s * len(returns) / count
    return 0.0


def intradayJumpTest(data: np.ndarray,
                     *,
                     window: int = 78,
                     alpha: float = 0.999,
                     make_returns: bool = False) -> dict:
    """Lee & Mykland (2008) intraday jump detection.

    Tests each return for being a jump using local bipower variation.

    Parameters
    ----------
    data : ndarray
        Log-returns or prices.
    window : int
        Window size for local volatility estimation.
    alpha : float
        Confidence level for critical value.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    dict
        Contains test statistics, critical value, jump indicators, and
        jump sizes for each observation.
    """
    from vortex_kit.utils.returns import log_returns

    if make_returns:
        returns = log_returns(data)
    elif data.ndim == 1 and len(data) > 0 and data[0] > 0:
        returns = log_returns(data)
    else:
        returns = np.asarray(data, dtype=np.float64)

    r = returns
    n = len(r)

    # Constants
    c_n = math.sqrt(2.0 * math.log(n))
    s_n = c_n / (math.sqrt(2.0 * math.log(c_n)))

    # Critical value
    beta_star = stats.norm.ppf(alpha)
    critical_value = c_n / s_n + beta_star / s_n

    # Test statistics and jump indicators
    test_stats = np.zeros(n, dtype=np.float64)
    jump_indicators = np.zeros(n, dtype=bool)
    jump_sizes = np.zeros(n, dtype=np.float64)
    local_vol = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sigma_sq = _local_bpv(r, window, i)
        sigma = math.sqrt(sigma_sq + 1e-15)
        local_vol[i] = sigma

        # Standardized return
        if sigma > 1e-15:
            test_stats[i] = abs(r[i]) / sigma
        else:
            test_stats[i] = 0.0

        # Jump detection
        if test_stats[i] > critical_value and i > 0:
            jump_indicators[i] = True
            jump_sizes[i] = r[i]

    n_jumps = np.sum(jump_indicators)

    return {
        "test_statistics": test_stats,
        "critical_value": critical_value,
        "jump_indicators": jump_indicators,
        "jump_mask": jump_indicators,  # Alias for backward compatibility
        "jump_sizes": jump_sizes,
        "n_jumps": n_jumps,
        "local_volatility": local_vol,
    }
