"""
vortex_kit.inference.IVinference
=================================
Standard errors and confidence bands for integrated variance.

Provides asymptotic inference for realized variance estimators,
including standard errors and confidence intervals.

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2002).
    "Econometric analysis of realized volatility and its use in
    estimating stochastic volatility models." Journal of the Royal
    Statistical Society: Series B, 64, 253-280.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats


def IVinference(data: np.ndarray,
                *,
                IVestimator: str = "BV",
                IQestimator: str = "rTPQuar",
                confidence: float = 0.95,
                make_returns: bool = False) -> dict:
    """Standard errors and confidence bands for integrated variance.

    Computes asymptotic standard error and confidence interval for
    realized variance estimates.

    Parameters
    ----------
    data : ndarray
        Log-returns or prices.
    IVestimator : str
        Integrated variance estimator (default "BV").
    IQestimator : str
        Integrated quarticity estimator (default "rTPQuar").
    confidence : float
        Confidence level (default 0.95).
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    dict
        Integrated variance estimate, standard error, and confidence
        interval.

    Reference
    ---------
    Barndorff-Nielsen, O.E. and Shephard, N. (2002).
        "Econometric analysis of realized volatility and its use in
        estimating stochastic volatility models." JRSS: Series B, 64, 253-280.
    """
    from vortex_kit.realized_measures.rRVar import rRVar
    from vortex_kit.realized_measures.rTPQuar import rTPQuar
    from vortex_kit.utils.returns import log_returns

    if make_returns:
        r = log_returns(data)
    elif data.ndim == 1 and len(data) > 0 and data[0] > 0:
        r = log_returns(data)
    else:
        r = np.asarray(data, dtype=np.float64)

    n = len(r)

    # Realized variance
    rv = rRVar(r)

    # Integrated quarticity for standard error
    iq = rTPQuar(r)

    # Asymptotic variance: 2 × IQ
    if iq > 0 and n > 0:
        var_rv = 2.0 * iq / n
        se = math.sqrt(var_rv)
    else:
        se = 0.0

    # Confidence interval
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = max(0.0, rv - z * se)
    ci_upper = rv + z * se

    return {
        "IV": rv,
        "hat_iv": rv,  # Alias for backward compatibility
        "std_error": se,
        "confidence_level": confidence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "integrated_quarticity": iq,
    }
