"""
vortex_kit.jump_tests.BNSjumpTest
====================================
Barndorff-Nielsen & Shephard (2006) jump test.

Statistical test for detecting the presence of price jumps in
high-frequency data.

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2006).
    "Econometrics of testing for jumps in financial economics using
    bipower variation." Journal of Financial Econometrics, 4, 1-30.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats

from vortex_kit.realized_measures import rRVar, rBPVar, rMedRVar, rMinRVar
from vortex_kit.realized_measures import rQuar, rTPQuar, rMedRQuar, rMinRQuar


_THETA_MAP = {
    "RV": 2.0,
    "BV": math.pi ** 2 / 4.0 + math.pi - 3.0,
    "rMedRVar": 2.96,
    "rMinRVar": 3.81,
}


def _get_theta(iv_estimator: str) -> float:
    """Return θ for the given IV estimator (see BNS 2004, Table 1)."""
    if iv_estimator not in _THETA_MAP:
        raise ValueError(
            f"Unknown IVestimator '{iv_estimator}'. "
            f"Available: {list(_THETA_MAP)}"
        )
    return _THETA_MAP[iv_estimator]


def _hat_iv(returns: np.ndarray, estimator: str) -> float:
    """Dispatch to the appropriate integrated-variance estimator."""
    if estimator == "RV":
        return rRVar(returns)
    elif estimator == "BV":
        return rBPVar(returns)
    elif estimator == "rMedRVar":
        return rMedRVar(returns)
    elif estimator == "rMinRVar":
        return rMinRVar(returns)
    else:
        raise ValueError(f"Unknown IV estimator: {estimator}")


def _hat_iq(returns: np.ndarray, estimator: str) -> float:
    """Dispatch to the appropriate integrated-quarticity estimator."""
    if estimator == "rQuar":
        return rQuar(returns)
    elif estimator == "rTPQuar":
        return rTPQuar(returns)
    elif estimator == "rMedRQuar":
        return rMedRQuar(returns)
    elif estimator == "rMinRQuar":
        return rMinRQuar(returns)
    else:
        raise ValueError(f"Unknown IQ estimator: {estimator}")


def BNSjumpTest(data: np.ndarray,
                *,
                IVestimator: str = "BV",
                IQestimator: str = "rTPQuar",
                test_type: str = "linear",
                log_transform: bool = False,
                use_max: bool = False,
                alpha: float = 0.975,
                make_returns: bool = False) -> dict:
    """Barndorff-Nielsen & Shephard (2006) jump test.

    Tests H₀: No jumps vs H₁: At least one jump.

    Parameters
    ----------
    data : ndarray
        Log-returns or prices.
    IVestimator : str
        Integrated variance estimator ("RV", "BV", "rMedRVar", "rMinRVar").
    IQestimator : str
        Integrated quarticity estimator ("rQuar", "rTPQuar", "rMedRQuar", "rMinRQuar").
    test_type : str
        Test formulation: "linear" or "ratio".
    log_transform : bool
        If True, use log-transformed test.
    use_max : bool
        If True, use maximum over sub-intervals.
    alpha : float
        Confidence level for critical value.
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    dict
        Contains ztest, pvalue, critical_value, IV, IQ, and test statistic components.
    """
    from vortex_kit.utils.returns import log_returns

    if make_returns:
        r = log_returns(data)
    elif data.ndim == 1 and len(data) > 0 and data[0] > 0:
        # Heuristic: if values look like prices (positive), convert
        r = log_returns(data)
    else:
        r = np.asarray(data, dtype=np.float64)

    # Compute IV and IQ
    iv = _hat_iv(r, IVestimator)
    iq = _hat_iq(r, IQestimator)

    # Test statistic
    if test_type == "linear":
        if log_transform:
            # Log-transformed linear test
            if iv <= 0 or iq <= 0:
                z = 0.0
            else:
                theta = _get_theta(IVestimator)
                var = theta * iq
                z = (math.log(iv) - math.log(iv)) / math.sqrt(var / (len(r) * iv * iv))
        else:
            # Standard linear test
            theta = _get_theta(IVestimator)
            var = theta * iq
            if var > 0 and iv > 0:
                z = (rRVar(r) - iv) / math.sqrt(var / len(r))
            else:
                z = 0.0
    else:
        # Ratio test
        rv = rRVar(r)
        if iv > 0 and rv > 0:
            theta = _get_theta(IVestimator)
            z = (rv / iv - 1.0) / math.sqrt(theta * iq / (len(r) * iv * iv))
        else:
            z = 0.0

    # Two-sided test
    pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    critical_value = stats.norm.ppf(alpha)

    return {
        "ztest": z,
        "pvalue": pvalue,
        "critical_value": critical_value,
        "IV": iv,
        "IQ": iq,
        "test_type": test_type,
        "IVestimator": IVestimator,
        "IQestimator": IQestimator,
    }
