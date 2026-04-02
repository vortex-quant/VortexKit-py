"""
vortex_kit.har_model.fit
=========================
HAR model fitting function.

Provides the main interface for estimating HAR family models.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .core import _har_agg, _abd_jump_test


def _ols_fit(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with Newey-West HAC standard errors.

    Returns coefficient estimates, standard errors, t-statistics,
    and p-values.
    """
    # Remove rows with NaN
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y_clean = y[mask]
    X_clean = X[mask]

    n = len(y_clean)
    k = X_clean.shape[1]

    if n < k + 1:
        raise ValueError(f"Insufficient observations: {n} < {k + 1}")

    # OLS estimates
    XtX = X_clean.T @ X_clean
    XtY = X_clean.T @ y_clean

    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]

    # Residuals
    y_hat = X_clean @ beta
    resid = y_clean - y_hat

    # Simple standard errors (could be upgraded to HAC)
    sigma2 = np.sum(resid ** 2) / (n - k)
    var_beta = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))

    # t-statistics and p-values
    t_stats = beta / (se + 1e-15)
    pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # R-squared
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "coefficients": beta,
        "standard_errors": se,
        "t_stats": t_stats,
        "pvalues": pvalues,
        "residuals": resid,
        "fitted_values": y_hat,
        "r2": r2,
        "nobs": n,
        "nparam": k,
    }


def fit_har(rv: np.ndarray,
            model_type: str = "HAR",
            bpv: np.ndarray | None = None,
            tq: np.ndarray | None = None) -> dict:
    """Fit Heterogeneous AutoRegressive (HAR) model.

    Fits the HAR family of models for realized volatility forecasting.

    Supported model types:
    - "HAR": Basic HAR-RV (Corsi 2009)
    - "HARJ": HAR with separate jump component
    - "HARCJ": HAR with continuous/jump decomposition
    - "HARQ": HAR with realized quarticity (Bollerslev et al. 2016)
    - "HARQJ": HAR-Q with jump component
    - "CHAR": Continuous HAR (uses BPV)
    - "CHARQ": Continuous HAR-Q

    Parameters
    ----------
    rv : ndarray
        Daily realized variance series.
    model_type : str
        Type of HAR model (see above).
    bpv : ndarray or None
        Daily bipower variation (required for jump models).
    tq : ndarray or None
        Daily tri-power quarticity (required for Q models).

    Returns
    -------
    dict
        Model fit results including coefficients, standard errors,
        fitted values, and residuals.

    Reference
    ---------
    Corsi, F. (2009). "A simple approximate long memory model of
        realized volatility." Journal of Financial Econometrics, 7, 174-196.

    Bollerslev, T., Patton, A., and Quaedvlieg, R. (2016).
        "Exploiting the errors: A simple approach for improved volatility
        forecasting." Journal of Econometrics, 192, 1-18.
    """
    rv = np.asarray(rv, dtype=np.float64)
    n = len(rv)

    # Standard HAR periods: daily, weekly, monthly (22 days)
    periods = [1, 5, 22]

    # Build HAR components
    har_components = _har_agg(rv, periods)

    if model_type == "HAR":
        # Basic HAR: RV ~ RV(daily) + RV(weekly) + RV(monthly)
        y = rv[22:]  # First 22 observations lost to lagging
        X = np.column_stack([
            np.ones(len(y)),  # Constant
            har_components[22:, 0],  # Daily
            har_components[22:, 1],  # Weekly
            har_components[22:, 2],  # Monthly
        ])
        coef_names = ["const", "daily", "weekly", "monthly"]

    elif model_type in ["HARJ", "HARCJ"]:
        if bpv is None:
            raise ValueError(f"model_type={model_type} requires bpv argument")
        bpv = np.asarray(bpv, dtype=np.float64)

        # Jump component
        jump = np.maximum(rv - bpv, 0.0)

        # ABD jump test for weighting
        if tq is not None:
            tq_arr = np.asarray(tq, dtype=np.float64)
            z_abd = _abd_jump_test(rv, bpv, tq_arr)
        else:
            z_abd = np.ones_like(rv)

        y = rv[22:]
        X = np.column_stack([
            np.ones(len(y)),
            har_components[22:, 0],
            har_components[22:, 1],
            har_components[22:, 2],
            jump[22:],
        ])
        coef_names = ["const", "daily", "weekly", "monthly", "jump"]

    elif model_type in ["HARQ", "HARQJ"]:
        # HAR with realized quarticity
        if tq is None:
            raise ValueError(f"model_type={model_type} requires tq argument")
        tq = np.asarray(tq, dtype=np.float64)

        # sqrt(RQ) component
        rq_sqrt = np.sqrt(tq)

        y = rv[22:]
        X = np.column_stack([
            np.ones(len(y)),
            har_components[22:, 0],
            har_components[22:, 1],
            har_components[22:, 2],
            har_components[22:, 0] * rq_sqrt[22:],  # HAR-Q interaction
        ])
        coef_names = ["const", "daily", "weekly", "monthly", "rq_interaction"]

    elif model_type in ["CHAR", "CHARQ"]:
        # Continuous HAR: use BPV instead of RV
        if bpv is None:
            raise ValueError(f"model_type={model_type} requires bpv argument")
        bpv = np.asarray(bpv, dtype=np.float64)

        char_components = _har_agg(bpv, periods)

        if model_type == "CHAR":
            y = rv[22:]
            X = np.column_stack([
                np.ones(len(y)),
                char_components[22:, 0],
                char_components[22:, 1],
                char_components[22:, 2],
            ])
            coef_names = ["const", "daily", "weekly", "monthly"]
        else:  # CHARQ
            if tq is None:
                raise ValueError("CHARQ requires tq argument")
            tq = np.asarray(tq, dtype=np.float64)
            rq_sqrt = np.sqrt(tq)

            y = rv[22:]
            X = np.column_stack([
                np.ones(len(y)),
                char_components[22:, 0],
                char_components[22:, 1],
                char_components[22:, 2],
                char_components[22:, 0] * rq_sqrt[22:],
            ])
            coef_names = ["const", "daily", "weekly", "monthly", "rq_interaction"]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Fit model
    fit_results = _ols_fit(y, X)
    fit_results["model_type"] = model_type
    fit_results["coef_names"] = coef_names

    return fit_results
