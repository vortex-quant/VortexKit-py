"""
vortex_kit.heavy_model.fit
===========================
HEAVY model fitting function.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats

from .core import _calc_rec_var, _heavy_llh, _compute_hessian_opg


def fit_heavy(returns: np.ndarray,
              realized_measure: np.ndarray,
              *,
              variance_starting: np.ndarray | None = None,
              rm_starting: np.ndarray | None = None,
              multi_start: bool = True) -> dict:
    """Fit the two-equation HEAVY model of Shephard & Sheppard (2010).

    The HEAVY model consists of:
    - Variance equation:  h_t = ω + α × RM_{t-1} + β × h_{t-1}
    - RM equation:        μ_t = ω_R + α_R × RM_{t-1} + β_R × μ_{t-1}

    Parameters
    ----------
    returns : ndarray
        Daily returns.
    realized_measure : ndarray
        Daily realized measure (e.g., RV or BPV).
    variance_starting : ndarray or None
        Starting values [omega, alpha, beta] for variance equation.
    rm_starting : ndarray or None
        Starting values [omega_R, alpha_R, beta_R] for RM equation.
    multi_start : bool
        If True, use multi-start optimization for near-unit-root cases.

    Returns
    -------
    dict
        Model fit results including coefficients, standard errors,
        conditional variances, and log-likelihoods.

    Reference
    ---------
    Shephard, N. and Sheppard, K. (2010).
        "Realising the future: Forecasting with high frequency based volatility
        (HEAVY) models." Journal of Applied Econometrics, 25, 197-231.
    """
    ret = np.asarray(returns, dtype=np.float64)
    rm = np.asarray(realized_measure, dtype=np.float64)
    n = len(ret)

    # Square returns for variance equation
    ret_sq = ret ** 2

    # ─── Variance Equation ─────────────────────────────────────────────

    # Bounds: omega > 0, alpha >= 0, beta >= 0
    # R only enforces beta <= 1 (not alpha + beta < 1)
    bounds_var = [(1e-10, 10.0), (0.0, 5.0), (0.0, 0.9999)]

    # Default starting values: AR(1) fit
    if variance_starting is None:
        # AR(1)-based starting values
        mean_ret_sq = np.mean(ret_sq)
        # Simple AR(1) coefficient
        if n > 1:
            cov = np.cov(ret_sq[:-1], ret_sq[1:])[0, 1]
            var = np.var(ret_sq[:-1])
            phi = cov / var if var > 0 else 0.5
            phi = np.clip(phi, 0.0, 0.95)
        else:
            phi = 0.5

        alpha_start = phi * 0.5  # Split between alpha and beta
        beta_start = phi * 0.5
        omega_start = max(mean_ret_sq * (1 - phi), 1e-8)

        variance_starting = np.array([omega_start, alpha_start, beta_start])

    # Optimize variance equation
    if multi_start:
        # Try multiple starting points for robustness
        best_llh = np.inf
        best_result = None

        for _ in range(5):
            # Random perturbation around starting values
            x0 = variance_starting * np.random.uniform(0.5, 1.5, 3)
            x0[0] = max(x0[0], 1e-8)  # Ensure omega > 0
            x0[1] = np.clip(x0[1], 0.0, 5.0)
            x0[2] = np.clip(x0[2], 0.0, 0.9999)

            result = minimize(
                lambda p: _heavy_llh(p, ret_sq),
                x0,
                method="SLSQP",
                bounds=bounds_var,
                options={"ftol": 1e-9, "maxiter": 500}
            )

            if result.fun < best_llh:
                best_llh = result.fun
                best_result = result

        result_var = best_result
    else:
        result_var = minimize(
            lambda p: _heavy_llh(p, ret_sq),
            variance_starting,
            method="SLSQP",
            bounds=bounds_var,
            options={"ftol": 1e-9, "maxiter": 500}
        )

    par_var = result_var.x
    h = _calc_rec_var(par_var, ret_sq)

    # Standard errors
    cov_var = _compute_hessian_opg(par_var, ret_sq)
    se_var = np.sqrt(np.maximum(np.diag(cov_var), 0))

    # t-stats and p-values
    t_var = par_var / (se_var + 1e-15)
    p_var = 2.0 * (1.0 - stats.t.cdf(np.abs(t_var), df=n - 3))

    # ─── RM Equation ─────────────────────────────────────────────────

    bounds_rm = [(1e-10, 10.0), (0.0, 5.0), (0.0, 0.9999)]

    if rm_starting is None:
        # AR(1)-based starting values for RM equation
        mean_rm = np.mean(rm)
        if n > 1:
            cov = np.cov(rm[:-1], rm[1:])[0, 1]
            var = np.var(rm[:-1])
            phi = cov / var if var > 0 else 0.7
            phi = np.clip(phi, 0.0, 0.95)
        else:
            phi = 0.7

        alpha_r_start = phi * 0.5
        beta_r_start = phi * 0.5
        omega_r_start = max(mean_rm * (1 - phi), 1e-8)

        rm_starting = np.array([omega_r_start, alpha_r_start, beta_r_start])

    # Optimize RM equation
    if multi_start:
        best_llh = np.inf
        best_result = None

        for _ in range(5):
            x0 = rm_starting * np.random.uniform(0.5, 1.5, 3)
            x0[0] = max(x0[0], 1e-8)
            x0[1] = np.clip(x0[1], 0.0, 5.0)
            x0[2] = np.clip(x0[2], 0.0, 0.9999)

            result = minimize(
                lambda p: _heavy_llh(p, rm),
                x0,
                method="SLSQP",
                bounds=bounds_rm,
                options={"ftol": 1e-9, "maxiter": 500}
            )

            if result.fun < best_llh:
                best_llh = result.fun
                best_result = result

        result_rm = best_result
    else:
        result_rm = minimize(
            lambda p: _heavy_llh(p, rm),
            rm_starting,
            method="SLSQP",
            bounds=bounds_rm,
            options={"ftol": 1e-9, "maxiter": 500}
        )

    par_rm = result_rm.x
    mu = _calc_rec_var(par_rm, rm)

    # Standard errors
    cov_rm = _compute_hessian_opg(par_rm, rm)
    se_rm = np.sqrt(np.maximum(np.diag(cov_rm), 0))

    t_rm = par_rm / (se_rm + 1e-15)
    p_rm = 2.0 * (1.0 - stats.t.cdf(np.abs(t_rm), df=n - 3))

    return {
        "variance_equation": {
            "parameters": par_var,
            "param_names": ["omega", "alpha", "beta"],
            "std_errors": se_var,
            "t_stats": t_var,
            "pvalues": p_var,
            "conditional_variance": h,
            "log_likelihood": -result_var.fun,
        },
        "rm_equation": {
            "parameters": par_rm,
            "param_names": ["omega_R", "alpha_R", "beta_R"],
            "std_errors": se_rm,
            "t_stats": t_rm,
            "pvalues": p_rm,
            "conditional_mean": mu,
            "log_likelihood": -result_rm.fun,
        },
        "total_log_likelihood": -(result_var.fun + result_rm.fun),
        "nobs": n,
    }
