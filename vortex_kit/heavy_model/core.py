"""
vortex_kit.heavy_model.core
============================
Core HEAVY model estimation functions.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _calc_rec_var(par: np.ndarray, rm: np.ndarray) -> np.ndarray:
    """Recursively compute conditional variances.

    h_t = omega + alpha × rm_{t-1} + beta × h_{t-1}
    Initialised at h_0 = mean(rm) (exact C++ port of calcRecVarEq).

    Parameters
    ----------
    par : ndarray, shape (3,)
        [omega, alpha, beta]
    rm : ndarray
        Realized measure (or squared returns for variance equation).

    Returns
    -------
    ndarray, shape (N,)
        Conditional variances h_0, ..., h_{N-1}.
    """
    omega, alpha, beta = par[0], par[1], par[2]
    n = len(rm)
    h = np.empty(n, dtype=np.float64)
    # Exact match of R C++ calcRecVarEq: h[0] = mean(rm)
    mean_rm = 0.0
    for i in range(n):
        mean_rm += rm[i]
    mean_rm /= n
    h[0] = mean_rm
    for t in range(1, n):
        h[t] = omega + alpha * rm[t - 1] + beta * h[t - 1]
        if h[t] < 1e-30:
            h[t] = 1e-30
    return h


def _heavy_llh(par: np.ndarray, rm: np.ndarray) -> float:
    """HEAVY log-likelihood (Gaussian): -0.5 × sum(log(h) + rm/h).

    Used for BOTH equations:
    - Variance eq:  rm = ret² → quasi log-normal likelihood
    - RM eq:        rm = RM_t → realized-measure likelihood

    Parameters
    ----------
    par : ndarray, shape (3,)
        [omega, alpha, beta]
    rm : ndarray
        Realized measure input.

    Returns
    -------
    float  (negative log-likelihood for use in minimizers)
    """
    omega, alpha, beta = par
    if omega <= 0 or alpha < 0 or beta < 0:
        return 1e10
    h = _calc_rec_var(par, rm)
    log_h = np.log(h)
    if np.any(~np.isfinite(log_h)):
        return 1e10
    # sum_t [ -0.5×log(2π) - 0.5×(log(h_t) + rm_t/h_t) ]
    llh = -0.5 * (math.log(2.0 * math.pi) * len(rm) +
                  np.sum(log_h + rm / h))
    if not math.isfinite(llh):
        return 1e10
    return -llh  # return negative (we minimise)


def _compute_hessian_opg(par: np.ndarray, rm: np.ndarray,
                         step: float = 1e-5) -> np.ndarray:
    """Compute Hessian and OPG sandwich for robust standard errors.

    Returns robust covariance matrix of parameter estimates.
    """
    n = len(rm)
    k = len(par)

    # Numerical gradient at each observation
    def ll_i(p, i):
        """Log-likelihood contribution for observation i."""
        h = _calc_rec_var(p, rm)
        return -0.5 * (math.log(2.0 * math.pi) + math.log(h[i]) + rm[i] / h[i])

    # Score contributions
    scores = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            p_plus = par.copy()
            p_minus = par.copy()
            p_plus[j] += step
            p_minus[j] -= step
            scores[i, j] = (ll_i(p_plus, i) - ll_i(p_minus, i)) / (2.0 * step)

    # OPG (outer product of gradients)
    opg = scores.T @ scores / n

    # Hessian (numerical)
    hess = np.zeros((k, k))
    ll0 = _heavy_llh(par, rm)
    for i in range(k):
        for j in range(k):
            p_ip = par.copy()
            p_jp = par.copy()
            p_ijp = par.copy()
            p_im = par.copy()
            p_jm = par.copy()
            p_ijm = par.copy()

            p_ip[i] += step
            p_jp[j] += step
            p_ijp[i] += step
            p_ijp[j] += step
            p_im[i] -= step
            p_jm[j] -= step
            p_ijm[i] -= step
            p_ijm[j] -= step

            hess[i, j] = (_heavy_llh(p_ijp, rm) - _heavy_llh(p_ip, rm)
                         - _heavy_llh(p_jp, rm) + 2.0 * ll0
                         - _heavy_llh(p_im, rm) - _heavy_llh(p_jm, rm)
                         + _heavy_llh(p_ijm, rm)) / (step * step)

    hess = hess / n

    # Robust covariance: (Hess⁻¹ × OPG × Hess⁻¹) / n
    try:
        hess_inv = np.linalg.inv(hess)
        robust_cov = hess_inv @ opg @ hess_inv / n
    except np.linalg.LinAlgError:
        robust_cov = np.eye(k) * 1e-6

    return robust_cov
