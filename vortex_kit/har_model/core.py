"""
vortex_kit.har_model.core
==========================
Core Heterogeneous AutoRegressive (HAR) model fitting.

Contains internal helpers for HAR model estimation.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _har_agg_col(rm: np.ndarray, period: int) -> np.ndarray:
    """Rolling average of `rm` over the last `period` observations.

    Output[i] = mean(rm[i-period+1 : i+1]).
    Leading ``period-1`` values are NaN.
    """
    n = len(rm)
    out = np.full(n, np.nan, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += rm[i]
        if i >= period:
            running -= rm[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def _har_agg(rm: np.ndarray, periods: list[int]) -> np.ndarray:
    """Build regressor matrix: one column per period (rolling averages).

    Returns
    -------
    ndarray, shape (N, len(periods))
    """
    cols = [_har_agg_col(rm, p) for p in periods]
    return np.column_stack(cols)


@njit(cache=True)
def _abd_jump_test(rv: np.ndarray, bpv: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """Andersen-Bollerslev-Diebold ABD jump test statistic.

    z_t = (1/N)^(-1/2) × (RV - BPV)/RV / sqrt( (μ₁⁻⁴+2μ₁⁻²-5) ×
          max(1, TQ/(BPV²)) )

    where μ₁ = sqrt(2/π) = E[|Z|] for Z~N(0,1).
    """
    mu1 = math.sqrt(2.0 / math.pi)
    n = len(rv)
    denom_factor = (mu1 ** (-4) + 2.0 * mu1 ** (-2) - 5.0)
    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        bpv2 = bpv[i] ** 2
        ratio = np.maximum(1.0, tq[i] / bpv2 if bpv2 > 1e-30 else 1.0)
        denom = math.sqrt(denom_factor * ratio) if denom_factor * ratio > 0 else 1e-10
        if rv[i] > 1e-30:
            z[i] = (n ** 0.5) * (rv[i] - bpv[i]) / rv[i] / denom
        else:
            z[i] = 0.0
    return z
