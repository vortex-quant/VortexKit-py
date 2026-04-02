"""
vortex_kit.realized_measures.rMPVar
======================================
Realized multipower variation.

rMPVar(m, p) = dₘ,ₚ × N^(p/2) / (N-m+1) ×
               Σᵢ Πⱼ₌₀ᵐ⁻¹ |rᵢ₊ⱼ|^(p/m)

where dₘ,ₚ = μₚ/ₘ⁻ᵐ.

Requires m > p/2.
"""

from __future__ import annotations

import numpy as np

from vortex_kit.utils.returns import log_returns
from vortex_kit.utils.stats import mu_p
from vortex_kit.utils.rolling import roll_prod


def rMPVar(data: np.ndarray,
           *,
           m: int = 2,
           p: float = 2.0,
           make_returns: bool = False) -> float:
    """Realized multipower variation.

    rMPVar(m, p) = dₘ,ₚ × N^(p/2) / (N-m+1) ×
                   Σᵢ Πⱼ₌₀ᵐ⁻¹ |rᵢ₊ⱼ|^(p/m)

    where dₘ,ₚ = μₚ/ₘ⁻ᵐ.

    Requires m > p/2.

    Parameters
    ----------
    data : ndarray
        Returns or prices.
    m : int
        Number of blocks (default 2).
    p : float
        Power (default 2.0).
    make_returns : bool
        If True, compute log-returns from prices.

    Returns
    -------
    float
        Realized multipower variation.
    """
    if m <= p / 2.0:
        raise ValueError(f"Require m > p/2, got m={m}, p={p}")

    r = log_returns(data) if make_returns else np.asarray(data, dtype=np.float64)
    n = len(r)
    if n < m:
        return 0.0

    power = p / m
    mu = mu_p(power)
    d = mu ** (-m)

    # Compute |r|^(p/m) then rolling product of m
    abs_r_pow = np.abs(r) ** power
    rp = roll_prod(abs_r_pow, m)
    return d * (n ** (p / 2.0)) / (n - m + 1) * np.sum(rp)
