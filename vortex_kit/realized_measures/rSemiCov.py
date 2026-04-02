"""
vortex_kit.realized_measures.rSemiCov
======================================
Realized semicovariance decomposition.

Decomposes the realized covariance into positive, negative, and mixed
components based on the signs of returns.

Returns
-------
dict with keys:
- "positive": Covariance from positive returns
- "negative": Covariance from negative returns
- "mixed": Cross-component covariance
- "concordant": Positive + Negative
- "rCov": Total realized covariance (= concordant + mixed)
"""

from __future__ import annotations

import numpy as np


def rSemiCov(returns_list: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Realized semicovariance decomposition.

    Decomposes the realized covariance into positive, negative, and mixed
    components.

    Parameters
    ----------
    returns_list : list of ndarray
        K aligned return series of equal length N.

    Returns
    -------
    dict
        Keys: ``"positive"``, ``"negative"``, ``"mixed"``,
              ``"concordant"`` (= pos + neg), ``"rCov"`` (= concordant + mixed).
    """
    k = len(returns_list)
    n = len(returns_list[0])
    R = np.column_stack(returns_list)  # (N, K)

    Rpos = np.maximum(R, 0.0)
    Rneg = np.minimum(R, 0.0)

    pos = Rpos.T @ Rpos
    neg = Rneg.T @ Rneg
    mixed = Rpos.T @ Rneg + Rneg.T @ Rpos
    concordant = pos + neg

    return {
        "positive": pos,
        "negative": neg,
        "mixed": mixed,
        "concordant": concordant,
        "rCov": concordant + mixed,
    }
