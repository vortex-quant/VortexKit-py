"""
vortex_kit.realized_measures.knChooseReMeDI
============================================
Select optimal kn for ReMeDI via the criterion from Li & Linton.

Chooses the smallest kn where
(R(0,kn) - R(1,kn) - R(2,kn) + R(3,kn))²
is below tolerance.
"""

from __future__ import annotations

import numpy as np

from .ReMeDI import _remedi_core


def knChooseReMeDI(prices: np.ndarray,
                   *,
                   max_kn: int = 30,
                   tol: float = 1e-10) -> int:
    """Select optimal kn for ReMeDI via the criterion from Li & Linton.

    Chooses the smallest kn where
    (R(0,kn) - R(1,kn) - R(2,kn) + R(3,kn))²
    is below *tol*.

    Parameters
    ----------
    prices : ndarray
    max_kn : int
        Maximum kn to try.
    tol : float
        Tolerance.

    Returns
    -------
    int
        Optimal kn.
    """
    lp = np.asarray(prices, dtype=np.float64)
    for kn in range(1, max_kn + 1):
        r0 = _remedi_core(lp, kn, 0)
        r1 = _remedi_core(lp, kn, 1)
        r2 = _remedi_core(lp, kn, 2)
        r3 = _remedi_core(lp, kn, 3)
        crit = (r0 - r1 - r2 + r3) ** 2
        if crit < tol:
            return kn
    return max_kn
