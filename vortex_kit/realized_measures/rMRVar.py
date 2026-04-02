"""
vortex_kit.realized_measures.rMRVar
====================================
Modulated (pre-averaged) realized variance.

Christensen, Kinnebrock, Podolskij (2010).

Uses the g(x)=min(x,1-x) weight function for pre-averaging, then
subtracts a noise bias correction.
"""

from __future__ import annotations

import numpy as np

from vortex_kit.utils.returns import log_returns
from vortex_kit.utils.pre_average import pre_average_returns, compute_psi_constants


def rMRVar(prices: np.ndarray, *, theta: float = 0.8) -> float:
    """Modulated (pre-averaged) realized variance.

    Christensen, Kinnebrock, Podolskij (2010).

    Uses the g(x)=min(x,1-x) weight function for pre-averaging, then
    subtracts a noise bias correction.

    Parameters
    ----------
    prices : ndarray
        Tick-level prices.
    theta : float
        Bandwidth parameter (default 0.8).

    Returns
    -------
    float
        Modulated realized variance.
    """
    prices = np.asarray(prices, dtype=np.float64)
    N = len(prices)
    kn = int(np.floor(theta * np.sqrt(N)))
    if kn < 2:
        kn = 2

    psi1, psi2 = compute_psi_constants(kn)
    raw_ret = log_returns(prices)
    pa_ret = pre_average_returns(raw_ret, kn)

    crv = (
        (1.0 / (np.sqrt(N) * theta * psi2)) * np.sum(pa_ret ** 2)
        - (psi1 / (2.0 * theta ** 2 * psi2 * N)) * np.sum(raw_ret ** 2)
    )
    return crv
