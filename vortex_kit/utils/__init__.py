"""
vortex_kit.utils
=================
Low-level utility functions shared across the library.

Contains Numba-accelerated helpers for:
- log-return computation
- rolling statistics (min, median, product)
- pre-averaging
- refresh-time synchronisation
- g-function (min(x, 1-x))
- alignment / aggregation helpers

All public functions accept and return plain numpy arrays to ensure
compatibility with any DataFrame library (Polars, Pandas, etc.).
"""

from .returns import log_returns, simple_returns
from .rolling import roll_min2, roll_median3, roll_prod
from .stats import mu_p, gfunction
from .sync import refresh_time
from .matrices import make_psd
from .pre_average import pre_average_returns, compute_psi_constants

__all__ = [
    "log_returns",
    "simple_returns",
    "roll_min2",
    "roll_median3",
    "roll_prod",
    "mu_p",
    "gfunction",
    "refresh_time",
    "make_psd",
    "pre_average_returns",
    "compute_psi_constants",
]
