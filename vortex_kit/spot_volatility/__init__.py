"""
vortex_kit.spot_volatility
===========================
Spot volatility, drift estimation, and drift-burst testing.

Implements:
    spotVol      – Nonparametric spot volatility estimation.
    spotDrift    – Spot drift (mean return) estimation.
    driftBursts  – Drift burst detection test.

Reference
---------
Kristensen, D. (2010).
    "Nonparametric filtering of the realized spot volatility."
    Econometric Theory, 26, 60-93.

Christensen, K., Oomen, R. and Reno, R. (2018).
    "The drift burst hypothesis." Journal of Econometrics, 208, 25-48.
"""

from .spotVol import spotVol
from .spotDrift import spotDrift
from .driftBursts import driftBursts

__all__ = ["spotVol", "spotDrift", "driftBursts"]
