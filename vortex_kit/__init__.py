"""
vortex_kit — High-Frequency Financial Data Analysis Library
============================================================

A Python library for statistical analysis of high-frequency financial data,
providing Numba-accelerated implementations of realized measures, jump tests,
spot volatility estimation, HAR/HEAVY models, lead-lag estimation, liquidity
measures, and inference tools.

This library is a Python reimplementation of the algorithms in
the R ``highfrequency`` package (Boudt, Cornelissen, Payseur et al.),
designed for industrial use in quantitative finance.

All functions accept and return **numpy** arrays, making them compatible
with any DataFrame library (Polars, Pandas, etc.). Internally, Numba JIT
compilation ensures near-C performance.

Modules
-------
realized_measures
    Realized variance, covariance, quarticity, semivariance, noise-robust,
    and pre-averaged estimators.

covariance
    Extended covariance estimators: Hayashi-Yoshida (rHYCov), threshold
    (rThresholdCov), subsampled (rAVGCov), outlyingness-weighted (rOWCov),
    robust two-scale (rRTSCov), realized skewness and kurtosis.

jump_tests
    Statistical tests for price jumps (BNS, AJ, intraday, rank co-jump).

spot_volatility
    Spot volatility, drift estimation, and drift-burst testing.

har_model
    HAR model family (HAR, HARJ, HARCJ, HARQ, HARQJ, CHAR, CHARQ) for
    realized volatility forecasting.

heavy_model
    HEAVY model (Shephard & Sheppard 2010) for high-frequency-based
    conditional variance estimation.

lead_lag
    Lead-lag estimation via Hayashi-Yoshida shifted contrast function.

liquidity
    23 market microstructure liquidity measures and Lee-Ready trade
    direction classifier.

inference
    Standard errors and confidence bands for integrated variance.

kernels
    Kernel weight functions for realized kernel estimators.

utils
    Low-level helpers (returns, rolling stats, pre-averaging, refresh-time
    synchronisation, etc.).

Quick Start
-----------
>>> import numpy as np
>>> from vortex_kit.realized_measures import rRVar, rBPVar
>>> from vortex_kit.jump_tests import BNSjumpTest
>>> from vortex_kit.har_model import fit_har
>>>
>>> # Realized variance & bipower variation
>>> prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
>>> rv = rRVar(prices, make_returns=True)
>>> bpv = rBPVar(prices, make_returns=True)
>>>
>>> # Jump test
>>> result = BNSjumpTest(prices, make_returns=True)
>>> print(f"z = {result['ztest']:.4f}, p = {result['pvalue']:.4f}")
"""

__version__ = "1.0.0"
__author__ = "VortexKit"

# ─── Public API (lazy imports for fast startup) ───────────────────────────

from . import realized_measures
from . import covariance
from . import jump_tests
from . import spot_volatility
from . import har_model
from . import heavy_model
from . import lead_lag
from . import liquidity
from . import inference
from . import kernels
from . import utils

# Direct function imports for most common use cases
from .realized_measures import (
    rRVar,
    rBPVar,
    rBPCov,
    rMedRVar,
    rMinRVar,
    rSVar,
    rSemiCov,
    rQuar,
    rTPQuar,
    rTSVar,
    rTSCov,
    rKernelVar,
    rKernelCov,
    rMRVar,
    rMRCov,
    ReMeDI,
    knChooseReMeDI,
)

from .covariance import (
    rHYCov,
    rThresholdCov,
    rAVGCov,
    rOWCov,
    rRTSCov,
    rSkew,
    rKurt,
)

from .jump_tests import (
    BNSjumpTest,
    AJjumpTest,
    intradayJumpTest,
    rankJumpTest,
)

from .spot_volatility import spotVol, spotDrift, driftBursts
from .har_model import fit_har
from .heavy_model import fit_heavy
from .lead_lag import leadLag
from .liquidity import getLiquidityMeasures, getTradeDirection
from .inference import IVinference

__all__ = [
    # Submodules
    "realized_measures",
    "covariance",
    "jump_tests",
    "spot_volatility",
    "har_model",
    "heavy_model",
    "lead_lag",
    "liquidity",
    "inference",
    "kernels",
    "utils",
    # Realized measures
    "rRVar",
    "rBPVar",
    "rBPCov",
    "rMedRVar",
    "rMinRVar",
    "rSVar",
    "rSemiCov",
    "rQuar",
    "rTPQuar",
    "rTSVar",
    "rTSCov",
    "rKernelVar",
    "rKernelCov",
    "rMRVar",
    "rMRCov",
    "ReMeDI",
    "knChooseReMeDI",
    # Covariance
    "rHYCov",
    "rThresholdCov",
    "rAVGCov",
    "rOWCov",
    "rRTSCov",
    "rSkew",
    "rKurt",
    # Jump tests
    "BNSjumpTest",
    "AJjumpTest",
    "intradayJumpTest",
    "rankJumpTest",
    # Spot volatility
    "spotVol",
    "spotDrift",
    "driftBursts",
    # Models
    "fit_har",
    "fit_heavy",
    # Lead-lag
    "leadLag",
    # Liquidity
    "getLiquidityMeasures",
    "getTradeDirection",
    # Inference
    "IVinference",
]
