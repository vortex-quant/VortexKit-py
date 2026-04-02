"""
vortex_kit.covariance
======================
Extended covariance estimators not covered in realized_measures.

Implements:
    rHYCov          – Hayashi-Yoshida covariance (non-synchronous trading).
    rThresholdCov   – Threshold (jump-truncated) realized covariance.
    rAVGCov         – Subsampled/averaged realized covariance.
    rOWCov          – Outlyingness-weighted realized covariance.
    rRTSCov         – Robust two-scale realized covariance.
    rSkew           – Realized skewness.
    rKurt           – Realized kurtosis.

All functions accept plain **numpy** arrays. Timestamp-based functions
additionally accept numeric timestamp arrays (e.g. epoch seconds).

Reference
---------
Hayashi, T. and Yoshida, N. (2005).
    "On covariance estimation of non-synchronously observed diffusion
    processes." Bernoulli, 11, 359-379.

Podolskij, M. and Vetter, M. (2009).
    "Bipower-type estimation in a noisy diffusion setting." Stochastic
    Processes and their Applications, 119, 2803-2831.

Boudt, K. and Zhang, J. (2010).
    "Jump robust two time scale covariance estimation and realized
    volatility budgets." Quantitative Finance, 15, 1041-1054.

Boudt, K., Croux, C. and Laurent, S. (2011).
    "Robust estimation of intraweek periodicity in volatility and jump
    detection." Journal of Empirical Finance, 18, 353-367.

Liu, L. Y., Patton, A. J. and Sheppard, K. (2015).
    "Does anything beat 5-minute RV?" Journal of Econometrics,
    187, 293-311.

Amaya, D. et al. (2015).
    "Does realized skewness and kurtosis predict the cross-section of
    equity returns?" Journal of Financial Economics, 118, 135-167.
"""

from .rHYCov import rHYCov
from .rThresholdCov import rThresholdCov
from .rAVGCov import rAVGCov
from .rOWCov import rOWCov
from .rRTSCov import rRTSCov
from .rSkew import rSkew
from .rKurt import rKurt

__all__ = [
    "rHYCov",
    "rThresholdCov",
    "rAVGCov",
    "rOWCov",
    "rRTSCov",
    "rSkew",
    "rKurt",
]
