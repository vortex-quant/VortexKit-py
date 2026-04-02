"""
vortex_kit.realized_measures
=============================
Realized variance, covariance, quarticity, and semivariance estimators.

This module provides Numba-accelerated implementations of the estimators
described in the *highfrequency* R package (Boudt, Cornelissen, Payseur
et al.).

Functions
---------
**Univariate variance / volatility:**
    rRVar          – Realized variance (sum of squared returns).
    rBPVar         – Realized bipower variation (jump-robust, Barndorff-Nielsen & Shephard 2004).
    rMedRVar       – Median realized variance (Andersen et al. 2012).
    rMinRVar       – Minimum realized variance (Andersen et al. 2012).
    rSVar          – Realized semivariance (positive / negative).
    rMPVar         – Realized multipower variation.

**Univariate quarticity:**
    rQuar          – Realized quarticity.
    rTPQuar        – Realized tri-power quarticity (jump-robust).
    rMedRQuar      – Median realized quarticity.
    rMinRQuar      – Minimum realized quarticity.
    rQPVar         – Realized quad-power variation.

**Noise-robust univariate:**
    rTSVar         – Two-Scale realized variance (Zhang et al. 2005).
    rKernelVar     – Realized kernel variance.
    rMRVar         – Modulated (pre-averaged) realized variance (Christensen et al. 2010).

**Multivariate covariance:**
    rCov           – Realized covariance (outer-product of returns).
    rBPCov         – Realized bipower covariance.
    rTSCov         – Two-Scale realized covariance.
    rKernelCov     – Realized kernel covariance.
    rMRCov         – Modulated realized covariance (pre-averaging).
    rSemiCov       – Realized semicovariance decomposition.

**Noise estimation:**
    ReMeDI         – Realized Method-of-moments estimator of noise auto-covariance.
    knChooseReMeDI – Optimal kn selection for ReMeDI.

Reference
---------
Barndorff-Nielsen, O.E. and Shephard, N. (2004).
    "Power and bipower variation with stochastic volatility and jumps."
    Journal of Financial Econometrics, 2, 1-37.

Andersen, T.G., Dobrev, D. and Schaumburg, E. (2012).
    "Jump-robust volatility estimation using nearest neighbor truncation."
    Journal of Econometrics, 169, 75-93.

Zhang, L., Mykland, P. A. and Aït-Sahalia, Y. (2005).
    "A tale of two time scales: Determining integrated volatility with
    noisy high-frequency data." JASA, 100, 1394-1411.

Christensen, K., Kinnebrock, S. and Podolskij, M. (2010).
    "Pre-averaging estimators of the ex-post covariance matrix in noisy
    diffusion models with non-synchronous data." JoE, 159, 116-133.
"""

from .rRVar import rRVar
from .rBPVar import rBPVar
from .rBPCov import rBPCov
from .rMedRVar import rMedRVar
from .rMinRVar import rMinRVar
from .rSVar import rSVar
from .rSemiCov import rSemiCov
from .rQuar import rQuar
from .rTPQuar import rTPQuar
from .rMedRQuar import rMedRQuar
from .rMinRQuar import rMinRQuar
from .rQPVar import rQPVar
from .rMPVar import rMPVar
from .rCov import rCov
from .rTSVar import rTSVar
from .rTSCov import rTSCov
from .rKernelVar import rKernelVar
from .rKernelCov import rKernelCov
from .rMRVar import rMRVar
from .rMRCov import rMRCov
from .ReMeDI import ReMeDI
from .knChooseReMeDI import knChooseReMeDI

__all__ = [
    "rRVar",
    "rBPVar",
    "rBPCov",
    "rMedRVar",
    "rMinRVar",
    "rSVar",
    "rSemiCov",
    "rQuar",
    "rTPQuar",
    "rMedRQuar",
    "rMinRQuar",
    "rQPVar",
    "rMPVar",
    "rCov",
    "rTSVar",
    "rTSCov",
    "rKernelVar",
    "rKernelCov",
    "rMRVar",
    "rMRCov",
    "ReMeDI",
    "knChooseReMeDI",
]
