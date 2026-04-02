"""
vortex_kit.heavy_model
=======================
High-frEquency-bAsed VolatilitY (HEAVY) model estimation.

Implements the two-equation HEAVY model of Shephard & Sheppard (2010):

    Variance equation:  h_t  = ω  + α  × RM_{t-1} + β  × h_{t-1}
    RM equation:        μ_t = ω_R + α_R × RM_{t-1} + β_R × μ_{t-1}

Both equations are estimated by maximum likelihood separately.
Robust standard errors are computed via an inverted Hessian–OPG sandwich.

Reference
---------
Shephard, N. and Sheppard, K. (2010).
    "Realising the future: Forecasting with high frequency based volatility
    (HEAVY) models." Journal of Applied Econometrics, 25, 197-231.
"""

from .fit import fit_heavy

__all__ = ["fit_heavy"]
