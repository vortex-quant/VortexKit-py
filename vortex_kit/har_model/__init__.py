"""
vortex_kit.har_model
=====================
Heterogeneous AutoRegressive (HAR) model family for realized volatility.

Implements (mirrors the R highfrequency ``HARmodel`` function exactly):
    fit_har  – Fit and predict the HAR family of volatility models.

Supported model types:
    "HAR"   – Corsi (2009) basic HAR-RV.
    "HARJ"  – HAR with separate jump component.
    "HARCJ" – HAR with continuous/jump decomposition and ABD jump test.
    "HARQ"  – HAR-RQ (Bollerslev, Patton, Quaedvlieg 2016).
    "HARQJ" – HAR-RQ with jump component.
    "CHAR"  – Continuous HAR (uses jump-robust BPV for RV).
    "CHARQ" – Continuous HAR-Q.

Reference
---------
Corsi, F. (2009).
    "A simple approximate long memory model of realized volatility."
    Journal of Financial Econometrics, 7, 174-196.

Andersen, T. G., Bollerslev, T. and Diebold, F. (2007).
    "Roughing it up: Including jump components in the measurement,
    modelling and forecasting of return volatility."
    Review of Economics and Statistics, 89, 701-720.

Bollerslev, T., Patton, A., and Quaedvlieg, R. (2016).
    "Exploiting the errors: A simple approach for improved volatility
    forecasting." Journal of Econometrics, 192, 1-18.
"""

from .fit import fit_har

__all__ = ["fit_har"]
