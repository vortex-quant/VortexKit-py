"""
vortex_kit.kernels
===================
Kernel weight functions for realized kernel estimators.

Implements 12 kernel functions used in realized kernel variance/covariance
calculations.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def kernel_weight(x: float, kernel_type: int) -> float:
    """Kernel weight function for realized kernel estimation.

    Parameters
    ----------
    x : float
        Normalized lag value in [0, 1].
    kernel_type : int
        Integer code for kernel type.

    Returns
    -------
    float
        Kernel weight at x.

    Kernel Types
    ------------
    0 : Bartlett (Triangular)
    1 : Parzen
    2 : Quadratic Spectral (QS)
    3 : Tukey-Hanning
    4 : Cubic
    5 : 5th order
    6 : 6th order
    7 : 7th order
    8 : 8th order
    9 : Parzen (polynomial)
    10 : Tukey-Hanning (polynomial)
    11 : Flat-top
    """
    if kernel_type == 0:  # Bartlett / Triangular
        return 1.0 - x

    elif kernel_type == 1:  # Parzen
        if x < 0.5:
            return 1.0 - 6.0 * x * x + 6.0 * x * x * x
        else:
            return 2.0 * (1.0 - x) * (1.0 - x) * (1.0 - x)

    elif kernel_type == 2:  # Quadratic Spectral (QS)
        if x == 0.0:
            return 1.0
        else:
            arg = 6.0 * math.pi * x / 5.0
            return 25.0 / (12.0 * math.pi * math.pi * x * x) * (
                math.sin(arg) / arg - math.cos(arg)
            )

    elif kernel_type == 3:  # Tukey-Hanning
        return (1.0 + math.cos(math.pi * x)) / 2.0

    elif kernel_type == 4:  # Cubic
        return 1.0 - 3.0 * x * x + 2.0 * x * x * x

    elif kernel_type == 5:  # 5th order
        return 1.0 - 10.0 * x * x + 15.0 * x * x * x - 6.0 * x * x * x * x

    elif kernel_type == 6:  # 6th order
        return 1.0 - 15.0 * x * x + 35.0 * x * x * x - 21.0 * x * x * x * x

    elif kernel_type == 7:  # 7th order
        return 1.0 - 21.0 * x * x + 70.0 * x * x * x - 84.0 * x * x * x * x + 35.0 * x * x * x * x * x

    elif kernel_type == 8:  # 8th order
        return 1.0 - 28.0 * x * x + 126.0 * x * x * x - 210.0 * x * x * x * x + 120.0 * x * x * x * x * x

    elif kernel_type == 9:  # Parzen (polynomial)
        if x < 0.5:
            return 1.0 - 6.0 * x * x * (1.0 - x)
        else:
            return 2.0 * (1.0 - x) ** 3

    elif kernel_type == 10:  # Tukey-Hanning (polynomial)
        return (1.0 + math.cos(math.pi * x)) / 2.0

    elif kernel_type == 11:  # Flat-top
        if x < 0.1:
            return 1.0
        else:
            return (1.0 - x) / (1.0 - 0.1)

    else:
        return 1.0 - x  # Default to Bartlett


def kernel_name_to_int(name: str) -> int:
    """Convert kernel name string to integer code.

    Parameters
    ----------
    name : str
        Kernel name (case-insensitive).

    Returns
    -------
    int
        Integer kernel code.

    Supported Names
    -----------------
    "bartlett", "triangular" → 0
    "parzen" → 1
    "qs", "quadratic" → 2
    "tukey", "hanning", "tukey-hanning" → 3
    "cubic" → 4
    "5th", "5", "fifth" → 5
    "6th", "6" → 6
    "7th", "7" → 7
    "8th", "8" → 8
    "parzen_poly" → 9
    "tukey_poly" → 10
    "flat-top", "flattop" → 11
    """
    name = name.lower().replace("-", "").replace("_", "")
    mapping = {
        "bartlett": 0,
        "triangular": 0,
        "parzen": 1,
        "qs": 2,
        "quadratic": 2,
        "tukey": 3,
        "hanning": 3,
        "tukeyhanning": 3,
        "cubic": 4,
        "5th": 5,
        "5": 5,
        "fifth": 5,
        "6th": 6,
        "6": 6,
        "7th": 7,
        "7": 7,
        "8th": 8,
        "8": 8,
        "parzenpoly": 9,
        "tukeypoly": 10,
        "flattop": 11,
        "flattop": 11,
    }
    return mapping.get(name, 1)  # Default to Parzen (1)


def list_available_kernels() -> list[str]:
    """Return a list of available kernel names.

    Returns
    -------
    list[str]
        List of supported kernel names.
    """
    return [
        "bartlett", "triangular",
        "parzen",
        "qs", "quadratic",
        "tukey", "hanning", "tukey-hanning",
        "cubic",
        "5th", "fifth",
        "6th",
        "7th",
        "8th",
        "parzen_poly",
        "tukey_poly",
        "flat-top", "flattop",
    ]


__all__ = ["kernel_weight", "kernel_name_to_int", "list_available_kernels"]
