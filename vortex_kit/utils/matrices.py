"""
vortex_kit.utils.matrices
==========================
Matrix utility functions.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def make_psd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite matrix.

    Uses eigendecomposition with eigenvalue clipping.

    Parameters
    ----------
    mat : ndarray, shape (K, K)
        Symmetric matrix (may have small negative eigenvalues due to
        numerical errors).
    eps : float
        Minimum eigenvalue threshold. Eigenvalues below this are set to eps.

    Returns
    -------
    ndarray, shape (K, K)
        Nearest PSD matrix.
    """
    # Note: eigendecomposition is not in Numba; do in pure Python wrapper
    # This is a placeholder for the actual implementation
    return _make_psd_core(mat, eps)


def _make_psd_core(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Core implementation (not numba-compiled due to np.linalg.eigh)."""
    mat = np.asarray(mat, dtype=np.float64)
    # Symmetrize
    mat = (mat + mat.T) / 2.0
    # Eigenvalue decomposition
    eigs, vecs = np.linalg.eigh(mat)
    # Clip eigenvalues
    eigs = np.maximum(eigs, eps)
    # Reconstruct
    return vecs @ np.diag(eigs) @ vecs.T
