"""
vortex_kit.jump_tests.rankJumpTest
=====================================
Li, Todorov, Tauchen (2017) co-jump rank test.

Tests for co-jumps across multiple assets using rank correlation of
detected jumps.

Reference
---------
Li, J., Todorov, V. and Tauchen, G. (2017).
    "Adaptive estimation of continuous-time regression models using
    high-frequency data." Journal of Econometrics, 200, 36-47.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .intradayJumpTest import intradayJumpTest


def rankJumpTest(returns_list: list[np.ndarray],
                 *,
                 window: int = 78,
                 alpha: float = 0.999) -> dict:
    """Li, Todorov, Tauchen (2017) co-jump rank test.

    Tests for co-jumps across multiple assets.

    Parameters
    ----------
    returns_list : list of ndarray
        List of return series for each asset (same length).
    window : int
        Window size for local volatility estimation.
    alpha : float
        Confidence level.

    Returns
    -------
    dict
        Contains test results for co-jump detection and rank statistics.
    """
    k = len(returns_list)
    n = len(returns_list[0])

    # Detect jumps for each asset
    jump_results = []
    all_jumps = np.zeros((n, k), dtype=bool)

    for i in range(k):
        result = intradayJumpTest(returns_list[i], window=window, alpha=alpha)
        jump_results.append(result)
        all_jumps[:, i] = result["jump_indicators"]

    # Co-jump detection: jumps occurring simultaneously
    cojump_counts = np.sum(all_jumps, axis=1)
    cojump_indicators = cojump_counts >= 2  # At least 2 assets jump together
    n_cojumps = np.sum(cojump_indicators)

    # Jump sizes correlation
    jump_sizes = np.column_stack([r["jump_sizes"] for r in jump_results])

    # Rank correlation of jump sizes (only where jumps detected)
    rank_corr = np.eye(k, dtype=np.float64)
    pvalues = np.eye(k, dtype=np.float64)

    for i in range(k):
        for j in range(i + 1, k):
            # Get observations with jumps in at least one series
            mask = (jump_sizes[:, i] != 0) | (jump_sizes[:, j] != 0)
            if np.sum(mask) > 5:  # Need enough observations
                try:
                    corr, pval = stats.spearmanr(
                        jump_sizes[mask, i], jump_sizes[mask, j]
                    )
                    rank_corr[i, j] = corr
                    rank_corr[j, i] = corr
                    pvalues[i, j] = pval
                    pvalues[j, i] = pval
                except:
                    rank_corr[i, j] = 0.0
                    rank_corr[j, i] = 0.0

    return {
        "n_cojumps": n_cojumps,
        "cojump_indicators": cojump_indicators,
        "cojump_counts": cojump_counts,
        "rank_correlation": rank_corr,
        "rank_pvalues": pvalues,
        "individual_results": jump_results,
    }
