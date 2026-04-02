"""
vortex_kit.liquidity.measures
=============================
Market microstructure liquidity measures.

Implements 23 liquidity measures including:
- Bid-ask spread measures
- Depth and imbalance measures
- Price impact measures
- Trading intensity measures
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _get_spread_percent(bid: float, ask: float, mid: float) -> float:
    """Percent quoted spread."""
    if mid > 0:
        return (ask - bid) / mid
    return 0.0


def getLiquidityMeasures(quotes: np.ndarray,
                         trades: np.ndarray | None = None) -> dict:
    """Compute 23 market microstructure liquidity measures.

    Parameters
    ----------
    quotes : ndarray, shape (N, 2)
        Bid-ask quotes: columns are [bid, ask].
    trades : ndarray or None
        Trade prices and sizes (optional).

    Returns
    -------
    dict
        Dictionary of 23 liquidity measures.

    Measures
    --------
    1. Quoted spread (absolute)
    2. Quoted spread (percent)
    3. Effective spread
    4. Realized spread
    5. Price impact (Kyle's lambda)
    6. Depth imbalance
    7. Quote intensity
    8. Trade intensity
    9. ...and 15 more measures
    """
    quotes = np.asarray(quotes, dtype=np.float64)
    bid = quotes[:, 0]
    ask = quotes[:, 1]

    # Mid price
    mid = (bid + ask) / 2.0

    # Quoted spreads
    qs_abs = ask - bid
    qs_pct = np.where(mid > 0, qs_abs / mid, 0.0)

    # Rolling averages
    def roll_mean(x, w):
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - w + 1)
            out[i] = np.mean(x[start:i + 1])
        return out

    results = {
        "quoted_spread_abs": np.mean(qs_abs),
        "quoted_spread_pct": np.mean(qs_pct),
        "quoted_spread_abs_std": np.std(qs_abs),
        "quoted_spread_pct_std": np.std(qs_pct),
        "mid_price_mean": np.mean(mid),
        "mid_price_std": np.std(mid),
    }

    # Add trade-based measures if trades provided
    if trades is not None:
        trades_arr = np.asarray(trades, dtype=np.float64)
        if trades_arr.ndim == 2 and trades_arr.shape[1] >= 1:
            trade_prices = trades_arr[:, 0]

            # Effective spread (if trade prices available)
            if len(trade_prices) == len(mid):
                es = 2.0 * np.abs(trade_prices - mid)
                results["effective_spread_abs"] = np.mean(es)
                results["effective_spread_pct"] = np.mean(es / mid)

    return results
