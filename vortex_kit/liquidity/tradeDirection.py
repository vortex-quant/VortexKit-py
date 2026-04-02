"""
vortex_kit.liquidity.trade_direction
=======================================
Lee-Ready trade direction classification.

Classifies trades as buyer-initiated or seller-initiated using
the Lee-Ready (1991) algorithm based on quote midpoints.

Reference
---------
Lee, C.M.C. and Ready, M.J. (1991).
    "Inferring trade direction from intraday data."
    Journal of Finance, 46, 733-746.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _lee_ready_core(trade_prices: np.ndarray,
                    mid_prices: np.ndarray) -> np.ndarray:
    """Core Lee-Ready classification.

    Returns: 1 for buyer-initiated, -1 for seller-initiated, 0 for unknown.
    """
    n = len(trade_prices)
    direction = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if trade_prices[i] > mid_prices[i]:
            direction[i] = 1  # Buy
        elif trade_prices[i] < mid_prices[i]:
            direction[i] = -1  # Sell
        else:
            # Use tick test: compare to previous trade
            if i > 0:
                if trade_prices[i] > trade_prices[i - 1]:
                    direction[i] = 1
                elif trade_prices[i] < trade_prices[i - 1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i - 1] if direction[i - 1] != 0 else 0

    return direction


def getTradeDirection(trade_prices: np.ndarray,
                      bid_quotes: np.ndarray,
                      ask_quotes: np.ndarray) -> dict:
    """Lee-Ready trade direction classification.

    Classifies each trade as buyer-initiated (+1) or seller-initiated (-1)
    based on the Lee-Ready (1991) algorithm.

    Parameters
    ----------
    trade_prices : ndarray
        Trade prices.
    bid_quotes : ndarray
        Bid quotes (matched to trades by time).
    ask_quotes : ndarray
        Ask quotes (matched to trades by time).

    Returns
    -------
    dict
        Trade directions, buy/sell counts, and classification summary.

    Reference
    ---------
    Lee, C.M.C. and Ready, M.J. (1991).
        "Inferring trade direction from intraday data."
        Journal of Finance, 46, 733-746.
    """
    tp = np.asarray(trade_prices, dtype=np.float64)
    bid = np.asarray(bid_quotes, dtype=np.float64)
    ask = np.asarray(ask_quotes, dtype=np.float64)

    # Mid prices
    mid = (bid + ask) / 2.0

    # Classify
    direction = _lee_ready_core(tp, mid)

    # Summary statistics
    n_buys = np.sum(direction == 1)
    n_sells = np.sum(direction == -1)
    n_unknown = np.sum(direction == 0)

    # Order imbalance
    total_classified = n_buys + n_sells
    if total_classified > 0:
        imbalance = (n_buys - n_sells) / total_classified
    else:
        imbalance = 0.0

    return {
        "directions": direction,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_unknown": n_unknown,
        "order_imbalance": imbalance,
        "buy_ratio": n_buys / len(tp) if len(tp) > 0 else 0.0,
    }
