"""
Aggregate raw trades into aggTrades.

in Exchanges aggTrades group consecutive trades that share the same
``price``, ``timestamp``, and ``is_buyer_maker`` flag. Each group
represents a single order execution at the same price and side.

**Limitation**: Binance internally groups by *order id*, which can split
trades with identical (price, timestamp, maker) into separate aggTrades
when they originate from different orders. This information is not
available in the raw trades data, so the output of
:func:`aggregate_trades` may produce **fewer rows** than the official
Binance aggTrades file (typically <0.1% difference). The aggregated
quantities, however, remain identical — the split rows are simply
combined.

The input DataFrame must use canonical column names as defined in
:mod:`aggregation.schemas` (i.e. as returned by
:func:`aggregation.loaders.load_trades`).
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from .schemas import AGG_TRADES_CANONICAL

# Binance stores all financial values with 8 decimal places.
_ROUND_PLACES = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def aggregate_trades(
    trades: pl.DataFrame,
    start_agg_id: int = 0,
) -> pl.DataFrame:
    """Aggregate raw trades into aggTrades.

    Trades are sorted by ``trade_id`` and then grouped into consecutive
    runs where ``price``, ``time``, and ``is_buyer_maker`` are all
    identical. Each run becomes one aggTrade row.

    Args:
        trades: DataFrame with canonical trades columns.
        start_agg_id: Starting value for the sequential ``agg_trade_id``
            column (default ``0``).

    Returns:
        DataFrame with canonical aggTrades columns, sorted by
        ``agg_trade_id``.
    """
    t = trades.sort("trade_id")

    # --- Detect group boundaries ---
    # A new group starts whenever price, time, or is_buyer_maker changes.
    price_changed = (pl.col("price") != pl.col("price").shift(1)).fill_null(True)
    time_changed = (pl.col("time") != pl.col("time").shift(1)).fill_null(True)
    maker_changed = (pl.col("is_buyer_maker") != pl.col("is_buyer_maker").shift(1)).fill_null(True)

    t = t.with_columns(
        (price_changed | time_changed | maker_changed)
        .cast(pl.Int32)
        .alias("_boundary")
    )
    t = t.with_columns(pl.col("_boundary").cum_sum().alias("_group_id"))

    # --- Aggregate each group ---
    result = t.group_by("_group_id", maintain_order=True).agg([
        pl.col("price").first().alias("price"),
        pl.col("qty").sum().alias("quantity"),
        pl.col("trade_id").min().alias("first_trade_id"),
        pl.col("trade_id").max().alias("last_trade_id"),
        pl.col("time").first().alias("timestamp"),
        pl.col("is_buyer_maker").first().alias("was_buyer_maker"),
        pl.col("is_best_match").first().alias("was_best_match"),
    ])

    # Drop internal columns and add sequential agg_trade_id
    result = result.drop("_group_id")
    n = result.height
    result = result.with_columns(
        (pl.int_range(0, n) + start_agg_id).alias("agg_trade_id")
    )

    # Round float columns to Binance precision (8 decimal places)
    result = result.with_columns(
        pl.col("quantity").round(_ROUND_PLACES)
    )

    # Reorder to canonical column order
    result = result.select(AGG_TRADES_CANONICAL)
    return result
