"""
Aggregate trades into klines (candlestick / OHLCV data).

Klines (also called candles or OHLCV bars) are produced by bucketing
trades into fixed time intervals. This module supports any interval
expressible as a Binance-style string (e.g. ``"1m"``, ``"5m"``,
``"1h"``, ``"1d"``).

The output schema matches the Binance klines format exactly:

- **open_time** – start of the interval (aligned to interval boundary)
- **open** – price of the first trade in the interval
- **high** – highest trade price in the interval
- **low** – lowest trade price in the interval
- **close** – price of the last trade in the interval
- **volume** – sum of base-asset quantities (qty)
- **close_time** – last tick of the interval (same unit as input)
- **quote_volume** – sum of quote-asset quantities (quote_qty)
- **num_trades** – count of trades in the interval
- **taker_buy_base_volume** – sum of qty for taker-buy trades
  (``is_buyer_maker == False``)
- **taker_buy_quote_volume** – sum of quote_qty for taker-buy trades
- **ignore** – always 0 (Binance placeholder)

**Precision preservation**: timestamps in the output klines use the
same unit as the input trades. If the input ``time`` column is in
microseconds, the output ``open_time`` / ``close_time`` will be in
microseconds. If the input is in milliseconds, the output stays in
milliseconds. No conversion is performed.

The input DataFrame must use canonical column names as defined in
:mod:`aggregation.schemas` (i.e. as returned by
:func:`aggregation.loaders.load_trades`).
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from .schemas import KLINES_CANONICAL
from .utils import TimestampPrecision, detect_timestamp_precision, interval_to_unit

# Binance stores all financial values with 8 decimal places.
# Rounding eliminates IEEE 754 floating-point accumulation errors
# that arise when summing thousands of small values.
_ROUND_PLACES = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def aggregate_klines(
    trades: pl.DataFrame,
    interval: str = "5m",
    precision: Optional[TimestampPrecision] = None,
) -> pl.DataFrame:
    """Aggregate trades into klines for a given time interval.

    Trades are assigned to intervals by flooring their timestamp to the
    nearest interval boundary. Each interval produces one kline row.

    Intervals with **no trades** are **not** included in the output
    (matching Binance behaviour for historical kline exports).

    Args:
        trades: DataFrame with canonical trades columns.
        interval: Binance-style interval string (e.g. ``"1m"``,
            ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``).
        precision: Timestamp precision of the ``time`` column. If ``None``,
            it is auto-detected. The output timestamps will use the same
            unit as the input.

    Returns:
        DataFrame with canonical klines columns, sorted by ``open_time``.
    """
    if precision is None:
        precision = detect_timestamp_precision(trades["time"])

    interval_ticks = interval_to_unit(interval, precision)

    # Assign each trade to its interval (floor-aligned open_time)
    t = trades.with_columns(
        (pl.col("time") // interval_ticks * interval_ticks).alias("open_time")
    )

    # Taker buy flag: buyer is taker when is_buyer_maker == False
    taker_buy_expr = pl.col("is_buyer_maker") == False  # noqa: E712

    # Aggregate
    result = t.group_by("open_time", maintain_order=True).agg([
        # OHLC — first/last by trade_id ordering within the group
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        # Volume
        pl.col("qty").sum().alias("volume"),
        # Quote volume
        pl.col("quote_qty").sum().alias("quote_volume"),
        # Trade count
        pl.col("trade_id").count().alias("num_trades"),
        # Taker buy volumes
        pl.when(taker_buy_expr)
            .then(pl.col("qty"))
            .otherwise(0.0)
            .sum()
            .alias("taker_buy_base_volume"),
        pl.when(taker_buy_expr)
            .then(pl.col("quote_qty"))
            .otherwise(0.0)
            .sum()
            .alias("taker_buy_quote_volume"),
    ])

    # Add computed columns (close_time = last tick in the interval)
    result = result.with_columns([
        (pl.col("open_time") + interval_ticks - 1).alias("close_time"),
        pl.lit(0).cast(pl.Int64).alias("ignore"),
    ])

    # Round float columns to Binance precision (8 decimal places)
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    result = result.with_columns([
        pl.col(c).round(_ROUND_PLACES) for c in float_cols if c in result.columns
    ])

    # Reorder to canonical column order
    result = result.select(KLINES_CANONICAL)
    return result
