"""
Aggregate trades into klines (candlestick / OHLCV data).

Klines (also called candles or OHLCV bars) are produced by bucketing
trades into fixed time intervals. Intervals can be strings such as
``"2.5m"`` and ``"20s"``, or numeric values paired with ``interval_scale``.

The output schema matches the Binance klines format exactly:

- ``open_time``: start of the interval, aligned to interval boundary.
- ``open``: price of the first trade in the interval.
- ``high``: highest trade price in the interval.
- ``low``: lowest trade price in the interval.
- ``close``: price of the last trade in the interval.
- ``volume``: sum of base-asset quantities.
- ``close_time``: last tick of the interval, in the same unit as input.
- ``quote_volume``: sum of quote-asset quantities.
- ``num_trades``: count of trades in the interval.
- ``taker_buy_base_volume``: base quantity where buyer was taker.
- ``taker_buy_quote_volume``: quote quantity where buyer was taker.
- ``ignore``: always 0, matching Binance's placeholder column.

The preferred input schema is produced by :func:`aggregation.normalize_trades`.
The legacy Binance-style canonical schema (``time``, ``qty``, ``quote_qty``,
``trade_id``) is also accepted for compatibility.
"""

from __future__ import annotations

from typing import Optional, Union

import polars as pl

from .schemas import KLINES_CANONICAL, normalize_trades
from .utils import (
    IntervalValue,
    TimestampPrecision,
    detect_timestamp_precision,
    interval_to_unit,
)


DataFrameLike = Union[pl.DataFrame, pl.LazyFrame]

# Binance stores all financial values with 8 decimal places. Rounding avoids
# visible IEEE 754 accumulation artifacts after summing many small trades.
_ROUND_PLACES = 8


def aggregate_klines(
    trades: DataFrameLike,
    interval: IntervalValue = 5,
    precision: Optional[TimestampPrecision] = None,
    interval_scale: str = "m",
) -> DataFrameLike:
    """Aggregate trades into klines for a fixed time interval.

    Trades are sorted by timestamp and stable sequence id before open/close
    prices are selected. Intervals with no trades are not included, matching
    exchange historical exports.

    Args:
        trades: DataFrame or LazyFrame with normalized trade columns
            (``timestamp``, ``price``, ``quantity``, ``quote_quantity``,
            ``sequence_id``) or legacy columns (``time``, ``price``, ``qty``,
            ``quote_qty``, ``trade_id``).
        interval: Fixed-duration interval. Use a compact string such as
            ``"2.5m"`` or a number with ``interval_scale``.
        interval_scale: Unit for numeric intervals. Supports seconds,
            minutes, hours, days, and weeks using common aliases such as
            ``"s"``, ``"m"``, ``"h"``, ``"d"``, and ``"w"``.
        precision: Timestamp precision of the timestamp column. If ``None``,
            it is auto-detected from integer timestamp magnitude.

    Returns:
        Klines sorted by ``open_time``. A DataFrame input returns a DataFrame;
        a LazyFrame input returns a LazyFrame.
    """
    if not isinstance(trades, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("trades must be a polars DataFrame or LazyFrame")

    normalized = normalize_trades(trades)
    timestamp_col = "timestamp"

    if precision is None:
        precision = _detect_precision(normalized, timestamp_col)

    interval_ticks = interval_to_unit(interval, precision, interval_scale)
    t = _to_lazy(normalized).with_row_index("_source_row")
    t = t.sort(["timestamp", "sequence_id", "_source_row"]).with_columns(
        (pl.col("timestamp") // interval_ticks * interval_ticks).alias("open_time")
    )

    result = t.group_by("open_time", maintain_order=True).agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("quantity").sum().alias("volume"),
        pl.col("quote_quantity").sum().alias("quote_volume"),
        pl.len().cast(pl.Int64).alias("num_trades"),
        _taker_buy_sum("quantity").alias("taker_buy_base_volume"),
        _taker_buy_sum("quote_quantity").alias("taker_buy_quote_volume"),
    ])

    result = result.with_columns([
        (pl.col("open_time") + interval_ticks - 1).alias("close_time"),
        pl.lit(0).cast(pl.Int64).alias("ignore"),
    ])

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    result = result.with_columns([
        pl.col(col).round(_ROUND_PLACES) for col in float_cols
    ])
    result = result.select(KLINES_CANONICAL).sort("open_time")
    return result if isinstance(trades, pl.LazyFrame) else result.collect()


def _to_lazy(data: DataFrameLike) -> pl.LazyFrame:
    """Represent a DataFrame or LazyFrame as a LazyFrame."""
    return data if isinstance(data, pl.LazyFrame) else data.lazy()


def _detect_precision(data: DataFrameLike, timestamp_col: str) -> TimestampPrecision:
    """Auto-detect timestamp precision from an integer timestamp column."""
    if isinstance(data, pl.DataFrame):
        return detect_timestamp_precision(data[timestamp_col].cast(pl.Int64))

    sample = (
        data.select(
            pl.col(timestamp_col).cast(pl.Int64).head(1_000).alias(timestamp_col)
        )
        .collect()
        .get_column(timestamp_col)
    )
    return detect_timestamp_precision(sample)


def _taker_buy_sum(value_col: str) -> pl.Expr:
    """Aggregate taker-buy volume, returning null when side is unknown."""
    taker_buy_expr = pl.col("is_buyer_maker") == False  # noqa: E712
    return (
        pl.when(pl.col("is_buyer_maker").is_null().any())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(
            pl.when(taker_buy_expr)
            .then(pl.col(value_col))
            .otherwise(0.0)
            .sum()
        )
    )
