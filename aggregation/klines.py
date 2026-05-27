"""
Aggregate trades into klines (candlestick / OHLCV data).

Klines (also called candles or OHLCV bars) are produced by bucketing
trades into fixed time intervals. This module supports fixed-duration
intervals expressible as Binance-style strings, such as ``"1m"``,
``"5m"``, ``"1h"``, and ``"1d"``.

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

from typing import Literal, Optional, Union, overload

import polars as pl

from .schemas import KLINES_CANONICAL, TradeColumnMapping, normalize_trades
from .utils import TimestampPrecision, detect_timestamp_precision, interval_to_unit


DataFrameLike = Union[pl.DataFrame, pl.LazyFrame]

# Binance stores all financial values with 8 decimal places. Rounding avoids
# visible IEEE 754 accumulation artifacts after summing many small trades.
_ROUND_PLACES = 8


def aggregate_klines(
    trades: DataFrameLike,
    interval: str = "5m",
    precision: Optional[TimestampPrecision] = None,
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
        interval: Fixed-duration interval string (for example ``"1m"``,
            ``"5m"``, ``"1h"``, ``"1d"``, ``"1w"``). Calendar-month
            intervals are rejected because their duration is not fixed.
        precision: Timestamp precision of the timestamp column. If ``None``,
            it is auto-detected from integer timestamp magnitude.

    Returns:
        Klines sorted by ``open_time``. A DataFrame input returns a DataFrame;
        a LazyFrame input returns a LazyFrame.
    """
    if not isinstance(trades, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("trades must be a polars DataFrame or LazyFrame")

    schema = _collect_schema(trades)
    if not _has_kline_trade_columns(schema):
        trades = normalize_trades(trades, _infer_kline_mapping(schema))
        schema = _collect_schema(trades)

    cols = _resolve_trade_columns(schema)

    if precision is None:
        precision = _detect_precision(trades, cols.timestamp)

    interval_ticks = interval_to_unit(interval, precision)
    t = _canonicalize_trade_columns(trades, schema, cols)
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


class _TradeColumns:
    """Resolved source columns for kline aggregation."""

    def __init__(
        self,
        *,
        timestamp: str,
        price: str,
        quantity: str,
        quote_quantity: str | None,
        sequence_id: str | None,
        is_buyer_maker: str | None,
    ) -> None:
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity
        self.quote_quantity = quote_quantity
        self.sequence_id = sequence_id
        self.is_buyer_maker = is_buyer_maker


def _collect_schema(data: DataFrameLike) -> pl.Schema:
    """Return a schema without collecting full lazy input."""
    return data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema


def _to_lazy(data: DataFrameLike) -> pl.LazyFrame:
    """Represent a DataFrame or LazyFrame as a LazyFrame."""
    return data if isinstance(data, pl.LazyFrame) else data.lazy()


def _has_kline_trade_columns(schema: pl.Schema) -> bool:
    """Return true when data is already normalized or legacy-canonical."""
    columns = set(schema.keys())
    return (
        _first_available(columns, "timestamp", "time", required=False) is not None
        and "price" in columns
        and _first_available(columns, "quantity", "qty", required=False) is not None
    )


def _infer_kline_mapping(schema: pl.Schema) -> TradeColumnMapping:
    """Infer an accurate mapping for common raw trade schemas."""
    columns = list(schema.keys())
    timestamp = _match_column(columns, ["timestamp", "time", "ts"], "timestamp")
    price = _match_column(columns, ["price", "tradePrice", "execPrice"], "price")

    # homeNotional/foreignNotional is an exchange-native base/quote pair.
    # Prefer it over size because size may be contracts on some venues.
    if _has_columns(columns, "homeNotional", "foreignNotional"):
        quantity = _match_column(columns, ["homeNotional"], "quantity")
        quote_quantity = _match_column(columns, ["foreignNotional"], "quote_quantity")
    else:
        quantity = _match_column(
            columns,
            ["quantity", "qty", "volume", "baseQuantity", "baseQty", "size"],
            "quantity",
        )
        quote_quantity = _match_column(
            columns,
            [
                "quote_quantity",
                "quote_qty",
                "quoteQty",
                "quoteVolume",
                "quoteNotional",
                "foreignNotional",
                "execValue",
                "notional",
            ],
            "quote_quantity",
            required=False,
        )

    side = _match_column(
        columns,
        ["is_buyer_maker", "isBuyerMaker", "side", "takerSide", "tradeSide"],
        "side",
        required=False,
    )
    trade_id = _match_column(
        columns,
        ["trade_id", "trade Id", "tradeId", "id", "execId", "trdMatchID"],
        "trade_id",
        required=False,
    )
    symbol = _match_column(
        columns,
        ["symbol", "pair", "instrument", "instrumentName"],
        "symbol",
        required=False,
    )
    side_semantics = (
        "is_buyer_maker"
        if side is not None and _normalize_name(side) in {"isbuyermaker"}
        else "taker"
    )

    return TradeColumnMapping(
        timestamp=timestamp,
        price=price,
        quantity=quantity,
        quote_quantity=quote_quantity,
        side=side,
        trade_id=trade_id,
        symbol=symbol,
        side_semantics=side_semantics,
    )


def _resolve_trade_columns(schema: pl.Schema) -> _TradeColumns:
    """Resolve normalized and legacy trade column names."""
    columns = set(schema.keys())

    return _TradeColumns(
        timestamp=_first_available(columns, "timestamp", "time"),
        price=_first_available(columns, "price"),
        quantity=_first_available(columns, "quantity", "qty"),
        quote_quantity=_first_available(
            columns,
            "quote_quantity",
            "quote_qty",
            required=False,
        ),
        sequence_id=_first_available(
            columns,
            "sequence_id",
            "trade_id",
            required=False,
        ),
        is_buyer_maker=_first_available(
            columns,
            "is_buyer_maker",
            required=False,
        ),
    )


@overload
def _first_available(
    columns: set[str],
    *candidates: str,
    required: Literal[True] = True,
) -> str:
    ...


@overload
def _first_available(
    columns: set[str],
    *candidates: str,
    required: Literal[False],
) -> str | None:
    ...


def _first_available(
    columns: set[str],
    *candidates: str,
    required: bool = True,
) -> str | None:
    """Return the first present candidate column."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        names = ", ".join(candidates)
        raise ValueError(f"Trades data is missing required column: one of {names}")
    return None


@overload
def _match_column(
    columns: list[str],
    aliases: list[str],
    field_name: str,
    *,
    required: Literal[True] = True,
) -> str:
    ...


@overload
def _match_column(
    columns: list[str],
    aliases: list[str],
    field_name: str,
    *,
    required: Literal[False],
) -> str | None:
    ...


def _match_column(
    columns: list[str],
    aliases: list[str],
    field_name: str,
    *,
    required: bool = True,
) -> str | None:
    """Find one unambiguous source column by normalized aliases."""
    normalized_aliases = {_normalize_name(alias) for alias in aliases}
    matches = [col for col in columns if _normalize_name(col) in normalized_aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Raw trades data has ambiguous {field_name} columns {matches}; "
            "call normalize_trades with an explicit TradeColumnMapping."
        )
    if required:
        names = ", ".join(aliases)
        raise ValueError(f"Raw trades data is missing required {field_name}: {names}")
    return None


def _has_columns(columns: list[str], *required: str) -> bool:
    """Return true when all required normalized column names are present."""
    normalized_columns = {_normalize_name(col) for col in columns}
    return all(_normalize_name(col) in normalized_columns for col in required)


def _normalize_name(name: str) -> str:
    """Normalize column names for source-schema matching."""
    return "".join(char for char in name.lower() if char.isalnum())


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


def _canonicalize_trade_columns(
    trades: DataFrameLike,
    schema: pl.Schema,
    cols: _TradeColumns,
) -> pl.LazyFrame:
    """Build the canonical expressions used by the kline aggregator."""
    source = _to_lazy(trades).with_row_index("_source_row")

    timestamp_expr = pl.col(cols.timestamp).cast(pl.Int64)
    price_expr = pl.col(cols.price).cast(pl.Float64)
    quantity_expr = pl.col(cols.quantity).cast(pl.Float64)
    quote_expr = (
        pl.col(cols.quote_quantity).cast(pl.Float64)
        if cols.quote_quantity is not None
        else price_expr * quantity_expr
    )

    if cols.sequence_id is not None and schema[cols.sequence_id].is_integer():
        sequence_expr = pl.col(cols.sequence_id).cast(pl.Int64)
    else:
        sequence_expr = pl.col("_source_row").cast(pl.Int64)

    side_expr = (
        pl.col(cols.is_buyer_maker).cast(pl.Boolean)
        if cols.is_buyer_maker is not None
        else pl.lit(None, dtype=pl.Boolean)
    )

    return source.select(
        [
            timestamp_expr.alias("timestamp"),
            price_expr.alias("price"),
            quantity_expr.alias("quantity"),
            quote_expr.alias("quote_quantity"),
            sequence_expr.alias("sequence_id"),
            side_expr.alias("is_buyer_maker"),
            pl.col("_source_row"),
        ]
    )


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
