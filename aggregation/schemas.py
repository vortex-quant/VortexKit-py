"""
Schema definitions for exchange trade data.

Each exchange has different column names and types for their trade data.
This module provides schema mappings so the aggregation module can work
with data from any supported exchange.

All schemas define columns in the order they appear in the source CSV files
(headerless). Timestamps are always stored as Int64 (unix-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Union, cast, overload

import polars as pl

from .utils import TimestampPrecision


# ---------------------------------------------------------------------------
# Canonical column names used internally by the aggregation module
# ---------------------------------------------------------------------------
TRADES_CANONICAL: List[str] = [
    "trade_id", "price", "qty", "quote_qty", "time",
    "is_buyer_maker", "is_best_match",
]
"""Canonical trades columns (Binance-style order)."""

AGG_TRADES_CANONICAL: List[str] = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "timestamp", "was_buyer_maker", "was_best_match",
]
"""Canonical aggTrades columns (Binance-style order)."""

KLINES_CANONICAL: List[str] = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]
"""Canonical klines columns (Binance-style order)."""


# ---------------------------------------------------------------------------
# Polars dtype maps for canonical schemas
# ---------------------------------------------------------------------------
TRADES_DTYPES: Dict[str, pl.DataType] = {
    "trade_id": pl.Int64(),
    "price": pl.Float64(),
    "qty": pl.Float64(),
    "quote_qty": pl.Float64(),
    "time": pl.Int64(),
    "is_buyer_maker": pl.Boolean(),
    "is_best_match": pl.Boolean(),
}

AGG_TRADES_DTYPES: Dict[str, pl.DataType] = {
    "agg_trade_id": pl.Int64(),
    "price": pl.Float64(),
    "quantity": pl.Float64(),
    "first_trade_id": pl.Int64(),
    "last_trade_id": pl.Int64(),
    "timestamp": pl.Int64(),
    "was_buyer_maker": pl.Boolean(),
    "was_best_match": pl.Boolean(),
}

KLINES_DTYPES: Dict[str, pl.DataType] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": pl.Int64(),
    "quote_volume": pl.Float64(),
    "num_trades": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
    "ignore": pl.Int64(),
}


# ---------------------------------------------------------------------------
# Normalized trade schema for exchange-neutral kline aggregation
# ---------------------------------------------------------------------------
DataFrameLike = Union[pl.DataFrame, pl.LazyFrame]
TimestampUnit = Union[
    Literal[
        "auto",
        "s",
        "sec",
        "second",
        "seconds",
        "ms",
        "millisecond",
        "milliseconds",
        "us",
        "microsecond",
        "microseconds",
    ],
    TimestampPrecision,
]
SideSemantics = Literal["taker", "maker", "is_buyer_maker"]

NORMALIZED_TRADES_CANONICAL: List[str] = [
    "timestamp",
    "price",
    "quantity",
    "quote_quantity",
    "sequence_id",
]
"""Required canonical columns produced by :func:`normalize_trades`."""

_BUY_VALUES = ["buy", "buyer", "b"]
_SELL_VALUES = ["sell", "seller", "s"]
_TRUE_VALUES = ["true", "1", "yes", "y", "t"]
_FALSE_VALUES = ["false", "0", "no", "n", "f"]

_FLOAT_TIMESTAMP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TradeColumnMapping:
    """Map source trade columns into the normalized trade schema.

    Args:
        timestamp: Numeric Unix timestamp column. Integer timestamps are
            preserved in their native unit. Float timestamps are interpreted
            as Unix seconds when ``timestamp_unit="auto"`` and converted to
            the smallest lossless integer unit.
        price: Trade price column.
        quantity: Base-asset trade quantity column.
        quote_quantity: Optional quote-asset quantity/notional column. When
            absent, quote quantity is derived as ``price * quantity``.
        side: Optional side column. By default this is interpreted as taker
            side, where ``Buy`` means the buyer was taker and ``Sell`` means
            the buyer was maker.
        trade_id: Optional source trade id. Numeric ids are used as stable
            sequence ids; otherwise row order is used.
        symbol: Optional symbol column to preserve in normalized output.
        side_semantics: Meaning of ``side``. Supported values are
            ``"taker"``, ``"maker"``, and ``"is_buyer_maker"``.
        timestamp_unit: ``"auto"`` or an explicit native integer timestamp
            unit (``"s"``, ``"ms"``, or ``"us"``).
    """

    timestamp: str
    price: str
    quantity: str
    quote_quantity: str | None = None
    side: str | None = None
    trade_id: str | None = None
    symbol: str | None = None
    side_semantics: SideSemantics = "taker"
    timestamp_unit: TimestampUnit = "auto"


def normalize_trades(
    data: DataFrameLike,
    mapping: TradeColumnMapping | None = None,
    *,
    round_decimals: int | None = None,
) -> DataFrameLike:
    """Normalize loaded trade data for exchange-neutral kline aggregation.

    The function accepts a :class:`polars.DataFrame` or
    :class:`polars.LazyFrame` and returns the same kind of object. It does not
    read files; callers should use Polars readers such as ``read_parquet`` or
    ``scan_parquet`` first.

    Args:
        data: Source trade data already loaded with Polars.
        mapping: Explicit source-to-canonical column mapping. If omitted,
            common aliases are inferred only when each required field is
            unambiguous.
        round_decimals: Optional decimal precision applied to normalized
            floating-point price and quantity columns.

    Returns:
        Data with canonical columns: ``timestamp``, ``price``, ``quantity``,
        ``quote_quantity``, ``sequence_id``, plus optional ``is_buyer_maker``
        and ``symbol``.

    Raises:
        ValueError: If required columns are missing, aliases are ambiguous,
            timestamp precision cannot be preserved, or unsupported side /
            timestamp policies are requested.
        TypeError: If the timestamp column is not numeric.
    """
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("data must be a polars DataFrame or LazyFrame")

    if round_decimals is not None and round_decimals < 0:
        raise ValueError("round_decimals must be non-negative")

    schema = _collect_schema(data)
    mapping = _infer_mapping(schema) if mapping is None else mapping
    _validate_mapping(schema, mapping)

    timestamp_expr = _timestamp_expr(data, schema, mapping)
    price_expr = pl.col(mapping.price).cast(pl.Float64)
    quantity_expr = pl.col(mapping.quantity).cast(pl.Float64)
    quote_expr = (
        pl.col(mapping.quote_quantity).cast(pl.Float64)
        if mapping.quote_quantity is not None
        else price_expr * quantity_expr
    )
    sequence_expr = _sequence_expr(schema, mapping)

    source = _to_lazy(data).with_row_index("_source_row")
    columns = [
        timestamp_expr.alias("timestamp"),
        price_expr.alias("price"),
        quantity_expr.alias("quantity"),
        quote_expr.alias("quote_quantity"),
        sequence_expr.alias("sequence_id"),
    ]

    if mapping.side is not None:
        columns.append(_side_expr(schema, mapping).alias("is_buyer_maker"))
    if mapping.symbol is not None:
        columns.append(pl.col(mapping.symbol).cast(pl.String).alias("symbol"))

    normalized = source.select(columns)

    if round_decimals is not None:
        rounded_cols = ["price", "quantity", "quote_quantity"]
        normalized = normalized.with_columns(
            pl.col(col).round(round_decimals).alias(col) for col in rounded_cols
        )

    return normalized if isinstance(data, pl.LazyFrame) else normalized.collect()


def _collect_schema(data: DataFrameLike) -> pl.Schema:
    """Return a schema without collecting full lazy data."""
    return data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema


def _to_lazy(data: DataFrameLike) -> pl.LazyFrame:
    """Represent a DataFrame or LazyFrame as a LazyFrame."""
    return data if isinstance(data, pl.LazyFrame) else data.lazy()


def _infer_mapping(schema: pl.Schema) -> TradeColumnMapping:
    """Infer a conservative mapping from common column aliases."""
    columns = list(schema.keys())
    timestamp = _match_alias(columns, _TIMESTAMP_ALIASES, "timestamp", required=True)
    price = _match_alias(columns, _PRICE_ALIASES, "price", required=True)
    quantity = _match_alias(columns, _QUANTITY_ALIASES, "quantity", required=True)
    quote_quantity = _match_alias(columns, _QUOTE_QUANTITY_ALIASES, "quote_quantity")
    side = _match_alias(columns, _SIDE_ALIASES, "side")
    trade_id = _match_alias(columns, _TRADE_ID_ALIASES, "trade_id")
    symbol = _match_alias(columns, _SYMBOL_ALIASES, "symbol")

    side_semantics: SideSemantics = (
        "is_buyer_maker"
        if side is not None and _normalize_name(side) in _BUYER_MAKER_FLAG_ALIASES
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


@overload
def _match_alias(
    columns: List[str],
    aliases: set[str],
    field_name: str,
    *,
    required: Literal[True],
) -> str:
    ...


@overload
def _match_alias(
    columns: List[str],
    aliases: set[str],
    field_name: str,
    *,
    required: Literal[False] = False,
) -> str | None:
    ...


def _match_alias(
    columns: List[str],
    aliases: set[str],
    field_name: str,
    *,
    required: bool = False,
) -> str | None:
    """Find one unambiguous source column for a normalized field."""
    matches = [col for col in columns if _normalize_name(col) in aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {field_name} columns {matches}; pass an explicit "
            "TradeColumnMapping."
        )
    if required:
        raise ValueError(
            f"Missing required {field_name} column; pass an explicit "
            "TradeColumnMapping."
        )
    return None


def _validate_mapping(schema: pl.Schema, mapping: TradeColumnMapping) -> None:
    """Validate mapped columns and policy values before building expressions."""
    for field_name in ("timestamp", "price", "quantity"):
        column = getattr(mapping, field_name)
        if column not in schema:
            raise ValueError(f"Mapped {field_name} column '{column}' is missing")

    for field_name in ("quote_quantity", "side", "trade_id", "symbol"):
        column = getattr(mapping, field_name)
        if column is not None and column not in schema:
            raise ValueError(f"Mapped {field_name} column '{column}' is missing")

    if mapping.side_semantics not in ("taker", "maker", "is_buyer_maker"):
        raise ValueError(
            "side_semantics must be one of: 'taker', 'maker', 'is_buyer_maker'"
        )

    _normalize_timestamp_unit(mapping.timestamp_unit)


def _timestamp_expr(
    data: DataFrameLike,
    schema: pl.Schema,
    mapping: TradeColumnMapping,
) -> pl.Expr:
    """Build the normalized integer timestamp expression."""
    dtype = schema[mapping.timestamp]
    if not dtype.is_numeric():
        raise TypeError(
            f"Mapped timestamp column '{mapping.timestamp}' must be numeric; "
            "string-only timestamps are not supported."
        )

    timestamp_unit = _normalize_timestamp_unit(mapping.timestamp_unit)
    source = pl.col(mapping.timestamp)

    if timestamp_unit == "auto":
        if dtype.is_float():
            precision = _detect_float_second_precision(data, mapping.timestamp)
            multiplier = _precision_multiplier(precision)
            return (source * multiplier).round(0).cast(pl.Int64)
        return source.cast(pl.Int64)

    if dtype.is_float():
        _validate_float_integral(data, mapping.timestamp)
        return source.round(0).cast(pl.Int64)

    return source.cast(pl.Int64)


def _detect_float_second_precision(
    data: DataFrameLike,
    column: str,
) -> TimestampPrecision:
    """Detect the smallest integer unit that preserves float Unix seconds."""
    expr = pl.col(column)
    summary = _to_lazy(data).select(
        [
            (expr - expr.round(0)).abs().max().alias("seconds_error"),
            ((expr * 1_000) - (expr * 1_000).round(0)).abs().max().alias("ms_error"),
            ((expr * 1_000_000) - (expr * 1_000_000).round(0))
            .abs()
            .max()
            .alias("us_error"),
        ]
    ).collect()
    seconds_error = summary["seconds_error"][0]
    ms_error = summary["ms_error"][0]
    us_error = summary["us_error"][0]

    if seconds_error is None:
        raise ValueError(f"Cannot detect timestamp precision for empty column '{column}'")
    if seconds_error <= _FLOAT_TIMESTAMP_TOLERANCE:
        return TimestampPrecision.SECONDS
    if ms_error <= _FLOAT_TIMESTAMP_TOLERANCE:
        return TimestampPrecision.MILLISECONDS
    if us_error <= _FLOAT_TIMESTAMP_TOLERANCE:
        return TimestampPrecision.MICROSECONDS
    raise ValueError(
        f"Float timestamp column '{column}' cannot be represented losslessly "
        "at seconds, milliseconds, or microseconds precision."
    )


def _validate_float_integral(data: DataFrameLike, column: str) -> None:
    """Ensure explicitly unit-tagged float timestamps are integral."""
    expr = pl.col(column)
    max_error = _to_lazy(data).select((expr - expr.round(0)).abs().max()).collect()[0, 0]
    if max_error is None:
        raise ValueError(f"Cannot validate empty timestamp column '{column}'")
    if max_error > _FLOAT_TIMESTAMP_TOLERANCE:
        raise ValueError(
            f"Float timestamp column '{column}' contains fractional values; "
            "use timestamp_unit='auto' for float Unix seconds."
        )


def _sequence_expr(schema: pl.Schema, mapping: TradeColumnMapping) -> pl.Expr:
    """Use numeric trade ids when available; otherwise preserve row order."""
    if mapping.trade_id is not None and schema[mapping.trade_id].is_integer():
        return pl.col(mapping.trade_id).cast(pl.Int64)
    return pl.col("_source_row").cast(pl.Int64)


def _side_expr(schema: pl.Schema, mapping: TradeColumnMapping) -> pl.Expr:
    """Convert side representations into Binance-style buyer-maker booleans."""
    if mapping.side is None:
        raise ValueError("mapping.side is required to build a side expression")

    dtype = schema[mapping.side]
    raw = pl.col(mapping.side)

    if mapping.side_semantics == "is_buyer_maker" and dtype == pl.Boolean:
        return raw.cast(pl.Boolean)

    value = raw.cast(pl.String).str.strip_chars().str.to_lowercase()

    if mapping.side_semantics == "is_buyer_maker":
        return (
            pl.when(value.is_in(_TRUE_VALUES))
            .then(pl.lit(True))
            .when(value.is_in(_FALSE_VALUES))
            .then(pl.lit(False))
            .otherwise(pl.lit(None))
            .cast(pl.Boolean)
        )

    if mapping.side_semantics == "taker":
        buy_value = False
        sell_value = True
    else:
        buy_value = True
        sell_value = False

    return (
        pl.when(value.is_in(_BUY_VALUES))
        .then(pl.lit(buy_value))
        .when(value.is_in(_SELL_VALUES))
        .then(pl.lit(sell_value))
        .otherwise(pl.lit(None))
        .cast(pl.Boolean)
    )


def _precision_multiplier(precision: TimestampPrecision) -> int:
    """Return the integer multiplier from seconds to *precision*."""
    if precision == TimestampPrecision.SECONDS:
        return 1
    if precision == TimestampPrecision.MILLISECONDS:
        return 1_000
    if precision == TimestampPrecision.MICROSECONDS:
        return 1_000_000
    raise ValueError(f"Unsupported timestamp precision: {precision}")


def _normalize_timestamp_unit(unit: TimestampUnit) -> Literal["auto", "s", "ms", "us"]:
    """Normalize timestamp unit aliases used by mappings."""
    if isinstance(unit, TimestampPrecision):
        return cast(Literal["s", "ms", "us"], unit.value)

    normalized = str(unit).lower()
    aliases: dict[str, Literal["auto", "s", "ms", "us"]] = {
        "auto": "auto",
        "s": "s",
        "sec": "s",
        "second": "s",
        "seconds": "s",
        "ms": "ms",
        "millisecond": "ms",
        "milliseconds": "ms",
        "us": "us",
        "microsecond": "us",
        "microseconds": "us",
    }
    if normalized not in aliases:
        raise ValueError("timestamp_unit must be 'auto', 's', 'ms', or 'us'")
    return aliases[normalized]


def _normalize_name(name: str) -> str:
    """Normalize source names for conservative alias matching."""
    return "".join(char for char in name.lower() if char.isalnum())


def _alias_set(*aliases: str) -> set[str]:
    """Create a normalized alias set."""
    return {_normalize_name(alias) for alias in aliases}


_TIMESTAMP_ALIASES = _alias_set(
    "timestamp",
    "time",
    "ts",
    "execTime",
    "tradeTime",
    "transactTime",
)
_PRICE_ALIASES = _alias_set("price", "tradePrice", "execPrice")
_QUANTITY_ALIASES = _alias_set(
    "quantity",
    "qty",
    "volume",
    "size",
    "baseQuantity",
    "baseQty",
    "homeNotional",
    "execQty",
    "amount",
)
_QUOTE_QUANTITY_ALIASES = _alias_set(
    "quote_quantity",
    "quoteQty",
    "quoteVolume",
    "quoteNotional",
    "foreignNotional",
    "execValue",
    "notional",
)
_SIDE_ALIASES = _alias_set(
    "side",
    "takerSide",
    "tradeSide",
    "is_buyer_maker",
    "isBuyerMaker",
    "was_buyer_maker",
    "wasBuyerMaker",
)
_BUYER_MAKER_FLAG_ALIASES = _alias_set(
    "is_buyer_maker",
    "isBuyerMaker",
    "was_buyer_maker",
    "wasBuyerMaker",
)
_TRADE_ID_ALIASES = _alias_set(
    "trade_id",
    "trade Id",
    "tradeId",
    "id",
    "execId",
    "trdMatchID",
)
_SYMBOL_ALIASES = _alias_set("symbol", "pair", "instrument", "instrumentName")


# ---------------------------------------------------------------------------
# Exchange schema definition
# ---------------------------------------------------------------------------
@dataclass
class ExchangeSchema:
    """Column mapping for a specific exchange.

    Attributes:
        name: Exchange identifier (e.g. ``"binance"``, ``"bybit"``).
        trades_columns: Column names in the source CSV, ordered left-to-right.
            Must map 1:1 to :data:`TRADES_CANONICAL`.
        agg_trades_columns: Column names in the source CSV, ordered left-to-right.
            Must map 1:1 to :data:`AGG_TRADES_CANONICAL`.
        klines_columns: Column names in the source CSV, ordered left-to-right.
            Must map 1:1 to :data:`KLINES_CANONICAL`.
    """

    name: str
    trades_columns: List[str]
    agg_trades_columns: List[str]
    klines_columns: List[str]

    def trades_rename_map(self) -> Dict[str, str]:
        """Mapping from source column names → canonical trades names."""
        return dict(zip(self.trades_columns, TRADES_CANONICAL))

    def agg_trades_rename_map(self) -> Dict[str, str]:
        """Mapping from source column names → canonical aggTrades names."""
        return dict(zip(self.agg_trades_columns, AGG_TRADES_CANONICAL))

    def klines_rename_map(self) -> Dict[str, str]:
        """Mapping from source column names → canonical klines names."""
        return dict(zip(self.klines_columns, KLINES_CANONICAL))


# ---------------------------------------------------------------------------
# Pre-defined exchange schemas
# ---------------------------------------------------------------------------
BINANCE_SCHEMA = ExchangeSchema(
    name="binance",
    trades_columns=[
        "trade Id", "price", "qty", "quoteQty", "time",
        "isBuyerMaker", "isBestMatch",
    ],
    agg_trades_columns=[
        "Aggregate tradeId", "Price", "Quantity", "First tradeId", "Last tradeId",
        "Timestamp", "Was the buyer the maker", "Was the trade the best price match",
    ],
    klines_columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
        "Quote asset volume", "Number of trades", "Taker buy base asset volume",
        "Taker buy quote asset volume", "Ignore",
    ],
)

# sample schema — placeholder for future use.
# Column names based on Bybit V5 API trade / kline fields.
BYBIT_SCHEMA = ExchangeSchema(
    name="bybit",
    trades_columns=[
        "execId", "execPrice", "execQty", "execValue", "execTime",
        "isMaker", "isBestMatch",
    ],
    agg_trades_columns=[
        "aggId", "execPrice", "execQty", "firstExecId", "lastExecId",
        "execTime", "isMaker", "isBestMatch",
    ],
    klines_columns=[
        "startTime", "open", "high", "low", "close", "volume", "endTime",
        "quoteVolume", "count", "takerBuyBaseVol", "takerBuyQuoteVol", "ignore",
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_EXCHANGE_REGISTRY: Dict[str, ExchangeSchema] = {
    "binance": BINANCE_SCHEMA,
    "bybit": BYBIT_SCHEMA,
}


def get_schema(exchange: str) -> ExchangeSchema:
    """Look up an exchange schema by name.

    Args:
        exchange: Case-insensitive exchange identifier.

    Returns:
        The matching :class:`ExchangeSchema`.

    Raises:
        ValueError: If the exchange is not registered.
    """
    key = exchange.lower()
    if key not in _EXCHANGE_REGISTRY:
        available = ", ".join(sorted(_EXCHANGE_REGISTRY))
        raise ValueError(f"Unknown exchange '{exchange}'. Available: {available}")
    return _EXCHANGE_REGISTRY[key]


def register_schema(schema: ExchangeSchema) -> None:
    """Register a custom exchange schema.

    Args:
        schema: The schema to register. Overwrites any existing entry
            with the same ``name``.
    """
    _EXCHANGE_REGISTRY[schema.name.lower()] = schema
