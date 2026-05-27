"""
VortexKit Aggregation Module
=============================

data aggregation for financial markets data.

Supports:
- Loading headerless CSV files from exchanges
- Normalizing loaded trade datasets into exchange-neutral kline input
- Aggregating raw trades → aggTrades (compressed trades)
- Aggregating raw trades → klines (OHLCV candles) at fixed intervals

All timestamps are handled as unix-based integers in their native precision without any assumptions about the unit
(seconds, milliseconds, or microseconds). No forced conversion is performed
data stays in its original unit throughout the pipeline.

Quick start::

    import polars as pl
    from aggregation import TradeColumnMapping, normalize_trades, aggregate_klines

    raw = pl.read_parquet("dataset/trades.parquet")
    trades = normalize_trades(raw, TradeColumnMapping(
        timestamp="timestamp",
        price="price",
        quantity="volume",
    ))
    klines_5m = aggregate_klines(trades, interval="5m")
    klines_20s = aggregate_klines(trades, interval=20, interval_scale="s")
    klines_1h = aggregate_klines(trades, interval="1h")
"""

from .agg_trades import aggregate_trades
from .klines import aggregate_klines
from .loaders import load_agg_trades, load_klines, load_trades
from .schemas import (
    AGG_TRADES_CANONICAL,
    BINANCE_SCHEMA,
    BYBIT_SCHEMA,
    KLINES_CANONICAL,
    NORMALIZED_TRADES_CANONICAL,
    TRADES_CANONICAL,
    ExchangeSchema,
    TradeColumnMapping,
    get_schema,
    normalize_trades,
    register_schema,
)
from .utils import (
    TimestampPrecision,
    add_datetime_column,
    align_to_interval,
    detect_timestamp_precision,
    interval_to_microseconds,
    interval_to_unit,
)

__all__ = [
    # Loaders
    "load_trades",
    "load_agg_trades",
    "load_klines",
    # Aggregation
    "aggregate_trades",
    "aggregate_klines",
    "normalize_trades",
    "TradeColumnMapping",
    "NORMALIZED_TRADES_CANONICAL",
    # Schemas
    "ExchangeSchema",
    "BINANCE_SCHEMA",
    "BYBIT_SCHEMA",
    "TRADES_CANONICAL",
    "AGG_TRADES_CANONICAL",
    "KLINES_CANONICAL",
    "get_schema",
    "register_schema",
    # Utils
    "TimestampPrecision",
    "add_datetime_column",
    "detect_timestamp_precision",
    "interval_to_microseconds",
    "interval_to_unit",
    "align_to_interval",
]
