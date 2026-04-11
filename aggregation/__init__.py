"""
VortexKit Aggregation Module
=============================

Institutional-grade trade data aggregation for cryptocurrency markets.

Supports:
- Loading headerless CSV files from Binance (and other exchanges)
- Aggregating raw trades → aggTrades (compressed trades)
- Aggregating raw trades → klines (OHLCV candles) at any interval

All timestamps are handled as unix-based integers in their **native precision**
(seconds, milliseconds, or microseconds). No forced conversion is performed —
data stays in its original unit throughout the pipeline.

Quick start::

    from aggregation import load_trades, aggregate_trades, aggregate_klines

    trades = load_trades("dataset/BTCUSDT-trades-2026-04-10.csv")
    agg = aggregate_trades(trades)
    klines_5m = aggregate_klines(trades, interval="5m")
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
    TRADES_CANONICAL,
    ExchangeSchema,
    get_schema,
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
