"""
VortexKit Aggregation Module
=============================

data aggregation for financial markets data.

Supports:
- Normalizing Polars trade DataFrames into exchange-neutral kline input
- Aggregating raw trades → aggTrades (compressed trades)
- Aggregating trades → klines (OHLCV candles) at fixed custom intervals

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
from .schemas import (
    AGG_TRADES_CANONICAL,
    KLINES_CANONICAL,
    NORMALIZED_TRADES_CANONICAL,
    TradeColumnMapping,
    normalize_trades,
)
from .utils import (
    TimestampPrecision,
    detect_timestamp_precision,
    interval_to_unit,
)

__all__ = [
    # Aggregation
    "aggregate_trades",
    "aggregate_klines",
    "normalize_trades",
    "TradeColumnMapping",
    "NORMALIZED_TRADES_CANONICAL",
    # Schemas
    "AGG_TRADES_CANONICAL",
    "KLINES_CANONICAL",
    # Utils
    "TimestampPrecision",
    "detect_timestamp_precision",
    "interval_to_unit",
]
