"""
Schema definitions for exchange trade data.

Each exchange has different column names and types for their trade data.
This module provides schema mappings so the aggregation module can work
with data from any supported exchange.

All schemas define columns in the order they appear in the source CSV files
(headerless). Timestamps are always stored as Int64 (unix-based).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import polars as pl


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
    "trade_id": pl.Int64,
    "price": pl.Float64,
    "qty": pl.Float64,
    "quote_qty": pl.Float64,
    "time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
    "is_best_match": pl.Boolean,
}

AGG_TRADES_DTYPES: Dict[str, pl.DataType] = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "timestamp": pl.Int64,
    "was_buyer_maker": pl.Boolean,
    "was_best_match": pl.Boolean,
}

KLINES_DTYPES: Dict[str, pl.DataType] = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "num_trades": pl.Int64,
    "taker_buy_base_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Int64,
}


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
