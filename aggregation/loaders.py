"""
CSV loaders for exchange trade data.

Loads headerless CSV files produced by Binance (and other exchanges) into
polars DataFrames with canonical column names and correct dtypes.

Timestamps are **preserved in their native precision** — no conversion
is performed. If the source file uses microseconds, the DataFrame will
contain microseconds; if milliseconds, it stays in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import polars as pl

from .schemas import (
    AGG_TRADES_CANONICAL,
    AGG_TRADES_DTYPES,
    KLINES_CANONICAL,
    KLINES_DTYPES,
    TRADES_CANONICAL,
    TRADES_DTYPES,
    ExchangeSchema,
    get_schema,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _apply_schema(
    df: pl.DataFrame,
    canonical_names: list[str],
    dtype_map: dict[str, pl.DataType],
    schema: ExchangeSchema,
) -> pl.DataFrame:
    """Rename columns to canonical names and cast dtypes."""
    # Build rename map: position-based (headerless CSVs)
    rename_map = {f"column_{i+1}": name for i, name in enumerate(canonical_names)}
    df = df.rename(rename_map)

    # Cast dtypes (only columns that exist).
    df = df.with_columns(
        pl.col(col).cast(dtype).alias(col)
        for col, dtype in dtype_map.items()
        if col in df.columns
    )

    # Reorder columns to canonical order
    df = df.select([c for c in canonical_names if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_trades(
    path: Union[str, Path],
    exchange: str = "binance",
) -> pl.DataFrame:
    """Load a trades CSV file into a polars DataFrame.

    The CSV is assumed to be **headerless** with columns in the exchange's
    standard order. Columns are renamed to the canonical names defined in
    :mod:`aggregation.schemas`. Timestamps are preserved in their native
    precision (no conversion).

    Args:
        path: Path to the trades CSV file.
        exchange: Exchange identifier (default ``"binance"``).

    Returns:
        Polars DataFrame with canonical trades columns.
    """
    schema = get_schema(exchange)
    df = pl.read_csv(str(path), has_header=False)
    return _apply_schema(df, TRADES_CANONICAL, TRADES_DTYPES, schema)


def load_agg_trades(
    path: Union[str, Path],
    exchange: str = "binance",
) -> pl.DataFrame:
    """Load an aggTrades CSV file into a polars DataFrame.

    Timestamps are preserved in their native precision.

    Args:
        path: Path to the aggTrades CSV file.
        exchange: Exchange identifier (default ``"binance"``).

    Returns:
        Polars DataFrame with canonical aggTrades columns.
    """
    schema = get_schema(exchange)
    df = pl.read_csv(str(path), has_header=False)
    return _apply_schema(df, AGG_TRADES_CANONICAL, AGG_TRADES_DTYPES, schema)


def load_klines(
    path: Union[str, Path],
    exchange: str = "binance",
) -> pl.DataFrame:
    """Load a klines CSV file into a polars DataFrame.

    Timestamps are preserved in their native precision.

    Args:
        path: Path to the klines CSV file.
        exchange: Exchange identifier (default ``"binance"``).

    Returns:
        Polars DataFrame with canonical klines columns.
    """
    schema = get_schema(exchange)
    df = pl.read_csv(str(path), has_header=False)
    return _apply_schema(df, KLINES_CANONICAL, KLINES_DTYPES, schema)
