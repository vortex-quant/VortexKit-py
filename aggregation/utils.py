"""
Timestamp and interval utilities for trade data aggregation.

All timestamps in this module are unix-based integers. Different exchanges
and data sources use different precisions:

- **Seconds** (10-digit, e.g. ``1735689600``)
- **Milliseconds** (13-digit, e.g. ``1735689600108``)
- **Microseconds** (16-digit, e.g. ``1735689600010866``)

This module detects and preserves the native precision of the data.
No forced conversion is performed — data stays in its original unit
throughout the entire pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, cast

import polars as pl

# EST (US Eastern) fixed offset: UTC-5.
# Note: this does **not** account for DST (EDT = UTC-4).
# For DST-aware conversion use a proper timezone database.
_EST_OFFSET_HOURS = -5


# ---------------------------------------------------------------------------
# Timestamp precision
# ---------------------------------------------------------------------------
class TimestampPrecision(Enum):
    """Unix timestamp precision."""

    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"


# Multiplier to convert 1 second into each precision unit.
_SECONDS_PER_UNIT: dict[TimestampPrecision, int] = {
    TimestampPrecision.SECONDS: 1,
    TimestampPrecision.MILLISECONDS: 1_000,
    TimestampPrecision.MICROSECONDS: 1_000_000,
}


def detect_timestamp_precision(
    series: pl.Series,
    sample_size: int = 1000,
) -> TimestampPrecision:
    """Detect whether timestamps should be handled as seconds, ms, or us.

    Integer timestamps are detected by median value magnitude:

    - **Seconds**: ~1.77 × 10⁹ (10 digits)
    - **Milliseconds**: ~1.77 × 10¹² (13 digits)
    - **Microseconds**: ~1.77 × 10¹⁵ (16 digits)

    Float timestamps in Unix seconds are additionally checked for fractional
    precision. For example, ``1777852800.082`` is stored in seconds but must
    be handled at millisecond precision, while ``1779235200.2682`` requires
    microsecond precision.

    Args:
        series: A numeric Polars series of Unix timestamps.
        sample_size: Number of rows to sample for detection.

    Returns:
        The detected :class:`TimestampPrecision`.

    Raises:
        ValueError: If the series is empty or values are unexpected.
    """
    if series.is_empty():
        raise ValueError("Cannot detect precision on an empty series")

    n = min(sample_size, len(series))
    sampled = series.sample(n, seed=42) if len(series) > n else series
    median = sampled.median()
    if median is None:
        raise ValueError("Cannot detect precision on a timestamp series with no median")
    median_val = float(cast(int | float, median))

    if median_val <= 0:
        raise ValueError(f"Unexpected timestamp values (median={median_val})")

    if series.dtype.is_float() and 1e8 <= median_val < 1e11:
        return _detect_fractional_second_precision(sampled)

    # 10^14 ≈ year 5138 in ms, so anything >= 10^14 must be microseconds
    if median_val >= 1e14:
        return TimestampPrecision.MICROSECONDS
    elif median_val >= 1e11:
        return TimestampPrecision.MILLISECONDS
    elif median_val >= 1e8:
        return TimestampPrecision.SECONDS
    else:
        raise ValueError(f"Cannot determine timestamp precision (median={median_val})")


def _detect_fractional_second_precision(series: pl.Series) -> TimestampPrecision:
    """Find the smallest integer unit that preserves float Unix seconds."""
    if _max_scaled_rounding_error(series, 1) <= 1e-6:
        return TimestampPrecision.SECONDS
    if _max_scaled_rounding_error(series, 1_000) <= 1e-6:
        return TimestampPrecision.MILLISECONDS
    if _max_scaled_rounding_error(series, 1_000_000) <= 1e-6:
        return TimestampPrecision.MICROSECONDS
    raise ValueError(
        "Float timestamp values cannot be represented losslessly at seconds, "
        "milliseconds, or microseconds precision."
    )


def _max_scaled_rounding_error(series: pl.Series, multiplier: int) -> float:
    """Return max error after scaling a float timestamp series to integers."""
    scaled = series.cast(pl.Float64) * multiplier
    error = (scaled - scaled.round(0)).abs().max()
    if error is None:
        raise ValueError("Cannot detect precision on a timestamp series with no data")
    return float(cast(int | float, error))


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------
def _parse_interval_seconds(interval: str) -> int:
    """Parse a fixed-duration interval string into seconds.

    Supported formats: ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``,
    ``"1d"``, and ``"1w"``. Calendar-month intervals are intentionally not
    supported because their duration depends on the calendar month.

    Args:
        interval: Interval string (e.g. ``"5m"``, ``"1h"``).

    Returns:
        Interval duration in seconds.

    Raises:
        ValueError: If the interval format is not recognised.
    """
    units = {
        "m": 60,          # seconds per minute
        "h": 3_600,       # seconds per hour
        "d": 86_400,      # seconds per day
        "w": 604_800,     # seconds per week
    }

    if len(interval) < 2:
        raise ValueError(f"Invalid interval format: '{interval}'")

    unit = interval[-1]
    if unit not in units:
        if unit == "M":
            raise ValueError(
                "Calendar-month intervals are not supported because their "
                "duration is not fixed."
            )
        raise ValueError(
            f"Unknown interval unit '{unit}'. Supported: {list(units.keys())}"
        )

    try:
        amount = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Invalid interval amount in '{interval}'")

    return amount * units[unit]


def interval_to_unit(
    interval: str,
    precision: TimestampPrecision,
) -> int:
    """Convert interval string to timestamp unit.

    This is the key function for precision-preserving interval arithmetic.
    The returned value is in the same unit as the data's timestamps, so
    floor-alignment and close_time calculations stay in native precision.

    Args:
        interval: Interval string (e.g. ``"5m"``, ``"1h"``).
        precision: The timestamp precision to convert to.

    Returns:
        Interval duration in the specified unit.

    Example::

        interval_to_unit("5m", TimestampPrecision.MICROSECONDS)  # 300_000_000
        interval_to_unit("5m", TimestampPrecision.MILLISECONDS)  # 300_000
        interval_to_unit("5m", TimestampPrecision.SECONDS)       # 300
    """
    seconds = _parse_interval_seconds(interval)
    return seconds * _SECONDS_PER_UNIT[precision]


def interval_to_microseconds(interval: str) -> int:
    """Convert a Binance-style interval string to microseconds.

    Convenience wrapper around :func:`interval_to_unit` for the
    microseconds precision. Kept for backward compatibility.

    Args:
        interval: Interval string (e.g. ``"5m"``, ``"1h"``).

    Returns:
        Interval duration in microseconds.
    """
    return interval_to_unit(interval, TimestampPrecision.MICROSECONDS)


def compute_kline_times(
    open_time: int,
    interval: int,
) -> tuple[int, int]:
    """Compute the (open_time, close_time) pair for a kline interval.

    The open_time is the **start** of the interval (aligned to the interval
    boundary). The close_time is the **last tick** of the interval in the
    same unit (e.g. last microsecond, last millisecond, or last second).

    Args:
        open_time: Aligned open time (in the data's native unit).
        interval: Interval duration (in the same unit).

    Returns:
        Tuple of ``(open_time, close_time)``.
    """
    close_time = open_time + interval - 1
    return open_time, close_time


def align_to_interval(
    timestamp: int,
    interval: int,
) -> int:
    """Floor-align a timestamp to the nearest interval boundary.

    Args:
        timestamp: Timestamp in the data's native unit.
        interval: Interval duration in the same unit.

    Returns:
        Aligned open-time in the same unit.
    """
    return (timestamp // interval) * interval


# ---------------------------------------------------------------------------
# Datetime conversion
# ---------------------------------------------------------------------------
def add_datetime_column(
    df: pl.DataFrame,
    time_col: str = "time",
    timezone: str = "UTC",
    precision: Optional[TimestampPrecision] = None,
    new_col_name: Optional[str] = None,
) -> pl.DataFrame:
    """Add a datetime column converted from a unix-timestamp column.

    The original integer timestamp column is preserved. A new column is
    appended with the converted datetime values.

    Args:
        df: DataFrame with a unix-timestamp integer column.
        time_col: Name of the source timestamp column (default ``"time"``).
        timezone: Target timezone — ``"UTC"`` or ``"EST"`` (default ``"UTC"``).
            EST is interpreted as a fixed UTC-5 offset (no DST adjustment).
        precision: Explicit timestamp precision, or ``None`` to auto-detect.
        new_col_name: Name for the new datetime column. Defaults to
            ``"<time_col>_dt"`` (e.g. ``"time_dt"``).

    Returns:
        DataFrame with an additional datetime column.

    Raises:
        ValueError: If *timezone* is not ``"UTC"`` or ``"EST"``.
    """
    tz = timezone.upper()
    if tz not in ("UTC", "EST"):
        raise ValueError(f"Unsupported timezone '{timezone}'. Use 'UTC' or 'EST'.")

    if new_col_name is None:
        new_col_name = f"{time_col}_dt"

    # Ensure we know the precision
    if precision is None:
        precision = detect_timestamp_precision(df[time_col])

    # Convert to milliseconds for pl.from_epoch
    if precision == TimestampPrecision.MICROSECONDS:
        epoch_expr = pl.col(time_col) // 1_000
        time_unit = "ms"
    elif precision == TimestampPrecision.MILLISECONDS:
        epoch_expr = pl.col(time_col)
        time_unit = "ms"
    else:  # SECONDS
        epoch_expr = pl.col(time_col)
        time_unit = "s"

    # Build datetime column: from_epoch returns a datetime without timezone.
    # We first create it, then set the timezone.
    result = df.with_columns(
        pl.from_epoch(epoch_expr, time_unit=time_unit).alias(new_col_name)
    )

    # Apply timezone
    if tz == "EST":
        result = result.with_columns(
            pl.col(new_col_name)
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Etc/GMT+5")
            .alias(new_col_name)
        )
    else:
        result = result.with_columns(
            pl.col(new_col_name)
            .dt.replace_time_zone("UTC")
            .alias(new_col_name)
        )

    return result
