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

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Union, cast

import polars as pl

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
IntervalValue = Union[int, float, str]

_SECONDS_PER_INTERVAL_SCALE: dict[str, Decimal] = {
    "s": Decimal("1"),
    "sec": Decimal("1"),
    "secs": Decimal("1"),
    "second": Decimal("1"),
    "seconds": Decimal("1"),
    "m": Decimal("60"),
    "min": Decimal("60"),
    "mins": Decimal("60"),
    "minute": Decimal("60"),
    "minutes": Decimal("60"),
    "h": Decimal("3600"),
    "hr": Decimal("3600"),
    "hrs": Decimal("3600"),
    "hour": Decimal("3600"),
    "hours": Decimal("3600"),
    "d": Decimal("86400"),
    "day": Decimal("86400"),
    "days": Decimal("86400"),
    "w": Decimal("604800"),
    "wk": Decimal("604800"),
    "wks": Decimal("604800"),
    "week": Decimal("604800"),
    "weeks": Decimal("604800"),
}


def interval_to_unit(
    interval: IntervalValue,
    precision: TimestampPrecision,
    interval_scale: str = "m",
) -> int:
    """Convert a custom interval to the timestamp unit used by the data.

    ``interval`` may be a compact string with its unit, such as ``"2.5m"``,
    ``"20s"``, or ``"1.1h"``. It may also be numeric; in that case
    ``interval_scale`` supplies the unit, for example ``interval=20`` and
    ``interval_scale="s"``.

    Decimal arithmetic is used to avoid float drift. The result must be an
    exact integer number of timestamp ticks; otherwise the interval cannot be
    represented accurately at the requested timestamp precision.

    Args:
        interval: Interval amount, or compact interval string.
        precision: The timestamp precision to convert to.
        interval_scale: Unit for numeric intervals. Supported units are
            seconds, minutes, hours, days, and weeks using common aliases.

    Returns:
        Interval duration in the specified unit.

    Example::

        interval_to_unit("5m", TimestampPrecision.MICROSECONDS)  # 300_000_000
        interval_to_unit("2.5m", TimestampPrecision.MILLISECONDS)  # 150_000
        interval_to_unit(20, TimestampPrecision.SECONDS, "s")      # 20
    """
    seconds = _interval_seconds(interval, interval_scale)
    ticks = seconds * Decimal(_SECONDS_PER_UNIT[precision])
    integer_ticks = ticks.to_integral_value()
    if ticks != integer_ticks:
        raise ValueError(
            f"Interval {interval!r} {interval_scale!r} is {ticks} ticks at "
            f"{precision.value} precision, which cannot be represented exactly."
        )
    if integer_ticks <= 0:
        raise ValueError("Interval must be greater than zero")
    return int(integer_ticks)


def _interval_seconds(interval: IntervalValue, interval_scale: str) -> Decimal:
    """Return interval duration in seconds as an exact Decimal."""
    if isinstance(interval, str):
        amount, scale = _split_interval_string(interval, interval_scale)
    else:
        amount = _decimal_from_value(interval)
        scale = interval_scale

    if amount <= 0:
        raise ValueError("Interval must be greater than zero")
    return amount * _seconds_per_scale(scale)


def _split_interval_string(
    interval: str,
    default_scale: str,
) -> tuple[Decimal, str]:
    """Split strings such as '2.5m' or '20 sec' into amount and scale."""
    compact = interval.strip().replace(" ", "")
    if not compact:
        raise ValueError("Interval cannot be empty")

    split_at = len(compact)
    while split_at > 0 and compact[split_at - 1].isalpha():
        split_at -= 1

    amount_text = compact[:split_at]
    scale = compact[split_at:] or default_scale
    return _decimal_from_value(amount_text), scale


def _decimal_from_value(value: int | float | str) -> Decimal:
    """Convert an interval value to Decimal without binary float artifacts."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid interval amount: {value!r}") from exc


def _seconds_per_scale(scale: str) -> Decimal:
    """Return the number of seconds for an interval scale alias."""
    if scale == "M" or scale.lower() in {"mo", "mon", "month", "months"}:
        raise ValueError(
            "Calendar-month intervals are not supported because their duration "
            "is not fixed."
        )

    key = scale.lower()
    if key not in _SECONDS_PER_INTERVAL_SCALE:
        supported = ", ".join(["s", "m", "h", "d", "w"])
        raise ValueError(f"Unknown interval scale '{scale}'. Supported: {supported}")
    return _SECONDS_PER_INTERVAL_SCALE[key]
