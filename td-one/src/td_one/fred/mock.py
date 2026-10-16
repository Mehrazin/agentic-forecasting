"""FRED mock: deterministic fixture series for development and tests.

Covers the macro series the M2 agent consults:

- ``DGS5``: 5-year Government of Canada bond yield (%)
- ``CPI_YOY``: CPI year-over-year inflation (%)
- ``UNRATE``: unemployment rate (%)

Monthly, 2020-01 through 2025-12. Each series is a smooth curve through
plausible Canadian anchor points plus fixed-seed AR(1) noise — deterministic
per (series_id, start, end), no network, no API key.

Live FRED access is out of scope: see :func:`get_series_live`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

#: First and last month of the fixture range.
FIXTURE_START = date(2020, 1, 1)
FIXTURE_END = date(2025, 12, 1)

SUPPORTED_SERIES = ("DGS5", "CPI_YOY", "UNRATE")

#: (month_index, value) anchors; month_index counts months from 2020-01.
_ANCHORS: dict[str, list[tuple[int, float]]] = {
    "DGS5": [
        (0, 0.45),
        (6, 0.55),
        (12, 1.05),
        (18, 1.35),
        (24, 2.90),
        (30, 3.55),
        (36, 3.75),
        (42, 3.45),
        (48, 3.20),
        (54, 3.05),
        (60, 3.00),
        (66, 3.10),
        (71, 3.05),
    ],
    "CPI_YOY": [
        (0, 2.30),
        (3, 0.70),
        (9, 1.00),
        (15, 3.40),
        (21, 6.80),
        (29, 8.10),
        (35, 5.90),
        (41, 3.60),
        (47, 2.90),
        (53, 2.20),
        (59, 1.90),
        (65, 2.10),
        (71, 2.00),
    ],
    "UNRATE": [
        (0, 5.50),
        (3, 7.80),
        (4, 13.00),
        (5, 13.70),
        (8, 8.80),
        (14, 8.10),
        (20, 6.00),
        (26, 5.10),
        (32, 5.00),
        (38, 5.50),
        (44, 5.80),
        (50, 6.20),
        (56, 6.50),
        (62, 6.90),
        (68, 6.70),
        (71, 6.60),
    ],
}

#: Fixed seed per series so fixtures never change between runs.
_SEEDS = {"DGS5": 101, "CPI_YOY": 202, "UNRATE": 303}


def _build_fixture(series_id: str) -> pd.Series:
    """Build the full 2020-01..2025-12 monthly fixture for a series."""
    idx = pd.date_range(FIXTURE_START, FIXTURE_END, freq="MS")
    months = np.arange(len(idx), dtype=float)
    anchor_m = np.array([a[0] for a in _ANCHORS[series_id]], dtype=float)
    anchor_v = np.array([a[1] for a in _ANCHORS[series_id]], dtype=float)
    smooth = np.interp(months, anchor_m, anchor_v)
    rng = np.random.default_rng(_SEEDS[series_id])
    noise = np.zeros(len(idx))
    for t in range(1, len(idx)):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0.0, 0.06)
    values = np.clip(smooth + noise, 0.0, None)
    return pd.Series(values, index=idx, name=series_id)


_FIXTURES: dict[str, pd.Series] = {
    series_id: _build_fixture(series_id) for series_id in SUPPORTED_SERIES
}


def get_series(series_id: str, start: date, end: date) -> pd.Series:
    """Return the deterministic mock monthly series for [start, end].

    Args:
        series_id: One of "DGS5", "CPI_YOY", "UNRATE".
        start: First month wanted (any day; snapped to month start).
        end: Last month wanted (any day; snapped to month start).

    Returns:
        Monthly pd.Series with a DatetimeIndex (month-start), named by
        series_id.

    Raises:
        ValueError: Unknown series_id, start > end, or window outside the
            2020-01..2025-12 fixture range.

    """
    if series_id not in _FIXTURES:
        raise ValueError(f"unknown series {series_id!r}; expected one of {SUPPORTED_SERIES}")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    lo = date(start.year, start.month, 1)
    hi = date(end.year, end.month, 1)
    if lo < FIXTURE_START or hi > FIXTURE_END:
        raise ValueError(
            f"window [{lo}, {hi}] outside fixture range [{FIXTURE_START}, {FIXTURE_END}]"
        )
    fixture = _FIXTURES[series_id]
    return fixture.loc[pd.Timestamp(lo) : pd.Timestamp(hi)].copy()


def get_series_live(
    series_id: str,
    start: date,
    end: date,
    api_key: str | None = None,
) -> pd.Series:
    """Live FRED access (not implemented).

    Raises:
        NotImplementedError: Always — live FRED requires ``FRED_API_KEY``
            and is out of scope for development; use
            :func:`get_series` (mock) or :class:`td_one.fred.cache.FileCache`.

    """
    raise NotImplementedError(
        "Live FRED access is out of scope: it requires a FRED_API_KEY and "
        "network access. Use td_one.fred.get_series (deterministic mock) or "
        "td_one.fred.cache.FileCache for development and tests."
    )
