"""Data-loading and canonical-baseline tools.

``load_series`` eliminates the CSV date-loading boilerplate every agent
code block repeats. ``naive_forecasts`` standardizes baseline forecasts
(naive / seasonal-naive / trailing-mean) so experiments stay comparable —
previously each experiment reimplemented these with slightly different
sigma estimators. ``series_features`` is the tsfeatures-lite bundle
(exp42's 10-feature set) used by the regime router.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from td_one.agent.tool_tracker import tracked

# ---------------------------------------------------------------------------
# load_series
# ---------------------------------------------------------------------------


class LoadedSeries(BaseModel):
    """Structured output of :func:`load_series`."""

    values: list[float] = Field(description="Series values as floats")
    dates: list[str] = Field(description="ISO date strings, same length as values")
    n: int = Field(description="Number of observations")
    value_col: str = Field(description="Column used as the series")
    origin_applied: str | None = Field(
        default=None, description="PIT cutoff applied, if any"
    )


@tracked()
def load_series(
    csv_path: str,
    value_col: str | None = None,
    origin: str | None = None,
    date_col: str = "date",
) -> LoadedSeries:
    """Load a time series from CSV with PIT-safe date filtering.

    Handles the two layouts agents encounter: a named date column, or
    dates in the index (``index_col=0`` with ``parse_dates``).

    Args:
        csv_path: Path to the CSV file.
        value_col: Column to use as the series. Defaults to the first
            numeric column.
        origin: Optional PIT cutoff (ISO date string). Rows with a date
            strictly after ``origin`` are dropped.
        date_col: Name of the date column. If absent, the index is tried.

    Returns:
        LoadedSeries with values, ISO date strings, and metadata.

    """
    try:
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        dates = pd.to_datetime(df[date_col])
        data = df
    except (ValueError, KeyError):
        # Fall back to dates-in-index layout.
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        dates = pd.to_datetime(df.index)
        data = df

    if value_col is None:
        numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            raise ValueError(f"no numeric columns found in {csv_path}")
        value_col = numeric[0]
    if value_col not in data.columns:
        raise ValueError(f"value_col={value_col!r} not in {csv_path}")

    values = pd.to_numeric(data[value_col], errors="coerce")
    mask = dates.notna() & values.notna()
    dates = dates[mask]
    values = values[mask].to_numpy(dtype=float)

    origin_applied: str | None = None
    if origin is not None:
        cutoff = pd.Timestamp(origin)
        keep = dates <= cutoff
        dates = dates[keep]
        values = values[keep.to_numpy()]
        origin_applied = cutoff.date().isoformat()

    if values.size == 0:
        raise ValueError(f"no usable observations in {csv_path} (origin={origin})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values remain in {csv_path}")

    order = np.argsort(dates.to_numpy())
    dates = dates.to_numpy()[order]
    values = values[order]
    return LoadedSeries(
        values=[float(v) for v in values],
        dates=[pd.Timestamp(d).date().isoformat() for d in dates],
        n=int(values.size),
        value_col=value_col,
        origin_applied=origin_applied,
    )


# ---------------------------------------------------------------------------
# naive_forecasts
# ---------------------------------------------------------------------------

#: Canonical quantile grid used across the TD-One benchmark suite.
NAIVE_LEVELS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class NaiveForecasts(BaseModel):
    """Structured output of :func:`naive_forecasts`."""

    levels: list[float] = Field(description="Quantile levels, ascending")
    naive: list[list[float]] = Field(description="(h, n_levels) quantile lattice")
    seasonal_naive: list[list[float]] = Field(description="(h, n_levels) lattice")
    trailing_mean: list[list[float]] = Field(description="(h, n_levels) lattice")
    sigma: float = Field(description="Canonical residual scale (std of diffs)")
    horizon: int


def _z_scores(levels: list[float]) -> np.ndarray:
    from scipy.stats import norm  # local import: scipy is optional at module load

    return norm.ppf(np.asarray(levels, dtype=float))


@tracked()
def naive_forecasts(
    train: Any,
    horizon: int,
    period: int = 12,
    levels: list[float] | None = None,
) -> NaiveForecasts:
    """Canonical naive / seasonal-naive / trailing-mean quantile forecasts.

    One implementation for every experiment — previously each experiment
    reimplemented these with slightly different sigma estimators, a
    comparability hazard. Sigma is the std of first differences (the
    convention from exp23/exp42), growing with ``sqrt(h)`` per step.

    Args:
        train: 1-D numeric training series (min 2 observations).
        horizon: Forecast horizon (number of steps, >= 1).
        period: Seasonal period for seasonal-naive (default 12).
        levels: Quantile levels; defaults to the 0.1..0.9 grid.

    Returns:
        NaiveForecasts with an (h, n_levels) lattice per baseline.

    """
    x = np.asarray(train, dtype=float).ravel()
    if x.size < 2:
        raise ValueError(f"train needs at least 2 observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("train contains NaN or inf")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    lvls = list(levels) if levels else list(NAIVE_LEVELS)
    z = _z_scores(lvls)
    diffs = np.diff(x)
    sigma = float(np.std(diffs)) if diffs.size else 0.0

    last = float(x[-1])
    seas = float(x[-period]) if x.size >= period else last
    trailing = float(np.mean(x[-min(period, x.size) :]))

    def _lattice(point: float) -> list[list[float]]:
        steps = np.sqrt(np.arange(1, horizon + 1))
        grid = point + (sigma * z[:, None]) * steps[None, :]
        return [[float(v) for v in grid[:, s]] for s in range(horizon)]

    return NaiveForecasts(
        levels=lvls,
        naive=_lattice(last),
        seasonal_naive=_lattice(seas),
        trailing_mean=_lattice(trailing),
        sigma=sigma,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# series_features
# ---------------------------------------------------------------------------


class SeriesFeatures(BaseModel):
    """Structured output of :func:`series_features` (exp42's 10-feature set)."""

    n_changepoints: float = 0.0
    n_recent_cps: float = 0.0
    trend_strength: float = 0.0
    seasonal_strength: float = 0.0
    vol_ratio: float = 0.0
    is_stationary: float = 0.0
    is_nonstationary: float = 0.0
    acf1: float = 0.0
    spectral_entropy: float = 1.0
    hurst: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


def _acf1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    xc = x - np.mean(x)
    denom = float(np.sum(xc**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(xc[:-1] * xc[1:]) / denom)


def _spectral_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return 1.0
    xc = x - np.mean(x)
    spec = np.abs(np.fft.rfft(xc)) ** 2
    total = float(np.sum(spec[1:]))  # exclude DC
    if total < 1e-12:
        return 0.0
    p = spec[1:] / total
    p = p[p > 1e-12]
    ent = -float(np.sum(p * np.log(p)))
    return ent / np.log(len(p)) if len(p) > 1 else 0.0


def _hurst_rs(x: np.ndarray) -> float:
    """Estimate Hurst via rescaled range (rough but cheap)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 16:
        return 0.5
    chunk = n // 4
    rs_vals = []
    for i in range(4):
        seg = x[i * chunk : (i + 1) * chunk]
        if len(seg) < 4:
            continue
        mean_adj = seg - np.mean(seg)
        cumdev = np.cumsum(mean_adj)
        r = float(np.max(cumdev) - np.min(cumdev))
        s = float(np.std(seg))
        if s > 1e-12 and r > 0:
            rs_vals.append(r / s)
    if not rs_vals:
        return 0.5
    rs_mean = float(np.mean(rs_vals))
    if rs_mean <= 0:
        return 0.5
    return float(np.clip(np.log(rs_mean) / np.log(chunk), 0.05, 0.95))


@tracked()
def series_features(series: Any, period: int = 12) -> SeriesFeatures:
    """Compute the tsfeatures-lite bundle (exp42's 10-feature set).

    PIT-safe: uses the training window only. Each component is guarded —
    short series degrade gracefully to neutral defaults instead of
    raising.

    Args:
        series: 1-D numeric series.
        period: Seasonal period for the decomposition component.

    Returns:
        SeriesFeatures with the 10 canonical features.

    """
    from td_one.agent.tools.analysis import decompose_series, test_stationarity
    from td_one.agent.tools.regimes import detect_changepoints
    # NOTE: regimes import is local to avoid a tools.<module> cycle
    # (regimes.py does not import data.py, so this is safe at call time).

    x = np.asarray(series, dtype=float).ravel()
    if x.size < 3:
        raise ValueError(f"series needs at least 3 observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or inf")

    feats = SeriesFeatures()

    with contextlib.suppress(Exception):
        cp = detect_changepoints(x, min_distance=6)
        feats.n_changepoints = float(cp.n_segments - 1)
        feats.n_recent_cps = float(sum(1 for c in cp.changepoints if c >= len(x) - 6))

    with contextlib.suppress(Exception):
        dc = decompose_series(x, period=period)
        feats.trend_strength = float(dc.trend_strength)
        feats.seasonal_strength = float(dc.seasonal_strength)

    with contextlib.suppress(Exception):
        # Full ADF+KPSS (not fast) to match the canonical exp42 feature set.
        st = test_stationarity(x)
        v = str(st.verdict)
        feats.is_stationary = 1.0 if v == "stationary" else 0.0
        feats.is_nonstationary = 1.0 if v == "non_stationary" else 0.0

    sx = float(np.std(x))
    feats.vol_ratio = float(np.std(np.diff(x)) / sx) if sx > 1e-9 else 0.0
    feats.acf1 = _acf1(x)
    with contextlib.suppress(Exception):
        feats.spectral_entropy = _spectral_entropy(x)
    with contextlib.suppress(Exception):
        feats.hurst = _hurst_rs(x)
    return feats
