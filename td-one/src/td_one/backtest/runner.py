"""Rolling-origin backtest runner over segment count panels.

For each segment and origin, the runner:

1. Filters the panel to ``month <= origin`` (training window) and applies a
   defensive cutoff check — any month past the origin in the training frame
   raises :class:`ValueError`.
2. Calls the injected ``forecast_fn`` for the *next* month's switching
   **rate** as a quantile lattice ``{level: rate}`` with
   ``0 < level < 1``. The forecast function is decoupled from model
   implementations; the scale-up script wires concrete models.
3. Converts the rate lattice to a count predictive distribution via
   inverse-CDF sampling + ``Binomial(N, rate)`` (``N`` = next-month
   renewals, known from scheduled maturities) and scores the realized
   next-month outcome with CRPS (counts, proper), MAE of the median count,
   and 80% prediction-interval coverage.

Returns a tidy DataFrame with one row per scored segment x origin.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from td_one.evaluation.metrics import crps_counts

#: Monte Carlo draws for the count predictive distribution per cell.
N_MC_SAMPLES = 2000

#: Tidy result schema.
RESULT_COLUMNS = ["segment", "origin", "crps", "mae", "coverage_80", "n_renewals"]

#: A forecast function maps (training panel, origin) to a rate quantile
#: lattice ``{level: rate_quantile}`` for the next month.
ForecastFn = Callable[[pd.DataFrame, date], dict[float, float]]


def _check_cutoff(train: pd.DataFrame, origin: pd.Timestamp, segment: str) -> None:
    """Defensive guard: the training frame must not extend past the origin.

    Raises:
        ValueError: If any training month is after the origin (leakage).

    """
    if len(train) == 0:
        return
    latest = train["month"].max()
    if latest > origin:
        raise ValueError(
            f"cutoff violation in segment {segment!r}: training month {latest} "
            f"is after origin {origin.date()}"
        )


def _validate_forecast(
    forecast: dict[float, float], segment: str, origin: date
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a rate quantile lattice; return sorted (levels, values)."""
    if not forecast:
        raise ValueError(f"empty forecast lattice for segment {segment!r} at {origin}")
    levels = np.array(sorted(forecast.keys()), dtype=float)
    values = np.array([forecast[float(lvl)] for lvl in levels], dtype=float)
    if bool((levels <= 0).any() or (levels >= 1).any()):
        raise ValueError(f"forecast levels must be in (0, 1) for segment {segment!r} at {origin}")
    if bool((values < 0).any() or (values > 1).any()):
        raise ValueError(
            f"forecast rate quantiles must be in [0, 1] for segment {segment!r} at {origin}"
        )
    if not bool((np.diff(values) >= 0).all()):
        raise ValueError(
            f"forecast quantiles must be non-decreasing for segment {segment!r} at {origin}"
        )
    return levels, values


def _validate_panel(panel: pd.DataFrame, segment: str) -> None:
    """Validate a segment panel's schema and count coherence."""
    required = {"month", "n_renewals", "n_switches"}
    if not required.issubset(panel.columns):
        raise ValueError(
            f"segment {segment!r} panel missing columns: {required - set(panel.columns)}"
        )
    if panel["month"].duplicated().any():
        raise ValueError(f"segment {segment!r} panel has duplicate months")
    bad = (panel["n_switches"] < 0) | (panel["n_switches"] > panel["n_renewals"])
    if bool(bad.any()):
        raise ValueError(f"segment {segment!r} panel violates 0 <= switches <= renewals")


def run_backtest(
    panels: dict[str, pd.DataFrame],
    forecast_fn: ForecastFn,
    origins: list[date],
    seed: int = 0,
) -> pd.DataFrame:
    """Run a rolling-origin backtest over segment panels.

    Args:
        panels: Mapping of segment key to monthly panel DataFrame with
            columns ``[month, n_renewals, n_switches]``.
        forecast_fn: ``(train_panel, origin) -> {level: rate_quantile}``
            for the next month. The training panel is already filtered to
            ``month <= origin``; ``origin`` is a :class:`datetime.date`.
        origins: Forecast origins (dates). Each origin is scored against
            the next panel month after it; segment x origin cells with no
            training data or no future month are skipped.
        seed: Base RNG seed; each cell derives its own seed from
            ``(seed, segment, origin)`` so results are reproducible and
            independent of dict ordering.

    Returns:
        Tidy DataFrame with columns
        ``[segment, origin, crps, mae, coverage_80, n_renewals]``.

    Raises:
        ValueError: On empty panels/origins, malformed panels, cutoff
            violations, or invalid forecast lattices.

    """
    if not panels:
        raise ValueError("panels is empty")
    if not origins:
        raise ValueError("origins is empty")

    rows: list[dict[str, object]] = []
    for segment, panel in panels.items():
        _validate_panel(panel, segment)
        p = panel.copy()
        p["month"] = pd.to_datetime(p["month"])
        p = p.sort_values("month").reset_index(drop=True)

        for origin in origins:
            ts = pd.Timestamp(origin)
            train = p[p["month"] <= ts].reset_index(drop=True)
            _check_cutoff(train, ts, segment)
            if train.empty:
                continue  # no training data before this origin
            future = p[p["month"] > ts]
            if future.empty:
                continue  # no next month to score
            tgt_pos = int(future["month"].to_numpy().argmin())
            target = future.iloc[tgt_pos]
            n = int(target["n_renewals"])
            actual = int(target["n_switches"])

            forecast = forecast_fn(train, ts.date())
            levels, qvals = _validate_forecast(forecast, segment, ts.date())

            # Rate lattice -> count predictive distribution.
            cell_key = f"{segment}|{ts.date().isoformat()}".encode()
            cell_seed = (seed + zlib.crc32(cell_key)) & 0x7FFFFFFF
            rng = np.random.default_rng(cell_seed)
            u = rng.random(N_MC_SAMPLES)
            rate_samples = np.clip(np.interp(u, levels, qvals), 0.0, 1.0)
            count_samples = rng.binomial(n, rate_samples)

            crps = float(crps_counts(count_samples[None, :], np.array([actual]), seed=cell_seed)[0])
            median_rate = float(np.interp(0.5, levels, qvals))
            mae = abs(median_rate * n - actual)
            q10, q90 = np.quantile(count_samples, [0.10, 0.90])
            coverage = 1.0 if q10 <= actual <= q90 else 0.0

            rows.append(
                {
                    "segment": segment,
                    "origin": ts.date(),
                    "crps": crps,
                    "mae": mae,
                    "coverage_80": coverage,
                    "n_renewals": n,
                }
            )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
