"""Forecasting Tool: one-step-ahead quantile forecasts for a segment panel.

``model_name`` selects the forecaster:

- ``"m0"``: trailing-6-month Jeffreys Beta-Binomial posterior predictive
  quantiles for the switching *rate* (analytical, via
  :class:`td_one.models.baseline.HistoricalBaseline`). Exposure for the
  upcoming month is the trailing-window mean renewal count (the exposure is
  not yet known at forecast time). Levels match the TimesFM 9-grid so M0 and
  M1 outputs are directly comparable.
- ``"chronos2_mock"`` / ``"timesfm_mock"``: deterministic, fixed-seed mock
  quantile forecasts on the real models' native grids (21-level Chronos-2
  lattice ``[0.01, 0.05, 0.10, ..., 0.99]``; 9-level TimesFM grid
  ``[0.1, ..., 0.9]``). Mocks are interface-only: they center on the last
  observed rate with probit-spaced spreads and are NOT statistical
  surrogates of the real models.

Returns a one-step-ahead forecast: ``{level: rate_quantile}``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from scipy import stats

from td_one.models.baseline import HistoricalBaseline
from td_one.models.tsfm import CHRONOS2_LEVELS, TIMESFM_LEVELS

#: Forecasters available to the agent.
SUPPORTED_MODELS = ("m0", "chronos2_mock", "timesfm_mock")

#: Fixed seeds so mock forecasts are deterministic across runs.
_MOCK_SEEDS = {"chronos2_mock": 20261015, "timesfm_mock": 20261016}

_MOCK_LEVELS = {
    "chronos2_mock": CHRONOS2_LEVELS,
    "timesfm_mock": TIMESFM_LEVELS,
}


def _panel_rates(panel: pd.DataFrame, run_date: date) -> tuple[np.ndarray, np.ndarray]:
    """Filter panel to month <= run_date; return (n_renewals, n_switches).

    Zero-exposure months are dropped (no rate is defined); an empty result
    raises ValueError.
    """
    if run_date > datetime.now(UTC).date():
        raise ValueError(f"run_date {run_date} is in the future")
    if panel.empty:
        raise ValueError("panel is empty")
    p = panel.copy()
    p["month"] = pd.to_datetime(p["month"]).dt.date
    p = p[p["month"] <= run_date].sort_values("month")
    p = p[p["n_renewals"] > 0]
    if p.empty:
        raise ValueError("panel has no visible rows with n_renewals > 0")
    return (
        p["n_renewals"].to_numpy(dtype=float),
        p["n_switches"].to_numpy(dtype=float),
    )


def _mock_quantiles(model_name: str, last_rate: float) -> dict[float, float]:
    """Deterministic seeded mock quantiles, monotone by construction.

    Quantiles are centered on the last observed rate with spreads from the
    probit of the level (so monotonicity is structural, not luck), scaled by
    a fixed-seed draw, then clipped to [0, 1].
    """
    levels = np.asarray(_MOCK_LEVELS[model_name], dtype=float)
    rng = np.random.default_rng(_MOCK_SEEDS[model_name])
    scale = float(rng.uniform(0.015, 0.025))
    center = float(np.clip(last_rate, 0.0, 1.0))
    values = np.clip(center + scale * stats.norm.ppf(levels), 0.0, 1.0)
    return {float(level): float(value) for level, value in zip(levels, values, strict=True)}


def forecast(model_name: str, panel: pd.DataFrame, run_date: date) -> dict[float, float]:
    """One-step-ahead switching-rate quantile forecast for a segment panel.

    Args:
        model_name: One of "m0", "chronos2_mock", "timesfm_mock".
        panel: DataFrame with columns month, n_renewals, n_switches.
        run_date: Point-in-time cutoff (inclusive); only month <= run_date
            rows are used.

    Returns:
        Mapping quantile level -> forecast rate (one step ahead).

    Raises:
        ValueError: Unknown model, future run_date, or empty/unusable panel.

    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model {model_name!r}; expected one of {SUPPORTED_MODELS}")
    n, a = _panel_rates(panel, run_date)
    rates = a / n

    if model_name == "m0":
        baseline = HistoricalBaseline()
        baseline.fit(n, a)
        n_upcoming = max(1, int(round(float(n[-baseline.trailing_window :].mean()))))
        return baseline.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))
    return _mock_quantiles(model_name, float(rates[-1]))
