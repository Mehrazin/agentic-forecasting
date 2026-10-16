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
- ``"chronos2"`` / ``"timesfm25"``: the real zero-shot TSFM forecasters
  (``td_one.models.tsfm``) applied to the panel's historical switching
  rates. Weights resolve from ``models/MANIFEST.json`` (written by
  ``scripts/bootstrap_models.py``); missing weights raise instead of
  downloading. Wrapper instances are cached per process; call
  :func:`unload_real_models` to free them between runs.

Returns a one-step-ahead forecast: ``{level: rate_quantile}``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import stats

from td_one._paths import _repo_root
from td_one.models.baseline import HistoricalBaseline
from td_one.models.tsfm import CHRONOS2_LEVELS, TIMESFM_LEVELS

#: Forecasters available to the agent.
SUPPORTED_MODELS = ("m0", "chronos2_mock", "timesfm_mock", "chronos2", "timesfm25")

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
    n = p["n_renewals"].to_numpy(dtype=float)
    a = p["n_switches"].to_numpy(dtype=float)
    # NaN switch counts would silently produce NaN rates (and NaN model
    # input); fail loudly at the panel boundary instead.
    if bool(np.isnan(n).any() or np.isnan(a).any()):
        raise ValueError("panel contains NaN in n_renewals/n_switches")
    return n, a


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


class _TSFMWrapper(Protocol):
    """Structural type for the real TSFM wrappers (mypy-strict).

    ``Chronos2Wrapper`` and ``TimesFMWrapper`` both satisfy this; the
    cache below is typed against the protocol instead of ``object`` so
    attribute access type-checks without ignores.
    """

    def forecast_quantiles(
        self, series: np.ndarray, horizon: int | None = ...
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast quantiles; see td_one.models.tsfm for the contract."""
        ...

    def unload(self) -> None:
        """Free model weights."""
        ...


#: Cache of real TSFM wrapper instances (one process-wide load each).
_real_wrappers: dict[str, _TSFMWrapper] = {}


def _real_weights_dir(model_name: str) -> str:
    """Resolve the local weights dir from models/MANIFEST.json (fail fast)."""
    root = _repo_root()
    manifest_path = root / "models" / "MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"real model {model_name!r} needs weights: {manifest_path} missing; "
            "run scripts/bootstrap_models.py first"
        )
    import json

    manifest = json.loads(manifest_path.read_text())
    try:
        local_path = manifest["models"][model_name]["local_path"]
    except KeyError as exc:
        raise RuntimeError(f"manifest has no entry for {model_name!r}") from exc
    wdir = root / local_path
    if not wdir.is_dir():
        raise RuntimeError(f"weights dir missing for {model_name!r}: {wdir}")
    return str(wdir)


#: Quantile-crossing diagnostics for the monotone rearrangement in
#: `_real_forecast`. Per model: forecasts seen, forecasts where the raw
#: wrapper output had any crossing, and the total number of pairwise
#: inversions repaired. Crossing is itself a tail-calibration signal, so
#: reports should disclose these counts rather than silently repairing.
_crossing_stats: dict[str, dict[str, int]] = {}


def crossing_stats() -> dict[str, dict[str, int]]:
    """Snapshot of quantile-crossing diagnostics (per real model)."""
    return {m: dict(s) for m, s in _crossing_stats.items()}


def reset_crossing_stats() -> None:
    """Clear quantile-crossing diagnostics (e.g. between experiments)."""
    _crossing_stats.clear()


def _count_inversions(vals: np.ndarray) -> int:
    """Count pairwise inversions (i<j but vals[i] > vals[j])."""
    # O(n^2); n <= 21 levels, trivial cost.
    inv = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if vals[i] > vals[j]:
                inv += 1
    return inv


def _real_forecast(model_name: str, rates: np.ndarray) -> dict[float, float]:
    """Zero-shot TSFM forecast of the next-month switching rate.

    Args:
        model_name: "chronos2" or "timesfm25".
        rates: Historical monthly switching rates in [0, 1].

    Returns:
        Mapping quantile level -> forecast rate, one step ahead, on the
        model's native grid. Values are sorted ascending to enforce
        monotone quantiles — neural forecasters can emit quantile
        crossing, which the backtest validator (correctly) rejects.

    """
    if model_name not in _real_wrappers:
        from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper

        wdir = _real_weights_dir(model_name)
        cls = Chronos2Wrapper if model_name == "chronos2" else TimesFMWrapper
        _real_wrappers[model_name] = cls(use_mock=False, weights_dir=wdir)
    wrapper = _real_wrappers[model_name]
    levels, values = wrapper.forecast_quantiles(rates, horizon=1)
    raw = np.asarray(values[:, 0], dtype=float)
    inv = _count_inversions(raw)
    stat = _crossing_stats.setdefault(model_name, {"forecasts": 0, "crossed": 0, "inversions": 0})
    stat["forecasts"] += 1
    if inv:
        stat["crossed"] += 1
        stat["inversions"] += inv
    vals = np.sort(raw)
    lvls = np.asarray(levels, dtype=float)
    order = np.argsort(lvls, kind="stable")
    return {float(lvls[i]): float(vals[j]) for j, i in enumerate(order)}


def unload_real_models() -> None:
    """Unload cached real TSFM wrappers (free RAM between runs)."""
    for wrapper in _real_wrappers.values():
        wrapper.unload()
    _real_wrappers.clear()


def forecast(model_name: str, panel: pd.DataFrame, run_date: date) -> dict[float, float]:
    """One-step-ahead switching-rate quantile forecast for a segment panel.

    Args:
        model_name: One of SUPPORTED_MODELS — "m0", "chronos2_mock",
            "timesfm_mock" (interface-only mocks) and the real-weight
            "chronos2" / "timesfm25".
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
    if model_name in ("chronos2", "timesfm25"):
        return _real_forecast(model_name, rates)
    return _mock_quantiles(model_name, float(rates[-1]))
