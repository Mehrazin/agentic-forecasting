"""Experiment 10: classical track — promoted baselines under one roof.

Compares the overnight winners on 2000x66 simulations (small: sibling
tracks are running):
  - m0_w6, m0_w3        : plain M0 (window envelope)
  - conf_m0             : static split-conformal M0 (published baseline)
  - aci_m0              : ACI adaptive conformal M0 (gamma=0.05)
  - macro_m0            : M0 + FRED macro overlay (OLS beta)
  - hier_thin           : hierarchical pooling on thin (subsampled) panels

Two DGPs:
  - standard: no macro structure. macro_m0 must NOT help (honest null).
  - macro   : FRED-driven overlay (exp8). macro_m0 should recover most
              of the oracle gain; ACI vs static on the shock dynamics.

Thin segments: panels binomial-subsampled at p=0.1; hier vs m0_w6 there
tests whether partial pooling has a niche on small-n segments.

Usage:
    cd td-one && uv run python experiments/exp10_classical.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import _default_origins, _summarize  # noqa: E402

from td_one.backtest.runner import ForecastFn, run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.conformal import AdaptiveConformalM0, ConformalM0  # noqa: E402
from td_one.models.hierarchical import HierarchicalBaseline  # noqa: E402
from td_one.models.macro_m0 import MacroM0, load_fred_z  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

N_CUSTOMERS = 2000
N_MONTHS = 66
SEED = 0
BETA_MACRO = 0.6
THIN_P = 0.1
LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)


def _n_upcoming(train: pd.DataFrame, window: int) -> int:
    n = train["n_renewals"].to_numpy()
    return max(1, int(round(float(n[-window:].mean()))))


def _m0_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    m = HistoricalBaseline(trailing_window=window)
    m.fit(n, a)
    return m.predict_rate_quantiles(_n_upcoming(train, window), LEVELS)


def _conf_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    m = ConformalM0(trailing_window=window)
    m.fit(n, a)
    return m.predict_rate_quantiles(_n_upcoming(train, window), LEVELS)


def _aci_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    m = AdaptiveConformalM0(trailing_window=window, gamma=0.05)
    m.fit(n, a)
    return m.predict_rate_quantiles(_n_upcoming(train, window), LEVELS)


def _macro_fn(
    train: pd.DataFrame,
    origin: date,
    fred_z: dict[date, float],
    window: int = 6,
) -> dict[float, float]:
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    months = pd.to_datetime(train["month"])
    m = MacroM0(trailing_window=window, fred_z=fred_z)
    m.fit(n, a, months)
    # Forecast month = month after origin.
    fmonth = (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()
    return m.predict_rate_quantiles(_n_upcoming(train, window), fmonth, LEVELS)


def _fit_hierarchical(
    panels: dict[str, pd.DataFrame], origin: date, window: int = 6
) -> tuple[HierarchicalBaseline, list[str]]:
    """Fit hier on PIT-masked training months; returns (model, valid segments).

    Segments with no trailing-window renewals are skipped (they carry no
    signal for the posterior); the caller must score only valid segments.
    """
    ts = pd.Timestamp(origin)
    hier = HierarchicalBaseline(trailing_window=window)
    seg_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seg, panel in panels.items():
        p = panel.copy()
        p["month"] = pd.to_datetime(p["month"])
        tr = p[p["month"] <= ts]
        n_w = tr["n_renewals"].to_numpy()[-window:]
        if float(n_w.sum()) <= 0:
            continue
        seg_data[seg] = (
            tr["n_renewals"].to_numpy(),
            tr["n_switches"].to_numpy(),
        )
    if not seg_data:
        raise ValueError("no segments with trailing renewals")
    hier.fit(seg_data)
    return hier, list(seg_data)


def _hier_fn(train: pd.DataFrame, origin: date, hier: HierarchicalBaseline) -> dict[float, float]:
    del origin
    seg = str(train["_seg"].iloc[0])
    return hier.predict_rate_quantiles(seg, _n_upcoming(train, 6), LEVELS)


def _thin_panels(panels: dict[str, pd.DataFrame], p: float, seed: int) -> dict[str, pd.DataFrame]:
    """Binomial-subsample panels to make thin segments (deterministic)."""
    rng = np.random.default_rng(seed)
    thin: dict[str, pd.DataFrame] = {}
    for seg, panel in panels.items():
        q = panel.copy()
        n = q["n_renewals"].to_numpy().astype(int)
        a = q["n_switches"].to_numpy().astype(int)
        # Subsample renewals; switches are a subset via hypergeometric.
        n_thin = rng.binomial(n, p)
        a_thin = np.array(
            [
                rng.hypergeometric(ai, ni - ai, nti) if nti > 0 else 0
                for ai, ni, nti in zip(a, n, n_thin, strict=True)
            ]
        )
        q["n_renewals"] = n_thin
        q["n_switches"] = a_thin
        thin[seg] = q
    return thin


def _run_dgp(macro: bool, fred_z: dict[date, float]) -> dict[str, float]:
    """Backtest all models on one DGP; returns model -> mean CRPS etc."""
    sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED)
    histories = sim.generate_history()
    if macro:
        macro_shifts = {d: BETA_MACRO * fred_z.get(d, 0.0) for d in sim._month_dates()}
        outcomes = sim.generate_outcomes(histories, macro_shifts=macro_shifts)
    else:
        outcomes = sim.generate_outcomes(histories)

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(
        f"[exp10] {'macro' if macro else 'standard'} DGP: "
        f"{len(panels_full)} segments, {len(origins)} origins",
        flush=True,
    )

    # Per-origin panels (PIT-masked), built once and reused.
    panels_by_origin: dict[date, dict[str, pd.DataFrame]] = {}
    for origin in origins:
        try:
            panels_by_origin[origin] = discover_segments(histories, outcomes, observed_asof=origin)
        except ValueError:
            continue
    origins = [o for o in origins if o in panels_by_origin]
    del histories, outcomes, panels_full
    gc.collect()

    frames: dict[str, list[pd.DataFrame]] = {
        "m0_w6": [],
        "m0_w3": [],
        "conf_m0": [],
        "aci_m0": [],
        "macro_m0": [],
        "hier_thin": [],
        "m0_w6_thin": [],
    }

    def _scored(
        name: str, fn: ForecastFn, panels: dict[str, pd.DataFrame], origin: date
    ) -> pd.DataFrame:
        r = run_backtest(panels, fn, [origin], seed=SEED)
        r["model"] = name
        return r

    for origin in origins:
        panels = panels_by_origin[origin]
        first = _scored("m0_w6", partial(_m0_fn, window=6), panels, origin)
        if first.empty:
            continue
        frames["m0_w6"].append(first)
        frames["m0_w3"].append(_scored("m0_w3", partial(_m0_fn, window=3), panels, origin))
        frames["conf_m0"].append(_scored("conf_m0", partial(_conf_fn, window=6), panels, origin))
        frames["aci_m0"].append(_scored("aci_m0", partial(_aci_fn, window=6), panels, origin))
        frames["macro_m0"].append(
            _scored("macro_m0", partial(_macro_fn, fred_z=fred_z, window=6), panels, origin)
        )

        # Thin-segment hierarchical test. _fit_hierarchical skips
        # segments whose trailing window emptied out under subsampling;
        # score hier and the thin M0 baseline on the same valid set.
        thin = _thin_panels(panels, THIN_P, seed=SEED)
        try:
            hier, valid_segs = _fit_hierarchical(thin, origin)
        except ValueError:
            continue
        thin_valid = {seg: thin[seg] for seg in valid_segs}
        tagged = {seg: p.assign(_seg=seg) for seg, p in thin_valid.items()}
        frames["hier_thin"].append(
            _scored("hier_thin", partial(_hier_fn, hier=hier), tagged, origin)
        )
        frames["m0_w6_thin"].append(
            _scored("m0_w6_thin", partial(_m0_fn, window=6), tagged, origin)
        )
        del hier, tagged, thin, thin_valid
        print(f"[exp10] origin {origin.isoformat()} done", flush=True)

    summary = _summarize(
        pd.concat([df for frames in frames.values() for df in frames], ignore_index=True)
    )
    out: dict[str, float] = {}
    for name in frames:
        seg = summary[summary["model"] == name]
        if seg.empty:
            continue
        out[f"{name}_crps"] = round(float(seg["crps"].mean()), 3)
        out[f"{name}_mae"] = round(float(seg["mae"].mean()), 3)
        out[f"{name}_cov"] = round(float(seg["coverage_80"].mean()), 3)
    return out


def main() -> int:
    t0 = time.time()
    print("[exp10] Loading FRED z-series ...", flush=True)
    fred_z = load_fred_z()

    results: dict[str, dict[str, float]] = {}
    for macro in (False, True):
        key = "macro_dgp" if macro else "standard_dgp"
        results[key] = _run_dgp(macro, fred_z)
        for k, v in sorted(results[key].items()):
            print(f"[exp10] {key}.{k} = {v}", flush=True)
        gc.collect()

    out = {
        "experiment": "exp10_classical",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seed": SEED,
            "beta_macro": BETA_MACRO,
            "thin_p": THIN_P,
        },
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp10_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[exp10] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
