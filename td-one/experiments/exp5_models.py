"""Experiment 5: better classical baselines vs M0.

Backtests four new baseline families (plus M0 window 3/6 controls) on the
standard scale-up config (10k customers x 66 months, seed 0, 6 origins):

- m0_w3 / m0_w6 : plain M0 controls (window 3 and 6)
- hier          : hierarchical partial-pooling Beta-Binomial (per-origin fit)
- ewm_hl2/hl4   : exponentially-weighted M0 (half-life 2 / 4 months)
- conf_w3/w6    : split-conformalized M0 (targets the undercoverage)
- ens_equal     : equal-weight lattice average of m0_w3 + m0_w6 + ewm_hl2
- ens_crpsw     : inverse-trailing-CRPS-weighted version of the above

Pure-Python (no weights, no LLM); runs in ~7 minutes.

Usage:
    cd td-one && uv run python experiments/exp5_models.py
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
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
from td_one.models.conformal import ConformalM0  # noqa: E402
from td_one.models.ensemble import average_lattices, crps_weights  # noqa: E402
from td_one.models.ewm import EWMHistoricalBaseline  # noqa: E402
from td_one.models.hierarchical import HierarchicalBaseline  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)
N_CUSTOMERS = 10_000
N_MONTHS = 66
SEED = 0
N_UPCOMING_K = 6
HIER_WINDOW = 6


def _n_upcoming(train: pd.DataFrame, k: int = N_UPCOMING_K) -> int:
    n = train["n_renewals"].to_numpy()
    return max(1, int(round(float(n[-k:].mean()))))


def _m0_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    model = HistoricalBaseline(trailing_window=window)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    # n_upcoming from the trailing *window* mean: mirrors the tools path
    # (forecast("m0") uses n[-trailing_window:]), verified to reproduce
    # the published M0 numbers exactly.
    return model.predict_rate_quantiles(_n_upcoming(train, k=window), LEVELS)


def _ewm_fn(train: pd.DataFrame, origin: date, half_life: float) -> dict[float, float]:
    del origin
    model = EWMHistoricalBaseline(half_life=half_life)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    return model.predict_rate_quantiles(_n_upcoming(train), LEVELS)


def _conf_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    model = ConformalM0(trailing_window=window)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    return model.predict_rate_quantiles(_n_upcoming(train, k=window))


def _fit_hierarchical(panels: dict[str, pd.DataFrame], origin: date) -> HierarchicalBaseline:
    """Fit the hierarchical model on PIT-masked training months only.

    ``panels`` may contain future (scoring) months; the fit slices to
    month <= origin exactly as run_backtest's train split does.
    """
    ts = pd.Timestamp(origin)
    hier = HierarchicalBaseline(trailing_window=HIER_WINDOW)
    seg_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seg, panel in panels.items():
        p = panel.copy()
        p["month"] = pd.to_datetime(p["month"])
        tr = p[p["month"] <= ts]
        seg_data[seg] = (
            tr["n_renewals"].to_numpy(),
            tr["n_switches"].to_numpy(),
        )
    hier.fit(seg_data)
    return hier


def _hier_fn(train: pd.DataFrame, origin: date, hier: HierarchicalBaseline) -> dict[float, float]:
    del origin
    seg = str(train["_seg"].iloc[0])  # tagged by the caller
    return hier.predict_rate_quantiles(seg, _n_upcoming(train), LEVELS)


def _ens_fn(
    train: pd.DataFrame,
    origin: date,
    weights: list[float] | None,
) -> dict[float, float]:
    lattices = [
        _m0_fn(train, origin, 3),
        _m0_fn(train, origin, 6),
        _ewm_fn(train, origin, 2.0),
    ]
    return average_lattices(lattices, weights)


def main() -> int:
    ap = argparse.ArgumentParser(description="exp5: better classical baselines")
    ap.add_argument(
        "--panels",
        default=None,
        help="Pickle from exp5_phase1.py (skips the simulation).",
    )
    args = ap.parse_args()
    t0 = time.time()

    origins: list[date]
    panels_by_origin: dict[date, dict[str, pd.DataFrame]]
    if args.panels:
        print(f"[exp5] Loading cached panels from {args.panels} ...", flush=True)
        with open(args.panels, "rb") as f:
            payload = pickle.load(f)  # noqa: S301 - own cache file
        origins = [date.fromisoformat(s) for s in payload["origins"]]
        panels_by_origin = {
            date.fromisoformat(k): v for k, v in payload["panels_by_origin"].items()
        }
        cfg = payload["config"]
        print(
            f"[exp5] {len(panels_by_origin[origins[0]])} segments, "
            f"{len(origins)} origins (seed={cfg['seed']})",
            flush=True,
        )
    else:
        print(f"[exp5] Simulating {N_CUSTOMERS} x {N_MONTHS} (seed={SEED}) ...", flush=True)
        sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED)
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories)
        del sim
        gc.collect()

        panels_full = discover_segments(histories, outcomes)
        origins = _default_origins(panels_full)
        print(f"[exp5] {len(panels_full)} segments, {len(origins)} origins", flush=True)
        panels_by_origin = {}
        for origin in origins:
            try:
                panels_by_origin[origin] = discover_segments(
                    histories, outcomes, observed_asof=origin
                )
            except ValueError:
                continue
        origins = [o for o in origins if o in panels_by_origin]
        del histories, outcomes, panels_full
        gc.collect()

    # Per-origin outer loop: flat memory (this VM is shared with sibling
    # research tracks). Each origin's panels are built, scored for all
    # models, then released.
    model_names = (
        "m0_w3",
        "m0_w6",
        "hier",
        "ewm_hl2",
        "ewm_hl4",
        "conf_w3",
        "conf_w6",
        "ens_equal",
        "ens_crpsw",
    )
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in model_names}
    comp_hist: dict[str, list[float]] = {"m0_w3": [], "m0_w6": [], "ewm_hl2": []}

    def _scored(
        name: str, fn: ForecastFn, panels: dict[str, pd.DataFrame], origin: date
    ) -> pd.DataFrame:
        r = run_backtest(panels, fn, [origin], seed=SEED)
        r["model"] = name
        return r

    n_origins_run = 0
    for origin in origins:
        panels = panels_by_origin[origin]
        # Score the first control; an empty frame means this origin has no
        # future month to score against (the known "5 of 6 origins" pattern).
        first = _scored("m0_w3", partial(_m0_fn, window=3), panels, origin)
        if first.empty:
            print(f"[exp5] origin {origin.isoformat()} skipped (no scoring month)", flush=True)
            continue
        n_origins_run += 1
        frames["m0_w3"].append(first)
        frames["m0_w6"].append(_scored("m0_w6", partial(_m0_fn, window=6), panels, origin))
        frames["ewm_hl2"].append(
            _scored("ewm_hl2", partial(_ewm_fn, half_life=2.0), panels, origin)
        )
        frames["ewm_hl4"].append(
            _scored("ewm_hl4", partial(_ewm_fn, half_life=4.0), panels, origin)
        )
        frames["conf_w3"].append(_scored("conf_w3", partial(_conf_fn, window=3), panels, origin))
        frames["conf_w6"].append(_scored("conf_w6", partial(_conf_fn, window=6), panels, origin))

        # Hierarchical: per-origin fit on PIT-masked training months only.
        hier = _fit_hierarchical(panels, origin)
        tagged = {seg: p.assign(_seg=seg) for seg, p in panels.items()}
        frames["hier"].append(_scored("hier", partial(_hier_fn, hier=hier), tagged, origin))
        del hier, tagged

        # Ensembles.
        frames["ens_equal"].append(
            _scored("ens_equal", partial(_ens_fn, weights=None), panels, origin)
        )
        if not comp_hist["m0_w3"]:
            cw: list[float] | None = None  # no history yet: equal weights
        else:
            cw = crps_weights([comp_hist[c] for c in ("m0_w3", "m0_w6", "ewm_hl2")])
        frames["ens_crpsw"].append(
            _scored("ens_crpsw", partial(_ens_fn, weights=cw), panels, origin)
        )

        # Trailing-origin CRPS feeds the adaptive ensemble (expanding window).
        for cname in comp_hist:
            per_origin_crps = frames[cname][-1].groupby("origin")["crps"].mean()
            comp_hist[cname].append(float(per_origin_crps.iloc[0]))

        del panels
        gc.collect()
        print(f"[exp5] origin {origin.isoformat()} done", flush=True)

    print(f"[exp5] scored {n_origins_run} origins", flush=True)
    results = {name: pd.concat(flist, ignore_index=True) for name, flist in frames.items()}

    # --- Summarize: mean over segments (one row per segment x model) ---
    summary = _summarize(pd.concat(results.values(), ignore_index=True))
    means = summary.groupby("model").agg(
        crps=("crps", "mean"),
        mae=("mae", "mean"),
        coverage_80=("coverage_80", "mean"),
        n_cells=("n_origins", "sum"),
    )
    table_rows: list[dict[str, object]] = []
    print(f"\n{'model':<10} {'CRPS':>7} {'MAE':>7} {'cov80':>6}  beats_m0_w3", flush=True)
    m0_w3_crps = float(means.loc["m0_w3", "crps"])
    for name in results:
        crps = float(means.loc[name, "crps"])
        mae = float(means.loc[name, "mae"])
        cov = float(means.loc[name, "coverage_80"])
        table_rows.append(
            {
                "model": name,
                "mean_crps": round(crps, 4),
                "mean_mae": round(mae, 4),
                "mean_coverage_80": round(cov, 4),
                "n_cells": int(means.loc[name, "n_cells"]),
                "beats_m0_w3": bool(crps < m0_w3_crps),
            }
        )
        print(
            f"{name:<10} {crps:7.3f} {mae:7.3f} {cov:6.3f}  {crps < m0_w3_crps}",
            flush=True,
        )

    out = {
        "experiment": "exp5_models",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seed": SEED,
            "hier_window": HIER_WINDOW,
        },
        "models": table_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp5_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[exp5] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
