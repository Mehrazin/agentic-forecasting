"""R2-10b: Multi-seed classical macro with window confounding resolved.

Argon flagged that r2_exp10's "-0.357" is confounded with window length:
macro_m0 uses trailing_window=6 and beats m0_w6, but m0_w3 (1.457) beats
macro_m0 (1.923) by 32% on seed=0. The 3-month z[t] may just compensate for
m0_w6's stale 6-month average rather than adding true macro value.

This experiment runs 5 seeds (0-4) on the macro DGP with four models:
  - m0_w3            : responsive baseline (no macro)
  - m0_w6            : sluggish baseline (no macro)
  - macro_m0_w6      : window=6 + lagged macro (original R2 finding)
  - macro_m0_w3      : window=3 + lagged macro (the critical test)

Critical comparison: macro_m0_w3 vs m0_w3. If macro can't beat the
responsive baseline, the "exploratory positive" becomes "no evidence."

PIT safety: same as r2_exp10 — DGP uses full-history z (ground truth),
model uses per-origin load_fred_z(asof=origin).

Usage:
    cd td-one && .venv/bin/python experiments/r2_exp10_multiseed.py
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import _default_origins  # noqa: E402

from td_one.backtest.runner import ForecastFn, run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.macro_m0 import MacroM0, load_fred_z  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

N_CUSTOMERS = 2000
N_MONTHS = 66
SEEDS = (0, 1, 2, 3, 4)
BETA_MACRO = 0.6
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


def _macro_fn(
    train: pd.DataFrame,
    origin: date,
    fred_z: dict[date, float],
    window: int,
) -> dict[float, float]:
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    months = pd.to_datetime(train["month"])
    m = MacroM0(trailing_window=window, fred_z=fred_z)
    m.fit(n, a, months)
    fmonth = (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()
    return m.predict_rate_quantiles(_n_upcoming(train, window), fmonth, LEVELS)


def _run_seed(
    seed: int, fred_z_full: dict[date, float]
) -> dict[str, dict[str, float | list[float]]]:
    """Backtest 4 models on macro DGP for one seed.

    Returns model -> {mean_crps, se_crps, per_origin_crps}.
    """
    sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed)
    histories = sim.generate_history()
    macro_shifts = {d: BETA_MACRO * fred_z_full.get(d, 0.0) for d in sim._month_dates()}
    outcomes = sim.generate_outcomes(histories, macro_shifts=macro_shifts)

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(f"[r2_exp10_ms] seed {seed}: {len(panels_full)} segs, {len(origins)} origins", flush=True)

    panels_by_origin: dict[date, dict[str, pd.DataFrame]] = {}
    for origin in origins:
        try:
            panels_by_origin[origin] = discover_segments(histories, outcomes, observed_asof=origin)
        except ValueError:
            continue
    origins = [o for o in origins if o in panels_by_origin]
    fred_z_by_origin: dict[date, dict[date, float]] = {o: load_fred_z(asof=o) for o in origins}
    print(f"[r2_exp10_ms] seed {seed}: PIT-safe fred_z for {len(origins)} origins", flush=True)
    del histories, outcomes, panels_full
    gc.collect()

    per_origin: dict[str, list[float]] = {
        "m0_w3": [],
        "m0_w6": [],
        "macro_m0_w6": [],
        "macro_m0_w3": [],
    }

    for origin in origins:
        panels = panels_by_origin[origin]
        fz = fred_z_by_origin[origin]
        fns: dict[str, ForecastFn] = {
            "m0_w3": partial(_m0_fn, window=3),
            "m0_w6": partial(_m0_fn, window=6),
            "macro_m0_w6": partial(_macro_fn, fred_z=fz, window=6),
            "macro_m0_w3": partial(_macro_fn, fred_z=fz, window=3),
        }
        for name, fn in fns.items():
            r = run_backtest(panels, fn, [origin], seed=seed)
            if r.empty:
                continue
            r["model"] = name
            per_origin[name].append(float(r["crps"].mean()))

    out: dict[str, dict[str, float | list[float]]] = {}
    for name, vals in per_origin.items():
        arr = np.asarray(vals, dtype=float)
        out[name] = {
            "mean_crps": round(float(arr.mean()), 4),
            "se_crps": round(float(arr.std(ddof=1) / np.sqrt(len(arr))), 4)
            if len(arr) > 1
            else 0.0,
            "n_origins": len(arr),
            "per_origin_crps": [round(float(v), 5) for v in arr],
        }
    print(f"[r2_exp10_ms] seed {seed} done", flush=True)
    return out


def main() -> int:
    t0 = time.time()
    print("[r2_exp10_ms] Loading FRED z-series (full history, DGP ground truth) ...", flush=True)
    fred_z_full = load_fred_z()

    per_seed: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = _run_seed(seed, fred_z_full)
        gc.collect()

    # Aggregate across seeds: mean of per-seed means, SE of per-seed means.
    model_names = ["m0_w3", "m0_w6", "macro_m0_w6", "macro_m0_w3"]
    aggregate: dict[str, dict[str, float]] = {}
    for name in model_names:
        means = np.array([per_seed[str(s)][name]["mean_crps"] for s in SEEDS], dtype=float)
        aggregate[name] = {
            "mean_crps": round(float(means.mean()), 4),
            "se_crps": round(float(means.std(ddof=1) / np.sqrt(len(means))), 4),
            "n_seeds": len(SEEDS),
        }

    # Paired t-test: macro_m0_w3 vs m0_w3 (per-seed means as paired obs).
    w3 = np.array([per_seed[str(s)]["m0_w3"]["mean_crps"] for s in SEEDS])
    mw3 = np.array([per_seed[str(s)]["macro_m0_w3"]["mean_crps"] for s in SEEDS])
    t_stat, p_val = stats.ttest_rel(mw3, w3)

    # Replicability check: original -0.357 = macro_m0_w6 - m0_w6 on seed 0.
    w6 = np.array([per_seed[str(s)]["m0_w6"]["mean_crps"] for s in SEEDS])
    mw6 = np.array([per_seed[str(s)]["macro_m0_w6"]["mean_crps"] for s in SEEDS])
    t62, p62 = stats.ttest_rel(mw6, w6)

    out = {
        "experiment": "r2_exp10_multiseed",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seeds": list(SEEDS),
            "beta_macro": BETA_MACRO,
            "models": model_names,
            "pit_safe": True,
            "dgp_z": "full-history (ground truth)",
            "model_z": "per-origin load_fred_z(asof=origin)",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "tests": {
            "macro_m0_w3_vs_m0_w3": {
                "delta": round(float((mw3 - w3).mean()), 4),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_val), 4),
                "n": len(SEEDS),
            },
            "macro_m0_w6_vs_m0_w6": {
                "delta": round(float((mw6 - w6).mean()), 4),
                "t_stat": round(float(t62), 3),
                "p_value": round(float(p62), 4),
                "n": len(SEEDS),
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "r2_exp10_multiseed_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[r2_exp10_ms] wrote {out_path}", flush=True)

    print("\n=== Aggregate (5 seeds) ===", flush=True)
    for name in model_names:
        a = aggregate[name]
        print(f"  {name:12s} {a['mean_crps']:.4f} ± {a['se_crps']:.4f}", flush=True)
    tests: dict[str, dict[str, dict[str, float]]] = out["tests"]  # type: ignore[assignment]
    t_w3 = tests["macro_m0_w3_vs_m0_w3"]
    t_w6 = tests["macro_m0_w6_vs_m0_w6"]
    print(
        f"\n  macro_m0_w3 vs m0_w3: delta={t_w3['delta']:+.4f}, "
        f"t={t_stat:.2f}, p={p_val:.4f}",
        flush=True,
    )
    print(
        f"  macro_m0_w6 vs m0_w6: delta={t_w6['delta']:+.4f}, "
        f"t={t62:.2f}, p={p62:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
