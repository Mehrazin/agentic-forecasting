"""Experiment 8: macro-grounded DGP — does macro structure reward macro-aware models.

The default simulator DGP has no macro dynamics, so macro-aware models
CANNOT win by construction. Here we overlay a FRED-driven component:
the 30y mortgage rate's 3-month change (z-scored) shifts the switching
logit each month, aligned by calendar month (simulator runs 2021-04 ->
2026-09, 1:1 with FRED history).

Compares:
  M0            — macro-blind (trailing-6 Jeffreys Beta-Binomial)
  M0-macro-or   — oracle: knows beta and the FRED series (upper bound on
                  the value of macro information)
  M0-macro-est  — realistic: OLS of trailing-24m logit(rates) on z,
                  then forecasts with z[forecast_month]

If the oracle wins big and the estimated version captures part of it,
macro-aware modeling is justified follow-up. If the oracle barely wins,
macro structure doesn't move the needle.

Usage:
    cd td-one && uv run python experiments/exp8_macro.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import _default_origins, _summarize  # noqa: E402

from td_one.backtest.runner import run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

N_CUSTOMERS = int(os.environ.get("EXP8_N_CUSTOMERS", "10000"))
N_MONTHS = 66
SEED = 0
BETA_MACRO = 0.6
FRED_ID = "MORTGAGE30US"


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def load_macro_z() -> dict[date, float]:
    """Month-end 3m change in 30y mortgage rate, z-scored, keyed by month-end."""
    df = pd.read_parquet(ROOT / "data" / "fred" / f"{FRED_ID}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date")["value"].resample("ME").last().dropna()
    mom3 = monthly.pct_change(3)  # 3-month relative change
    z = (mom3 - mom3.mean()) / mom3.std()
    out: dict[date, float] = {}
    for ts, val in z.items():
        d = ts.date()
        # month-end key
        out[d] = float(val)
    return out


def main() -> int:
    t0 = time.time()
    print("[exp8] Loading FRED macro series ...", flush=True)
    z_by_month = load_macro_z()

    print(f"[exp8] Simulating {N_CUSTOMERS} x {N_MONTHS} with macro overlay ...", flush=True)
    sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED)
    histories = sim.generate_history()
    month_dates = sim._month_dates()
    macro_shifts = {d: BETA_MACRO * z_by_month.get(d, 0.0) for d in month_dates}
    n_nonzero = sum(1 for v in macro_shifts.values() if v != 0.0)
    print(
        f"[exp8] macro shifts: {n_nonzero}/{len(macro_shifts)} months nonzero, "
        f"std={np.std(list(macro_shifts.values())):.3f} logit",
        flush=True,
    )
    outcomes = sim.generate_outcomes(histories, macro_shifts=macro_shifts)

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(f"[exp8] {len(panels_full)} segments, {len(origins)} origins", flush=True)

    # Precompute z aligned to panel months for the estimated version.
    z_series = pd.Series(macro_shifts) / BETA_MACRO  # back to z-scores

    def m0_fn(train, origin):
        n = train["n_renewals"].to_numpy()
        a = train["n_switches"].to_numpy()
        model = HistoricalBaseline(trailing_window=6)
        model.fit(n, a)
        n_up = max(1, int(round(float(n[-6:].mean()))))
        return model.predict_rate_quantiles(n_up, np.asarray(TIMESFM_LEVELS, dtype=float))

    def m0_oracle_fn(train, origin):
        base = m0_fn(train, origin)
        # Forecast month = month after origin.
        fmonth = (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()
        shift = macro_shifts.get(fmonth, 0.0)
        return {lvl: _sigmoid(_logit(rate) + shift) for lvl, rate in base.items()}

    def m0_est_fn(train, origin):
        n = train["n_renewals"].to_numpy()
        a = train["n_switches"].to_numpy()
        # OLS of trailing-24m logit(rates) on z.
        months = pd.to_datetime(train["month"])
        z_vals, y_vals = [], []
        for i in range(max(0, len(train) - 24), len(train)):
            d = months.iloc[i].date()
            z = z_series.get(d)
            if z is None or np.isnan(z):
                continue
            r = a[i] / max(n[i], 1)
            z_vals.append(z)
            y_vals.append(_logit(r))
        beta_hat = 0.0
        if len(z_vals) >= 12:
            design = np.column_stack([np.ones(len(z_vals)), np.asarray(z_vals)])
            y = np.asarray(y_vals)
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            beta_hat = float(coef[1])
        base = m0_fn(train, origin)
        fmonth = (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()
        z_f = z_series.get(fmonth, 0.0)
        if z_f is None or np.isnan(z_f):
            z_f = 0.0
        return {lvl: _sigmoid(_logit(rate) + beta_hat * z_f) for lvl, rate in base.items()}

    fns = {"m0": m0_fn, "m0_macro_oracle": m0_oracle_fn, "m0_macro_est": m0_est_fn}
    all_results = []
    for name, fn in fns.items():
        for origin in origins:
            try:
                panels = discover_segments(histories, outcomes, observed_asof=origin)
            except ValueError:
                continue
            r = run_backtest(panels, fn, [origin], seed=SEED)
            r["model"] = name
            all_results.append(r)
        print(f"[exp8] {name}: done", flush=True)

    summary = _summarize(pd.concat(all_results, ignore_index=True))
    table = []
    for name in fns:
        seg = summary[summary["model"] == name]
        table.append(
            {
                "model": name,
                "mean_crps": round(float(seg["crps"].mean()), 3),
                "mean_mae": round(float(seg["mae"].mean()), 3),
                "mean_coverage_80": round(float(seg["coverage_80"].mean()), 3),
            }
        )
        print(f"[exp8] {table[-1]}", flush=True)

    out = {
        "experiment": "exp8_macro",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seed": SEED,
            "beta_macro": BETA_MACRO,
            "fred_series": FRED_ID,
        },
        "results": table,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp8_macro_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[exp8] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
