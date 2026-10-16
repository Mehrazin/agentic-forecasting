"""Experiment 13: TSFM + FRED rematch — the fair fight.

Real Chronos-2 / TimesFM-2.5 never significantly beat M0 on 5 seeds
(multiseed_report.md), but they were tested WITHOUT macro information
while exp8 showed macro structure is worth 32% CRPS when present.
Neither TSFM accepts exogenous covariates natively (univariate
zero-shot APIs), so macro-conditioning uses the residual approach in
``td_one.models.tsfm_macro`` (documented there).

Compares on two DGPs x seeds 0-2 (2000 customers x 66 months):
  standard: v1 simulator, no macro overlay (macro TSFM should NOT win —
            honest null; beta_hat ~ 0 reduces to plain TSFM)
  macro:    v1 simulator + FRED macro_shifts overlay (exp8-style,
            beta=0.6; macro-conditioned models should win)

Models:
  conf_m0_auto  — conformal M0, window auto (3/6/12); the published baseline
  macro_m0      — MacroM0 (OLS FRED overlay, window 6)
  chronos2 / timesfm25 — plain real-weight TSFM
  chronos2_macro / timesfm25_macro — residual macro-conditioned TSFM

Usage:
    cd td-one && uv run python experiments/exp13_tsfm_macro.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import (  # noqa: E402
    _default_origins,
    _m0_auto_window_forecast,
    _summarize,
)

from td_one.backtest.runner import run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.models.macro_m0 import MacroM0, load_fred_z  # noqa: E402
from td_one.models.tsfm import (  # noqa: E402
    Chronos2Wrapper,
    TimesFMWrapper,
)
from td_one.models.tsfm_macro import MacroConditionedTSFM  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402
from td_one.tools.forecasting import _panel_rates, _real_weights_dir  # noqa: E402

N_CUSTOMERS = 2000
N_MONTHS = 66
SEEDS = (0, 1, 2)
BETA_MACRO = 0.6


def _month_dates(train: pd.DataFrame) -> list[date]:
    return [d.date() for d in pd.to_datetime(train["month"])]


def _forecast_month(origin: date) -> date:
    return (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()


def _plain_tsfm_fn(wrapper: Chronos2Wrapper | TimesFMWrapper):
    """forecast_fn running a TSFM wrapper one step ahead (tools-parity)."""

    def fn(train: pd.DataFrame, origin: date) -> dict[float, float]:
        n, a = _panel_rates(train, origin)
        rates = np.asarray(a, dtype=float) / np.maximum(np.asarray(n, dtype=float), 1.0)
        levels, values = wrapper.forecast_quantiles(rates, horizon=1)
        raw = np.sort(np.asarray(values[:, 0], dtype=float))
        lvls = np.asarray(levels, dtype=float)
        order = np.argsort(lvls, kind="stable")
        return {float(lvls[i]): float(raw[j]) for j, i in enumerate(order)}

    return fn


def _macro_tsfm_fn(mc: MacroConditionedTSFM):
    """forecast_fn for the residual macro-conditioned TSFM."""

    def fn(train: pd.DataFrame, origin: date) -> dict[float, float]:
        n, a = _panel_rates(train, origin)
        n_arr = np.asarray(n, dtype=float)
        a_arr = np.asarray(a, dtype=float)
        return mc.forecast_quantiles(n_arr, a_arr, _month_dates(train), _forecast_month(origin))

    return fn


def _macro_m0_fn(
    train: pd.DataFrame, origin: date, fred_z: dict[date, float]
) -> dict[float, float]:
    n = np.asarray(train["n_renewals"].to_numpy(), dtype=float)
    a = np.asarray(train["n_switches"].to_numpy(), dtype=float)
    m = MacroM0(trailing_window=6, fred_z=fred_z)
    m.fit(n, a, _month_dates(train))
    n_up = max(1, int(round(float(n[-6:].mean()))))
    return m.predict_rate_quantiles(n_up, _forecast_month(origin))


def main() -> int:
    t0 = time.time()
    print("[exp13] Loading FRED z-series ...", flush=True)
    fred_z = load_fred_z()

    results: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        for dgp in ("standard", "macro"):
            key = f"{dgp}/seed{seed}"
            print(f"[exp13] {key}: simulating {N_CUSTOMERS}x{N_MONTHS} ...", flush=True)
            sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed)
            histories = sim.generate_history()
            shifts = (
                {d: BETA_MACRO * fred_z.get(d, 0.0) for d in sim._month_dates()}
                if dgp == "macro"
                else None
            )
            outcomes = sim.generate_outcomes(histories, macro_shifts=shifts)
            panels_full = discover_segments(histories, outcomes)
            origins = _default_origins(panels_full)
            print(f"[exp13] {key}: {len(panels_full)} segments, {len(origins)} origins", flush=True)

            fns: dict[str, object] = {
                "conf_m0_auto": lambda train, origin: _m0_auto_window_forecast(
                    train, origin, conformal=True
                ),
                "macro_m0": lambda train, origin: _macro_m0_fn(train, origin, fred_z),
            }
            # One TSFM at a time (weights ~1GB each); unload between.
            for wname, wcls, mname in (
                ("chronos2", Chronos2Wrapper, "chronos2"),
                ("timesfm25", TimesFMWrapper, "timesfm25"),
            ):
                print(f"[exp13] {key}: loading {wname} ...", flush=True)
                wrapper = wcls(use_mock=False, weights_dir=_real_weights_dir(wname))
                mc = MacroConditionedTSFM(wrapper, fred_z=fred_z)
                fns[mname] = _plain_tsfm_fn(wrapper)
                fns[f"{mname}_macro"] = _macro_tsfm_fn(mc)

                for name in (mname, f"{mname}_macro"):
                    all_r = []
                    for origin in origins:
                        try:
                            panels = discover_segments(histories, outcomes, observed_asof=origin)
                        except ValueError:
                            continue
                        r = run_backtest(panels, fns[name], [origin], seed=seed)  # type: ignore[arg-type]
                        r["model"] = name
                        all_r.append(r)
                    summary = _summarize(pd.concat(all_r, ignore_index=True))
                    seg = summary[summary["model"] == name]
                    results.setdefault(key, {})[name] = {
                        "mean_crps": round(float(seg["crps"].mean()), 3),
                        "mean_mae": round(float(seg["mae"].mean()), 3),
                        "mean_coverage_80": round(float(seg["coverage_80"].mean()), 3),
                        "n_segments": int(seg["segment"].nunique()),
                    }
                    print(f"[exp13] {key} {name}: {results[key][name]}", flush=True)
                wrapper.unload()
                del wrapper, mc
                import gc

                gc.collect()

            # Non-TSFM models (no weights needed).
            for name in ("conf_m0_auto", "macro_m0"):
                all_r = []
                for origin in origins:
                    try:
                        panels = discover_segments(histories, outcomes, observed_asof=origin)
                    except ValueError:
                        continue
                    r = run_backtest(panels, fns[name], [origin], seed=seed)  # type: ignore[arg-type]
                    r["model"] = name
                    all_r.append(r)
                summary = _summarize(pd.concat(all_r, ignore_index=True))
                seg = summary[summary["model"] == name]
                results.setdefault(key, {})[name] = {
                    "mean_crps": round(float(seg["crps"].mean()), 3),
                    "mean_mae": round(float(seg["mae"].mean()), 3),
                    "mean_coverage_80": round(float(seg["coverage_80"].mean()), 3),
                    "n_segments": int(seg["segment"].nunique()),
                }
                print(f"[exp13] {key} {name}: {results[key][name]}", flush=True)

    out = {
        "experiment": "exp13_tsfm_macro",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seeds": list(SEEDS),
            "beta_macro": BETA_MACRO,
            "fred_series": "MORTGAGE30US",
            "baseline": "conf_m0_auto (conformal, window auto 3/6/12)",
            "tsfm_conditioning": "residual (no native covariate support)",
        },
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp13_tsfm_macro_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[exp13] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
