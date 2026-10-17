"""R2 Experiment 15: PIT-safe macro overlay with corrected sign.

R1 (exp15_macro_corrected.py) used ``load_fred_z()`` without ``asof`` for
the estimated model — z-score mean/std on full FRED history (1986-2026),
leaking future into every forecast origin (DATA_AUDIT.md P0-1), plus
``z[forecast_month]`` lookahead.

R2 fixes the ESTIMATED model to per-origin ``load_fred_z(asof=origin)``.
The DGP (simulator v2's internal ``load_fred_macro_z``) and the ORACLE
(which knows the true DGP component) are unchanged — ground truth.

R1 result: oracle gain ~0, estimated overlay HURT by 0.15 CRPS. R2 tests
whether PIT safety changes this (it shouldn't — the damage was from
unstable beta_hat, not the leak — but let's verify).

Usage:
    cd td-one && uv run python experiments/r2_exp15_macro_corrected.py
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

from scale_up import _default_origins  # noqa: E402

from td_one.backtest.runner import run_backtest  # noqa: E402
from td_one.data_gen.simulator_v2 import (  # noqa: E402
    FRED_BETA,
    FRED_Z_CAP,
    MortgageSimulatorV2,
    load_fred_macro_z,
)
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.macro_m0 import MacroM0, load_fred_z  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

N_CUSTOMERS = int(os.environ.get("EXP15_N_CUSTOMERS", "2000"))
N_MONTHS = 66
SEEDS = (0, 1, 2)
LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def _n_upcoming(train: pd.DataFrame) -> int:
    n = train["n_renewals"].to_numpy()
    return max(1, int(round(float(n[-6:].mean()))))


def _forecast_month(origin: date) -> date:
    return (pd.Timestamp(origin) + pd.offsets.MonthEnd(1)).date()


def main() -> int:
    t0 = time.time()

    # The DGP's true macro component (what the oracle knows). v2 applies
    # -FRED_BETA * winsorized_z with the CORRECTED negative lock-in sign.
    z_raw = load_fred_macro_z()

    def oracle_shift(d: date) -> float:
        zval = z_raw.get(d, 0.0)
        zval = max(-FRED_Z_CAP, min(FRED_Z_CAP, zval))
        return -FRED_BETA * zval

    # The estimated model sees only the raw (unwinsorized) z-series.
    # R2: PIT-safe per-origin (cached). DGP/oracle unchanged from R1.
    _fred_z_cache: dict[date, dict[date, float]] = {}

    def _fred_z_pit(origin: date) -> dict[date, float]:
        if origin not in _fred_z_cache:
            _fred_z_cache[origin] = load_fred_z(asof=origin)
        return _fred_z_cache[origin]

    def m0_fn(train: pd.DataFrame, origin: date) -> dict[float, float]:
        del origin
        model = HistoricalBaseline(trailing_window=6)
        model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
        return model.predict_rate_quantiles(_n_upcoming(train), LEVELS)

    def oracle_fn(train: pd.DataFrame, origin: date) -> dict[float, float]:
        base = m0_fn(train, origin)
        shift = oracle_shift(_forecast_month(origin))
        if shift == 0.0:
            return base
        return {lvl: _sigmoid(_logit(rate) + shift) for lvl, rate in base.items()}

    def est_fn(train: pd.DataFrame, origin: date) -> dict[float, float]:
        model = MacroM0(trailing_window=6, fred_z=_fred_z_pit(origin))
        model.fit(
            train["n_renewals"].to_numpy(),
            train["n_switches"].to_numpy(),
            train["month"],
        )
        return model.predict_rate_quantiles(_n_upcoming(train), _forecast_month(origin), LEVELS)

    fns = {
        "m0": m0_fn,
        "m0_macro_oracle": oracle_fn,
        "m0_macro_est": est_fn,
    }

    per_seed: dict[int, dict[str, float]] = {}
    beta_hats: dict[int, list[float]] = {}
    for seed in SEEDS:
        print(f"[r2_exp15] seed {seed}: simulating v2 DGP ...", flush=True)
        sim = MortgageSimulatorV2(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed)
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories)  # FRED macro on by default
        panels_full = discover_segments(histories, outcomes)
        origins = _default_origins(panels_full)
        print(
            f"[r2_exp15] seed {seed}: {len(panels_full)} segments, {len(origins)} origins",
            flush=True,
        )

        seed_crps: dict[str, list[float]] = {name: [] for name in fns}
        for name, fn in fns.items():
            for origin in origins:
                try:
                    panels = discover_segments(histories, outcomes, observed_asof=origin)
                except ValueError:
                    continue
                r = run_backtest(panels, fn, [origin], seed=seed)
                seed_crps[name].extend(r["crps"].tolist())
            print(f"[r2_exp15] seed {seed} {name}: done", flush=True)

        per_seed[seed] = {name: round(float(np.mean(vals)), 4) for name, vals in seed_crps.items()}
        print(f"[r2_exp15] seed {seed}: {per_seed[seed]}", flush=True)

        # Diagnostic: what beta does MacroM0 actually estimate on a
        # representative segment? (Negative confirms it found the
        # lock-in direction.) R2: uses PIT-safe z at the last origin.
        first_key = next(iter(panels_full))
        first_panel = panels_full[first_key]
        diag_origin = origins[-1] if origins else date(2026, 1, 31)
        model = MacroM0(trailing_window=6, fred_z=_fred_z_pit(diag_origin))
        model.fit(
            first_panel["n_renewals"].to_numpy(),
            first_panel["n_switches"].to_numpy(),
            first_panel["month"],
        )
        beta_hats[seed] = [round(model.beta_hat_, 4)]
        print(
            f"[r2_exp15] seed {seed}: MacroM0 beta_hat={model.beta_hat_:.4f} "
            f"(true DGP beta={-FRED_BETA})",
            flush=True,
        )

    # Aggregate across seeds: mean CRPS per model, oracle-gain capture.
    models = list(fns)
    mean_crps = {m: round(float(np.mean([per_seed[s][m] for s in SEEDS])), 4) for m in models}
    gain_oracle = mean_crps["m0"] - mean_crps["m0_macro_oracle"]
    gain_est = mean_crps["m0"] - mean_crps["m0_macro_est"]
    capture = round(gain_est / gain_oracle, 3) if gain_oracle > 0 else None

    print("\n[r2_exp15] === cross-seed means ===")
    for m in models:
        print(f"[r2_exp15]   {m}: CRPS {mean_crps[m]}")
    print(f"[r2_exp15]   oracle gain: {gain_oracle:.4f}, est gain: {gain_est:.4f}")
    print(f"[r2_exp15]   fraction of oracle gain captured: {capture}")

    out = {
        "experiment": "r2_exp15_macro_corrected",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seeds": list(SEEDS),
            "dgp": "simulator_v2, use_fred_macro=True (default)",
            "true_beta": -FRED_BETA,
            "fred_z_cap": FRED_Z_CAP,
            "note": "PIT-safe R2: estimated model uses per-origin "
            "load_fred_z(asof=origin). DGP/oracle unchanged from R1.",
            "pit_safe": True,
        },
        "per_seed_crps": per_seed,
        "beta_hats": beta_hats,
        "mean_crps": mean_crps,
        "oracle_gain": round(gain_oracle, 4),
        "est_gain": round(gain_est, 4),
        "fraction_oracle_gain_captured": capture,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "r2_exp15_macro_corrected_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[r2_exp15] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
