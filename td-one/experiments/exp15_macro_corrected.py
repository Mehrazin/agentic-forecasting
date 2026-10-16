"""Experiment 15: macro overlay with the CORRECTED sign (negative lock-in).

exp8 (`experiments/exp8_macro.py`) showed a FRED-driven macro overlay
capturing "93% of oracle gain" — but it used the WRONG economics: rising
mortgage rates -> MORE switching (beta=+0.6 on the v1 simulator).

Simulator v2 corrected this to negative lock-in: rising rates -> LESS
switching (borrowers retain cheaper existing loans). The v2 default DGP
bakes this in: ``macro_shifts[d] = -FRED_BETA * winsorized_z[d]`` with
``FRED_BETA = 0.3``.

This experiment reruns the exp8 question on v2 with the corrected sign:

  m0              — macro-blind trailing-6 Jeffreys Beta-Binomial
  m0_macro_oracle — knows the true DGP component (-0.3 * winsorized z)
  m0_macro_est    — MacroM0: OLS-estimated beta on the FRED z-series
                    (does NOT know the winsorization or the true beta)

Key question: does the estimated overlay still capture ~93% of the
oracle gain with the corrected sign, or does the economics change the
story? Multiple seeds (0, 1, 2) per the "no single-seed claims" rule.

Usage:
    cd td-one && uv run python experiments/exp15_macro_corrected.py
    EXP15_N_CUSTOMERS=3000 uv run python experiments/exp15_macro_corrected.py
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
    fred_z = load_fred_z()

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
        model = MacroM0(trailing_window=6, fred_z=fred_z)
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
        print(f"[exp15] seed {seed}: simulating v2 DGP ...", flush=True)
        sim = MortgageSimulatorV2(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed)
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories)  # FRED macro on by default
        panels_full = discover_segments(histories, outcomes)
        origins = _default_origins(panels_full)
        print(
            f"[exp15] seed {seed}: {len(panels_full)} segments, {len(origins)} origins",
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
            print(f"[exp15] seed {seed} {name}: done", flush=True)

        per_seed[seed] = {name: round(float(np.mean(vals)), 4) for name, vals in seed_crps.items()}
        print(f"[exp15] seed {seed}: {per_seed[seed]}", flush=True)

        # Diagnostic: what beta does MacroM0 actually estimate on a
        # representative segment? (Negative confirms it found the
        # lock-in direction.)
        first_key = next(iter(panels_full))
        first_panel = panels_full[first_key]
        model = MacroM0(trailing_window=6, fred_z=fred_z)
        model.fit(
            first_panel["n_renewals"].to_numpy(),
            first_panel["n_switches"].to_numpy(),
            first_panel["month"],
        )
        beta_hats[seed] = [round(model.beta_hat_, 4)]
        print(
            f"[exp15] seed {seed}: MacroM0 beta_hat={model.beta_hat_:.4f} "
            f"(true DGP beta={-FRED_BETA})",
            flush=True,
        )

    # Aggregate across seeds: mean CRPS per model, oracle-gain capture.
    models = list(fns)
    mean_crps = {m: round(float(np.mean([per_seed[s][m] for s in SEEDS])), 4) for m in models}
    gain_oracle = mean_crps["m0"] - mean_crps["m0_macro_oracle"]
    gain_est = mean_crps["m0"] - mean_crps["m0_macro_est"]
    capture = round(gain_est / gain_oracle, 3) if gain_oracle > 0 else None

    print("\n[exp15] === cross-seed means ===")
    for m in models:
        print(f"[exp15]   {m}: CRPS {mean_crps[m]}")
    print(f"[exp15]   oracle gain: {gain_oracle:.4f}, est gain: {gain_est:.4f}")
    print(f"[exp15]   fraction of oracle gain captured: {capture}")

    out = {
        "experiment": "exp15_macro_corrected",
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "seeds": list(SEEDS),
            "dgp": "simulator_v2, use_fred_macro=True (default)",
            "true_beta": -FRED_BETA,
            "fred_z_cap": FRED_Z_CAP,
            "note": "Corrected negative lock-in sign. exp8's +0.6 beta "
            "on v1 was the wrong economics.",
        },
        "per_seed_crps": per_seed,
        "beta_hats": beta_hats,
        "mean_crps": mean_crps,
        "oracle_gain": round(gain_oracle, 4),
        "est_gain": round(gain_est, 4),
        "fraction_oracle_gain_captured": capture,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp15_macro_corrected_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[exp15] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
