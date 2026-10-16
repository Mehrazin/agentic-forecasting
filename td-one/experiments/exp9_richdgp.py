"""Experiment 9: richer DGP (simulator v2) — do conclusions change?.

Runs the current best models on the v2 data-generating process, which adds:
  - renewal offers with TD-vs-competitor spread driving switching (S-curved),
  - customer heterogeneity (utilization trend, life events, channel, advisor),
  - FRED macro (MORTGAGE30US 3m change) baked in by default,
  - prepayment-canon mechanics: burnout, 30-month seasoning, arctan incentive.

Compares:
  m0_w3   — trailing-3 Jeffreys Beta-Binomial (best CRPS on v1: 3.196)
  m0_w6   — trailing-6 (published M0: 5.360)
  conf_w3 — conformalized M0 w3 (best coverage on v1: 0.914)

Question: does the richer structure (observable heterogeneity + macro +
offers) change the model ranking, or does M0 still win?

Result (2000 customers, seed 0): m0_w6 CRPS 0.848 BEATS m0_w3 (0.986) —
a ranking reversal from v1 where w3 won. The richer DGP (FRED macro +
smoother S-curved incentives) rewards the longer window. conf_w3 matches
m0_w3 on CRPS (0.990) with similar coverage (0.900).

Kept small (2000 customers) for the shared VM.

Usage:
    cd td-one && uv run python experiments/exp9_richdgp.py
"""

from __future__ import annotations

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
from td_one.data_gen.simulator_v2 import MortgageSimulatorV2  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.conformal import ConformalM0  # noqa: E402
from td_one.models.tsfm import TIMESFM_LEVELS  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

N_CUSTOMERS = int(os.environ.get("EXP9_N_CUSTOMERS", "2000"))
N_MONTHS = 66
SEED = 0
LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)
N_UPCOMING_K = 6


def _n_upcoming(train: pd.DataFrame, k: int = N_UPCOMING_K) -> int:
    n = train["n_renewals"].to_numpy()
    return max(1, int(round(float(n[-k:].mean()))))


def _m0_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    model = HistoricalBaseline(trailing_window=window)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    return model.predict_rate_quantiles(_n_upcoming(train, k=window), LEVELS)


def _conf_fn(train: pd.DataFrame, origin: date, window: int) -> dict[float, float]:
    del origin
    model = ConformalM0(trailing_window=window)
    model.fit(train["n_renewals"].to_numpy(), train["n_switches"].to_numpy())
    return model.predict_rate_quantiles(_n_upcoming(train, k=window))


def main() -> int:
    t0 = time.time()
    print(f"[exp9] Simulating v2 DGP: {N_CUSTOMERS} x {N_MONTHS} ...", flush=True)
    sim = MortgageSimulatorV2(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED)
    histories = sim.generate_history()
    print(f"[exp9] history: {len(histories)} records in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    outcomes = sim.generate_outcomes(histories)  # FRED macro on by default
    sw = sum(1 for o in outcomes if o.switched_lender_flag)
    print(
        f"[exp9] outcomes: {len(outcomes)} ({sw / len(outcomes):.3f} switch) "
        f"in {time.time() - t0:.1f}s; offers: {len(sim.offers)}",
        flush=True,
    )

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(f"[exp9] {len(panels_full)} segments, {len(origins)} origins", flush=True)

    fns = {
        "m0_w3": lambda tr, og: _m0_fn(tr, og, 3),
        "m0_w6": lambda tr, og: _m0_fn(tr, og, 6),
        "conf_w3": lambda tr, og: _conf_fn(tr, og, 3),
    }
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
        print(f"[exp9] {name}: done", flush=True)

    summary = _summarize(pd.concat(all_results, ignore_index=True))
    print("\n[exp9] Summary (mean CRPS / coverage):")
    print(summary.to_string())
    out_path = ROOT / "experiments" / "exp9_results.json"
    summary.to_json(out_path, orient="index", indent=2)
    print(f"[exp9] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
