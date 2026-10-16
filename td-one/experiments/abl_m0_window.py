"""Ablation: M0 trailing-window sensitivity.

Question: how much does the baseline's trailing window matter?
M0 is defined with trailing_window=6. We backtest trailing_window in
{3, 6, 12} on the standard scale-up configuration (10k customers x 66
months, 6 origins) and report mean CRPS / MAE / 80% coverage per window.

Pure-Python (no weights, no LLM); runs in ~1-2 minutes.

Usage:
    cd td-one && uv run python experiments/abl_m0_window.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import _default_origins, _summarize  # noqa: E402

from td_one.backtest.runner import run_backtest  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402

WINDOWS = [3, 6, 12]
N_CUSTOMERS = 10_000
N_MONTHS = 66
SEED = 0


def _m0_window_forecast(train, origin: date, window: int) -> dict[float, float]:
    """M0 forecast_fn with a parametrized trailing window.

    Mirrors the ``forecast("m0", ...)`` path in
    ``td_one.tools.forecasting`` (the scale-up's M0) exactly, except the
    trailing window is parametrized instead of hardcoded to 6.
    """
    from td_one.tools.forecasting import TIMESFM_LEVELS

    del origin
    n = train["n_renewals"].to_numpy()
    a = train["n_switches"].to_numpy()
    model = HistoricalBaseline(trailing_window=window)
    model.fit(n, a)
    n_upcoming = max(1, int(round(float(n[-window:].mean()))))
    return model.predict_rate_quantiles(n_upcoming, np.asarray(TIMESFM_LEVELS, dtype=float))


def main() -> int:
    t0 = time.time()
    print(f"[abl] Simulating {N_CUSTOMERS} x {N_MONTHS} (seed={SEED}) ...", flush=True)
    sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=SEED)
    histories = sim.generate_history()
    outcomes = sim.generate_outcomes(histories)

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(f"[abl] {len(panels_full)} segments, {len(origins)} origins", flush=True)

    rows: list[dict[str, object]] = []
    for window in WINDOWS:
        fn = partial(_m0_window_forecast, window=window)
        all_results = []
        for origin in origins:
            try:
                panels = discover_segments(histories, outcomes, observed_asof=origin)
            except ValueError:
                continue
            r = run_backtest(panels, fn, [origin], seed=SEED)
            r["model"] = f"m0_w{window}"
            all_results.append(r)
        import pandas as pd

        summary = _summarize(pd.concat(all_results, ignore_index=True))
        seg_means = summary[summary["model"] == f"m0_w{window}"]
        mean_crps = float(seg_means["crps"].mean())
        mean_mae = float(seg_means["mae"].mean())
        mean_cov = float(seg_means["coverage_80"].mean())
        rows.append(
            {
                "window": window,
                "mean_crps": mean_crps,
                "mean_mae": mean_mae,
                "mean_coverage_80": mean_cov,
                "n_cells": int(seg_means["n_origins"].sum()),
            }
        )
        print(
            f"[abl] window={window}: CRPS={mean_crps:.3f} MAE={mean_mae:.3f} cov80={mean_cov:.3f}",
            flush=True,
        )

    out = {
        "experiment": "abl_m0_window",
        "config": {"n_customers": N_CUSTOMERS, "n_months": N_MONTHS, "seed": SEED},
        "windows": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "abl_m0_window_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[abl] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
