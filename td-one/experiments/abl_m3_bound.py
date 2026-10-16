"""Ablation: M3 delta-bound sensitivity.

Question: does the +/-0.75 bound on the agent's logit adjustment matter?
We re-run the 6 exp4 macro scenarios in MOCK mode (scripted deltas, no
LLM) with HybridReadout.MAX_DELTA in {0.25, 0.50, 0.75, 1.50, 3.00}
and report M3 CRPS per scenario per bound.

The bound is the methodological point of M3 (prevent runaway
hallucinated adjustments); this ablation measures its empirical bite.

Usage:
    cd td-one && uv run python experiments/abl_m3_bound.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from exp4_m3 import (  # noqa: E402
    N_TEST,
    SCENARIOS,
    apply_m3_forecast,
    build_scenario_panels,
    logit,
    sigmoid,
)

from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.hybrid import HybridReadout  # noqa: E402

BOUNDS = [0.25, 0.50, 0.75, 1.50, 3.00]
SEED = 42


def main() -> int:
    t0 = time.time()
    panels = build_scenario_panels()
    rng = np.random.default_rng(SEED)
    original_bound = HybridReadout.MAX_DELTA

    # Precompute scenario truth once (same seed as exp4 for comparability).
    scenarios: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        sid = str(scenario["id"])
        n = panels[sid]["N"]
        a = panels[sid]["A"]
        m0 = HistoricalBaseline()
        m0.fit(n, a)
        base_rate = m0.expected_rate
        true_rate = float(
            sigmoid(
                logit(float(np.clip(base_rate, 1e-6, 1 - 1e-6))) + float(scenario["true_shift"])
            )
        )
        actual = int(rng.binomial(N_TEST, true_rate))
        m0_samples = m0.predict_counts(N_TEST, seed=SEED)
        m0_crps = float(crps_counts(m0_samples[None, :], np.array([actual]))[0])
        scenarios.append(
            {
                "id": sid,
                "n": n,
                "a": a,
                "delta": float(scenario["mock_delta"]),
                "actual": actual,
                "m0_crps": round(m0_crps, 4),
            }
        )

    rows: list[dict[str, object]] = []
    try:
        for bound in BOUNDS:
            HybridReadout.MAX_DELTA = bound
            per_scenario: dict[str, float] = {}
            for s in scenarios:
                m3 = apply_m3_forecast(
                    s["n"],
                    s["a"],
                    N_TEST,
                    s["delta"],
                    seed=SEED,  # type: ignore[arg-type]
                )
                per_scenario[str(s["id"])] = round(
                    float(crps_counts(m3["samples"][None, :], np.array([s["actual"]]))[0]),
                    4,
                )
            mean_crps = round(float(np.mean(list(per_scenario.values()))), 4)
            rows.append(
                {"bound": bound, "m3_crps_by_scenario": per_scenario, "mean_crps": mean_crps}
            )
            print(
                f"[abl] bound={bound}: "
                + " ".join(f"{k}={v:.2f}" for k, v in per_scenario.items())
                + f" | mean={mean_crps:.2f}",
                flush=True,
            )
    finally:
        HybridReadout.MAX_DELTA = original_bound

    out = {
        "experiment": "abl_m3_bound",
        "mode": "mock (scripted deltas; bound is the only varied factor)",
        "bounds": BOUNDS,
        "m0_crps_by_scenario": {s["id"]: s["m0_crps"] for s in scenarios},
        "results": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "abl_m3_bound_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[abl] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
