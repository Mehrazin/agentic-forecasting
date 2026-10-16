"""Prototype: regime-classifier hybrid (M3c).

Hypothesis: LLMs are better at categorical judgment (UP/DOWN/NONE) than
at precise numerical deltas (which they undershoot and sometimes sign-flip).
M3c = classify the regime, then apply a DATA-calibrated delta per class,
instead of trusting the LLM's raw logit number.

This prototype tests the MECHANISM in mock mode: we simulate a classifier
with a given accuracy via a confusion matrix and measure at what accuracy
classify-then-calibrate beats direct deltas.

Usage:
    cd td-one && uv run python experiments/exp7_regime.py
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

SEED = 42

# Calibrated deltas per predicted class: magnitudes from data (the exp2
# fixed-grid probe's best point was +0.25; symmetric for DOWN), not from
# the LLM's gut. NONE -> hard zero (the "do nothing" prior).
CLASS_DELTAS = {"UP": 0.25, "DOWN": -0.25, "NONE": 0.0}

# True classes for the 6 exp4 scenarios (from true_shift sign).
TRUE_CLASSES = {
    "rate_shock_up": "UP",
    "rate_shock_down": "DOWN",
    "osfi_tightening": "DOWN",  # true shift is negative (fewer can switch)
    "benign": "NONE",
    "family_squeeze": "UP",
    "competitor_rate_war": "UP",
}


def simulate_predictions(true_classes: list[str], accuracy: float, seed: int) -> list[str]:
    """Simulate a classifier with `accuracy` (errors -> uniform wrong class)."""
    rng = np.random.default_rng(seed)
    classes = ["UP", "DOWN", "NONE"]
    preds = []
    for t in true_classes:
        if rng.random() < accuracy:
            preds.append(t)
        else:
            wrong = [c for c in classes if c != t]
            preds.append(str(rng.choice(wrong)))
    return preds


def main() -> int:
    t0 = time.time()
    panels = build_scenario_panels()
    rng = np.random.default_rng(SEED)

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
        # Direct-delta baseline: the real-Groq deltas from exp4_results.json
        scenarios.append(
            {
                "id": sid,
                "n": n,
                "a": a,
                "actual": actual,
                "m0_crps": m0_crps,
                "true_class": TRUE_CLASSES[sid],
            }
        )

    # Direct deltas from the real exp4 run (for the mechanism comparison).
    direct_deltas = {
        "rate_shock_up": 0.18,
        "rate_shock_down": -0.25,
        "osfi_tightening": 0.30,
        "benign": 0.30,
        "family_squeeze": 0.20,
        "competitor_rate_war": 0.28,
    }

    def crps_for(s: dict, delta: float) -> float:
        m3 = apply_m3_forecast(
            s["n"],
            s["a"],
            N_TEST,
            delta,
            seed=SEED,  # type: ignore[arg-type]
        )
        return float(crps_counts(m3["samples"][None, :], np.array([s["actual"]]))[0])

    direct_mean = float(np.mean([crps_for(s, direct_deltas[str(s["id"])]) for s in scenarios]))
    m0_mean = float(np.mean([s["m0_crps"] for s in scenarios]))

    print(f"[m3c] M0 mean CRPS: {m0_mean:.3f}", flush=True)
    print(f"[m3c] M3 direct-delta mean CRPS: {direct_mean:.3f}", flush=True)

    true_classes = [str(s["true_class"]) for s in scenarios]
    for acc in [0.5, 0.67, 0.83, 1.0]:
        crpss = []
        for rep in range(20):
            preds = simulate_predictions(true_classes, acc, SEED * 1000 + rep)
            rep_crps = float(
                np.mean(
                    [crps_for(s, CLASS_DELTAS[p]) for s, p in zip(scenarios, preds, strict=True)]
                )
            )
            crpss.append(rep_crps)
        print(
            f"[m3c] accuracy={acc:.2f}: mean CRPS {np.mean(crpss):.3f} "
            f"± {np.std(crpss):.3f} (20 reps)",
            flush=True,
        )

    out = {
        "experiment": "exp7_regime_prototype",
        "mode": "mock (simulated classifier accuracy; mechanism test only)",
        "class_deltas": CLASS_DELTAS,
        "m0_mean_crps": round(m0_mean, 4),
        "m3_direct_mean_crps": round(direct_mean, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp7_regime_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[m3c] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
