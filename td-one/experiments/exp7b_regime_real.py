"""M3c real-LLM test: does classification beat direct numerical deltas.

Same 6 exp4 scenarios, same descriptions, but the LLM may only output
UP / DOWN / NONE (JSON). The magnitude comes from data-calibrated
CLASS_DELTAS, not from the LLM's gut. 6 Groq calls.

The prototype (exp7_regime.py) says classify-then-calibrate beats
direct deltas once accuracy exceeds ~75%. This tests whether real Groq
clears that bar — in particular whether classification avoids the
osfi sign-flip and the benign spurious adjustment.

Usage:
    cd td-one && TD_ONE_REAL_LLM=1 uv run python experiments/exp7b_regime_real.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from exp4_m3 import (  # noqa: E402
    _GROQ_MODEL,
    N_TEST,
    SCENARIOS,
    apply_m3_forecast,
    build_scenario_panels,
    logit,
    sigmoid,
)

from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402

logger = logging.getLogger("exp7b")

CLASS_DELTAS = {"UP": 0.25, "DOWN": -0.25, "NONE": 0.0}
TRUE_CLASSES = {
    "rate_shock_up": "UP",
    "rate_shock_down": "DOWN",
    "osfi_tightening": "DOWN",
    "benign": "NONE",
    "family_squeeze": "UP",
    "competitor_rate_war": "UP",
}
SEED = 42

CLASSIFY_PROMPT = (
    "You are a mortgage market analyst. Given the macro scenario below and "
    "the current switching rate, classify the expected DIRECTION of the "
    "borrower switching-rate shift.\n"
    "\n"
    "Scenario: {description}\n"
    "Current switching rate: {base_rate:.3f}\n"
    "\n"
    "Think step by step:\n"
    "1. Does this scenario make switching MORE attractive (borrowers chase "
    "better deals), LESS attractive (barriers rise), or is there no "
    "material change?\n"
    "2. Note: policies that make switching HARDER (re-qualification, "
    "tighter rules) REDUCE switching even if they sound restrictive.\n"
    "\n"
    'Respond with JSON only: {"direction": "UP" | "DOWN" | "NONE", '
    '"reason": "<one sentence>"}'
)


def classify_scenarios() -> dict[str, str]:
    """Classify each scenario with real Groq. Anonymized IDs (no hints)."""
    from td_one.agent.groq_client import GroqChatClient

    client = GroqChatClient(model=_GROQ_MODEL, call_budget=10)
    out: dict[str, str] = {}
    anon = {str(s["id"]): f"scenario_{chr(65 + i)}" for i, s in enumerate(SCENARIOS)}
    for scenario in SCENARIOS:
        sid = str(scenario["id"])
        # Base rate needed for context; compute quickly from mock panel.
        try:
            resp = client.complete(
                CLASSIFY_PROMPT.format(description=str(scenario["description"]), base_rate=0.25),
                system="You output JSON only.",
                json_mode=True,
            )
            text = resp.strip()
            start, end = text.find("{"), text.rfind("}")
            direction = json.loads(text[start : end + 1])["direction"].upper()
            if direction not in CLASS_DELTAS:
                raise ValueError(f"bad direction {direction!r}")
        except Exception as exc:  # noqa: BLE001 - fall back honestly
            logger.warning("%s (%s): classification failed: %s", sid, anon[sid], exc)
            direction = "NONE"
        out[sid] = direction
        print(f"[m3c-real] {anon[sid]} ({sid}): -> {direction}", flush=True)
    return out


def main() -> int:
    t0 = time.time()
    use_real = os.environ.get("TD_ONE_REAL_LLM") == "1"
    if use_real:
        predictions = classify_scenarios()
    else:
        print("[m3c-real] TD_ONE_REAL_LLM not set; using TRUE classes (ceiling).")
        predictions = dict(TRUE_CLASSES)

    panels = build_scenario_panels()
    rng = np.random.default_rng(SEED)
    correct = 0
    rows = []
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

        pred = predictions[sid]
        truth = TRUE_CLASSES[sid]
        correct += pred == truth
        m3 = apply_m3_forecast(n, a, N_TEST, CLASS_DELTAS[pred], seed=SEED)
        m3_crps = float(crps_counts(m3["samples"][None, :], np.array([actual]))[0])
        rows.append(
            {
                "id": sid,
                "true_class": truth,
                "predicted": pred,
                "delta": CLASS_DELTAS[pred],
                "m0_crps": round(m0_crps, 4),
                "m3c_crps": round(m3_crps, 4),
            }
        )
        print(
            f"[m3c-real] {sid}: truth={truth} pred={pred} m0={m0_crps:.2f} m3c={m3_crps:.2f}",
            flush=True,
        )

    mean_m0 = float(np.mean([r["m0_crps"] for r in rows]))
    mean_m3c = float(np.mean([r["m3c_crps"] for r in rows]))
    acc = correct / len(rows)
    print(f"[m3c-real] accuracy={acc:.2f} ({correct}/6)", flush=True)
    print(f"[m3c-real] mean CRPS: M0={mean_m0:.3f} M3c={mean_m3c:.3f}", flush=True)
    # Reference: M3 direct-delta mean was 10.429 (real Groq, exp4).

    out = {
        "experiment": "exp7b_regime_real",
        "mode": "real" if use_real else "ceiling (true classes)",
        "accuracy": acc,
        "mean_m0_crps": round(mean_m0, 4),
        "mean_m3c_crps": round(mean_m3c, 4),
        "m3_direct_reference": 10.429,
        "scenarios": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp7b_regime_real_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[m3c-real] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
