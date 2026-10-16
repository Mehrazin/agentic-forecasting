"""Experiment 11: M4 regime-reader agent vs M0.

Tests the redesigned agent role: the LLM is a noisy sensor that outputs
(class, confidence, reason) — never a number. Classical code decides
whether to intervene, requiring LLM confidence AND (optionally) panel
corroboration.

Scenario panels are synthetic but structured: shock scenarios shift the
switching rate in the last 3 panel months (visible to the panel gate)
and the shift persists into the test period; benign scenarios are flat.
True shifts reuse exp4's magnitudes.

Variants:
  M0   — macro-blind baseline (trailing-6 Jeffreys Beta-Binomial).
  M4a  — M0 + confidence-gated regime reading (LLM sensor only).
  M4b  — M0 + confidence gate + panel z-gate (two sensors must agree).

Mock mode sweeps the reader's accuracy and calibration. A real-Groq
validation runs at the end (6 calls, one per scenario).

Usage:
    cd td-one && uv run python experiments/exp11_agent_v2.py
    TD_ONE_REAL_LLM=1 uv run python experiments/exp11_agent_v2.py  # +real
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.regime_reader import (  # noqa: E402
    MockRegimeReader,
    RegimeClass,
    RegimeReading,
    read_regime_real,
)
from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.regime_blend import (  # noqa: E402
    _regime_class_for_shift,
    apply_regime_reading,
    m4_crps,
)

SEED = 42
N_TEST = 200
N_PER_MONTH = 200
PANEL_MONTHS = 12
BASE_RATE = 0.10

#: (id, description, true logit shift). Shifts from exp4.
SCENARIOS: list[tuple[str, str, float]] = [
    (
        "rate_shock_up",
        "BoC hikes 200bp; renewal payments jump, switching incentive surges",
        0.60,
    ),
    (
        "rate_shock_down",
        "BoC cuts 150bp; renewal offers improve, switching incentive falls",
        -0.50,
    ),
    (
        "osfi_tightening",
        "Stress-test exemption revoked; uninsured 720+ switchers must re-qualify",
        -0.35,
    ),
    (
        "benign",
        "No material macro/policy change; baseline conditions hold",
        0.00,
    ),
    (
        "family_squeeze",
        "Child-benefit clawback raises pressure on low-income 3+ kid households",
        0.45,
    ),
    (
        "competitor_rate_war",
        "Big-5 lenders discount switcher rates 75bp below incumbent renewals",
        0.70,
    ),
]


def build_panel(true_shift: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a 12-month (N, A) panel with the shift in the last 3 months."""
    rng = np.random.default_rng(seed)
    n = np.full(PANEL_MONTHS, N_PER_MONTH, dtype=float)
    p0 = BASE_RATE
    p1 = float(sigmoid(logit(p0) + true_shift))
    probs = np.full(PANEL_MONTHS, p0)
    probs[-3:] = p1
    a = rng.binomial(N_PER_MONTH, probs).astype(float)
    return n, a


def draw_actual(true_shift: float, seed: int) -> int:
    """Draw the test-period switch count at the shifted rate."""
    rng = np.random.default_rng(seed)
    p1 = float(sigmoid(logit(BASE_RATE) + true_shift))
    return int(rng.binomial(N_TEST, p1))


def m0_crps_for(n: np.ndarray, a: np.ndarray, actual: int, seed: int) -> float:
    """M0 CRPS on a panel."""
    m0 = HistoricalBaseline()
    m0.fit(n, a)
    samples = m0.predict_counts(N_TEST, seed=seed)
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def run_mock_sweep() -> dict[str, object]:
    """Sweep mock-reader accuracy x calibration; return summary table."""
    panels = {sid: build_panel(shift, SEED + i) for i, (sid, _, shift) in enumerate(SCENARIOS)}
    actuals = {
        sid: draw_actual(shift, SEED + 100 + i) for i, (sid, _, shift) in enumerate(SCENARIOS)
    }
    true_classes: dict[str, RegimeClass] = {
        sid: _regime_class_for_shift(shift) for sid, _, shift in SCENARIOS
    }
    m0_means = {
        sid: m0_crps_for(panels[sid][0], panels[sid][1], actuals[sid], SEED) for sid in panels
    }
    m0_mean = float(np.mean(list(m0_means.values())))
    print(f"[exp11] M0 mean CRPS: {m0_mean:.3f}", flush=True)

    rows = []
    for accuracy in (0.50, 0.67, 0.83):
        for calibrated in (True, False):
            for use_panel_gate in (False, True):
                variant = (
                    f"m4a_acc{accuracy}_cal{int(calibrated)}"
                    if not use_panel_gate
                    else f"m4b_acc{accuracy}_cal{int(calibrated)}"
                )
                crpss = []
                n_intervened = 0
                for rep in range(20):
                    reader = MockRegimeReader(
                        accuracy=accuracy,
                        calibrated=calibrated,
                        seed=SEED * 1000 + rep,
                    )
                    rep_crps = []
                    for sid in panels:
                        n, a = panels[sid]
                        reading = reader.read(true_classes[sid])
                        fc = apply_regime_reading(
                            n,
                            a,
                            N_TEST,
                            reading,
                            use_panel_gate=use_panel_gate,
                            seed=SEED,
                        )
                        n_intervened += int(fc["intervened"])
                        rep_crps.append(m4_crps(fc, actuals[sid]))
                    crpss.append(float(np.mean(rep_crps)))
                row = {
                    "variant": variant,
                    "accuracy": accuracy,
                    "calibrated": calibrated,
                    "panel_gate": use_panel_gate,
                    "mean_crps": round(float(np.mean(crpss)), 3),
                    "std_crps": round(float(np.std(crpss)), 3),
                    "intervention_rate": round(n_intervened / (20 * len(panels)), 3),
                }
                rows.append(row)
                print(
                    f"[exp11] {variant}: CRPS {row['mean_crps']:.3f} "
                    f"± {row['std_crps']:.3f} "
                    f"(intervened {row['intervention_rate']:.0%})",
                    flush=True,
                )
    return {"m0_mean_crps": round(m0_mean, 3), "mock_sweep": rows}


def run_real_validation() -> dict[str, object]:
    """One real-Groq reading per scenario (6 calls); score M4a/M4b vs M0."""
    from td_one.agent.groq_client import GroqChatClient

    client = GroqChatClient(call_budget=10)
    panels = {sid: build_panel(shift, SEED + i) for i, (sid, _, shift) in enumerate(SCENARIOS)}
    actuals = {
        sid: draw_actual(shift, SEED + 100 + i) for i, (sid, _, shift) in enumerate(SCENARIOS)
    }
    results = []
    for sid, description, shift in SCENARIOS:
        n, a = panels[sid]
        m0 = HistoricalBaseline()
        m0.fit(n, a)
        base_rate = float(m0.expected_rate)
        try:
            reading = read_regime_real(client, description, base_rate)
            ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[exp11] {sid}: LLM failed ({exc}); abstaining", flush=True)
            reading = RegimeReading(regime="NONE", confidence=0.0, reason="llm-error")
            ok = False
        m0_c = m0_crps_for(n, a, actuals[sid], SEED)
        fc_a = apply_regime_reading(n, a, N_TEST, reading, use_panel_gate=False, seed=SEED)
        fc_b = apply_regime_reading(n, a, N_TEST, reading, use_panel_gate=True, seed=SEED)
        row = {
            "scenario": sid,
            "true_shift": shift,
            "llm_ok": ok,
            "reading": {
                "regime": reading.regime,
                "confidence": round(reading.confidence, 3),
                "reason": reading.reason,
            },
            "m0_crps": round(m0_c, 3),
            "m4a_crps": round(m4_crps(fc_a, actuals[sid]), 3),
            "m4b_crps": round(m4_crps(fc_b, actuals[sid]), 3),
            "m4a_intervened": bool(fc_a["intervened"]),
            "m4b_intervened": bool(fc_b["intervened"]),
        }
        results.append(row)
        print(
            f"[exp11] {sid}: {reading.regime}@{reading.confidence:.2f} "
            f"M0={row['m0_crps']:.2f} M4a={row['m4a_crps']:.2f} "
            f"M4b={row['m4b_crps']:.2f}",
            flush=True,
        )
    means = {
        k: round(float(np.mean([r[k] for r in results])), 3)
        for k in ("m0_crps", "m4a_crps", "m4b_crps")
    }
    print(f"[exp11] real means: {means} ({client.calls} Groq calls)", flush=True)
    return {"per_scenario": results, "means": means, "groq_calls": client.calls}


def main() -> int:
    """Run the mock sweep and (optionally) the real-Groq validation."""
    t0 = time.time()
    out: dict[str, object] = {
        "experiment": "exp11_agent_v2",
        "config": {
            "seed": SEED,
            "n_test": N_TEST,
            "class_deltas": {"UP": 0.25, "DOWN": -0.25, "NONE": 0.0},
            "confidence_threshold": 0.7,
            "z_threshold": 1.25,
        },
    }
    out["mock"] = run_mock_sweep()
    if os.environ.get("TD_ONE_REAL_LLM") == "1":
        out["real"] = run_real_validation()
    else:
        print("[exp11] skipping real-Groq validation (TD_ONE_REAL_LLM!=1)", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp11_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[exp11] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
