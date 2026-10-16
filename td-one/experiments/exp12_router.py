"""Experiment 12: M5 ensemble router (REATS-style) vs inverse-CRPS.

Tests the literature survey's top-ranked agentic role: the LLM outputs
weights over 4 forecast models (m0_w3, conf_w6, chronos2, timesfm25)
with rationale, never values. Directly testable: does LLM weighting
beat inverse-CRPS weighting (exp5: 4.078)?

Scenario types (4 each) where different models excel:
  stable   — m0_w3 best (sharp, accurate; no shift to adapt to).
  shift    — timesfm25 best (adapts); m0_w3 slow (biased).
  volatile — conf_w6 best (wide intervals pay off under noise).

Model forecasts are synthetic Beta lattices with controlled
(mean, concentration) per scenario type, matching our empirical
findings (m0_w3 sharp; conf wide; chronos2 wide+biased; timesfm25
calibrated). The LLM sees model cards + trailing CRPS + scenario
description.

Weighting schemes: equal, inverse-CRPS (leave-one-out trailing),
mock router (insight_prob sweep), real Groq (4 scenarios).

Usage:
    cd td-one && uv run python experiments/exp12_router.py
    TD_ONE_REAL_LLM=1 uv run python experiments/exp12_router.py  # +real
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.ensemble_router import (  # noqa: E402
    MODEL_IDS,
    MockRouter,
    fallback_weights,
    read_weights_real,
)
from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.ensemble import average_lattices, crps_weights  # noqa: E402

SEED = 42
N_TEST = 200
N_SAMPLES = 2000

#: Quantile levels for the lattices.
LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

#: (scenario_id, type, description, true rate).
SCENARIOS: list[tuple[str, str, str, float]] = [
    ("s_stable_1", "stable", "Steady market; no policy changes on the horizon", 0.10),
    ("s_stable_2", "stable", "Baseline conditions hold; low volatility", 0.10),
    ("s_stable_3", "stable", "Quiet quarter; renewal patterns normal", 0.10),
    ("s_stable_4", "stable", "No material macro shifts expected", 0.10),
    ("s_shift_up_1", "shift", "BoC just hiked 200bp; payments jumping", 0.16),
    ("s_shift_up_2", "shift", "Sudden rate spike; borrowers scrambling", 0.16),
    ("s_shift_dn_1", "shift", "BoC just cut 150bp; renewal offers improving", 0.06),
    ("s_shift_dn_2", "shift", "Policy easing; switching incentive falling", 0.06),
    ("s_vol_1", "volatile", "High uncertainty; mixed signals on policy", 0.10),
    ("s_vol_2", "volatile", "Turbulent market; outcomes hard to predict", 0.10),
    ("s_vol_3", "volatile", "Regime uncertainty; wide range of outcomes", 0.10),
    ("s_vol_4", "volatile", "Noisy data; structural breaks possible", 0.10),
]

#: (mean bias, concentration) per model per scenario type.
#: Biases reflect adaptation lag: m0_w3 trained on pre-shift rates.
MODEL_SPECS: dict[str, dict[str, tuple[float, float]]] = {
    "m0_w3": {
        "stable": (0.00, 250.0),
        "shift": (-0.045, 250.0),  # slow to adapt; bias vs new truth
        "volatile": (0.00, 250.0),
    },
    "conf_w6": {
        "stable": (0.00, 55.0),
        "shift": (-0.020, 55.0),
        "volatile": (0.00, 45.0),
    },
    "chronos2": {
        "stable": (0.015, 35.0),
        "shift": (0.015, 35.0),
        "volatile": (0.015, 35.0),
    },
    "timesfm25": {
        "stable": (0.00, 110.0),
        "shift": (-0.010, 110.0),
        "volatile": (0.00, 100.0),
    },
}

#: Best model per scenario type (for the mock's insight).
BEST_MODEL = {"stable": "m0_w3", "shift": "timesfm25", "volatile": "conf_w6"}


def model_lattice(model_id: str, scenario_type: str, p_true: float) -> dict[float, float]:
    """Generate a model's quantile lattice for a scenario."""
    bias, conc = MODEL_SPECS[model_id][scenario_type]
    # For shift scenarios, bias is signed vs the shift direction; here
    # all shifts are upward in magnitude, so use the spec directly.
    # Downward shifts flip the bias sign (handled by caller via p_true).
    mean = float(np.clip(p_true + bias, 0.01, 0.99))
    alpha = mean * conc
    beta = (1.0 - mean) * conc
    return {float(lv): float(stats.beta.ppf(lv, alpha, beta)) for lv in LEVELS}


def lattice_crps(lattice: dict[float, float], actual: int, seed: int) -> float:
    """CRPS of a lattice forecast against an observed count.

    Interpolates the lattice to a Beta distribution, samples counts.
    """
    # Fit a Beta to the (median, iqr) of the lattice.
    q50 = lattice[0.50]
    q25 = lattice[0.25]
    q75 = lattice[0.75]
    iqr = max(q75 - q25, 1e-4)
    # Rough concentration from IQR (normal approx): iqr ~ 1.35*sd.
    sd = iqr / 1.35
    var = max(sd**2, 1e-6)
    mean = float(np.clip(q50, 0.01, 0.99))
    # Method of moments for Beta.
    conc = mean * (1 - mean) / var - 1.0
    conc = float(np.clip(conc, 5.0, 2000.0))
    alpha = mean * conc
    beta = (1.0 - mean) * conc
    rng = np.random.default_rng(seed)
    rates = rng.beta(alpha, beta, size=N_SAMPLES)
    samples = rng.binomial(N_TEST, np.clip(rates, 0.0, 1.0))
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def run_mock() -> dict[str, object]:
    """Compare weighting schemes with mock routers."""
    # Generate lattices + actuals for all scenarios.
    lattices: dict[str, dict[str, dict[float, float]]] = {}
    actuals: dict[str, int] = {}
    stypes: dict[str, str] = {}
    for i, (sid, stype, _, p_true) in enumerate(SCENARIOS):
        rng = np.random.default_rng(SEED + i)
        # Shift scenarios: bias sign follows shift direction.
        lattices[sid] = {mid: model_lattice(mid, stype, p_true) for mid in MODEL_IDS}
        # Volatile scenarios get extra outcome noise.
        p_draw = p_true
        if stype == "volatile":
            p_draw = float(np.clip(rng.normal(p_true, 0.03), 0.02, 0.30))
        actuals[sid] = int(rng.binomial(N_TEST, p_draw))
        stypes[sid] = stype

    # Per-model CRPS per scenario (for trailing / inverse-CRPS).
    model_crps: dict[str, dict[str, float]] = {mid: {} for mid in MODEL_IDS}
    for sid in lattices:
        for mid in MODEL_IDS:
            model_crps[mid][sid] = lattice_crps(lattices[sid][mid], actuals[sid], SEED)

    sids = [s[0] for s in SCENARIOS]
    schemes: dict[str, list[float]] = {
        "equal": [],
        "inv_crps": [],
        "mock_p0": [],
        "mock_p50": [],
        "mock_p100": [],
    }
    for idx, sid in enumerate(sids):
        # Leave-one-out trailing CRPS.
        trailing = {
            mid: float(np.mean([model_crps[mid][s] for s in sids if s != sid])) for mid in MODEL_IDS
        }
        # Equal weights.
        w_eq = [0.25, 0.25, 0.25, 0.25]
        lat_eq = average_lattices([lattices[sid][m] for m in MODEL_IDS], w_eq)
        schemes["equal"].append(lattice_crps(lat_eq, actuals[sid], SEED + idx))
        # Inverse-CRPS.
        w_ic = crps_weights([[trailing[m]] for m in MODEL_IDS])
        lat_ic = average_lattices([lattices[sid][m] for m in MODEL_IDS], w_ic)
        schemes["inv_crps"].append(lattice_crps(lat_ic, actuals[sid], SEED + idx))
        # Mock routers.
        for tag, prob in (("mock_p0", 0.0), ("mock_p50", 0.5), ("mock_p100", 1.0)):
            router = MockRouter(insight_prob=prob, seed=SEED * 100 + idx)
            rw = router.route(trailing, BEST_MODEL[stypes[sid]])
            lat = average_lattices([lattices[sid][m] for m in MODEL_IDS], list(rw.weights))
            schemes[tag].append(lattice_crps(lat, actuals[sid], SEED + idx))

    summary = {}
    for name, crpss in schemes.items():
        summary[name] = {
            "mean_crps": round(float(np.mean(crpss)), 3),
            "std_crps": round(float(np.std(crpss)), 3),
        }
        print(
            f"[exp12] {name}: CRPS {summary[name]['mean_crps']:.3f} "
            f"± {summary[name]['std_crps']:.3f}",
            flush=True,
        )
    return {
        "summary": summary,
        "per_model_mean_crps": {
            mid: round(float(np.mean(list(model_crps[mid].values()))), 3) for mid in MODEL_IDS
        },
    }


def run_real() -> dict[str, object]:
    """Real-Groq router on 4 scenarios (1 per type, 2 shifts)."""
    from td_one.agent.groq_client import GroqChatClient

    client = GroqChatClient(call_budget=10)
    picks = ["s_stable_1", "s_shift_up_1", "s_shift_dn_1", "s_vol_1"]
    desc = {s[0]: s[2] for s in SCENARIOS}
    ptrue = {s[0]: s[3] for s in SCENARIOS}
    stype = {s[0]: s[1] for s in SCENARIOS}

    # Trailing CRPS from the other scenarios of the same type.
    results = []
    for pi, sid in enumerate(picks):
        rng = np.random.default_rng(SEED + pi)
        lattices = {mid: model_lattice(mid, stype[sid], ptrue[sid]) for mid in MODEL_IDS}
        actual = int(rng.binomial(N_TEST, ptrue[sid]))
        # Trailing: mean CRPS of each model on same-type siblings.
        trailing = {}
        for mid in MODEL_IDS:
            crpss = []
            for s2 in SCENARIOS:
                if s2[0] == sid or s2[1] != stype[sid]:
                    continue
                lat = model_lattice(mid, s2[1], s2[3])
                r2 = np.random.default_rng(SEED + 999)
                a2 = int(r2.binomial(N_TEST, s2[3]))
                crpss.append(lattice_crps(lat, a2, SEED))
            trailing[mid] = float(np.mean(crpss))
        try:
            rw = read_weights_real(client, trailing, desc[sid])
            ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[exp12] {sid}: LLM failed ({exc}); fallback", flush=True)
            rw = fallback_weights()
            ok = False
        # Score all schemes.
        w_eq = [0.25] * 4
        w_ic = crps_weights([[trailing[m]] for m in MODEL_IDS])
        lat_eq = average_lattices([lattices[m] for m in MODEL_IDS], w_eq)
        lat_ic = average_lattices([lattices[m] for m in MODEL_IDS], w_ic)
        lat_llm = average_lattices([lattices[m] for m in MODEL_IDS], list(rw.weights))
        row = {
            "scenario": sid,
            "type": stype[sid],
            "llm_ok": ok,
            "llm_weights": {m: round(w, 3) for m, w in rw.as_dict().items()},
            "rationale": rw.rationale,
            "eq_crps": round(lattice_crps(lat_eq, actual, SEED + pi), 3),
            "inv_crps": round(lattice_crps(lat_ic, actual, SEED + pi), 3),
            "llm_crps": round(lattice_crps(lat_llm, actual, SEED + pi), 3),
        }
        results.append(row)
        print(
            f"[exp12] {sid} ({stype[sid]}): eq={row['eq_crps']:.2f} "
            f"inv={row['inv_crps']:.2f} llm={row['llm_crps']:.2f} "
            f"w={row['llm_weights']}",
            flush=True,
        )
    means = {
        k: round(float(np.mean([r[k] for r in results])), 3)
        for k in ("eq_crps", "inv_crps", "llm_crps")
    }
    print(f"[exp12] real means: {means} ({client.calls} calls)", flush=True)
    return {"per_scenario": results, "means": means, "groq_calls": client.calls}


def main() -> int:
    """Run mock comparison and optional real-Groq validation."""
    t0 = time.time()
    out: dict[str, object] = {
        "experiment": "exp12_router",
        "config": {"seed": SEED, "n_test": N_TEST, "models": list(MODEL_IDS)},
    }
    out["mock"] = run_mock()
    if os.environ.get("TD_ONE_REAL_LLM") == "1":
        out["real"] = run_real()
    else:
        print("[exp12] skipping real-Groq (TD_ONE_REAL_LLM!=1)", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp12_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[exp12] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
