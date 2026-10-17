"""Experiment 12b: M5 v2 critic-constrained router vs inverse-CRPS.

Re-runs the exp12 real-Groq scenarios (same 4 scenarios, same synthetic
Beta lattices, same seeds) but replaces the v1 LLM router
(``read_weights_real``, which output 4 weights directly and lost 2.60 vs
2.32) with the v2 critic (``read_weights_critic``): the LLM reviews the
inverse-CRPS baseline and may AGREE or propose ONE bounded adjustment.

Decision rule (from the task):
  - v2 >= inv-CRPS  -> critic pattern works, M5 lives.
  - v2 < inv-CRPS   -> LLM adds no value even constrained; kill M5.

Usage:
    cd td-one && uv run python experiments/exp12b_router_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.ensemble_router import (  # noqa: E402
    MODEL_IDS,
    fallback_weights,
    read_weights_critic,
)
from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.ensemble import average_lattices, crps_weights  # noqa: E402

# Same constants as exp12 for direct comparability.
SEED = 42
N_TEST = 200
N_SAMPLES = 2000
LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

SCENARIOS: list[tuple[str, str, str, float]] = [
    ("s_stable_1", "stable", "Steady market; no policy changes on the horizon", 0.10),
    ("s_shift_up_1", "shift", "BoC just hiked 200bp; payments jumping", 0.16),
    ("s_shift_dn_1", "shift", "BoC just cut 150bp; renewal offers improving", 0.06),
    ("s_vol_1", "volatile", "High uncertainty; mixed signals on policy", 0.10),
]

#: Same model specs as exp12.
MODEL_SPECS: dict[str, dict[str, tuple[float, float]]] = {
    "m0_w3": {
        "stable": (0.00, 250.0),
        "shift": (-0.045, 250.0),
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


def model_lattice(model_id: str, scenario_type: str, p_true: float) -> dict[float, float]:
    """Generate a model's quantile lattice for a scenario (same as exp12)."""
    from scipy import stats

    bias, conc = MODEL_SPECS[model_id][scenario_type]
    mean = float(np.clip(p_true + bias, 0.01, 0.99))
    alpha = mean * conc
    beta = (1.0 - mean) * conc
    return {float(lv): float(stats.beta.ppf(lv, alpha, beta)) for lv in LEVELS}


def lattice_crps(lattice: dict[float, float], actual: int, seed: int) -> float:
    """CRPS of a lattice forecast against an observed count (same as exp12)."""
    q50 = lattice[0.50]
    q25 = lattice[0.25]
    q75 = lattice[0.75]
    iqr = max(q75 - q25, 1e-4)
    sd = iqr / 1.35
    var = max(sd**2, 1e-6)
    mean = float(np.clip(q50, 0.01, 0.99))
    conc = float(np.clip(mean * (1 - mean) / var - 1.0, 5.0, 2000.0))
    alpha = mean * conc
    beta = (1.0 - mean) * conc
    rng = np.random.default_rng(seed)
    rates = rng.beta(alpha, beta, size=N_SAMPLES)
    samples = rng.binomial(N_TEST, np.clip(rates, 0.0, 1.0))
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def run_real_v2() -> dict[str, object]:
    """Real-Groq v2 critic on the same 4 scenarios as exp12's run_real."""
    from td_one.agent.groq_client import GroqChatClient

    client = GroqChatClient(call_budget=10)
    desc = {s[0]: s[2] for s in SCENARIOS}
    ptrue = {s[0]: s[3] for s in SCENARIOS}
    stype = {s[0]: s[1] for s in SCENARIOS}

    # Trailing CRPS from same-type siblings (identical construction to
    # exp12's run_real: leave-one-out over the 12-scenario list).
    full_scenarios: list[tuple[str, str, str, float]] = [
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

    results = []
    for pi, (sid, st, _, p) in enumerate(SCENARIOS):
        rng = np.random.default_rng(SEED + pi)
        lattices = {mid: model_lattice(mid, st, p) for mid in MODEL_IDS}
        actual = int(rng.binomial(N_TEST, p))
        trailing = {}
        for mid in MODEL_IDS:
            crpss = []
            for s2 in full_scenarios:
                if s2[0] == sid or s2[1] != st:
                    continue
                lat = model_lattice(mid, s2[1], s2[3])
                r2 = np.random.default_rng(SEED + 999)
                a2 = int(r2.binomial(N_TEST, s2[3]))
                crpss.append(lattice_crps(lat, a2, SEED))
            trailing[mid] = float(np.mean(crpss))
        try:
            rw = read_weights_critic(client, trailing, desc[sid])
            ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[exp12b] {sid}: LLM failed ({exc}); fallback", flush=True)
            rw = fallback_weights()
            ok = False
        w_eq = [0.25] * 4
        w_ic = crps_weights([[trailing[m]] for m in MODEL_IDS])
        lat_eq = average_lattices([lattices[m] for m in MODEL_IDS], w_eq)
        lat_ic = average_lattices([lattices[m] for m in MODEL_IDS], w_ic)
        lat_v2 = average_lattices([lattices[m] for m in MODEL_IDS], list(rw.weights))
        row = {
            "scenario": sid,
            "type": st,
            "llm_ok": ok,
            "v2_weights": {m: round(w, 3) for m, w in rw.as_dict().items()},
            "rationale": rw.rationale,
            "eq_crps": round(lattice_crps(lat_eq, actual, SEED + pi), 3),
            "inv_crps": round(lattice_crps(lat_ic, actual, SEED + pi), 3),
            "v2_crps": round(lattice_crps(lat_v2, actual, SEED + pi), 3),
        }
        results.append(row)
        print(
            f"[exp12b] {sid} ({st}): eq={row['eq_crps']:.2f} "
            f"inv={row['inv_crps']:.2f} v2={row['v2_crps']:.2f} "
            f"w={row['v2_weights']}",
            flush=True,
        )
    means = {
        k: round(float(np.mean([r[k] for r in results])), 3)
        for k in ("eq_crps", "inv_crps", "v2_crps")
    }
    print(f"[exp12b] v2 means: {means} ({client.calls} calls)", flush=True)
    return {"per_scenario": results, "means": means, "groq_calls": client.calls}


def main() -> int:
    """Run the v2 critic test with real Groq."""
    t0 = time.time()
    out: dict[str, object] = {
        "experiment": "exp12b_router_v2",
        "config": {
            "seed": SEED,
            "n_test": N_TEST,
            "models": list(MODEL_IDS),
            "router": "v2 critic-constrained (read_weights_critic)",
            "compares_against": "exp12 v1 (2.60) and inv-CRPS (2.32)",
        },
    }
    out["v2_real"] = run_real_v2()
    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp12b_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[exp12b] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
