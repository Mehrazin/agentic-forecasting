"""Experiment 6: M3+ — shrinkage and regime gating vs M3/M0.

Tests two mechanisms for M3's over-adjustment (exp4 real-Groq findings:
wrong-sign deltas, spurious benign adjustments, magnitude undershoot):

1. **Shrinkage**: :class:`ShrunkHybridReadout` pulls the proposed delta
   toward 0 by a factor ``k`` tuned on data.
2. **Regime gate**: :func:`gate_delta` zeroes the delta unless the panel
   shows a significant recent shift (|z| >= threshold).

Design (mock mode only — the "LLM" delta is scripted; we test the
*mechanism*, not the LLM):

- 12 NEW scenarios (6 shock + 6 benign), disjoint from exp4. Shock
  scenarios inject the true logit shift into the last 3 panel months so
  the gate can see it; the shift persists into the test period. Benign
  scenarios are stationary (true shift 0).
- Mock deltas replicate the real-Groq error model (seeded): magnitude
  undershoot (0.3–0.9x of the correct delta), 15% wrong-sign flips on
  shocks, spurious ±0.30 adjustments on half the benign scenarios.
- Split: tune (k, threshold) on 6 scenarios (3 shock + 3 benign) by
  joint grid search minimizing mean CRPS of M3+both; evaluate all five
  variants on the held-out 6.
- Variants: M0, M3 (bound 0.75), M3+shrinkage, M3+gate, M3+both.

Writes ``experiments/exp6_results.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from td_one.evaluation.metrics import crps_counts  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.hybrid import HybridReadout  # noqa: E402
from td_one.models.shrinkage import (  # noqa: E402
    ShrunkHybridReadout,
    gate_delta,
    regime_z_stat,
)

logger = logging.getLogger(__name__)

#: Upcoming renewals in the test period (matches exp4).
N_TEST = 200
#: Simulator size (chosen so the gate's z-test has power; see probe notes).
N_CUSTOMERS = 4000
N_MONTHS = 24
#: Panel length and shock-visibility window (months).
PANEL_MONTHS = 12
SHIFT_MONTHS = 3
#: Seeds (fixed; scenarios are disjoint from exp4's seed=1000+i).
_PANEL_SEED_BASE = 2000
_MOCK_SEED_BASE = 9000
_EVAL_SEED = 1234

#: 12 new scenarios: 6 shocks + 6 benign. Order matters: the first three
#: of each kind form the tuning set, the last three the held-out set.
SCENARIOS6: list[dict[str, object]] = [
    # --- shocks (tune) ---
    {
        "id": "boc_hike_175",
        "kind": "shock",
        "description": "BoC hikes 175bp; variable-rate cohort faces payment shock",
        "true_shift": 0.55,
    },
    {
        "id": "boc_cut_200",
        "kind": "shock",
        "description": "BoC cuts 200bp; renewal offers improve, switching incentive falls",
        "true_shift": -0.45,
    },
    {
        "id": "cashback_war",
        "kind": "shock",
        "description": "Big-6 lenders offer $3k cashback to switchers",
        "true_shift": 0.50,
    },
    # --- shocks (held-out) ---
    {
        "id": "osfi_relax",
        "kind": "shock",
        "description": "Stress-test threshold lowered; more switchers qualify",
        "true_shift": 0.40,
    },
    {
        "id": "recession_fear",
        "kind": "shock",
        "description": "Unemployment spikes; borrowers hunker down with incumbent",
        "true_shift": -0.38,
    },
    {
        "id": "portability_reform",
        "kind": "shock",
        "description": "New portable-mortgage rules cut switching frictions",
        "true_shift": 0.42,
    },
    # --- benign (tune) ---
    {
        "id": "benign_steady",
        "kind": "benign",
        "description": "No material macro or policy change; baseline holds",
        "true_shift": 0.0,
    },
    {
        "id": "benign_noise",
        "kind": "benign",
        "description": "Headline volatility, no policy change",
        "true_shift": 0.0,
    },
    {
        "id": "benign_seasonal",
        "kind": "benign",
        "description": "Normal seasonal renewal pattern",
        "true_shift": 0.0,
    },
    # --- benign (held-out) ---
    {
        "id": "benign_mixed",
        "kind": "benign",
        "description": "Mixed signals across lenders, net neutral",
        "true_shift": 0.0,
    },
    {
        "id": "benign_local",
        "kind": "benign",
        "description": "Regional news with no national rate impact",
        "true_shift": 0.0,
    },
    {
        "id": "benign_technical",
        "kind": "benign",
        "description": "Lender IT outage; no rate or policy impact",
        "true_shift": 0.0,
    },
]

#: Tuning grid.
K_GRID = [round(k, 1) for k in np.arange(0.0, 1.01, 0.1)]
THRESHOLD_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


class ScenarioPanel(TypedDict):
    """Panel plus scenario truth for one exp6 scenario."""

    n: np.ndarray
    a: np.ndarray
    base_rate: float
    true_test_rate: float
    correct_delta: float
    mock_delta: float
    z_stat: float
    is_shock: bool


def _clip01(x: float) -> float:
    return float(np.clip(x, 1e-6, 1 - 1e-6))


def build_exp6_panels() -> dict[str, ScenarioPanel]:
    """Build 12 scenario panels with shock injection and mock deltas.

    Mirrors ``exp4_m3.build_scenario_panels`` (simulator → monthly (N, A)
    buckets, last 12 months) then, for shocks, redraws the last
    ``SHIFT_MONTHS`` months at the shifted regime rate so the gate can
    see the shift. Mock deltas replicate the real-Groq error model with
    fixed seeds (they are scenario properties, not re-randomized).
    """
    from td_one.data_gen.simulator import MortgageSimulator

    panels: dict[str, ScenarioPanel] = {}
    for i, scenario in enumerate(SCENARIOS6):
        sid = str(scenario["id"])
        is_shock = scenario["kind"] == "shock"
        true_shift = cast(float, scenario["true_shift"])
        sim = MortgageSimulator(
            n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=_PANEL_SEED_BASE + i
        )
        outcomes = sim.generate_outcomes()
        buckets: dict[str, list[int]] = {}
        for o in outcomes:
            if o.switched_lender_flag is None:
                continue
            key = o.scheduled_maturity_date.strftime("%Y-%m")
            entry = buckets.setdefault(key, [0, 0])
            entry[0] += 1
            entry[1] += int(o.switched_lender_flag)
        months = sorted(buckets)[-PANEL_MONTHS:]
        n = np.array([buckets[m][0] for m in months], dtype=float)
        a = np.array([buckets[m][1] for m in months], dtype=float)

        rng = np.random.default_rng(_MOCK_SEED_BASE + i)
        r0 = float((a[:-SHIFT_MONTHS] / n[:-SHIFT_MONTHS]).mean())
        if is_shock:
            r_regime = float(sigmoid(logit(_clip01(r0)) + true_shift))
            a = a.copy()
            a[-SHIFT_MONTHS:] = rng.binomial(n[-SHIFT_MONTHS:].astype(int), r_regime)
            true_test_rate = r_regime
        else:
            true_test_rate = r0

        m0 = HistoricalBaseline()
        m0.fit(n, a)
        base_rate = m0.expected_rate
        correct_delta = (
            logit(_clip01(true_test_rate)) - logit(_clip01(base_rate)) if is_shock else 0.0
        )
        # Mock LLM: real-Groq error model (undershoot, sign flips, benign spurious).
        if is_shock:
            mock = float(correct_delta) * float(rng.uniform(0.3, 0.9))
            if rng.random() < 0.15:
                mock = -mock
            mock_delta = float(np.clip(mock, -0.75, 0.75))
        else:
            mock_delta = float(rng.uniform(-0.30, 0.30)) if rng.random() < 0.5 else 0.0

        panels[sid] = ScenarioPanel(
            n=n,
            a=a,
            base_rate=base_rate,
            true_test_rate=true_test_rate,
            correct_delta=float(correct_delta),
            mock_delta=mock_delta,
            z_stat=regime_z_stat(n, a),
            is_shock=is_shock,
        )
        logger.info(
            "scenario %s: shock=%s z=%+.2f mock_delta=%+.2f correct=%+.2f",
            sid,
            is_shock,
            panels[sid]["z_stat"],
            mock_delta,
            correct_delta,
        )
    return panels


def _m0_crps(n: np.ndarray, a: np.ndarray, actual: int, seed: int) -> float:
    m0 = HistoricalBaseline()
    m0.fit(n, a)
    samples = m0.predict_counts(N_TEST, seed=seed)
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def _m3_crps(
    n: np.ndarray,
    a: np.ndarray,
    delta: float,
    readout: HybridReadout,
    actual: int,
    seed: int,
) -> float:
    m0 = HistoricalBaseline()
    m0.fit(n, a)
    base_rate = m0.expected_rate
    concentration = 1.0 + float(np.asarray(n)[-6:].sum())
    samples = readout.predict_counts(
        base_rate, delta, N_TEST, concentration=concentration, seed=seed
    )
    return float(crps_counts(samples[None, :], np.array([actual]))[0])


def eval_variant(
    panels: dict[str, ScenarioPanel],
    sids: list[str],
    variant: str,
    k: float,
    threshold: float,
    seed: int,
) -> dict[str, float]:
    """CRPS per scenario for one variant.

    Variants: "m0", "m3", "m3_shrink", "m3_gate", "m3_both".
    """
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for sid in sids:
        p = panels[sid]
        actual = int(rng.binomial(N_TEST, p["true_test_rate"]))
        if variant == "m0":
            out[sid] = _m0_crps(p["n"], p["a"], actual, seed)
            continue
        delta = p["mock_delta"]
        if variant in ("m3_gate", "m3_both"):
            delta = gate_delta(delta, p["n"], p["a"], threshold=threshold)
        readout: HybridReadout
        if variant in ("m3_shrink", "m3_both"):
            readout = ShrunkHybridReadout(shrinkage=k)
        else:
            readout = HybridReadout()
        out[sid] = _m3_crps(p["n"], p["a"], delta, readout, actual, seed)
    return out


def tune(
    panels: dict[str, ScenarioPanel], tune_sids: list[str], seed: int
) -> tuple[float, float, list[dict[str, object]]]:
    """Joint grid search over (k, threshold) on the tuning set.

    Minimizes mean CRPS of the M3+both variant.
    """
    grid: list[dict[str, object]] = []
    best: tuple[float, float] = (1.0, 2.0)
    best_crps = float("inf")
    for k in K_GRID:
        for t in THRESHOLD_GRID:
            crps = eval_variant(panels, tune_sids, "m3_both", k, t, seed)
            mean = float(np.mean(list(crps.values())))
            grid.append({"k": k, "threshold": t, "mean_crps": round(mean, 4)})
            if mean < best_crps:
                best_crps = mean
                best = (k, t)
    logger.info("tuned k=%.1f threshold=%.2f (tune mean CRPS %.3f)", best[0], best[1], best_crps)
    return best[0], best[1], grid


def gate_confusion(
    panels: dict[str, ScenarioPanel], sids: list[str], threshold: float
) -> dict[str, int]:
    """Confusion matrix of the gate itself (fired vs is_shock)."""
    cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for sid in sids:
        p = panels[sid]
        fired = abs(p["z_stat"]) >= threshold
        if p["is_shock"] and fired:
            cm["tp"] += 1
        elif p["is_shock"]:
            cm["fn"] += 1
        elif fired:
            cm["fp"] += 1
        else:
            cm["tn"] += 1
    return cm


def run_exp6(out_path: Path, seed: int = _EVAL_SEED) -> Path:
    """Run the tuning + held-out evaluation and write results JSON."""
    panels = build_exp6_panels()
    shock_ids = [str(s["id"]) for s in SCENARIOS6 if s["kind"] == "shock"]
    benign_ids = [str(s["id"]) for s in SCENARIOS6 if s["kind"] == "benign"]
    tune_sids = shock_ids[:3] + benign_ids[:3]
    heldout_sids = shock_ids[3:] + benign_ids[3:]

    k_star, t_star, grid = tune(panels, tune_sids, seed)

    variants = ["m0", "m3", "m3_shrink", "m3_gate", "m3_both"]
    heldout: dict[str, dict[str, float]] = {
        v: eval_variant(panels, heldout_sids, v, k_star, t_star, seed) for v in variants
    }
    tune_eval: dict[str, dict[str, float]] = {
        v: eval_variant(panels, tune_sids, v, k_star, t_star, seed) for v in variants
    }

    scenario_rows: list[dict[str, object]] = []
    for sid in heldout_sids:
        p = panels[sid]
        scenario_rows.append(
            {
                "id": sid,
                "kind": "shock" if p["is_shock"] else "benign",
                "z_stat": round(p["z_stat"], 3),
                "gate_fired": bool(abs(p["z_stat"]) >= t_star),
                "mock_delta": round(p["mock_delta"], 3),
                "correct_delta": round(p["correct_delta"], 3),
                "crps": {v: round(heldout[v][sid], 4) for v in variants},
            }
        )

    def _means(d: dict[str, dict[str, float]]) -> dict[str, float]:
        return {v: round(float(np.mean(list(d[v].values()))), 4) for v in variants}

    results = {
        "experiment": "exp6_m3plus",
        "mode": "mock (mechanism test; deltas replicate the real-Groq error model)",
        "tuned": {"k_star": k_star, "threshold_star": t_star},
        "tuning_grid": grid,
        "tune_sids": tune_sids,
        "heldout_sids": heldout_sids,
        "tune_mean_crps": _means(tune_eval),
        "heldout_mean_crps": _means(heldout),
        "heldout_scenarios": scenario_rows,
        "gate_confusion_heldout": gate_confusion(panels, heldout_sids, t_star),
        "gate_confusion_all": gate_confusion(panels, shock_ids + benign_ids, t_star),
        "config": {
            "n_customers": N_CUSTOMERS,
            "n_months": N_MONTHS,
            "panel_months": PANEL_MONTHS,
            "shift_months": SHIFT_MONTHS,
            "n_test": N_TEST,
            "seed": seed,
        },
    }
    out_file = out_path / "exp6_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_file)
    return out_file


def main() -> int:
    """CLI entrypoint (mock mode only)."""
    out = run_exp6(Path(__file__).resolve().parent)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
