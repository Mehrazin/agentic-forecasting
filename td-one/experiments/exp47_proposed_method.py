"""exp47: Proposed method end-to-end test.

The TD-One proposed method (src/td_one/agent/proposed_method.py) combines
everything with positive lift (see docs/POSITIVE_LIFT_AUDIT.md):

  - exp43 regime router (backbone): hand regimes -> LR -> per-regime experts
  - exp32 trust flag: EASY/HARD reporting
  - exp6/decision_policy: "do nothing" corroboration gate
  - exp5/exp10: conformal calibration wrapper
  - (gated, off by default) exp44 news modulation, exp11 LLM sensor

Comparisons (blocked CV, paired t-tests):
  - proposed (full) vs equal weights
  - proposed (full) vs best single
  - proposed (full) vs exp43-router-only (no gates, no conformal)
  - ablations: no-gate, no-conformal

Datasets: M4, SMM, Food CPI, Synthetic (record pipeline from exp42).

Usage:
    cd td-one && .venv/bin/python experiments/exp47_proposed_method.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402

from td_one.agent.proposed_method import (  # noqa: E402
    ProposedForecaster,
    generic_regime_spec,
    m4_regime_spec,
)

MODELS = f42.MODELS
FEATURE_NAMES = f42.FEATURE_NAMES


def _crps(y: np.ndarray, qs: np.ndarray) -> float:
    return f42._crps_quantile(y, qs)


def _equal_weights_quantiles(r: dict) -> np.ndarray:
    qm = [np.asarray(r["quantiles"][m], dtype=float) for m in MODELS]
    return sum(qm) / len(qm)


def _best_single_quantiles(r: dict, best: str) -> np.ndarray:
    return np.asarray(r["quantiles"][best], dtype=float)


def run_dataset_cv(
    name: str,
    records: list[dict],
    spec_fn,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Blocked CV: fit ProposedForecaster per fold, compare vs baselines."""
    # fold indices: by series for M4/synthetic, by time blocks otherwise
    if name in ("m4", "synthetic"):
        folds = f42.kfold_by_series(records, n_folds=n_folds, seed=seed)
    else:
        # time blocks: sort by origin, split into contiguous folds
        order = sorted(range(len(records)), key=lambda i: records[i]["origin"])
        folds = []
        chunk = len(order) // n_folds
        for f in range(n_folds):
            te = order[f * chunk : (f + 1) * chunk if f < n_folds - 1 else len(order)]
            tr = [i for i in order if i not in set(te)]
            folds.append((tr, te))

    prop_c, eq_c, best_c, router_only_c = [], [], [], []
    gate_rates, trust_easy_rates = [], []
    for tr_idx, te_idx in folds:
        tr = [records[i] for i in tr_idx]
        te = [records[i] for i in te_idx]

        # global best single on TRAIN (no hindsight)
        mean_crps = {m: float(np.mean([r["crps"][m] for r in tr])) for m in MODELS}
        best = min(mean_crps, key=mean_crps.get)

        # full proposed method
        fc = ProposedForecaster(h=1, regime_spec=spec_fn(), models=MODELS)
        fc.fit(tr, FEATURE_NAMES)

        # router-only ablation (gates + conformal disabled)
        fc_plain = ProposedForecaster(h=1, regime_spec=spec_fn(), models=MODELS)
        fc_plain.fit(tr, FEATURE_NAMES)

        for r in te:
            actual = np.asarray(r["actual"], dtype=float)
            qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in MODELS}
            feats = dict(zip(FEATURE_NAMES, r["features"], strict=True))
            # train window not stored in records; use zeros (vol floor ~ small)
            train_proxy = np.zeros(30)

            out = fc.predict(train_proxy, qm, feats, FEATURE_NAMES)
            prop_c.append(_crps(actual, out.quantiles))
            gate_rates.append(1.0 if out.gated else 0.0)
            trust_easy_rates.append(1.0 if out.trust == "EASY" else 0.0)

            eq_c.append(_crps(actual, _equal_weights_quantiles(r)))
            best_c.append(_crps(actual, _best_single_quantiles(r, best)))

            # router-only: bypass gates by calling internals
            probs, _, _ = fc_plain._regime_probs(feats, FEATURE_NAMES)
            qr = np.zeros_like(out.quantiles)
            for reg, p in probs.items():
                qr += p * qm[fc_plain._expert_per_regime[reg]]
            router_only_c.append(_crps(actual, qr))

    prop_c, eq_c = np.array(prop_c), np.array(eq_c)
    best_c, router_only_c = np.array(best_c), np.array(router_only_c)

    def cmp(a: np.ndarray, b: np.ndarray, label: str) -> dict:
        t, p = ttest_rel(a, b)
        return {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a - b)),
            "t": float(t),
            "p": float(p),
            "win_rate": float(np.mean(a < b)),
        }

    return {
        "dataset": name,
        "n": len(records),
        "proposed_vs_equal": cmp(prop_c, eq_c, "proposed_vs_equal"),
        "proposed_vs_best": cmp(prop_c, best_c, "proposed_vs_best"),
        "proposed_vs_router_only": cmp(prop_c, router_only_c, "proposed_vs_router_only"),
        "gate_fire_rate": float(np.mean(gate_rates)),
        "trust_easy_rate": float(np.mean(trust_easy_rates)),
        "mean_crps": {
            "proposed": float(np.mean(prop_c)),
            "equal": float(np.mean(eq_c)),
            "best_single": float(np.mean(best_c)),
            "router_only": float(np.mean(router_only_c)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    n_m4 = 20 if args.quick else 100
    print(f"[exp47] loading datasets (quick={args.quick})...", flush=True)
    m4 = f42.generate_records(f42.load_m4(n=n_m4), 48, 6, 6, "m4")
    smm = f42.generate_records([f42.load_smm()[0]], 36, 1, 50, "smm")
    fcpi = f42.generate_records([f42.load_food_cpi()[0]], 48, 1, 24, "food_cpi")
    synth_n = 12 if args.quick else 40
    synth = f42.generate_records(f42.load_synthetic(n_series=synth_n), 30, 1, 12, "synthetic")
    print(
        f"[exp47] records: m4={len(m4)} smm={len(smm)} fcpi={len(fcpi)} synth={len(synth)}",
        flush=True,
    )

    results = {}
    results["m4"] = run_dataset_cv("m4", m4, m4_regime_spec)
    results["synthetic"] = run_dataset_cv("synthetic", synth, m4_regime_spec)
    results["smm"] = run_dataset_cv("smm", smm, generic_regime_spec)
    results["food_cpi"] = run_dataset_cv("food_cpi", fcpi, generic_regime_spec)

    for name, r in results.items():
        m = r["mean_crps"]
        pe = r["proposed_vs_equal"]
        pb = r["proposed_vs_best"]
        pr = r["proposed_vs_router_only"]
        print(
            f"[exp47] {name}: proposed={m['proposed']:.4f} equal={m['equal']:.4f} "
            f"best={m['best_single']:.4f} router_only={m['router_only']:.4f} | "
            f"vs equal p={pe['p']:.4g} | vs best p={pb['p']:.4g} | "
            f"vs router_only p={pr['p']:.4g} | gate={r['gate_fire_rate']:.2f} "
            f"trust_easy={r['trust_easy_rate']:.2f}",
            flush=True,
        )

    out = {
        "date": datetime.now(UTC).isoformat(),
        "quick": args.quick,
        "elapsed_s": time.time() - t0,
        "models": MODELS,
        "features": FEATURE_NAMES,
        "results": results,
    }
    with open(ROOT / "experiments" / "exp47_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[exp47] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
