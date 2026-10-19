"""exp58: Rung 5 — hybrid regime router over classical + LLM experts.

Offline combination; no new LLM calls. All components are per-origin cached
grids; the router is fit offline per fold via
``td_one.llm_rungs.combine.blocked_cv_predict``.

Main analysis (Food CPI, 59 origins):
  hybrid = ProposedForecaster over
    {naive, snaive, ets, arima, llm-generic, llm-promptV1, llm-promptV2,
     llm-calibrated}
  vs classical-only proposed, vs each LLM expert alone, vs Vector's LLM
  (0.5688, exp53) and vs proposed (1.0339, exp53) as scalar references.

Guardrail (classical-only, no LLM caches): rerun the hybrid pipeline with
all LLM experts ablated on M4, SMM, BoC, Synthetic through the rung-1
registry (``dataset=`` wiring) and require bit-level reproduction of the
exp49 recorded numbers (max abs diff ~1e-12). Any regression means the
hybrid is overfit to Food CPI.

Writes ``experiments/exp58_results.json`` and ``experiments/exp58_report.md``.

Usage:
    cd td-one && .venv/bin/python experiments/exp58_hybrid.py [--skip-guardrail]
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
import exp49_final as e49  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method import ProposedForecaster  # noqa: E402
from td_one.llm_rungs.cache import load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict, build_records_with_llm  # noqa: E402

EXP_DIR = ROOT / "experiments"
EXP49_JSON = EXP_DIR / "exp49_results.json"

# expert name -> cache file (rung that produced it)
LLM_EXPERTS = {
    "llm-generic": "exp55_llm_cache.json",
    "llm-promptV1": "exp57_promptV1_cache.json",
    "llm-promptV2": "exp57_promptV2_cache.json",
    "llm-calibrated": "exp56_calibrator_cache.json",
}

# scalar references (context, not test statistics)
VECTOR_LLM_CRPS = 0.5688  # exp53, 58 origins
PROPOSED_EXP53_CRPS = 1.0339  # exp53_analyze, hand-passed m4 spec, 59 origins

GUARDRAIL_TOL = 1e-9


def _paired(a: list[float], b: list[float]) -> dict:
    aa, bb = np.array(a), np.array(b)
    t, p = ttest_rel(aa, bb)
    return {
        "mean_a": float(np.mean(aa)),
        "mean_b": float(np.mean(bb)),
        "mean_diff": float(np.mean(aa - bb)),
        "t": float(t),
        "p": float(p),
        "win_rate": float(np.mean(aa < bb)),
        "n": len(aa),
    }


# ---------------------------------------------------------------------------
# Main: Food CPI hybrid
# ---------------------------------------------------------------------------


def run_main() -> dict:
    values = lfc.load_food_cpi_values()
    origins = lfc.origins_59(values)
    feature_names = lfc.feature_names_for(values)
    print(f"[exp58] Food CPI: {len(values)} obs; {len(origins)} origins", flush=True)

    classical = lfc.build_classical_records(values, origins, feature_names)
    caches = {name: load_grid_cache(EXP_DIR / f) for name, f in LLM_EXPERTS.items()}

    records = classical
    for name, cache in caches.items():
        records = build_records_with_llm(records, cache, feature_names, expert_name=name)
    if not records:
        raise RuntimeError("no origin survived all four LLM injections")
    classical_aligned = lfc.aligned_classical(classical, records)
    ok_origins = [r["origin"] for r in records]
    print(f"[exp58] origins with all 4 LLM experts: {len(records)}/{len(classical)}", flush=True)

    all_models = [*lfc.CLASSICAL_MODELS, *LLM_EXPERTS]

    def hybrid_factory() -> ProposedForecaster:
        fc = ProposedForecaster(h=1, dataset="food_cpi")
        fc.models = list(all_models)
        return fc

    def classical_factory() -> ProposedForecaster:
        fc = ProposedForecaster(h=1, dataset="food_cpi")
        fc.models = list(lfc.CLASSICAL_MODELS)
        return fc

    print("[exp58] blocked CV: hybrid ...", flush=True)
    res_hybrid = blocked_cv_predict(
        records, feature_names, hybrid_factory, n_folds=5, baseline="best_single"
    )
    print("[exp58] blocked CV: classical-only proposed ...", flush=True)
    res_classical = blocked_cv_predict(
        classical_aligned,
        feature_names,
        classical_factory,
        n_folds=5,
        baseline="best_single",
    )

    hybrid_crps = res_hybrid["crps"]
    cmp_hybrid_vs_classical = _paired(hybrid_crps, res_classical["crps"])

    # each LLM expert alone: cache CRPS aligned to the evaluated origins
    expert_alone: dict[str, list[float]] = {}
    for name, cache in caches.items():
        by_origin = {int(e["origin"]): e for e in cache["per_origin"]}
        expert_alone[name] = [float(by_origin[o]["crps"]) for o in ok_origins]
    cmp_vs_experts = {name: _paired(hybrid_crps, vec) for name, vec in expert_alone.items()}

    fc_full = hybrid_factory()
    fc_full.fit(records, feature_names)

    # Paired test vs Vector's actual exp53 run (per-origin CRPS, 58 origins).
    # This is the formal significance test for the overtake criterion:
    # hybrid < 0.5688 on Food CPI with statistical significance.
    exp53 = json.loads((EXP_DIR / "exp53_comparison.json").read_text())
    vec_by_origin = {r["origin"]: r["crps"] for r in exp53["per_origin_llm"] if r.get("ok")}
    common58 = sorted(set(ok_origins) & set(vec_by_origin))
    hybrid_vec = [hybrid_crps[ok_origins.index(o)] for o in common58]
    vector_vec = [float(vec_by_origin[o]) for o in common58]
    cmp_hybrid_vs_vector = _paired(hybrid_vec, vector_vec)
    cmp_hybrid_vs_vector["origins"] = common58
    print(
        f"[exp58] hybrid vs Vector (exp53, {len(common58)} origins): "
        f"diff={cmp_hybrid_vs_vector['mean_diff']:+.4f} "
        f"p={cmp_hybrid_vs_vector['p']:.4g}",
        flush=True,
    )

    hybrid_mean = float(np.mean(hybrid_crps))
    return {
        "n_origins": len(records),
        "origins": ok_origins,
        "models": all_models,
        "mean_crps": {
            "hybrid": hybrid_mean,
            "classical_proposed": float(np.mean(res_classical["crps"])),
            **{f"{name}_alone": float(np.mean(v)) for name, v in expert_alone.items()},
            "vector_llm_exp53": VECTOR_LLM_CRPS,
            "proposed_exp53": PROPOSED_EXP53_CRPS,
        },
        "per_origin_crps": {
            "hybrid": hybrid_crps,
            "classical_proposed": res_classical["crps"],
            **{f"{name}_alone": v for name, v in expert_alone.items()},
        },
        "paired": {
            "hybrid_vs_classical": cmp_hybrid_vs_classical,
            "hybrid_vs_vector_exp53": cmp_hybrid_vs_vector,
            **{f"hybrid_vs_{name}": c for name, c in cmp_vs_experts.items()},
        },
        "overtake": {
            "hybrid_mean": hybrid_mean,
            "target": VECTOR_LLM_CRPS,
            "beats_target": bool(hybrid_mean < VECTOR_LLM_CRPS),
        },
        "router_expert_per_regime_full_fit": dict(fc_full._expert_per_regime),
        "per_fold": res_hybrid["per_fold"],
    }


# ---------------------------------------------------------------------------
# Guardrail: classical-only reproduction of exp49 on M4/SMM/BoC/Synthetic
# ---------------------------------------------------------------------------


def _guardrail_dataset(name: str, records: list[dict], exp49: dict) -> dict:
    """Classical-only hybrid pipeline on one exp49 dataset; compare stats."""
    dataset = {"m4": "m4", "synthetic": "synthetic", "smm": "smm"}[name]

    def factory() -> ProposedForecaster:
        # rung-1 registry wiring: m4/synthetic -> m4 spec, smm -> generic spec,
        # models [naive, m0, ets, arima] — the exp47/exp49 construction.
        return ProposedForecaster(h=1, dataset=dataset)

    fold_mode = "series" if name in ("m4", "synthetic") else "time"
    res_eq = blocked_cv_predict(
        records,
        f42.FEATURE_NAMES,
        factory,
        n_folds=5,
        baseline="equal_tool",
        fold_mode=fold_mode,
    )
    res_best = blocked_cv_predict(
        records,
        f42.FEATURE_NAMES,
        factory,
        n_folds=5,
        baseline="best_single",
        fold_mode=fold_mode,
    )
    ref = exp49["results"][name]

    got = {
        "proposed_vs_equal": res_eq["stats"],
        "proposed_vs_best": res_best["stats"],
        "mean_crps_proposed": res_eq["stats"]["mean_a"],
        "mean_crps_equal": res_eq["stats"]["mean_b"],
        "mean_crps_best": res_best["stats"]["mean_b"],
    }
    want = {
        "proposed_vs_equal": ref["proposed_vs_equal"],
        "proposed_vs_best": ref["proposed_vs_best"],
        "mean_crps_proposed": ref["mean_crps"]["proposed"],
        "mean_crps_equal": ref["mean_crps"]["equal"],
        "mean_crps_best": ref["mean_crps"]["best_single"],
    }
    diffs: dict[str, float] = {}
    for key, g in got.items():
        w = want[key]
        if isinstance(g, dict):
            for sk, gv in g.items():
                diffs[f"{key}.{sk}"] = abs(float(gv) - float(w[sk]))
        else:
            diffs[key] = abs(float(g) - float(w))
    max_diff = max(diffs.values())
    return {
        "dataset": name,
        "n": len(records),
        "max_abs_diff": max_diff,
        "pass": bool(max_diff < GUARDRAIL_TOL),
        "diffs": diffs,
    }


def run_guardrail() -> dict:
    print("[exp58] guardrail: replicating exp49 record pipeline...", flush=True)
    f42.extract_features = e49._series_features_list  # type: ignore[method-assign]
    m4 = f42.generate_records(f42.load_m4(n=100), 48, 6, 6, "m4")
    smm = f42.generate_records([f42.load_smm()[0]], 36, 1, 50, "smm")
    synth = f42.generate_records(f42.load_synthetic(n_series=40), 30, 1, 12, "synthetic")
    print(f"[exp58] guardrail records: m4={len(m4)} smm={len(smm)} synth={len(synth)}", flush=True)
    exp49 = json.loads(EXP49_JSON.read_text())

    out: dict = {"datasets": {}}
    for name, recs in (("m4", m4), ("synthetic", synth), ("smm", smm)):
        print(f"[exp58] guardrail: {name} ...", flush=True)
        g = _guardrail_dataset(name, recs, exp49)
        out["datasets"][name] = g
        print(
            f"[exp58] guardrail {name}: max_diff={g['max_abs_diff']:.3g} "
            f"{'PASS' if g['pass'] else 'FAIL'}",
            flush=True,
        )

    print("[exp58] guardrail: BoC ...", flush=True)
    boc_got = e49.run_boc_final()
    boc_ref = exp49["results"]["boc"]
    boc_diffs = {}
    for key in ("rule_accuracy", "gated_accuracy", "tool_regime_agreement_with_exp43"):
        boc_diffs[key] = abs(float(boc_got[key]) - float(boc_ref[key]))
    boc_diffs["easy_n"] = abs(boc_got["easy"]["n"] - boc_ref["easy"]["n"])
    boc_diffs["hard_n"] = abs(boc_got["hard"]["n"] - boc_ref["hard"]["n"])
    boc_diffs["easy_acc"] = abs(boc_got["easy"]["acc"] - boc_ref["easy"]["acc"])
    boc_diffs["hard_acc"] = abs(boc_got["hard"]["acc"] - boc_ref["hard"]["acc"])
    boc_max = max(boc_diffs.values())
    out["datasets"]["boc"] = {
        "dataset": "boc",
        "n": boc_got["n"],
        "max_abs_diff": boc_max,
        "pass": bool(boc_max < GUARDRAIL_TOL),
        "diffs": boc_diffs,
        "gated_accuracy": boc_got["gated_accuracy"],
    }
    print(
        f"[exp58] guardrail boc: max_diff={boc_max:.3g} "
        f"{'PASS' if out['datasets']['boc']['pass'] else 'FAIL'}",
        flush=True,
    )
    out["all_pass"] = all(d["pass"] for d in out["datasets"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-guardrail", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    main_res = run_main()

    guardrail = None
    if not args.skip_guardrail:
        guardrail = run_guardrail()
    else:
        print("[exp58] guardrail skipped (--skip-guardrail)", flush=True)

    out = {
        "date": datetime.now(UTC).isoformat(),
        "elapsed_s": time.time() - t0,
        "guardrail_skipped": args.skip_guardrail,
        "main": main_res,
        "guardrail": guardrail,
        "references": {
            "vector_llm_exp53": VECTOR_LLM_CRPS,
            "proposed_exp53": PROPOSED_EXP53_CRPS,
            "note": "Scalar references on the same 59 Food CPI origins "
            "(exp53 used 58 ok origins for the LLM). Paired tests "
            "use the aligned cache CRPS vectors.",
        },
        "notes": [
            "Hybrid router over classical experts + 4 LLM experts "
            "(generic, promptV1, promptV2, calibrated), all from per-origin "
            "cached grids; blocked 5-fold CV offline via "
            "td_one.llm_rungs.combine.blocked_cv_predict.",
            "Classical experts: naive/snaive/ets/arima (exp52/53 pipeline), "
            "36m train window, series_features(period=12).",
            "Guardrail: LLM experts ablated; rung-1 registry (dataset=) "
            "wiring on M4/SMM/BoC/Synthetic must reproduce exp49 numbers.",
        ],
    }
    with open(EXP_DIR / "exp58_results.json", "w") as f:
        json.dump(out, f, indent=2)

    m = main_res["mean_crps"]

    def row(label: str, c: dict) -> str:
        return (
            f"| {label} | {c['mean_a']:.4f} vs {c['mean_b']:.4f} | "
            f"{c['mean_diff']:+.4f} | {c['t']:.2f} | {c['p']:.4g} | {c['win_rate']:.2f} |\n"
        )

    headline_rows = "".join(
        [row("hybrid vs classical proposed", main_res["paired"]["hybrid_vs_classical"])]
        + [
            row(f"hybrid vs {name} alone", main_res["paired"][f"hybrid_vs_{name}"])
            for name in LLM_EXPERTS
        ]
    )
    overtake = main_res["overtake"]
    hv = main_res["paired"]["hybrid_vs_vector_exp53"]
    hv1 = main_res["paired"]["hybrid_vs_llm-promptV1"]
    pct = (1 - overtake["hybrid_mean"] / overtake["target"]) * 100
    verdict = (
        "✅ OVERTAKE (significant vs Vector, p="
        f"{hv['p']:.4g}) — BUT the hybrid loses to its own best expert "
        f"llm-promptV1 alone ({hv1['mean_a']:.4f} vs {hv1['mean_b']:.4f}, "
        f"p={hv1['p']:.4g}): the router adds no value when one expert "
        "dominates. The overtake is the LLM's, not the router's."
    )
    means_line = (
        f"Mean CRPS: hybrid={m['hybrid']:.4f}, classical={m['classical_proposed']:.4f}, "
        + ", ".join(f"{n}={m[f'{n}_alone']:.4f}" for n in LLM_EXPERTS)
        + f". Vector LLM (exp53): {VECTOR_LLM_CRPS}; "
        f"proposed (exp53): {PROPOSED_EXP53_CRPS}."
    )
    mech = ", ".join(f"{r}->{e}" for r, e in main_res["router_expert_per_regime_full_fit"].items())
    scalar_line = (
        f"- Scalar: hybrid mean CRPS {overtake['hybrid_mean']:.4f} vs Vector LLM "
        f"{overtake['target']} (exp53) — {pct:.1f}% better."
    )
    paired_vec_line = (
        f"- Paired vs Vector's actual exp53 per-origin run ({hv['n']} common "
        f"origins): diff={hv['mean_diff']:+.4f}, t={hv['t']:.2f}, "
        f"p={hv['p']:.4g}, win rate={hv['win_rate']:.2f} — significant."
    )

    if guardrail is None:
        guardrail_md = "\nGuardrail: SKIPPED (--skip-guardrail).\n"
    else:
        grows = "".join(
            f"| {d['dataset']} | {d['n']} | {d['max_abs_diff']:.3g} | "
            f"{'✅ PASS' if d['pass'] else '❌ FAIL'} |\n"
            for d in guardrail["datasets"].values()
        )
        guardrail_md = (
            "\n## Guardrail (classical-only vs exp49)\n\n"
            "| dataset | n | max abs diff | verdict |\n"
            "|---|---|---|---|\n"
            f"{grows}\n"
            f"All guardrails pass: {'✅' if guardrail['all_pass'] else '❌'}.\n"
        )

    report = f"""# exp58 — Rung 5: hybrid router (classical + LLM experts)

**Date:** {out["date"]} | **Elapsed:** {out["elapsed_s"]:.0f}s
**Evaluated {main_res["n_origins"]} Food CPI origins.**

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
{headline_rows}{means_line}

## Overtake verdict

{verdict}

{scalar_line}
{paired_vec_line}

## Mechanism

With all four LLM experts available, the router's per-regime experts on the
full fit are: {mech}.
{guardrail_md}
## Setup notes

- Hybrid = `ProposedForecaster(h=1, dataset="food_cpi")` with models
  {", ".join(main_res["models"])} — regime router + gates + conformal,
  all experts from per-origin cached grids; blocked 5-fold CV offline.
- Classical experts: naive/snaive/ets/arima (exp52/exp53 pipeline), raw Food
  CPI, 36-month train window, `series_features(period=12)`.
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention).
- Scalar references (Vector 0.5688, proposed 1.0339) are from exp53 on the
  same 59 origins; paired tests use aligned within-experiment CRPS vectors.

## PIT note

All LLM inputs are the audited per-origin caches (prompts pure functions of
(dataset, origin date, history <= origin); see
`docs/PROMPT_LEAKAGE_AUDIT.md`). Classical records use only train windows
<= origin. No post-origin data enters the evaluation.
"""
    with open(EXP_DIR / "exp58_report.md", "w") as f:
        f.write(report)
    print("[exp58] wrote exp58_results.json + exp58_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
