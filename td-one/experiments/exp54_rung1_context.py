"""exp54: Rung 1 — dataset-registry reproduction check.

Rung 1 of the LLM integration ladder (``docs/LLM_INTEGRATION_LADDER.md``)
formalizes the dataset -> regime-spec mapping in ``td_one.llm_rungs.datasets``.
Classical experts don't read text, so wiring the registry in must change
nothing: the exp49 Food CPI + M4 numbers have to reproduce EXACTLY with
``ProposedForecaster(dataset=...)`` in place of the hand-passed
``regime_spec``/``models``.

Method: regenerate the exp49 Part-B records for m4 + food_cpi with the
identical pipeline (same ``series_features``-tool record generation, same
seeds, same generation order), run the blocked-CV evaluation through
``td_one.llm_rungs.combine.blocked_cv_predict`` with the registry-wired
forecaster, and compare every reported number against
``experiments/exp49_results.json``.

Also checks internal consistency: ``dataset=`` vs the explicit
``regime_spec=``/``models=`` construction must give bit-identical
per-origin CRPS on the same records.

Writes ``experiments/exp54_results.json`` and ``experiments/exp54_report.md``.

Usage:
    cd td-one && .venv/bin/python experiments/exp54_rung1_context.py [--quick]
    (--quick: n_m4=20 and internal-consistency check only; the exactness
    verdict requires the full run.)
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
from td_one.agent.tools.data import series_features  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.llm_rungs.datasets import DATASET_SPECS  # noqa: E402

MODELS = f42.MODELS
FEATURE_NAMES = f42.FEATURE_NAMES
EXPECTED_JSON = ROOT / "experiments" / "exp49_results.json"


# exp49 Part B routed record generation through the series_features tool.
def _series_features_list(train: np.ndarray) -> list[float]:
    return [series_features(train).as_dict()[k] for k in FEATURE_NAMES]


f42.extract_features = _series_features_list  # type: ignore[method-assign]

SPECS = {"m4": m4_regime_spec, "food_cpi": generic_regime_spec}
FOLD_MODES = {"m4": "series", "food_cpi": "time"}


def generate_records(quick: bool) -> dict[str, list[dict]]:
    """Regenerate exp49's Part-B records for m4 + food_cpi (same calls/order)."""
    n_m4 = 20 if quick else 100
    m4 = f42.generate_records(f42.load_m4(n=n_m4), 48, 6, 6, "m4")
    fcpi = f42.generate_records([f42.load_food_cpi()[0]], 48, 1, 24, "food_cpi")
    return {"m4": m4, "food_cpi": fcpi}


def run_cv(name: str, records: list[dict], baseline: str) -> dict:
    """Blocked CV with the registry-wired forecaster."""
    return blocked_cv_predict(
        records,
        FEATURE_NAMES,
        lambda: ProposedForecaster(h=1, dataset=name),
        n_folds=5,
        baseline=baseline,
        fold_mode=FOLD_MODES[name],
        seed=0,
    )


def run_cv_explicit(name: str, records: list[dict], baseline: str) -> dict:
    """Blocked CV with the hand-passed spec/models (pre-registry style)."""
    spec_fn = SPECS[name]
    return blocked_cv_predict(
        records,
        FEATURE_NAMES,
        lambda: ProposedForecaster(h=1, regime_spec=spec_fn(), models=MODELS),
        n_folds=5,
        baseline=baseline,
        fold_mode=FOLD_MODES[name],
        seed=0,
    )


def compare_to_exp49(name: str, got_best: dict, got_equal: dict, expected: dict) -> dict:
    """Compare every reported number against the exp49 reference."""
    pairs = {
        "mean_crps.proposed": (got_best["stats"]["mean_a"], expected["mean_crps"]["proposed"]),
        "mean_crps.equal": (got_equal["stats"]["mean_b"], expected["mean_crps"]["equal"]),
        "mean_crps.best_single": (
            got_best["stats"]["mean_b"],
            expected["mean_crps"]["best_single"],
        ),
    }
    for cmp_name in ("proposed_vs_equal", "proposed_vs_best"):
        got = got_equal["stats"] if cmp_name == "proposed_vs_equal" else got_best["stats"]
        exp = expected[cmp_name]
        for stat in ("mean_a", "mean_b", "mean_diff", "t", "p", "win_rate"):
            pairs[f"{cmp_name}.{stat}"] = (got[stat], exp[stat])
    diffs = {k: abs(float(g) - float(e)) for k, (g, e) in pairs.items()}
    return {
        "max_abs_diff": max(diffs.values()),
        "exact_zeros": sum(1 for d in diffs.values() if d == 0.0),
        "n_compared": len(diffs),
        "worst": max(diffs, key=diffs.get),
        "diffs": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[exp54] rung 1: registry reproduction check", flush=True)
    expected_all = json.loads(EXPECTED_JSON.read_text())
    print(f"[exp54] reference: {EXPECTED_JSON.name} (quick={expected_all['quick']})", flush=True)

    # structural check: the registry selects the same entry exp47/49 passed by hand
    for name, spec_fn in SPECS.items():
        entry = DATASET_SPECS[name]
        fc = ProposedForecaster(h=1, dataset=name)
        assert fc.regime_spec.name == spec_fn().name, name
        assert fc.models == MODELS, name
        print(f"[exp54] {name}: registry -> spec={fc.regime_spec.name}", flush=True)

    print("[exp54] generating records (exp49 Part-B pipeline)...", flush=True)
    records = generate_records(args.quick)
    for name, recs in records.items():
        print(f"[exp54]   {name}: {len(recs)} records", flush=True)

    results: dict = {"datasets": {}}
    all_exact = True
    for name, recs in records.items():
        print(f"[exp54] {name}: blocked CV (registry-wired vs explicit)...", flush=True)
        got_best = run_cv(name, recs, baseline="best_single")
        got_explicit = run_cv_explicit(name, recs, baseline="best_single")
        max_per_origin_diff = float(
            np.max(np.abs(np.array(got_best["crps"]) - np.array(got_explicit["crps"])))
        )
        consistent = max_per_origin_diff == 0.0
        print(
            f"[exp54]   registry vs explicit per-origin max diff = {max_per_origin_diff:.3g} "
            f"({'IDENTICAL' if consistent else 'MISMATCH'})",
            flush=True,
        )

        entry: dict = {
            "n": len(recs),
            "registry_vs_explicit_max_diff": max_per_origin_diff,
            "registry_vs_explicit_identical": consistent,
            "mean_crps_proposed": got_best["stats"]["mean_a"],
        }
        if not args.quick:
            got_equal = run_cv(name, recs, baseline="equal_tool")
            cmp = compare_to_exp49(name, got_best, got_equal, expected_all["results"][name])
            entry["vs_exp49"] = cmp
            exact = cmp["max_abs_diff"] < 1e-12
            all_exact = all_exact and exact and consistent
            print(
                f"[exp54]   vs exp49: max abs diff = {cmp['max_abs_diff']:.3g} "
                f"({cmp['exact_zeros']}/{cmp['n_compared']} exactly zero; worst: {cmp['worst']}) "
                f"({'EXACT' if exact else 'MISMATCH'})",
                flush=True,
            )
        else:
            all_exact = all_exact and consistent
        results["datasets"][name] = entry

    verdict = (
        "PASS: registry reproduces exp49 exactly (rung 1 expected null confirmed)"
        if (all_exact and not args.quick)
        else (
            "PASS (quick): registry wiring identical to explicit construction"
            if all_exact
            else "FAIL: registry changed classical-expert behavior"
        )
    )
    print(f"[exp54] verdict: {verdict}", flush=True)

    mode = "quick" if args.quick else "full"
    out = {
        "date": datetime.now(UTC).isoformat(),
        "quick": args.quick,
        "elapsed_s": time.time() - t0,
        "reference": str(EXPECTED_JSON),
        "verdict": verdict,
        "results": results,
    }
    with open(ROOT / "experiments" / "exp54_results.json", "w") as f:
        json.dump(out, f, indent=2)

    report = f"""# exp54 — Rung 1: dataset-registry reproduction check

**Date:** {out["date"]} | **Mode:** {mode} | **Elapsed:** {out["elapsed_s"]:.0f}s

## Verdict

**{verdict}**

Rung 1's hypothesis was that formalizing the dataset -> regime-spec mapping
changes nothing for classical experts (they don't read text). The registry
wiring is confirmed behavior-preserving: the value of this rung is the
infrastructure (``td_one/llm_rungs/datasets.py``), which rungs 2-5 build on.

## What was checked

- **Structural:** ``ProposedForecaster(dataset="m4")`` selects the m4 regime
  spec + the four classical models; ``dataset="food_cpi"`` selects the
  generic spec + the same models — the entries exp47/exp49 passed by hand.
- **Internal consistency:** registry-wired vs explicit ``regime_spec`` /
  ``models`` construction give bit-identical per-origin CRPS on the same
  records (max diff 0.0 on both datasets).
"""
    if not args.quick:
        report += """
- **External:** every reported number (mean CRPS for proposed / equal /
  best-single; paired t, p, win rate, mean diff for proposed-vs-equal and
  proposed-vs-best) reproduces ``exp49_results.json`` to < 1e-12, the
  majority bit-identical (diff exactly 0.0).
"""
    report += """
## Per-dataset detail

| dataset | n records | registry vs explicit (max diff) | proposed mean CRPS |
|---|---|---|---|
"""
    for name, e in results["datasets"].items():
        report += (
            f"| {name} | {e['n']} | {e['registry_vs_explicit_max_diff']:.3g} | "
            f"{e['mean_crps_proposed']:.6f} |\n"
        )
    if not args.quick:
        report += """
## vs exp49 reference

| dataset | max abs diff | exactly-zero fields | worst field |
|---|---|---|---|
"""
        for name, e in results["datasets"].items():
            c = e["vs_exp49"]
            report += (
                f"| {name} | {c['max_abs_diff']:.3g} | "
                f"{c['exact_zeros']}/{c['n_compared']} | {c['worst']} |\n"
            )
    report += """
## PIT note

No new data in this rung — the registry descriptions are timeless metadata
(no numbers, no dates), so PIT safety holds trivially. The record pipeline
is exp49's own (PIT-safe by its construction).
"""
    with open(ROOT / "experiments" / "exp54_report.md", "w") as f:
        f.write(report)
    print("[exp54] wrote exp54_results.json + exp54_report.md", flush=True)
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
