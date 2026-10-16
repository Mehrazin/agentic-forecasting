"""Collect multi-seed scale-up results into experiments/multiseed_results.json.

Reads experiments/scale_up_results_<tag>.json for tags
seed{s}-{m0w6,m0w3,chronos2,timesfm25}, s in [0..4], averages the
`summary` per-segment means (same convention as the published numbers),
runs paired t-tests per seed (M0_w6 vs each challenger), and writes the
consolidated table.

Usage:
    cd td-one && uv run python experiments/collect_multiseed.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [0, 1, 2, 3, 4]
# tag -> (seed, model_label)
TAGS = (
    {f"seed{s}-m0w6": (s, "m0_w6") for s in SEEDS}
    | {f"seed{s}-m0w3": (s, "m0_w3") for s in SEEDS}
    | {f"seed{s}-chronos2": (s, "chronos2") for s in SEEDS}
    | {f"seed{s}-timesfm25": (s, "timesfm25") for s in SEEDS}
)
# seed 0 published artifacts use different tags
SEED0_TAGS = {
    "": (0, "m0_w6"),  # scale_up_results.json
    "verify-m0w3": (0, "m0_w3"),
    "real-chronos2": (0, "chronos2"),
    "real-timesfm25": (0, "timesfm25"),
}


def _load_summary(tag: str) -> list[dict] | None:
    name = (
        "experiments/scale_up_results.json"
        if tag == ""
        else f"experiments/scale_up_results_{tag}.json"
    )
    p = ROOT / name
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return d["summary"]  # type: ignore[no-any-return]


def _mean(rows: list[dict], model: str, key: str) -> float:
    vals = [r[key] for r in rows if r["model"] == model]
    return float(statistics.mean(vals))


def main() -> int:
    per_seed: dict[str, dict[str, dict[str, float | int]]] = {}
    missing: list[str] = []

    # seed 0 from published artifacts
    for tag, (seed, label) in SEED0_TAGS.items():
        rows = _load_summary(tag)
        if rows is None:
            missing.append(f"seed0/{label}")
            continue
        # published files use model names m0 / chronos2 / timesfm25
        model_key = "m0" if label in ("m0_w6", "m0_w3") else label
        seg_rows = [r for r in rows if r["model"] == model_key]
        per_seed.setdefault(str(seed), {})[label] = {
            "mean_crps": round(_mean(rows, model_key, "crps"), 3),
            "mean_mae": round(_mean(rows, model_key, "mae"), 3),
            "mean_coverage_80": round(_mean(rows, model_key, "coverage_80"), 3),
            "n_segments": len(seg_rows),
        }

    # seeds 1-4 from tagged runs (model name in file is the --models arg)
    model_arg = {"m0_w6": "m0", "m0_w3": "m0", "chronos2": "chronos2", "timesfm25": "timesfm25"}
    for tag, (seed, label) in TAGS.items():
        if seed == 0:
            continue
        rows = _load_summary(tag)
        if rows is None:
            missing.append(tag)
            continue
        mk = model_arg[label]
        seg_rows = [r for r in rows if r["model"] == mk]
        per_seed.setdefault(str(seed), {})[label] = {
            "mean_crps": round(_mean(rows, mk, "crps"), 3),
            "mean_mae": round(_mean(rows, mk, "mae"), 3),
            "mean_coverage_80": round(_mean(rows, mk, "coverage_80"), 3),
            "n_segments": len(seg_rows),
        }

    # paired t-tests per seed: challenger vs m0_w6 on shared segments
    from scipy import stats as scipy_stats

    paired: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        s = str(seed)
        # reload raw summaries for pairing
        tag0 = "" if seed == 0 else f"seed{seed}-m0w6"
        rows0 = _load_summary(tag0)
        mk0 = "m0"  # model key in file
        if rows0 is None:
            continue
        base = {r["segment"]: r["crps"] for r in rows0 if r["model"] == mk0}
        labels = [
            ("m0_w3", "m0"),
            ("chronos2", "chronos2"),
            ("timesfm25", "timesfm25"),
        ]
        for label, file_label in labels:
            tagc = (
                {
                    "m0_w3": "verify-m0w3",
                    "chronos2": "real-chronos2",
                    "timesfm25": "real-timesfm25",
                }[label]
                if seed == 0
                else f"seed{seed}-{label.replace('m0_w3', 'm0w3')}"
            )
            rowsc = _load_summary(tagc)
            if rowsc is None:
                continue
            chal = {r["segment"]: r["crps"] for r in rowsc if r["model"] == file_label}
            shared = sorted(set(base) & set(chal))
            if len(shared) < 3:
                continue
            diffs = [chal[g] - base[g] for g in shared]
            t, p = scipy_stats.ttest_rel([chal[g] for g in shared], [base[g] for g in shared])
            wins = sum(1 for d in diffs if d < 0)
            paired.setdefault(s, {})[label] = {
                "n_shared": len(shared),
                "challenger_wins": wins,
                "mean_diff": round(float(statistics.mean(diffs)), 3),
                "t": round(float(t), 3),
                "p": round(float(p), 4),
            }

    out = {
        "experiment": "multiseed_scaleup",
        "seeds": SEEDS,
        "per_seed": per_seed,
        "paired_vs_m0w6": paired,
        "missing": missing,
    }
    out_path = ROOT / "experiments" / "multiseed_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[collect] wrote {out_path}")
    if missing:
        print(f"[collect] MISSING ({len(missing)}): {missing}")
    # print compact table
    print(f"{'seed':>4} | {'m0_w6':>7} | {'m0_w3':>7} | {'chronos2':>8} | {'timesfm25':>9}")
    for s in SEEDS:
        r = per_seed.get(str(s), {})

        def _cell(k: str, row: dict = r) -> str:
            v = row.get(k, {}).get("mean_crps")
            return f"{v:>7.3f}" if isinstance(v, float) else "    ---"

        print(
            f"{s:>4} | {_cell('m0_w6')} | {_cell('m0_w3')} | "
            f"{_cell('chronos2'):>8} | {_cell('timesfm25'):>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
