"""exp57: Rung 4 — per-dataset prompts (PIT-safe).

Loads the two rung-4 caches (``exp57_promptV1_cache.json``,
``exp57_promptV2_cache.json``) plus the rung-2 generic cache
(``exp55_llm_cache.json``) and evaluates:

1. Each prompt variant as a standalone expert (mean CRPS over the 59 Food
   CPI origins, recomputed from the 9-level subset grids) with paired
   t-tests: V1 vs generic, V2 vs generic, V1 vs V2. The key question: does
   more context (V2 = V1 + macro lookup) help or distract?
2. A mini-router over prompt variants: blocked 5-fold CV; within each fold
   pick the variant with the best train-fold mean CRPS and score the test
   origins with that variant. Paired comparison vs the best single variant.

Writes ``experiments/exp57_results.json`` and ``experiments/exp57_report.md``.

Usage:
    cd td-one && .venv/bin/python experiments/exp57_rung4_prompts.py
"""

from __future__ import annotations

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

from td_one.llm_rungs.cache import grid_to_crps, load_grid_cache  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402

EXP_DIR = ROOT / "experiments"

VARIANTS = {
    "generic": EXP_DIR / "exp55_llm_cache.json",
    "promptV1": EXP_DIR / "exp57_promptV1_cache.json",
    "promptV2": EXP_DIR / "exp57_promptV2_cache.json",
}

N_FOLDS = 5


def _subset_indices(cache_levels: list[float]) -> list[int]:
    idx = []
    for lv in LEVELS_9:
        hits = [i for i, c in enumerate(cache_levels) if abs(float(c) - lv) < 1e-9]
        if not hits:
            raise ValueError(f"level {lv} not present in cache levels {cache_levels}")
        idx.append(hits[0])
    return idx


def variant_crps(cache: dict) -> dict[int, float]:
    """Per-origin CRPS from the 9-level subset of each cached grid."""
    idx = _subset_indices(cache["levels"])
    out = {}
    for e in cache["per_origin"]:
        if not e.get("ok"):
            continue
        grid9 = np.asarray(e["grid"], dtype=float)[idx]
        out[int(e["origin"])] = grid_to_crps(grid9, LEVELS_9, float(e["actual"]))
    return out


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


def _time_block_folds(origins: list[int], n_folds: int) -> list[tuple[list[int], list[int]]]:
    """Contiguous time-block folds (mirrors combine._time_block_folds)."""
    order = sorted(origins)
    folds = []
    chunk = len(order) // n_folds
    for f in range(n_folds):
        te = order[f * chunk : (f + 1) * chunk if f < n_folds - 1 else len(order)]
        te_set = set(te)
        tr = [o for o in order if o not in te_set]
        folds.append((tr, te))
    return folds


def main() -> int:
    t0 = time.time()
    caches = {name: load_grid_cache(path) for name, path in VARIANTS.items()}
    per_variant = {name: variant_crps(c) for name, c in caches.items()}

    common = sorted(set.intersection(*(set(v) for v in per_variant.values())))
    print(
        f"[exp57] common origins: {len(common)} "
        f"(generic={len(per_variant['generic'])}, "
        f"V1={len(per_variant['promptV1'])}, V2={len(per_variant['promptV2'])})",
        flush=True,
    )
    if not common:
        sys.stderr.write("[exp57] FATAL: no common origins across the three caches.\n")
        return 2

    vec = {name: [per_variant[name][o] for o in common] for name in VARIANTS}
    means = {name: float(np.mean(vec[name])) for name in VARIANTS}
    stored_means = {
        name: float(np.mean([e["crps"] for e in caches[name]["per_origin"] if e.get("ok")]))
        for name in VARIANTS
    }

    pairs = {
        "promptV1_vs_generic": _paired(vec["promptV1"], vec["generic"]),
        "promptV2_vs_generic": _paired(vec["promptV2"], vec["generic"]),
        "promptV1_vs_promptV2": _paired(vec["promptV1"], vec["promptV2"]),
    }
    for k, v in pairs.items():
        print(
            f"[exp57] {k}: diff={v['mean_diff']:+.4f} p={v['p']:.4g} win={v['win_rate']:.2f}",
            flush=True,
        )

    # Mini-router: per fold, pick the variant with best train-fold mean CRPS.
    folds = _time_block_folds(common, N_FOLDS)
    router_scores: list[float] = []
    fold_picks: list[dict] = []
    for f, (tr, te) in enumerate(folds):
        tr_idx = [common.index(o) for o in tr]
        te_idx = [common.index(o) for o in te]
        train_means = {name: float(np.mean([vec[name][i] for i in tr_idx])) for name in VARIANTS}
        pick = min(train_means, key=train_means.get)  # type: ignore[arg-type]
        fold_picks.append({"fold": f, "pick": pick, "train_means": train_means})
        router_scores.extend(vec[pick][i] for i in te_idx)
        print(
            f"[exp57] fold {f}: train means="
            + ", ".join(f"{k}={v:.4f}" for k, v in train_means.items())
            + f" -> pick {pick}",
            flush=True,
        )

    best_variant = min(means, key=means.get)  # type: ignore[arg-type]
    vs_best = _paired(router_scores, vec[best_variant])
    # degenerate case: the router picks the same variant every fold, so the
    # two vectors are identical and the paired test is undefined
    identical_to_best = all(a == b for a, b in zip(router_scores, vec[best_variant], strict=True))
    if identical_to_best:
        vs_best = {
            **vs_best,
            "t": float("nan"),
            "p": float("nan"),
            "note": "identical to best variant (picked every fold)",
        }
    mini_router = {
        "mean": float(np.mean(router_scores)),
        "vs_best_single_variant": vs_best,
        "vs_generic": _paired(router_scores, vec["generic"]),
        "fold_picks": fold_picks,
        "best_single_variant": best_variant,
    }
    print(
        f"[exp57] mini-router vs {best_variant}: "
        f"diff={mini_router['vs_best_single_variant']['mean_diff']:+.4f} "
        f"p={mini_router['vs_best_single_variant']['p']:.4g}",
        flush=True,
    )

    out = {
        "date": datetime.now(UTC).isoformat(),
        "elapsed_s": time.time() - t0,
        "n_origins": len(common),
        "origins": common,
        "variants": {
            name: {
                "path": str(VARIANTS[name]),
                "variant_tag": caches[name]["variant"],
                "model": caches[name]["model"],
                "n_ok": sum(1 for e in caches[name]["per_origin"] if e.get("ok")),
                "mean_crps_recomputed": means[name],
                "mean_crps_stored": stored_means[name],
            }
            for name in VARIANTS
        },
        "per_origin_crps": vec,
        "paired": pairs,
        "mini_router": mini_router,
        "notes": [
            "Standalone means are recomputed from the 9-level (0.1..0.9) "
            "subset of each cached 11-level grid via grid_to_crps; "
            "'stored' means come from the worker's per-origin CRPS.",
            "The mini-router selects one prompt variant per fold by "
            "train-fold mean CRPS (no test-fold information).",
            "Generic-prompt cache is exp55_llm_cache.json (rung 2).",
        ],
    }
    with open(EXP_DIR / "exp57_results.json", "w") as f:
        json.dump(out, f, indent=2)

    def row(label: str, c: dict) -> str:
        return (
            f"| {label} | {c['mean_a']:.4f} vs {c['mean_b']:.4f} | "
            f"{c['mean_diff']:+.4f} | {c['t']:.2f} | {c['p']:.4g} | {c['win_rate']:.2f} |\n"
        )

    headline_rows = "".join(
        [
            row("promptV1 vs generic", pairs["promptV1_vs_generic"]),
            row("promptV2 vs generic", pairs["promptV2_vs_generic"]),
            row("promptV1 vs promptV2", pairs["promptV1_vs_promptV2"]),
        ]
    )
    mr = mini_router["vs_best_single_variant"]
    mr_gen = mini_router["vs_generic"]
    fold_pick_str = ", ".join(f"fold {fp['fold']}->{fp['pick']}" for fp in fold_picks)
    means_line = (
        f"Standalone mean CRPS (recomputed from 9-level grids): "
        f"generic={means['generic']:.4f}, promptV1={means['promptV1']:.4f}, "
        f"promptV2={means['promptV2']:.4f}."
    )
    stored_line = (
        f"(Worker-stored means: generic={stored_means['generic']:.4f}, "
        f"promptV1={stored_means['promptV1']:.4f}, "
        f"promptV2={stored_means['promptV2']:.4f}.)"
    )
    mr_line = f"- vs best single variant ({best_variant}): " + (
        "identical — the router picked it in every fold, selection adds nothing"
        if "note" in mr
        else f"diff={mr['mean_diff']:+.4f}, t={mr['t']:.2f}, "
        f"p={mr['p']:.4g}, win rate={mr['win_rate']:.2f}"
    )
    mr_gen_line = f"- vs generic: diff={mr_gen['mean_diff']:+.4f}, p={mr_gen['p']:.4g}"
    report = f"""# exp57 — Rung 4: per-dataset prompts (PIT-safe)

**Date:** {out["date"]} | **Elapsed:** {out["elapsed_s"]:.0f}s
Evaluated on **{len(common)} common origins** (generic / promptV1 / promptV2 caches).

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
{headline_rows}{means_line}
{stored_line}

## Mini-router over prompt variants

Blocked 5-fold CV: each fold picks the variant with the best *train-fold*
mean CRPS and scores the test fold with it.

- Fold picks: {fold_pick_str}
- Mini-router mean CRPS: {mini_router["mean"]:.4f}
{mr_line}
{mr_gen_line}

## Setup notes

- Per-origin CRPS is recomputed from the 9-level (0.1..0.9) subset of each
  cached 11-level grid with `grid_to_crps`, so all three variants are scored
  identically.
- Prompt templates are audited for PIT safety in
  `docs/PROMPT_LEAKAGE_AUDIT.md` (every template field traced to a pre-origin
  source; macro table versioned with as-of dates). No new data was used here.
"""
    with open(EXP_DIR / "exp57_report.md", "w") as f:
        f.write(report)
    print("[exp57] wrote exp57_results.json + exp57_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
