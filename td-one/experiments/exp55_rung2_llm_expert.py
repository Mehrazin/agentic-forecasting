"""exp55: Rung 2 — LLM as an expert in the regime router.

Reads ``experiments/exp55_llm_cache.json`` (written by the LLM worker) and
runs the offline evaluation:

- Build classical per-origin records for the 59 Food CPI origins (raw
  values; same origin set and classical quantile pipeline as exp52/exp53).
- Inject the cached LLM grids as an ``"llm"`` expert
  (``td_one.llm_rungs.combine.build_records_with_llm``).
- Blocked 5-fold CV (``blocked_cv_predict``):
    proposed+llm (router over classical experts + llm)
      vs proposed (same pipeline, classical experts only)
      vs LLM alone (cache CRPS)

Writes ``experiments/exp55_results.json`` and ``experiments/exp55_report.md``.

Fails with a clear message if the cache is missing — the LLM worker fills
it later (see the message text for the exact schema).

NOTE on comparability: the "proposed" arm here uses the rung-1 registry
pipeline (``dataset="food_cpi"`` -> generic regime spec, per-origin
records, blocked CV through the new offline framework). exp53_analyze's
1.0339 used a hand-passed m4 spec and per-fold expanding-window train
records, so it is context — not the test statistic. The valid comparisons
are the paired within-experiment ones below.

Usage:
    cd td-one && .venv/bin/python experiments/exp55_rung2_llm_expert.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp52_vector_baselines as e52  # noqa: E402

from td_one.agent.proposed_method import ProposedForecaster  # noqa: E402
from td_one.agent.tools.data import naive_forecasts, series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps, load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict, build_records_with_llm  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402

DATA_CSV = ROOT / "benchmarks" / "food_cpi" / "data.csv"
CACHE_PATH = ROOT / "experiments" / "exp55_llm_cache.json"

CLASSICAL_MODELS = ["naive", "snaive", "ets", "arima"]
Z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])


def _feature_names(values: np.ndarray) -> list[str]:
    feats = series_features(values[:50], period=12)
    fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
    return sorted(fd.keys())


def _origin_quantiles(train: np.ndarray) -> dict[str, np.ndarray]:
    """Classical 9-quantile grids for one origin (exp52/exp53 pipeline)."""
    qm: dict[str, np.ndarray] = {}
    nq = naive_forecasts(train, 1, period=12)
    q = np.asarray(nq.naive)
    qm["naive"] = q.T if q.shape == (1, 9) else q
    sp = train[-12] if len(train) >= 12 else train[-1]
    resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    qm["snaive"] = (sp + Z * sigma).reshape(9, 1)
    qm["ets"] = e52.fit_ets_like(train, 1)
    qm["arima"] = e52.fit_arima_like(train, 1)
    return qm


def build_classical_records(
    values: np.ndarray, origins: list[int], feature_names: list[str]
) -> list[dict]:
    """One classical record per origin (raw values, H=1)."""
    records = []
    for o in origins:
        train = values[o - 36 : o]
        actual = float(values[o])
        if len(train) < 36:
            continue
        feats = series_features(train, period=12)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        qm = _origin_quantiles(train)
        records.append(
            {
                "origin": int(o),
                "features": feat_vec,
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {m: grid_to_crps(q[:, 0], LEVELS_9, actual) for m, q in qm.items()},
                "actual": [actual],
            }
        )
    return records


def llm_factory() -> ProposedForecaster:
    """Registry-wired forecaster plus the LLM expert (rung 2)."""
    fc = ProposedForecaster(h=1, dataset="food_cpi")
    fc.models = [*CLASSICAL_MODELS, "llm"]
    return fc


def classical_factory() -> ProposedForecaster:
    """Registry-wired forecaster, classical experts only."""
    fc = ProposedForecaster(h=1, dataset="food_cpi")
    fc.models = list(CLASSICAL_MODELS)
    return fc


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


def main() -> int:
    t0 = time.time()
    if not CACHE_PATH.exists():
        sys.stderr.write(
            f"""[exp55] FATAL: LLM grid cache not found: {CACHE_PATH}
Rung 2 needs the LLM worker's cache before the offline evaluation can run.
The worker should write it with td_one.llm_rungs.cache.save_grid_cache:

    from td_one.llm_rungs.cache import save_grid_cache
    save_grid_cache(
        "experiments/exp55_llm_cache.json",
        variant="<prompt variant tag, e.g. 'groq120b-generic'>",
        dataset="food_cpi",
        model="<model id, e.g. 'openai/gpt-oss-120b (via Groq)'>",
        levels=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        per_origin=[
            {{"origin": int, "as_of": "YYYY-MM-DD", "actual": float,
              "grid": [11 floats], "crps": float, "ok": bool}},
            ...
        ],
    )

Schema: {{"variant", "dataset", "model", "levels", "per_origin"}} — see
td_one/llm_rungs/cache.py (save_grid_cache validates it and raises on
violations).

NOTE: experiments/exp53_results.json does NOT contain the 11-level grids
(only per-origin CRPS), so it cannot be repackaged into this cache — the
worker must (re)generate the grids for the 59 Food CPI origins (the same
origin set as exp53: range(n-60, n-1) into benchmarks/food_cpi/data.csv).
"""
        )
        return 2

    print(f"[exp55] loading cache: {CACHE_PATH}", flush=True)
    cache = load_grid_cache(CACHE_PATH)
    print(
        f"[exp55] cache: variant={cache['variant']} model={cache['model']} "
        f"levels={len(cache['levels'])} origins={len(cache['per_origin'])}",
        flush=True,
    )

    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    values = df["value"].to_numpy(dtype=float)
    n = len(values)
    origins = list(range(n - 60, n - 1))  # same 59 origins as exp52/exp53
    print(f"[exp55] data: {n} obs; {len(origins)} origins", flush=True)

    feature_names = _feature_names(values)
    print(f"[exp55] building classical records ({len(feature_names)} features)...", flush=True)
    classical = build_classical_records(values, origins, feature_names)
    print(f"[exp55] classical records: {len(classical)}", flush=True)

    records_llm = build_records_with_llm(classical, cache, feature_names)
    if not records_llm:
        sys.stderr.write(
            "[exp55] FATAL: no classical origin matched a usable (ok=True) LLM "
            f"cache entry ({len(classical)} classical records, "
            f"{len(cache['per_origin'])} cache entries).\n"
        )
        return 2
    ok_origins = [r["origin"] for r in records_llm]
    classical_aligned = [r for r in classical if r["origin"] in set(ok_origins)]
    assert [r["origin"] for r in classical_aligned] == ok_origins
    print(
        f"[exp55] origins with LLM grids: {len(records_llm)}/{len(classical)}",
        flush=True,
    )

    print("[exp55] blocked CV: proposed+llm ...", flush=True)
    res_llm = blocked_cv_predict(
        records_llm, feature_names, llm_factory, n_folds=5, baseline="best_single"
    )
    print("[exp55] blocked CV: proposed (classical only) ...", flush=True)
    res_classical = blocked_cv_predict(
        classical_aligned, feature_names, classical_factory, n_folds=5, baseline="best_single"
    )

    cache_by_origin = {int(e["origin"]): e for e in cache["per_origin"]}
    llm_alone = [float(cache_by_origin[o]["crps"]) for o in ok_origins]
    # cross-check: stored cache CRPS vs recomputed from the injected grids
    injected = [r["crps"]["llm"] for r in records_llm]
    cache_recompute_diff = float(np.max(np.abs(np.array(llm_alone) - np.array(injected))))

    cmp_llm_vs_classical = _paired(res_llm["crps"], res_classical["crps"])
    cmp_llm_vs_llmalone = _paired(res_llm["crps"], llm_alone)
    cmp_classical_vs_llmalone = _paired(res_classical["crps"], llm_alone)

    # mechanism: which experts does the router pick when llm is available?
    fc_full = llm_factory()
    fc_full.fit(records_llm, feature_names)

    c1, c2 = cmp_llm_vs_classical, cmp_llm_vs_llmalone
    print(
        f"[exp55] proposed+llm vs proposed: diff={c1['mean_diff']:.4f} p={c1['p']:.4g}",
        flush=True,
    )
    print(
        f"[exp55] proposed+llm vs LLM alone: diff={c2['mean_diff']:.4f} p={c2['p']:.4g}",
        flush=True,
    )

    out = {
        "date": datetime.now(UTC).isoformat(),
        "elapsed_s": time.time() - t0,
        "cache": {
            "path": str(CACHE_PATH),
            "variant": cache["variant"],
            "dataset": cache["dataset"],
            "model": cache["model"],
            "n_origins": len(cache["per_origin"]),
            "n_ok": sum(1 for e in cache["per_origin"] if e.get("ok")),
        },
        "n_origins_evaluated": len(records_llm),
        "origins": ok_origins,
        "classical_models": CLASSICAL_MODELS,
        "mean_crps": {
            "proposed_llm": float(np.mean(res_llm["crps"])),
            "proposed": float(np.mean(res_classical["crps"])),
            "llm_alone": float(np.mean(llm_alone)),
        },
        "per_origin_crps": {
            "proposed_llm": res_llm["crps"],
            "proposed": res_classical["crps"],
            "llm_alone": llm_alone,
        },
        "paired": {
            "proposed_llm_vs_proposed": cmp_llm_vs_classical,
            "proposed_llm_vs_llm_alone": cmp_llm_vs_llmalone,
            "proposed_vs_llm_alone": cmp_classical_vs_llmalone,
        },
        "cache_crps_vs_recomputed_max_diff": cache_recompute_diff,
        "router_expert_per_regime_full_fit": dict(fc_full._expert_per_regime),
        "notes": [
            "Classical records: raw Food CPI, 36m train window, "
            "series_features(period=12); experts naive/snaive/ets/arima (exp52/53 pipeline).",
            "Regime spec is the rung-1 registry entry for food_cpi "
            "(generic calm/stressed); exp53_analyze's 1.0339 used a hand-passed "
            "m4 spec, so it is context, not the test statistic.",
            "blocked_cv_predict scores with a zeros(30) train proxy "
            "(exp47 convention), not the real train window.",
        ],
    }
    with open(ROOT / "experiments" / "exp55_results.json", "w") as f:
        json.dump(out, f, indent=2)

    def row(label: str, c: dict) -> str:
        return (
            f"| {label} | {c['mean_a']:.4f} vs {c['mean_b']:.4f} | "
            f"{c['mean_diff']:+.4f} | {c['t']:.2f} | {c['p']:.4g} | {c['win_rate']:.2f} |\n"
        )

    m = out["mean_crps"]
    headline_rows = "".join(
        [
            row("proposed+llm vs proposed", cmp_llm_vs_classical),
            row("proposed+llm vs LLM alone", cmp_llm_vs_llmalone),
            row("proposed vs LLM alone", cmp_classical_vs_llmalone),
        ]
    )
    n_ok, n_all = out["cache"]["n_ok"], out["cache"]["n_origins"]
    headline_means = (
        f"Mean CRPS: proposed+llm={m['proposed_llm']:.4f}, "
        f"proposed={m['proposed']:.4f}, LLM alone={m['llm_alone']:.4f}."
    )
    report = f"""# exp55 — Rung 2: LLM as an expert in the regime router

**Date:** {out["date"]} | **Elapsed:** {out["elapsed_s"]:.0f}s
**Cache:** variant=`{cache["variant"]}`, model=`{cache["model"]}`,
{n_ok}/{n_all} origins ok; **evaluated {len(records_llm)} origins**.

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
{headline_rows}{headline_means}

## Mechanism

With the LLM available as an expert, the router's per-regime experts on the
full fit are: {", ".join(f"{r}->{e}" for r, e in fc_full._expert_per_regime.items())}.
(The hypothesis under test: the router learns *when* the LLM beats classical
experts — e.g. on smooth macro stretches — and routes accordingly.)

## Setup notes

- Classical records: raw Food CPI values, 36-month train window per origin,
  `series_features(period=12)`; experts naive/snaive/ets/arima — the
  exp52/exp53 classical pipeline.
- Regime spec comes from the rung-1 registry (`dataset="food_cpi"` ->
  generic calm/stressed). exp53_analyze's proposed number (1.0339) used a
  hand-passed m4 spec with per-fold expanding-window train records, so it is
  context here, not the test statistic; the paired comparisons above are.
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention) rather than the real train window.
- Cache CRPS vs recomputed-from-grid max diff: {cache_recompute_diff:.3g}
  (validates the worker's stored CRPS against the evaluation scorer).

## PIT note

The cache is per-origin from the LLM worker; prompts must be pure functions
of (dataset, origin date, history <= origin) per `docs/LLM_INTEGRATION_LADDER.md`
rung 2. The classical records use only train windows <= origin. No
post-origin data enters the evaluation.
"""
    with open(ROOT / "experiments" / "exp55_report.md", "w") as f:
        f.write(report)
    print("[exp55] wrote exp55_results.json + exp55_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
