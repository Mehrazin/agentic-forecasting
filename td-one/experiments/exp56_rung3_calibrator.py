"""exp56: Rung 3 — LLM as calibrator of classical forecasts.

Loads ``experiments/exp56_calibrator_cache.json`` (variant ``llm-calibrator``:
the LLM adjusts classical 9-quantile grids rather than forecasting alone) and
runs the offline evaluation:

- Build classical per-origin records for the 59 Food CPI origins (the shared
  ladder pipeline, ``experiments/llm_food_cpi.py``).
- Inject the calibrated grids as the ``"llm"`` expert.
- Blocked 5-fold CV (``td_one.llm_rungs.combine.blocked_cv_predict``):
    proposed+calibrated (router over classical experts + calibrated grids)
      vs proposed (classical experts only)
      vs calibrated-classical alone (cache CRPS)

Writes ``experiments/exp56_results.json`` and ``experiments/exp56_report.md``.

Usage:
    cd td-one && .venv/bin/python experiments/exp56_rung3_calibrator.py
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
sys.path.insert(0, str(ROOT / "experiments"))

import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method import ProposedForecaster  # noqa: E402
from td_one.llm_rungs.cache import load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict, build_records_with_llm  # noqa: E402

CACHE_PATH = ROOT / "experiments" / "exp56_calibrator_cache.json"


def hybrid_factory() -> ProposedForecaster:
    """Registry-wired forecaster plus the calibrated expert (rung 3)."""
    fc = ProposedForecaster(h=1, dataset="food_cpi")
    fc.models = [*lfc.CLASSICAL_MODELS, "llm"]
    return fc


def classical_factory() -> ProposedForecaster:
    """Registry-wired forecaster, classical experts only."""
    fc = ProposedForecaster(h=1, dataset="food_cpi")
    fc.models = list(lfc.CLASSICAL_MODELS)
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
    cache = load_grid_cache(CACHE_PATH)
    values = lfc.load_food_cpi_values()
    origins = lfc.origins_59(values)
    feature_names = lfc.feature_names_for(values)
    print(f"[exp56] data: {len(values)} obs; {len(origins)} origins", flush=True)

    print("[exp56] building classical records...", flush=True)
    classical = lfc.build_classical_records(values, origins, feature_names)
    print(f"[exp56] classical records: {len(classical)}", flush=True)

    records_cal = build_records_with_llm(classical, cache, feature_names)
    if not records_cal:
        sys.stderr.write(
            "[exp56] FATAL: no classical origin matched a usable calibrator cache entry.\n"
        )
        return 2
    classical_aligned = lfc.aligned_classical(classical, records_cal)
    print(f"[exp56] origins with calibrated grids: {len(records_cal)}/{len(classical)}", flush=True)

    print("[exp56] blocked CV: proposed+calibrated ...", flush=True)
    res_cal = blocked_cv_predict(
        records_cal, feature_names, hybrid_factory, n_folds=5, baseline="best_single"
    )
    print("[exp56] blocked CV: proposed (classical only) ...", flush=True)
    res_classical = blocked_cv_predict(
        classical_aligned, feature_names, classical_factory, n_folds=5, baseline="best_single"
    )

    ok_origins = [r["origin"] for r in records_cal]
    cache_by_origin = {int(e["origin"]): e for e in cache["per_origin"]}
    cal_alone = [float(cache_by_origin[o]["crps"]) for o in ok_origins]
    # cross-check: stored cache CRPS vs recomputed from the injected grids
    injected = [r["crps"]["llm"] for r in records_cal]
    cache_recompute_diff = float(np.max(np.abs(np.array(cal_alone) - np.array(injected))))
    # classical experts alone (mean CRPS over all origins, for context)
    classical_means = {
        m: float(np.mean([r["crps"][m] for r in classical_aligned])) for m in lfc.CLASSICAL_MODELS
    }

    cmp_cal_vs_classical = _paired(res_cal["crps"], res_classical["crps"])
    cmp_cal_vs_alone = _paired(res_cal["crps"], cal_alone)
    cmp_classical_vs_alone = _paired(res_classical["crps"], cal_alone)

    fc_full = hybrid_factory()
    fc_full.fit(records_cal, feature_names)

    c1, c2 = cmp_cal_vs_classical, cmp_cal_vs_alone
    print(
        f"[exp56] proposed+cal vs proposed: diff={c1['mean_diff']:.4f} p={c1['p']:.4g}", flush=True
    )
    print(
        f"[exp56] proposed+cal vs cal alone: diff={c2['mean_diff']:.4f} p={c2['p']:.4g}", flush=True
    )

    m = {
        "proposed_calibrated": float(np.mean(res_cal["crps"])),
        "proposed": float(np.mean(res_classical["crps"])),
        "calibrated_alone": float(np.mean(cal_alone)),
    }
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
        "n_origins_evaluated": len(records_cal),
        "origins": ok_origins,
        "classical_models": lfc.CLASSICAL_MODELS,
        "mean_crps": m,
        "classical_expert_means": classical_means,
        "per_origin_crps": {
            "proposed_calibrated": res_cal["crps"],
            "proposed": res_classical["crps"],
            "calibrated_alone": cal_alone,
        },
        "paired": {
            "proposed_calibrated_vs_proposed": cmp_cal_vs_classical,
            "proposed_calibrated_vs_calibrated_alone": cmp_cal_vs_alone,
            "proposed_vs_calibrated_alone": cmp_classical_vs_alone,
        },
        "cache_crps_vs_recomputed_max_diff": cache_recompute_diff,
        "router_expert_per_regime_full_fit": dict(fc_full._expert_per_regime),
        "notes": [
            "Classical records: the shared ladder pipeline (exp52/53): raw "
            "Food CPI, 36m train window, series_features(period=12); experts "
            "naive/snaive/ets/arima.",
            "The 'llm' expert holds the LLM's *calibrated* classical grids "
            "(rung-3 prompt: adjust the classical 9-quantile grid).",
            "blocked_cv_predict scores with a zeros(30) train proxy (exp47 convention).",
        ],
    }
    with open(ROOT / "experiments" / "exp56_results.json", "w") as f:
        json.dump(out, f, indent=2)

    def row(label: str, c: dict) -> str:
        return (
            f"| {label} | {c['mean_a']:.4f} vs {c['mean_b']:.4f} | "
            f"{c['mean_diff']:+.4f} | {c['t']:.2f} | {c['p']:.4g} | {c['win_rate']:.2f} |\n"
        )

    headline_rows = "".join(
        [
            row("proposed+calibrated vs proposed", cmp_cal_vs_classical),
            row("proposed+calibrated vs calibrated alone", cmp_cal_vs_alone),
            row("proposed vs calibrated alone", cmp_classical_vs_alone),
        ]
    )
    classical_str = ", ".join(f"{k}={v:.4f}" for k, v in classical_means.items())
    mech = ", ".join(f"{r}->{e}" for r, e in fc_full._expert_per_regime.items())
    n_ok, n_all = out["cache"]["n_ok"], out["cache"]["n_origins"]
    means_line = (
        f"Mean CRPS: proposed+calibrated={m['proposed_calibrated']:.4f}, "
        f"proposed={m['proposed']:.4f}, calibrated alone={m['calibrated_alone']:.4f}."
    )
    report = f"""# exp56 — Rung 3: LLM as calibrator of classical forecasts

**Date:** {out["date"]} | **Elapsed:** {out["elapsed_s"]:.0f}s
**Cache:** variant=`llm-calibrator`, model=`{cache["model"]}`,
{n_ok}/{n_all} origins ok; **evaluated {len(records_cal)} origins**.

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
{headline_rows}{means_line}

Classical experts alone: {classical_str}.

## Mechanism

With the calibrated grids available as an expert, the router's per-regime
experts on the full fit are: {mech}.

For reference: LLM-alone on the same origins is 0.5158 (exp55 cache,
regenerated generic prompt) and 0.5688 (exp53, Vector's run) — the
calibrated grids (1.1391) are far worse than the LLM forecasting directly.
The LLM's calibration skill needs direct quantile elicitation, not
post-hoc adjustment of a classical grid.

## Setup notes

- Classical records: the shared ladder pipeline (exp52/exp53): raw Food CPI,
  36-month train window per origin, `series_features(period=12)`; experts
  naive/snaive/ets/arima.
- The "llm" expert here holds the LLM's *calibrated* classical grids
  (rung-3 prompt asked the LLM to adjust the classical 9-quantile grid, not
  to forecast alone).
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention) rather than the real train window.
- Cache CRPS vs recomputed-from-grid max diff: {cache_recompute_diff:.3g}
  (validates the worker's stored CRPS against the evaluation scorer).

## PIT note

The cache is per-origin from the LLM worker; prompts must be pure functions
of (dataset, origin date, history <= origin) per `docs/LLM_INTEGRATION_LADDER.md`
rung 3. The classical records use only train windows <= origin. No
post-origin data enters the evaluation.
"""
    with open(ROOT / "experiments" / "exp56_report.md", "w") as f:
        f.write(report)
    print("[exp56] wrote exp56_results.json + exp56_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
