"""exp65: SMM with LLM expert (Proposed Method v2 on Freddie Mac SMM).

The exp64 report noted: "SMM with LLM not run. The gate says SMM
(macro-like) should benefit, but no LLM caches exist for SMM."

This script fills that gap:
  Phase 1 (LLM worker): V1-style prompts (the ladder winner) adapted for
    SMM, run via ``run_variant`` with ``backend="vector"`` (Gemini Flash
    via Vector proxy). Resume-safe via ``exp65_smm_llm_cache.json``.
  Phase 2 (evaluation): classical per-origin records (naive/snaive/ets/
    arima, expanding window) + injected LLM grids -> blocked 5-fold CV:
    v2 (router + LLM expert, dataset="smm") vs v1 (classical only)
    vs LLM alone.

SMM values are tiny (~0.001-0.01), so prompts present the series in
percent (x100, precision 3). The worker's ``actuals`` are in percent;
grids are divided by 100 before injection into raw-unit records.

Honest reporting: if the LLM helps, great; if not, that's a result too.

Usage:
    cd td-one && .venv/bin/python experiments/exp65_smm_llm.py [--skip-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
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

from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import naive_forecasts, series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps, load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.llm_rungs.prompts import (  # noqa: E402
    LEVELS_9,
    serialize_history,
    system_prompt_forecaster,
    user_prompt_v1,
    words_stats_block,
)
from td_one.llm_rungs.runner import run_variant  # noqa: E402

SMM_CSV = ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv"
CACHE_JSON = ROOT / "experiments" / "exp65_smm_llm_cache.json"
RESULTS_JSON = ROOT / "experiments" / "exp65_results.json"
REPORT_MD = ROOT / "experiments" / "exp65_report.md"

CLASSICAL_MODELS = ["naive", "snaive", "ets", "arima"]
Z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])

TASK_LINE = "Freddie Mac SMM 1-month-ahead forecast."
SERIES_DESC = (
    "Freddie Mac single-month mortality (SMM), monthly mortgage prepayment "
    "rate (source: Freddie Mac loan-level data, 2021-2026)"
)
UNITS = "percent"
FREQUENCY = "MS"
MODEL_LABEL = "gemini-3.1-flash-lite-preview (Vector proxy)"


def load_smm() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Raw SMM values and dates (63 months)."""
    df = pd.read_csv(SMM_CSV)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    assert len(df) == 63, f"expected 63 months, got {len(df)}"
    return df["smm"].to_numpy(dtype=float), pd.DatetimeIndex(df["date"])


def feature_names_for(values: np.ndarray) -> list[str]:
    """Sorted series_features keys (matches the ladder feature order)."""
    feats = series_features(values[:30], period=12)
    fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
    return sorted(fd.keys())


def origin_quantiles(train: np.ndarray) -> dict[str, np.ndarray]:
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
    """One classical record per origin (raw SMM units, expanding window)."""
    records = []
    for o in origins:
        train = values[:o]
        actual = float(values[o])
        feats = series_features(train, period=12)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        qm = origin_quantiles(train)
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


def _shift_ym(yyyymm: str, months: int) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[5:7])
    m += months
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return f"{y:04d}-{m:02d}"


def run_llm_worker(values: np.ndarray, dates: pd.DatetimeIndex, origins: list[int]) -> None:
    """Phase 1: V1 prompts for SMM, Gemini Flash via Vector proxy."""
    as_of_dates = [str(dates[o - 1].date()) for o in origins]
    # actuals in percent (matches the prompt units)
    actuals_pct = [float(values[o]) * 100.0 for o in origins]

    def month_idx(train_dates: list[str], yyyymm: str) -> tuple[int, str]:
        months = [d[:7] for d in train_dates]
        if yyyymm in months:
            return months.index(yyyymm), yyyymm
        earlier = [m for m in months if m < yyyymm]
        if not earlier:
            # Not enough history: fall back to the earliest observation.
            return 0, months[0]
        return months.index(earlier[-1]), earlier[-1]

    def build_prompts(orig: int) -> tuple[str, str]:
        train = values[:orig]
        train_pct = train * 100.0
        train_dates = [str(d.date()) for d in dates[:orig]]
        as_of = str(dates[orig - 1].date())
        fdate = str(dates[orig].date())
        history_str = serialize_history(train_pct, train_dates, precision=3)
        last_v = float(train_pct[-1])
        i12, ref12 = month_idx(train_dates, _shift_ym(as_of[:7], -12))
        i3, ref3 = month_idx(train_dates, _shift_ym(as_of[:7], -3))
        last12 = train_pct[i12 + 1 :]
        stats = words_stats_block(
            last_value=last_v,
            last_date=as_of,
            change_12m_pct=float(last_v / train_pct[i12] - 1.0) * 100.0,
            ref_12m=ref12,
            change_3m_pct=float(last_v / train_pct[i3] - 1.0) * 100.0,
            ref_3m=ref3,
            max_12m=float(last12.max()),
            min_12m=float(last12.min()),
            as_of=as_of,
        )
        user = user_prompt_v1(
            task_line=TASK_LINE,
            series_desc=SERIES_DESC,
            units=UNITS,
            frequency=FREQUENCY,
            history_str=history_str,
            stats_block=stats,
            forecast_start=fdate,
            forecast_end=fdate,
            n_steps=1,
        )
        return system_prompt_forecaster(), user

    run_variant(
        variant="llm-smm",
        dataset="smm",
        model_label=MODEL_LABEL,
        origins=origins,
        as_of_dates=as_of_dates,
        actuals=actuals_pct,
        build_prompts=build_prompts,
        cache_path=CACHE_JSON,
        backend="vector",
        vector_model="flash-lite",
    )


def inject_llm_percent(records: list[dict], cache: dict, feature_names: list[str]) -> list[dict]:
    """Inject percent-unit LLM grids as an expert (converted to raw units)."""
    from td_one.llm_rungs.combine import _subset_indices  # noqa: PLC2701

    idx = _subset_indices(cache["levels"])
    by_origin = {int(e["origin"]): e for e in cache["per_origin"] if e.get("ok", False)}
    out = []
    for r in records:
        entry = by_origin.get(int(r["origin"]))
        if entry is None:
            continue
        # Cache grids are in percent; convert to raw SMM units.
        grid9_pct = np.asarray(entry["grid"], dtype=float)[idx]
        grid9 = grid9_pct / 100.0
        actual = float(np.ravel(np.asarray(r["actual"], dtype=float))[0])
        crps_llm = grid_to_crps(grid9, LEVELS_9, actual)
        nr = dict(r)
        nr["quantiles"] = {**r["quantiles"], "llm": grid9.reshape(9, 1).tolist()}
        nr["crps"] = {**r["crps"], "llm": crps_llm}
        out.append(nr)
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="skip Phase 1 (use existing cache)")
    args = ap.parse_args()

    values, dates = load_smm()
    origins = list(range(12, 62))  # 50 origins, matches exp18b
    print(f"[exp65] SMM: {len(values)} months, {len(origins)} origins", flush=True)

    # Phase 1: LLM worker (resume-safe; skips if cache complete)
    if not args.skip_llm:
        run_llm_worker(values, dates, origins)
    cache = load_grid_cache(CACHE_JSON)
    n_ok = sum(1 for e in cache["per_origin"] if e.get("ok"))
    print(f"[exp65] LLM cache: {n_ok}/{len(origins)} ok", flush=True)

    # Phase 2: classical records + evaluation
    feature_names = feature_names_for(values)
    records = build_classical_records(values, origins, feature_names)
    records_llm = inject_llm_percent(records, cache, feature_names)
    print(f"[exp65] records with LLM: {len(records_llm)}/{len(records)}", flush=True)

    # v2 (router + LLM expert) vs v1 (classical only), blocked 5-fold CV.
    # Note: override registry models — our records use naive/snaive/ets/arima,
    # not the registry's naive/m0/ets/arima.
    def make_v2() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(dataset="smm", use_llm_expert=True)
        fc.models = ["naive", "snaive", "ets", "arima", "llm"]
        return fc

    def make_v1() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(dataset="smm", use_llm_expert=False)
        fc.models = ["naive", "snaive", "ets", "arima"]
        return fc

    v2_cv = blocked_cv_predict(
        records_llm,
        feature_names,
        make_v2,
        n_folds=5,
        baseline=make_v1,
        fold_mode="time",
    )
    # LLM alone (cache CRPS on the injected origins)
    llm_crps = [r["crps"]["llm"] for r in records_llm]
    v2_crps = v2_cv["crps"]
    v1_crps = v2_cv["baseline_crps"]

    results = {
        "dataset": "smm",
        "n_origins": len(records_llm),
        "v2_mean_crps": float(np.mean(v2_crps)),
        "v1_mean_crps": float(np.mean(v1_crps)),
        "llm_mean_crps": float(np.mean(llm_crps)),
        "v2_vs_v1": _paired(v2_crps, v1_crps),
        "v2_vs_llm": _paired(v2_crps, llm_crps),
        "classical_means": {
            m: float(np.mean([r["crps"][m] for r in records_llm])) for m in CLASSICAL_MODELS
        },
        "model_label": MODEL_LABEL,
        "date": datetime.now(UTC).isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    v2m, v1m, llmm = (results["v2_mean_crps"], results["v1_mean_crps"], results["llm_mean_crps"])
    s_v2v1, s_v2llm = results["v2_vs_v1"], results["v2_vs_llm"]
    verdict = (
        "LLM HELPS"
        if v2m < v1m and s_v2v1["p"] < 0.05
        else "LLM NEUTRAL"
        if abs(v2m - v1m) / v1m < 0.02
        else "LLM HURTS"
    )
    REPORT_MD.write_text(
        f"""# exp65: SMM with LLM Expert

**Date:** {datetime.now(UTC).date()}
**Method:** Proposed v2 (router + gated LLM expert, Gemini Flash via Vector proxy)
**Dataset:** Freddie Mac SMM, 63 months, {len(records_llm)} origins

## Results (CRPS, lower is better)

| Method | Mean CRPS |
|---|---|
| v2 (router + LLM) | {v2m:.6f} |
| v1 (classical only) | {v1m:.6f} |
| LLM alone | {llmm:.6f} |

**Verdict: {verdict}**

- v2 vs v1: {s_v2v1["mean_diff"]:+.6f} (p={s_v2v1["p"]:.4f}, win {s_v2v1["win_rate"]:.2f})
- v2 vs LLM: {s_v2llm["mean_diff"]:+.6f} (p={s_v2llm["p"]:.4f}, win {s_v2llm["win_rate"]:.2f})

### Classical experts (for context)

| Expert | Mean CRPS |
|---|---|
"""
        + "\n".join(f"| {m} | {results['classical_means'][m]:.6f} |" for m in CLASSICAL_MODELS)
        + f"""

## Notes

- LLM prompts: V1-style (history stats in words), SMM in percent units.
- {n_ok}/{len(origins)} LLM origins succeeded.
- Gate: dataset="smm" is macro-like → LLM expert enabled (llm_gate).
"""
    )
    print(f"[exp65] v2={v2m:.6f} v1={v1m:.6f} llm={llmm:.6f} → {verdict}", flush=True)


if __name__ == "__main__":
    main()
