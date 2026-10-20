"""exp70: OSFI M4 runoff as a real-data benchmark for the proposed method.

**Question:** can ProposedForecasterV2 beat naive on REAL Canadian mortgage
runoff — the first real-data test for the proposed method's runoff story?

**Data:** ``data/osfi/m4_runoff.parquet`` — implied monthly runoff
(outstanding[t-1] + advanced[t] - outstanding[t]) by segment,
2016-07 -> 2026-07 (121 months), StatCan table 10100006 (OSFI M4).
Target = runoff rate (runoff / outstanding[t-1]) — normalizes scale drift.

**Protocol:** per segment (insured / uninsured / variable_insured /
variable_uninsured / total): rolling origins over the last 60 months,
36-month train window, h=1. Classical records (naive/snaive/ets/arima)
built with the exp52 pipeline (``experiments/llm_food_cpi.py``);
5-fold blocked time CV of ProposedForecasterV2 vs best-single classical.

**Pre-registered kill criterion:** if v2 does not beat naive at p < 0.10
on TOTAL runoff h=1, the proposed method has a real-data problem —
record it honestly; do not tune on this target.

**PIT:** OSFI month M is observed as-of M+2 (publication lag). Origins use
only months <= origin month (conservative: the M+2 lag is not modeled in
the CV, flagged as a limitation — it affects all arms equally).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402

DATA_DIR = ROOT / "data" / "osfi"
TRAIN_WINDOW = 36
N_ORIGINS = 60
SEGMENTS = ["insured", "uninsured", "variable_insured", "variable_uninsured"]


def load_runoff_rates() -> dict[str, pd.Series]:
    """Monthly runoff rate per segment + total. Index = month."""
    bal = pd.read_parquet(DATA_DIR / "m4_balances.parquet")
    run = pd.read_parquet(DATA_DIR / "m4_runoff.parquet")
    out: dict[str, pd.Series] = {}
    for seg in SEGMENTS:
        b = bal[bal["segment"] == seg].set_index("month").sort_index()
        r = run[run["segment"] == seg].set_index("month").sort_index()
        rate = r["runoff_m"] / b["outstanding_m"].shift(1)
        rate = rate.dropna()
        out[seg] = rate
    # total: sum runoff / sum lagged outstanding
    btot = bal.groupby("month")["outstanding_m"].sum().sort_index()
    rtot = run.groupby("month")["runoff_m"].sum().sort_index()
    out["total"] = (rtot / btot.shift(1)).dropna()
    return out


def build_records(values: np.ndarray) -> tuple[list[dict], list[str]]:
    """One classical record per origin (last N_ORIGINS months, h=1)."""
    feature_names = lfc.feature_names_for(values)
    records = []
    start = len(values) - N_ORIGINS
    for o in range(start, len(values)):
        train = values[o - TRAIN_WINDOW : o]
        actual = float(values[o])
        if len(train) < TRAIN_WINDOW:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = series_features(train, period=12)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qm = lfc.origin_quantiles(train)
        records.append(
            {
                "origin": int(o),
                "features": feat_vec,
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {
                    m: grid_to_crps(q[:, 0], LEVELS_9, actual)
                    for m, q in qm.items()
                },
                "actual": [actual],
            }
        )
    return records, feature_names


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    t, p = ttest_rel(a, b)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_pct": float(100 * (np.mean(a) - np.mean(b)) / np.mean(b)),
        "p": float(p),
        "t": float(t),
        "win_rate": float(np.mean(a < b)),
        "n": int(len(a)),
    }


def run_segment(name: str, series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    records, feature_names = build_records(values)
    print(f"[exp70] {name}: {len(records)} origins", flush=True)
    if len(records) < 10:
        return {"segment": name, "n": len(records), "error": "too few origins"}

    def factory() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(h=1, use_llm_expert=False)
        # OSFI pipeline uses snaive, not m0 (matches the exp52 record set).
        fc.models = list(lfc.CLASSICAL_MODELS)
        return fc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = blocked_cv_predict(
            records, feature_names, factory, n_folds=5, baseline="best_single"
        )
    method = np.asarray(res["crps"])
    base = np.asarray(res["baseline_crps"])
    stats = _paired(method, base)
    # standalone naive for the kill criterion
    naive = np.array([r["crps"]["naive"] for r in records])
    # align: res crps are per-input-record order
    stats_naive = _paired(method, naive)
    return {
        "segment": name,
        "n": len(records),
        "v2_crps": stats["mean_a"],
        "best_single_crps": stats["mean_b"],
        "baseline_name": res.get("baseline_name", "?"),
        "vs_best_single": stats,
        "vs_naive": stats_naive,
    }


def main() -> None:
    series = load_runoff_rates()
    results = {}
    for name in ["total", *SEGMENTS]:
        results[name] = run_segment(name, series[name])
    out_path = Path(__file__).parent / "exp70_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    tot = results["total"]
    if "vs_naive" in tot:
        s = tot["vs_naive"]
        verdict = (
            "KILL CRITERION TRIGGERED: v2 does not beat naive on total runoff"
            if s["p"] >= 0.10 or s["mean_a"] >= s["mean_b"]
            else "v2 beats naive on total runoff"
        )
        print(f"[exp70] total: v2 {s['mean_a']:.5f} vs naive {s['mean_b']:.5f}, "
              f"p={s['p']:.3g}, win={s['win_rate']:.2f}")
        print(f"[exp70] {verdict}")


if __name__ == "__main__":
    main()
