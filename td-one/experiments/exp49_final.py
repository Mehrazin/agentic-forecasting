"""exp49: Final model run with the updated toolset.

The definitive validation of everything TD-One built:

  Part A — Toolset validation (parity checks):
    - series_features vs exp42.extract_features (must match)
    - naive_forecasts median vs exp42._fit_naive point (must match)
    - regime_probabilities vs exp43 label rules (must agree)
    - combine_quantiles vs manual mean (must match)

  Part B — Final proposed-method run (blocked CV, paired t-tests):
    - Records generated with series_features in the pipeline
    - ProposedForecaster on M4, Synthetic, SMM, Food CPI
    - Equal-weights baseline via the combine_quantiles tool
    - BoC: regime_probabilities (spec="boc") -> EASY/HARD -> rule accuracy

Comparisons: proposed vs equal weights, proposed vs best single.
Datasets: M4, Synthetic, SMM, Food CPI (continuous) + BoC (discrete).

Usage:
    cd td-one && .venv/bin/python experiments/exp49_final.py [--quick]
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
import exp47_proposed_method as e47  # noqa: E402
from exp43_regime_router import label_m4 as _exp43_label_m4  # noqa: E402

from td_one.agent.proposed_method import (  # noqa: E402
    generic_regime_spec,
    m4_regime_spec,
)
from td_one.agent.tools.data import (  # noqa: E402
    naive_forecasts,
    series_features,
)
from td_one.agent.tools.evaluation import combine_quantiles  # noqa: E402
from td_one.agent.tools.regimes import regime_probabilities  # noqa: E402

MODELS = f42.MODELS
FEATURE_NAMES = f42.FEATURE_NAMES


# ---------------------------------------------------------------------------
# Part A: toolset parity validation
# ---------------------------------------------------------------------------


def part_a_tool_parity(n_series: int = 20, seed: int = 7) -> dict:
    """Verify new tools reproduce the proven exp42/exp43 behavior."""
    out: dict = {}
    rng = np.random.default_rng(seed)

    # 1. series_features vs f42.extract_features
    max_diff = 0.0
    for _ in range(n_series):
        n = int(rng.integers(48, 120))
        x = np.cumsum(rng.normal(0, 1, n)) + 5 * np.sin(
            2 * np.pi * np.arange(n) / 12
        )
        ref = np.array(f42.extract_features(x))
        got = np.array(
            [series_features(x).as_dict()[k] for k in FEATURE_NAMES]
        )
        max_diff = max(max_diff, float(np.max(np.abs(ref - got))))
    out["series_features_max_abs_diff"] = max_diff
    out["series_features_parity"] = bool(max_diff < 1e-9)

    # 2. naive_forecasts median vs f42._fit_naive point
    max_pt_diff = 0.0
    for _ in range(n_series):
        n = int(rng.integers(24, 100))
        x = np.cumsum(rng.normal(0, 1, n))
        ref_pt, _ = f42._fit_naive(x, 3)
        got = naive_forecasts(x, horizon=3)
        got_med = np.array([row[4] for row in got.naive])  # 0.5 level
        max_pt_diff = max(max_pt_diff, float(np.max(np.abs(ref_pt - got_med))))
    out["naive_point_max_abs_diff"] = max_pt_diff
    out["naive_point_parity"] = bool(max_pt_diff < 1e-9)

    # 3. regime_probabilities vs exp43 label_m4 (on real M4 records)
    recs = f42.generate_records(f42.load_m4(n=20, seed=seed), 48, 6, 6, "m4")
    agree = 0
    for r in recs[:50]:
        feats = dict(zip(FEATURE_NAMES, r["features"], strict=True))
        if regime_probabilities(feats, spec="m4").label == _exp43_label_m4(feats):
            agree += 1
    out["regime_label_agreement"] = agree / 50
    out["regime_label_n"] = 50

    # 4. combine_quantiles vs manual mean
    mq = {
        m: {float(q): float(v) for q, v in zip([0.1, 0.5, 0.9], vals, strict=True)}
        for m, vals in [("a", [1.0, 2.0, 3.0]), ("b", [3.0, 4.0, 5.0])]
    }
    got = combine_quantiles(mq)
    manual = [(1.0 + 3.0) / 2, (2.0 + 4.0) / 2, (3.0 + 5.0) / 2]
    out["combine_max_abs_diff"] = float(
        np.max(np.abs(np.array(got.values) - np.array(manual)))
    )
    out["combine_parity"] = bool(out["combine_max_abs_diff"] < 1e-9)
    return out


# ---------------------------------------------------------------------------
# Part B: final proposed-method run (new tools in the pipeline)
# ---------------------------------------------------------------------------


def _series_features_list(train: np.ndarray) -> list[float]:
    """Drop-in replacement for f42.extract_features via the new tool."""
    return [series_features(train).as_dict()[k] for k in FEATURE_NAMES]


def _equal_weights_via_tool(r: dict) -> np.ndarray:
    """Equal-weights combination using the combine_quantiles tool."""
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in MODELS}
    h = next(iter(qm.values())).shape[1]
    out = np.zeros((len(levels), h))
    for s in range(h):
        mq = {
            m: {
                float(q): float(v)
                for q, v in zip(levels, qm[m][:, s], strict=True)
            }
            for m in MODELS
        }
        out[:, s] = np.array(combine_quantiles(mq).values)
    return out


def run_boc_final() -> dict:
    """BoC validation via the regime_probabilities tool (spec='boc').

    EASY -> rule, HARD -> climatology (exp32). Reports accuracy and the
    EASY/HARD split, replicating exp32's +7.2pp finding through the tool.
    """
    from exp43_regime_router import load_boc  # noqa: E402

    records = load_boc()
    n = len(records)
    correct_rule = correct_gate = 0
    easy_n = easy_correct = 0
    hard_n = hard_correct = 0
    agree_with_exp43 = 0
    for m in records:
        rm, ig, um, ys = (float(v) for v in m["features"])
        feats = {
            "rate_momentum": rm,
            "inflation_gap": ig,
            "unemployment_momentum": um,
            "yield_spread": ys,
        }
        rp = regime_probabilities(feats, spec="boc")
        if rp.label == m["regime"]:
            agree_with_exp43 += 1
        rule_pred = m["policies"]["rule"]
        clim_pred = m["policies"]["climatology"]
        gated_pred = rule_pred if rp.label == "EASY" else clim_pred
        if rule_pred == m["actual"]:
            correct_rule += 1
        if gated_pred == m["actual"]:
            correct_gate += 1
        if rp.label == "EASY":
            easy_n += 1
            easy_correct += int(rule_pred == m["actual"])
        else:
            hard_n += 1
            hard_correct += int(clim_pred == m["actual"])
    return {
        "n": n,
        "tool_regime_agreement_with_exp43": agree_with_exp43 / n,
        "rule_accuracy": correct_rule / n,
        "gated_accuracy": correct_gate / n,
        "easy": {"n": easy_n, "acc": easy_correct / easy_n if easy_n else 0.0},
        "hard": {"n": hard_n, "acc": hard_correct / hard_n if hard_n else 0.0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[exp49] Part A: toolset parity validation...", flush=True)
    parity = part_a_tool_parity()
    for k, v in parity.items():
        print(f"[exp49]   {k} = {v}", flush=True)

    print("[exp49] Part B: final run (series_features in pipeline)...", flush=True)
    # Route the record pipeline through the new series_features tool.
    f42.extract_features = _series_features_list  # type: ignore[method-assign]

    n_m4 = 20 if args.quick else 100
    m4 = f42.generate_records(f42.load_m4(n=n_m4), 48, 6, 6, "m4")
    smm = f42.generate_records([f42.load_smm()[0]], 36, 1, 50, "smm")
    fcpi = f42.generate_records([f42.load_food_cpi()[0]], 48, 1, 24, "food_cpi")
    synth_n = 12 if args.quick else 40
    synth = f42.generate_records(
        f42.load_synthetic(n_series=synth_n), 30, 1, 12, "synthetic"
    )
    print(
        f"[exp49] records: m4={len(m4)} smm={len(smm)} "
        f"fcpi={len(fcpi)} synth={len(synth)}",
        flush=True,
    )

    results: dict = {}
    # Reuse exp47's blocked-CV harness, but swap equal-weights to the tool.
    orig_eq = e47._equal_weights_quantiles
    e47._equal_weights_quantiles = _equal_weights_via_tool  # type: ignore[method-assign]
    try:
        results["m4"] = e47.run_dataset_cv("m4", m4, m4_regime_spec)
        results["synthetic"] = e47.run_dataset_cv("synthetic", synth, m4_regime_spec)
        results["smm"] = e47.run_dataset_cv("smm", smm, generic_regime_spec)
        results["food_cpi"] = e47.run_dataset_cv("food_cpi", fcpi, generic_regime_spec)
    finally:
        e47._equal_weights_quantiles = orig_eq

    for name, r in results.items():
        m_ = r["mean_crps"]
        pe = r["proposed_vs_equal"]
        pb = r["proposed_vs_best"]
        print(
            f"[exp49] {name}: proposed={m_['proposed']:.4f} "
            f"equal={m_['equal']:.4f} best={m_['best_single']:.4f} | "
            f"vs equal p={pe['p']:.4g} | vs best p={pb['p']:.4g}",
            flush=True,
        )

    print("[exp49] BoC via regime_probabilities tool...", flush=True)
    results["boc"] = run_boc_final()
    b = results["boc"]
    print(
        f"[exp49] boc: n={b['n']} rule_acc={b['rule_accuracy']:.3f} "
        f"gated_acc={b['gated_accuracy']:.3f} "
        f"EASY {b['easy']['n']}@{b['easy']['acc']:.3f} "
        f"HARD {b['hard']['n']}@{b['hard']['acc']:.3f} "
        f"tool_agree={b['tool_regime_agreement_with_exp43']:.3f}",
        flush=True,
    )

    out = {
        "date": datetime.now(UTC).isoformat(),
        "quick": args.quick,
        "elapsed_s": time.time() - t0,
        "models": MODELS,
        "features": FEATURE_NAMES,
        "toolset": [
            "load_series",
            "naive_forecasts",
            "series_features",
            "regime_probabilities",
            "combine_quantiles",
        ],
        "part_a_parity": parity,
        "results": results,
    }
    with open(ROOT / "experiments" / "exp49_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[exp49] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
