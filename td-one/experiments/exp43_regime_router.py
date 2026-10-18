"""exp43: Regime-probability router -- soft Markov-switching over per-regime experts.

Motivation: exp42 proved learned combination weights don't beat equal weights
at our scale (forecast combination puzzle: estimation error eats the gains).
The literature's biggest bet: a REGIME-PROBABILITY ROUTER. Instead of learning
4 continuous weights, learn P(regime | features) -- an easier classification
problem -- and use fixed per-regime experts. This is a structural prior:
combination weights are constrained to the simplex spanned by regime-expert
assignments, which should need far less data than exp42's flexible learner.

Method:
  1. Define 2-3 regimes per dataset (PIT-safe, from train window / pre-meeting
     data only).
  2. Train classifier: features -> P(regime). (LogisticRegression: simple,
     calibrated probabilities.)
  3. Per-regime expert = model with lowest mean CRPS on TRAIN records of that
     regime (fallback: global best single if regime has < 10 records).
  4. Test forecast = sum_r P(r | x) * expert_r quantiles (Vincentized).
  5. Compare vs equal weights (paired t-test), best single, hard router
     (argmax regime), oracle.

Datasets: M4, synthetic (true DGP kind = clean latent-regime test), SMM,
Food CPI (single-series, blocked time CV), BoC (EASY/HARD, accuracy metric).

Success criterion: beat equal weights with p < 0.05 on at least one dataset.

Usage:
    cd td-one && .venv/bin/python experiments/exp43_regime_router.py [--quick]
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
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = f42.MODELS
FEATURE_NAMES = f42.FEATURE_NAMES
MIN_REGIME_N = 10  # fallback to global best single below this

# ---------------------------------------------------------------------------
# Regime definitions (all PIT-safe: train window / pre-meeting data only)
# ---------------------------------------------------------------------------

M4_REGIMES = ["seasonal", "trending", "noisy"]
SYNTH_REGIMES = ["trending", "seasonal", "rw"]
SMM_REGIMES = ["calm", "stressed"]
FCPI_REGIMES = ["calm", "volatile"]
BOC_REGIMES = ["EASY", "HARD"]


def label_m4(feats: dict) -> str:
    if feats["seasonal_strength"] > 0.5:
        return "seasonal"
    if feats["trend_strength"] > 0.7:
        return "trending"
    return "noisy"


def label_smm(feats: dict) -> str:
    if feats["n_recent_cps"] >= 1 or feats["vol_ratio"] > 1.0:
        return "stressed"
    return "calm"


def label_fcpi(feats: dict, vol_thr: float = 0.0395) -> str:
    # NOTE: the default threshold is only a placeholder. run_continuous()
    # re-labels food_cpi per fold using the TRAIN records' median vol_ratio,
    # so no future information leaks into the regime definition.
    return "volatile" if feats["vol_ratio"] > vol_thr else "calm"


def label_synth(series_id: str) -> str:
    # load_synthetic: kind = i % 3 -> 0: trending, 1: seasonal, 2: rw
    i = int(series_id.split("_")[1])
    return SYNTH_REGIMES[i % 3]


def attach_regimes(records: list[dict], dataset: str) -> list[dict]:
    out = []
    for r in records:
        f = dict(zip(FEATURE_NAMES, r["features"], strict=True))
        if dataset == "m4":
            reg = label_m4(f)
        elif dataset == "synthetic":
            reg = label_synth(r["series_id"])
        elif dataset == "smm":
            reg = label_smm(f)
        elif dataset == "food_cpi":
            reg = label_fcpi(f)
        else:
            raise ValueError(dataset)
        r2 = dict(r)
        r2["regime"] = reg
        out.append(r2)
    return out


# ---------------------------------------------------------------------------
# Router: classifier features -> P(regime)
# ---------------------------------------------------------------------------


def train_router(x_train: np.ndarray, y_train: list[str], regimes: list[str]) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, C=1.0)
    # map regime names to 0..K-1 in canonical order
    y_idx = np.array([regimes.index(y) for y in y_train])
    clf.fit(x_train, y_idx)
    return clf


def regime_proba(clf: LogisticRegression, x: np.ndarray, regimes: list[str]) -> np.ndarray:
    """Return (n, K) probs aligned to canonical regime order."""
    raw = clf.predict_proba(x)  # columns follow clf.classes_
    out = np.zeros((x.shape[0], len(regimes)))
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = raw[:, j]
    return out


# ---------------------------------------------------------------------------
# Per-regime experts
# ---------------------------------------------------------------------------


def pick_experts(
    train_records: list[dict], regimes: list[str]
) -> tuple[dict[str, str], dict[str, int]]:
    """Best model per regime on training records (mean CRPS)."""
    experts: dict[str, str] = {}
    counts: dict[str, int] = {}
    # global best single as fallback
    global_mean = {m: float(np.mean([r["crps"][m] for r in train_records])) for m in MODELS}
    global_best = min(global_mean, key=global_mean.get)  # type: ignore[arg-type]
    for reg in regimes:
        recs = [r for r in train_records if r["regime"] == reg]
        counts[reg] = len(recs)
        if len(recs) < MIN_REGIME_N:
            experts[reg] = global_best
            continue
        mean_c = {m: float(np.mean([r["crps"][m] for r in recs])) for m in MODELS}
        experts[reg] = min(mean_c, key=mean_c.get)  # type: ignore[arg-type]
    return experts, counts


# ---------------------------------------------------------------------------
# Evaluation (continuous datasets)
# ---------------------------------------------------------------------------


def evaluate_router(
    test_records: list[dict],
    clf: LogisticRegression,
    experts: dict[str, str],
    regimes: list[str],
) -> dict:
    n = len(test_records)
    x = np.array([r["features"] for r in test_records])
    p_mat = regime_proba(clf, x, regimes)  # (n, K)

    router_c = np.zeros(n)
    hard_c = np.zeros(n)
    equal_c = np.zeros(n)
    best_c = np.zeros(n)
    oracle_c = np.zeros(n)
    regime_acc = np.zeros(n)

    for i, r in enumerate(test_records):
        actual = np.asarray(r["actual"], dtype=float)
        qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in MODELS}
        c = np.array([r["crps"][m] for m in MODELS])
        h = actual.shape[0]

        # soft router
        q_soft = np.zeros((9, h))
        for k, reg in enumerate(regimes):
            q_soft += p_mat[i, k] * qm[experts[reg]]
        router_c[i] = f42._crps_quantile(actual, q_soft)

        # hard router (argmax regime)
        hard_reg = regimes[int(np.argmax(p_mat[i]))]
        hard_c[i] = f42._crps_quantile(actual, qm[experts[hard_reg]])
        regime_acc[i] = 1.0 if hard_reg == r["regime"] else 0.0

        # equal weights
        q_eq = sum(qm[m] for m in MODELS) / len(MODELS)
        equal_c[i] = f42._crps_quantile(actual, q_eq)

        best_c[i] = float(np.min(c))
        oracle_c[i] = float(np.min(c))

    per_model_mean = {m: float(np.mean([r["crps"][m] for r in test_records])) for m in MODELS}
    best_single_name = min(per_model_mean, key=per_model_mean.get)  # type: ignore[arg-type]

    diff = router_c - equal_c
    t_stat, p_val = ttest_rel(router_c, equal_c)

    def gap_closed(base: float, meta: float) -> float:
        o = float(np.mean(oracle_c))
        denom = base - o
        return (base - meta) / denom if abs(denom) > 1e-12 else 0.0

    return {
        "n": n,
        "mean_crps": {
            "router_soft": float(np.mean(router_c)),
            "router_hard": float(np.mean(hard_c)),
            "equal_weights": float(np.mean(equal_c)),
            "best_single": float(np.mean(best_c)),
            "best_single_name": best_single_name,
            "per_model": per_model_mean,
            "oracle": float(np.mean(oracle_c)),
        },
        "experts": experts,
        "router_regime_accuracy": float(np.mean(regime_acc)),
        "router_vs_equal": {
            "mean_diff": float(np.mean(diff)),  # negative = router better
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "win_rate": float(np.mean(router_c < equal_c)),
        },
        "router_vs_hard": {
            "mean_diff": float(np.mean(router_c - hard_c)),
            "win_rate": float(np.mean(router_c < hard_c)),
        },
        "gap_closed_vs_equal": gap_closed(float(np.mean(equal_c)), float(np.mean(router_c))),
        "gap_closed_vs_best": gap_closed(float(np.mean(best_c)), float(np.mean(router_c))),
    }


def blocked_folds(n: int, n_folds: int) -> list[tuple[list[int], list[int]]]:
    """Contiguous time blocks (for single-series records, already chronological)."""
    idx = list(range(n))
    size = n // n_folds
    out = []
    for f in range(n_folds):
        te = idx[f * size : (f + 1) * size if f < n_folds - 1 else n]
        tr = [i for i in idx if i not in set(te)]
        out.append((tr, te))
    return out


def run_continuous(
    name: str,
    series: list[tuple[str, np.ndarray]],
    train_window: int,
    horizon: int,
    n_origins: int,
    regimes: list[str],
    by_series: bool,
    n_folds: int,
    quick: bool,
) -> dict:
    print(f"[exp43] {name}: generating records...", flush=True)
    records = attach_regimes(
        f42.generate_records(series, train_window, horizon, n_origins, name), name
    )
    print(f"[exp43] {name}: {len(records)} records", flush=True)
    dist = {reg: sum(1 for r in records if r["regime"] == reg) for reg in regimes}
    note = " (placeholder; relabeled per-fold from train median)" if name == "food_cpi" else ""
    print(f"[exp43] {name} regime distribution: {dist}{note}", flush=True)

    folds = (
        f42.kfold_by_series(records, n_folds=n_folds)
        if by_series
        else blocked_folds(len(records), n_folds)
    )
    fold_results = []
    all_router, all_equal = [], []
    for f, (tr, te) in enumerate(folds):
        if name == "food_cpi":
            # Per-fold regime threshold: median vol_ratio on TRAIN records only.
            # Fully PIT-safe (no future origins used); semantics ("volatile" =
            # above train-median vol) are stable across folds.
            thr = float(
                np.median(
                    [
                        dict(zip(FEATURE_NAMES, records[i]["features"], strict=True))["vol_ratio"]
                        for i in tr
                    ]
                )
            )
            for i in tr + te:
                ff = dict(zip(FEATURE_NAMES, records[i]["features"], strict=True))
                records[i]["regime"] = label_fcpi(ff, vol_thr=thr)
        tr_recs = [records[i] for i in tr]
        te_recs = [records[i] for i in te]
        x_tr = np.array([r["features"] for r in tr_recs])
        y_tr = [r["regime"] for r in tr_recs]
        clf = train_router(x_tr, y_tr, regimes)
        experts, counts = pick_experts(tr_recs, regimes)
        res = evaluate_router(te_recs, clf, experts, regimes)
        res["fold"] = f
        res["train_regime_counts"] = counts
        fold_results.append(res)
        # pooled per-record for the headline test
        x_te = np.array([r["features"] for r in te_recs])
        p_mat = regime_proba(clf, x_te, regimes)
        for i, r in enumerate(te_recs):
            actual = np.asarray(r["actual"])
            qm = {m: np.asarray(r["quantiles"][m]) for m in MODELS}
            q_soft = sum(p_mat[i, k] * qm[experts[reg]] for k, reg in enumerate(regimes))
            q_eq = sum(qm[m] for m in MODELS) / len(MODELS)
            all_router.append(f42._crps_quantile(actual, q_soft))
            all_equal.append(f42._crps_quantile(actual, q_eq))
        print(
            f"[exp43] {name} fold {f}: router={res['mean_crps']['router_soft']:.4f} "
            f"equal={res['mean_crps']['equal_weights']:.4f} "
            f"hard={res['mean_crps']['router_hard']:.4f} "
            f"p={res['router_vs_equal']['p_value']:.4f} "
            f"regacc={res['router_regime_accuracy']:.2f} experts={experts}",
            flush=True,
        )

    all_router = np.array(all_router)
    all_equal = np.array(all_equal)
    t_stat, p_val = ttest_rel(all_router, all_equal)
    pooled = {
        "n": len(all_router),
        "mean_router": float(np.mean(all_router)),
        "mean_equal": float(np.mean(all_equal)),
        "mean_diff": float(np.mean(all_router - all_equal)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "win_rate": float(np.mean(all_router < all_equal)),
    }
    print(
        f"[exp43] {name} POOLED: router={pooled['mean_router']:.4f} "
        f"equal={pooled['mean_equal']:.4f} p={pooled['p_value']:.4g}",
        flush=True,
    )
    return {
        "dataset": name,
        "regimes": regimes,
        "regime_distribution": dist,
        "folds": fold_results,
        "pooled": pooled,
        "n_records": len(records),
    }


# ---------------------------------------------------------------------------
# BoC: EASY/HARD router (accuracy metric)
# ---------------------------------------------------------------------------


def _implied_rm(rm: float) -> int:
    return -1 if rm < -0.25 else (1 if rm > 0.25 else 0)


def _implied_ig(ig: float) -> int:
    return -1 if ig < -1.0 else (1 if ig > 1.0 else 0)


def _implied_ys(ys: float) -> int:
    return -1 if ys < -0.5 else (1 if ys > 0.5 else 0)


def load_boc() -> list[dict]:
    """One record per meeting.

    PIT-safe macro features, EASY/HARD regime, policy predictions
    {rule, climatology, agent_override}, actual decision.
    """
    bench = ROOT / "benchmarks" / "boc_rates"
    df = pd.read_json(bench / "pilot_data.json")
    sys.path.insert(0, str(bench))
    from exp30_mnlogit import (  # noqa: E402
        build_features,
        load_meetings,
        load_parquet,
    )

    meetings = load_meetings()
    rates = load_parquet("IRSTCB01CAM156N.parquet")
    cpi = load_parquet("CPALTT01CAM659N.parquet")
    unemp = load_parquet("LRHUTTTTCAM156S.parquet")
    y2 = load_parquet("BD.CDN.2YR.DQ.YLD.parquet")
    ys_map = {}
    for md in meetings:
        feats = build_features(md, rates, cpi, unemp, y2)
        if feats is not None:
            ys_map[md.isoformat()] = feats["yield_spread"]

    records = []
    for _, row in df.iterrows():
        md = row["meeting_date"]
        ys = ys_map.get(md)
        if ys is None or not np.isfinite(ys):
            continue
        rm, ig, um = (
            float(row["rate_momentum"]),
            float(row["inflation_gap"]),
            float(row["unemployment_momentum"]),
        )
        implied = [_implied_rm(rm), _implied_ig(ig), _implied_ys(float(ys))]
        regime = "EASY" if len(set(implied)) == 1 else "HARD"
        rule = _implied_rm(rm)
        clim = 0
        agent = rule
        if ig > 2.0 and rule == 0:
            agent = 1
        records.append(
            {
                "meeting_date": md,
                "features": [rm, ig, um, float(ys)],
                "regime": regime,
                "policies": {"rule": rule, "climatology": clim, "agent_override": agent},
                "actual": int(row["decision"]),
            }
        )
    # already chronological
    return records


def run_boc(n_folds: int = 5) -> dict:
    print("[exp43] BoC: loading meetings...", flush=True)
    records = load_boc()
    print(f"[exp43] BoC: {len(records)} meetings", flush=True)
    dist = {r: sum(1 for m in records if m["regime"] == r) for r in BOC_REGIMES}
    print(f"[exp43] BoC regime distribution: {dist}", flush=True)
    policies = ["rule", "climatology", "agent_override"]

    folds = blocked_folds(len(records), n_folds)
    fold_results = []
    all_soft, all_rule, all_hard = [], [], []
    for f, (tr, te) in enumerate(folds):
        tr_recs = [records[i] for i in tr]
        te_recs = [records[i] for i in te]
        x_tr = np.array([m["features"] for m in tr_recs])
        y_tr = [m["regime"] for m in tr_recs]
        clf = train_router(x_tr, y_tr, BOC_REGIMES)
        # per-regime best policy on train
        experts: dict[str, str] = {}
        for reg in BOC_REGIMES:
            recs = [m for m in tr_recs if m["regime"] == reg]
            if len(recs) < 5:
                experts[reg] = "rule"
                continue
            acc = {
                p: float(np.mean([1.0 if m["policies"][p] == m["actual"] else 0.0 for m in recs]))
                for p in policies
            }
            experts[reg] = max(acc, key=acc.get)  # type: ignore[arg-type]

        x_te = np.array([m["features"] for m in te_recs])
        p_mat = regime_proba(clf, x_te, BOC_REGIMES)
        s_acc, r_acc, h_acc, c_acc = [], [], [], []
        reg_correct = 0
        for i, m in enumerate(te_recs):
            # soft router: weighted vote over policies
            votes = {-1: 0.0, 0: 0.0, 1: 0.0}
            for k, reg in enumerate(BOC_REGIMES):
                votes[m["policies"][experts[reg]]] += p_mat[i, k]
            soft_pred = max(votes, key=votes.get)  # type: ignore[arg-type]
            hard_reg = BOC_REGIMES[int(np.argmax(p_mat[i]))]
            hard_pred = m["policies"][experts[hard_reg]]
            s_acc.append(1.0 if soft_pred == m["actual"] else 0.0)
            r_acc.append(1.0 if m["policies"]["rule"] == m["actual"] else 0.0)
            h_acc.append(1.0 if hard_pred == m["actual"] else 0.0)
            c_acc.append(1.0 if m["policies"]["climatology"] == m["actual"] else 0.0)
            reg_correct += 1.0 if hard_reg == m["regime"] else 0.0
            all_soft.append(s_acc[-1])
            all_rule.append(r_acc[-1])
            all_hard.append(h_acc[-1])

        fold_results.append(
            {
                "fold": f,
                "n_test": len(te_recs),
                "experts": experts,
                "acc": {
                    "router_soft": float(np.mean(s_acc)),
                    "router_hard": float(np.mean(h_acc)),
                    "rule": float(np.mean(r_acc)),
                    "climatology": float(np.mean(c_acc)),
                },
                "regime_accuracy": float(reg_correct / max(len(te_recs), 1)),
            }
        )
        print(
            f"[exp43] BoC fold {f}: soft={np.mean(s_acc):.3f} hard={np.mean(h_acc):.3f} "
            f"rule={np.mean(r_acc):.3f} clim={np.mean(c_acc):.3f} experts={experts}",
            flush=True,
        )

    all_soft = np.array(all_soft)
    all_rule = np.array(all_rule)
    t_stat, p_val = ttest_rel(all_soft, all_rule)
    pooled = {
        "n": len(all_soft),
        "mean_soft": float(np.mean(all_soft)),
        "mean_rule": float(np.mean(all_rule)),
        "mean_hard": float(np.mean(all_hard)),
        "mean_diff_soft_vs_rule": float(np.mean(all_soft - all_rule)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "win_rate": float(np.mean(all_soft > all_rule)),
    }
    print(
        f"[exp43] BoC POOLED: soft={pooled['mean_soft']:.3f} "
        f"rule={pooled['mean_rule']:.3f} p={pooled['p_value']:.4g}",
        flush=True,
    )
    return {
        "dataset": "boc",
        "regimes": BOC_REGIMES,
        "regime_distribution": dist,
        "features": ["rate_momentum", "inflation_gap", "unemployment_momentum", "yield_spread"],
        "folds": fold_results,
        "pooled": pooled,
        "n_records": len(records),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run for smoke test")
    ap.add_argument("--skip-boc", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {
        "experiment": "exp43_regime_router",
        "date": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "features": FEATURE_NAMES,
        "method": (
            "Regime-probability router: LogisticRegression maps 10 PIT-safe "
            "train-window features -> P(regime). Per-regime expert = model with "
            "lowest mean CRPS on training records of that regime "
            "(fallback: global best single if <10 records). Test forecast = "
            "sum_r P(r|x) * expert_r quantiles (Vincentized). Baselines: equal "
            "weights, best single (hindsight), hard router (argmax), oracle."
        ),
    }

    # M4 (primary): 5-fold by series
    m4_series = f42.load_m4(n=20 if args.quick else 100, seed=42)
    out["m4"] = run_continuous("m4", m4_series, 60, 6, 6, M4_REGIMES, True, 5, args.quick)

    # Synthetic: true DGP kind as latent regime
    synth = f42.load_synthetic(n_series=12 if args.quick else 30, length=120)
    out["synthetic"] = run_continuous(
        "synthetic", synth, 60, 1, 4, SYNTH_REGIMES, True, 5, args.quick
    )

    # SMM: single series, blocked time CV
    smm = f42.load_smm()
    out["smm"] = run_continuous("smm", smm, 12, 1, 50, SMM_REGIMES, False, 5, args.quick)

    # Food CPI: single series, blocked time CV
    fcpi = f42.load_food_cpi()
    out["food_cpi"] = run_continuous(
        "food_cpi", fcpi, 60, 1, 24, FCPI_REGIMES, False, 4, args.quick
    )

    if not args.skip_boc:
        out["boc"] = run_boc()

    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp43_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[exp43] wrote {out_path} ({out['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
