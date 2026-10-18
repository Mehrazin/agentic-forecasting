"""exp45: Learned regime router -- unsupervised discovery replaces hand-rules.

Motivation: exp43's regime router beat equal weights (M4, p=0.018), but its
regimes are hand-written rules ("seasonal_strength > 0.5 -> seasonal").
Now learn the regimes from data:

  1. GMM(K) on standardized train features -> posterior P(regime|x) used
     directly as the router (soft, no separate classifier).
  2. KMeans(K) on standardized train features -> hard clusters ->
     LogisticRegression router (same structure as exp43).

Both are fit on TRAIN records only per fold (PIT-safe; no future leakage).
We also re-run exp43's hand-rule router on the SAME folds so the learned-vs-hand
comparison is apples-to-apples (paired per-record t-tests).

Datasets: M4 (primary), synthetic (true DGP kind = ground truth for
interpretability via ARI), SMM, Food CPI. BoC skipped (classification task,
different pipeline; exp43 showed router = rule there anyway).

Success criterion: learned (GMM or KMeans) beats exp43's 0.1282 on M4, or at
least beats the hand-rule router head-to-head with p < 0.05.

Usage:
    cd td-one && .venv/bin/python experiments/exp45_learned_regimes.py [--quick]
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402
import exp43_regime_router as e43  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = f42.MODELS
FEATURE_NAMES = f42.FEATURE_NAMES
MIN_REGIME_N = 10
SEED = 42

# K per dataset (matches exp43's hand-rule regime counts for fair comparison)
K_MAP = {"m4": 3, "synthetic": 3, "smm": 2, "food_cpi": 2}
REGIME_NAMES = {
    "m4": ["L0", "L1", "L2"],
    "synthetic": ["L0", "L1", "L2"],
    "smm": ["L0", "L1"],
    "food_cpi": ["L0", "L1"],
}


# ---------------------------------------------------------------------------
# Unsupervised discovery (fit on train features only)
# ---------------------------------------------------------------------------


def fit_gmm(x_train: np.ndarray, k: int) -> tuple[GaussianMixture, StandardScaler]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_train)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=5,
        random_state=SEED,
        max_iter=200,
    )
    gmm.fit(xs)
    return gmm, scaler


def fit_kmeans(x_train: np.ndarray, k: int) -> tuple[KMeans, StandardScaler]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_train)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(xs)
    return km, scaler


def train_router_kmeans(
    x_train: np.ndarray, km: KMeans, scaler: StandardScaler, regimes: list[str]
) -> LogisticRegression:
    y_tr = km.predict(scaler.transform(x_train)).tolist()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_train, y_tr)
    return clf


# ---------------------------------------------------------------------------
# Evaluation: soft combination from a (n, K) proba matrix
# ---------------------------------------------------------------------------


def soft_crps(
    test_records: list[dict], p_mat: np.ndarray, experts: dict[str, str], regimes: list[str]
) -> np.ndarray:
    out = np.zeros(len(test_records))
    for i, r in enumerate(test_records):
        actual = np.asarray(r["actual"], dtype=float)
        qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in MODELS}
        h = actual.shape[0]
        q_soft = np.zeros((9, h))
        for k, reg in enumerate(regimes):
            q_soft += p_mat[i, k] * qm[experts[reg]]
        out[i] = f42._crps_quantile(actual, q_soft)
    return out


def equal_crps(test_records: list[dict]) -> np.ndarray:
    out = np.zeros(len(test_records))
    for i, r in enumerate(test_records):
        actual = np.asarray(r["actual"], dtype=float)
        qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in MODELS}
        q_eq = sum(qm[m] for m in MODELS) / len(MODELS)
        out[i] = f42._crps_quantile(actual, q_eq)
    return out


def pick_experts_from_labels(
    train_records: list[dict], labels: list[str], regimes: list[str]
) -> dict[str, str]:
    """Per-regime best model on training records (mean CRPS)."""
    recs = [dict(r, regime=lab) for r, lab in zip(train_records, labels, strict=True)]
    return e43.pick_experts(recs, regimes)[0]


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def hand_labels(records: list[dict], dataset: str) -> list[str]:
    return [r["regime"] for r in e43.attach_regimes(records, dataset)]


def run_continuous(
    name: str,
    series: list[tuple[str, np.ndarray]],
    train_window: int,
    horizon: int,
    n_origins: int,
    by_series: bool,
    n_folds: int,
    quick: bool,
) -> dict:
    k = K_MAP[name]
    regimes = REGIME_NAMES[name]
    print(f"[exp45] {name}: generating records...", flush=True)
    records = f42.generate_records(series, train_window, horizon, n_origins, name)
    print(f"[exp45] {name}: {len(records)} records, K={k}", flush=True)

    # hand-rule labels (exp43) for the apples-to-apples comparison
    hand_all = hand_labels(records, name)
    for r, lab in zip(records, hand_all, strict=True):
        r["hand_regime"] = lab

    folds = (
        f42.kfold_by_series(records, n_folds=n_folds)
        if by_series
        else e43.blocked_folds(len(records), n_folds)
    )

    hand_c, gmm_c, km_c, eq_c = [], [], [], []
    hand_experts_all: list[dict] = []
    gmm_experts_all: list[dict] = []
    km_experts_all: list[dict] = []
    sil_gmm, sil_km = [], []
    ari_vs_hand: list[float] = []
    ari_vs_truth: list[float] = []  # synthetic only
    hand_reg_acc: list[float] = []

    for f, (tr, te) in enumerate(folds):
        if name == "food_cpi":
            # replicate exp43's PIT-safe per-fold hand threshold
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
                records[i]["hand_regime"] = e43.label_fcpi(ff, vol_thr=thr)
        tr_recs = [records[i] for i in tr]
        te_recs = [records[i] for i in te]
        x_tr = np.array([r["features"] for r in tr_recs])
        x_te = np.array([r["features"] for r in te_recs])
        eq = equal_crps(te_recs)
        eq_c.extend(eq.tolist())

        # --- hand-rule branch (exp43 replication) ---
        y_hand_tr = [r["hand_regime"] for r in tr_recs]
        hand_regimes = sorted(set(y_hand_tr))
        clf_hand = e43.train_router(x_tr, y_hand_tr, hand_regimes)
        exp_hand = pick_experts_from_labels(tr_recs, y_hand_tr, hand_regimes)
        p_hand = e43.regime_proba(clf_hand, x_te, hand_regimes)
        hand_c.extend(soft_crps(te_recs, p_hand, exp_hand, hand_regimes).tolist())
        hard_hand = [hand_regimes[int(a)] for a in np.argmax(p_hand, axis=1)]
        hand_reg_acc.append(
            float(
                np.mean([h_ == r["hand_regime"] for h_, r in zip(hard_hand, te_recs, strict=True)])
            )
        )
        hand_experts_all.append(exp_hand)

        # --- GMM branch ---
        gmm, scaler = fit_gmm(x_tr, k)
        xs_tr = scaler.transform(x_tr)
        y_gmm_tr = gmm.predict(xs_tr).tolist()
        sil_gmm.append(float(silhouette_score(xs_tr, y_gmm_tr)) if len(set(y_gmm_tr)) > 1 else 0.0)
        exp_gmm = pick_experts_from_labels(tr_recs, y_gmm_tr, regimes)
        p_gmm = gmm.predict_proba(scaler.transform(x_te))
        gmm_c.extend(soft_crps(te_recs, p_gmm, exp_gmm, regimes).tolist())
        gmm_experts_all.append(exp_gmm)
        ari_vs_hand.append(
            float(
                adjusted_rand_score(
                    y_gmm_tr,
                    [hand_regimes.index(y) if y in hand_regimes else -1 for y in y_hand_tr],
                )
            )
        )

        # --- KMeans branch ---
        km, kscaler = fit_kmeans(x_tr, k)
        xs_tr_k = kscaler.transform(x_tr)
        y_km_tr = km.labels_.tolist()
        sil_km.append(float(silhouette_score(xs_tr_k, y_km_tr)) if len(set(y_km_tr)) > 1 else 0.0)
        clf_km = train_router_kmeans(x_tr, km, kscaler, regimes)
        exp_km = pick_experts_from_labels(tr_recs, y_km_tr, regimes)
        p_km = e43.regime_proba(clf_km, x_te, regimes)
        km_c.extend(soft_crps(te_recs, p_km, exp_km, regimes).tolist())
        km_experts_all.append(exp_km)

        if name == "synthetic":
            truth_tr = [e43.label_synth(r["series_id"]) for r in tr_recs]
            truth_idx = [["trending", "seasonal", "rw"].index(t) for t in truth_tr]
            ari_vs_truth.append(float(adjusted_rand_score(y_gmm_tr, truth_idx)))

        print(
            f"[exp45] {name} fold {f}: hand={np.mean(hand_c[-len(te_recs) :]):.4f} "
            f"gmm={np.mean(gmm_c[-len(te_recs) :]):.4f} "
            f"km={np.mean(km_c[-len(te_recs) :]):.4f} "
            f"eq={np.mean(eq[-len(te_recs) :]):.4f} "
            f"sil_gmm={sil_gmm[-1]:.2f} sil_km={sil_km[-1]:.2f}",
            flush=True,
        )

    hand_c_a = np.array(hand_c)
    gmm_c_a = np.array(gmm_c)
    km_c_a = np.array(km_c)
    eq_c_a = np.array(eq_c)

    def vs(a: np.ndarray, b: np.ndarray) -> dict:
        t, p = ttest_rel(a, b)
        return {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a - b)),
            "t_stat": float(t),
            "p_value": float(p),
            "win_rate": float(np.mean(a < b)),
        }

    result = {
        "dataset": name,
        "k": k,
        "n_records": len(records),
        "means": {
            "hand_router": float(np.mean(hand_c_a)),
            "gmm_router": float(np.mean(gmm_c_a)),
            "kmeans_router": float(np.mean(km_c_a)),
            "equal_weights": float(np.mean(eq_c_a)),
        },
        "gmm_vs_equal": vs(gmm_c_a, eq_c_a),
        "gmm_vs_hand": vs(gmm_c_a, hand_c_a),
        "kmeans_vs_equal": vs(km_c_a, eq_c_a),
        "kmeans_vs_hand": vs(km_c_a, hand_c_a),
        "hand_vs_equal": vs(hand_c_a, eq_c_a),
        "silhouette": {"gmm_mean": float(np.mean(sil_gmm)), "kmeans_mean": float(np.mean(sil_km))},
        "ari_vs_hand": {"gmm_mean": float(np.mean(ari_vs_hand))},
        "hand_regime_accuracy": float(np.mean(hand_reg_acc)),
        "hand_experts_per_fold": hand_experts_all,
        "gmm_experts_per_fold": gmm_experts_all,
        "kmeans_experts_per_fold": km_experts_all,
    }
    if ari_vs_truth:
        result["ari_vs_truth"] = {"gmm_mean": float(np.mean(ari_vs_truth))}
    print(
        f"[exp45] {name} POOLED: hand={result['means']['hand_router']:.4f} "
        f"gmm={result['means']['gmm_router']:.4f} "
        f"km={result['means']['kmeans_router']:.4f} "
        f"eq={result['means']['equal_weights']:.4f} "
        f"gmm_vs_hand p={result['gmm_vs_hand']['p_value']:.4g}",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {
        "experiment": "exp45_learned_regimes",
        "date": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "features": FEATURE_NAMES,
        "seed": SEED,
        "method": (
            "Unsupervised regime discovery replaces exp43's hand-written rules. "
            "Per fold (fit on TRAIN records only, PIT-safe): "
            "(a) GMM(K) on standardized features -> posterior P(regime|x) used "
            "directly as the router; "
            "(b) KMeans(K) -> hard clusters -> LogisticRegression router. "
            "Per-regime experts = best model on train records of that regime. "
            "Soft combination: sum_r P(r|x) * expert_r quantiles. "
            "exp43's hand-rule router re-run on the same folds for paired "
            "head-to-head comparison."
        ),
    }
    m4_series = f42.load_m4(n=20 if args.quick else 100, seed=42)
    out["m4"] = run_continuous("m4", m4_series, 60, 6, 6, True, 5, args.quick)
    synth = f42.load_synthetic(n_series=12 if args.quick else 30, length=120)
    out["synthetic"] = run_continuous("synthetic", synth, 60, 1, 4, True, 5, args.quick)
    smm = f42.load_smm()
    out["smm"] = run_continuous("smm", smm, 12, 1, 50, False, 5, args.quick)
    fcpi = f42.load_food_cpi()
    out["food_cpi"] = run_continuous("food_cpi", fcpi, 60, 1, 24, False, 4, args.quick)

    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp45_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[exp45] wrote {out_path} ({out['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
