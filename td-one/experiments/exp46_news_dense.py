"""exp46: News-as-selector with DENSE news on real data.

Motivation: exp44 proved the news-as-selector mechanism on synthetic
(p=0.0068) but was blocked on real data by sparsity (47/143 BoC meetings
had news). The Perplexity backfill raised coverage to 123/143 (86%).
This experiment re-tests on real data with dense news.

Key design improvements over exp44:
1. PRIMARY target for BoC is rule-surprise (rule != actual), not EASY/HARD.
   EASY/HARD is a deterministic function of macro features, so predicting it
   from the same features is circular; surprise is what we actually care
   about (can news tell us when the rule will fail?).
2. News source variants to handle the temporal confound: raw article counts
   correlate with calendar year (r=0.42) because 2025-2026 has a general-news
   flood. We test all-sources vs perplexity_boc-only vs rate-keyword-filtered.
3. SMM: forecast-difficulty target (top-tercile |ETS error|) in addition to
   the calm/stressed regime replication, since backward features already hit
   96% on regime classification (nothing left for news to add).
4. Feature importance: standardized logistic coefficients for news features,
   and news_net correlation with decision direction.

PIT safety: for origin date D, only articles with pub_date in [D-30d, D)
are used -- strictly before D.

Usage:
    cd td-one && .venv/bin/python experiments/exp46_news_dense.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402
import exp43_regime_router as r43  # noqa: E402
from exp44_news_selector import (  # noqa: E402
    DOVE_KW,
    HAWK_KW,
    NEWS_BOC_NAMES,
    NEWS_SMM_NAMES,
    RATE_KW,
    STRESS_KW,
    _count_kw,
)

warnings.filterwarnings("ignore")

NEWS_DB = ROOT / "data" / "news" / "news_2026.db"
NEWS_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# News querying with source / keyword filters
# ---------------------------------------------------------------------------


def query_titles_filtered(
    start: date, end: date, source: str | None = None, require_kw: tuple[str, ...] | None = None
) -> list[str]:
    """Return article titles with start <= pub_date < end (PIT-safe).

    source: restrict to a source value (e.g. 'perplexity_boc'), or None for all.
    require_kw: keep only titles containing at least one keyword (case-insens).
    """
    con = sqlite3.connect(NEWS_DB)
    try:
        if source is None:
            rows = con.execute(
                "SELECT title FROM articles WHERE pub_date >= ? AND pub_date < ?",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT title FROM articles WHERE pub_date >= ? AND pub_date < ? AND source = ?",
                (start.isoformat(), end.isoformat(), source),
            ).fetchall()
    finally:
        con.close()
    titles = [r[0].lower() for r in rows]
    if require_kw is not None:
        titles = [t for t in titles if any(k in t for k in require_kw)]
    return titles


def news_features_boc_from_titles(titles: list[str]) -> list[float]:
    n = len(titles)
    hawk = _count_kw(titles, HAWK_KW)
    dove = _count_kw(titles, DOVE_KW)
    return [
        float(n),
        float(hawk),
        float(dove),
        1.0 if (hawk > 0 and dove > 0) else 0.0,
        float(hawk - dove) / float(n + 1),
    ]


def news_features_smm_from_titles(titles: list[str]) -> list[float]:
    n = len(titles)
    rate = _count_kw(titles, RATE_KW)
    stress = _count_kw(titles, STRESS_KW)
    return [float(n), float(rate), float(stress), float(stress) / float(n + 1)]


def coverage_report(meetings: list[date]) -> dict:
    """How many meetings have news under each variant."""
    cov = {"all": 0, "boc_only": 0, "rate_kw": 0}
    for md in meetings:
        start = md - timedelta(days=NEWS_WINDOW_DAYS)
        if query_titles_filtered(start, md):
            cov["all"] += 1
        if query_titles_filtered(start, md, source="perplexity_boc"):
            cov["boc_only"] += 1
        if query_titles_filtered(start, md, require_kw=RATE_KW):
            cov["rate_kw"] += 1
    cov["n"] = len(meetings)
    return cov


# ---------------------------------------------------------------------------
# Classifier comparison with class imbalance handling
# ---------------------------------------------------------------------------


def compare_classifiers_v2(
    feature_sets: dict[str, np.ndarray],
    y: list[int],
    folds: list[tuple[list[int], list[int]]],
    labels: tuple[str, str] = ("neg", "pos"),
) -> dict:
    """Compare multiple feature sets on binary classification.

    Trains LogisticRegression(class_weight='balanced') per fold, reports
    accuracy and AUC per feature set, plus paired t-tests of correctness
    indicators vs the 'base' feature set.
    """
    names = list(feature_sets.keys())
    y_arr = np.array(y)
    acc: dict[str, list[float]] = {n: [] for n in names}
    auc: dict[str, list[float]] = {n: [] for n in names}
    correct: dict[str, list[float]] = {n: [] for n in names}
    skipped = 0
    for tr, te in folds:
        if len(set(y_arr[tr].tolist())) < 2 or len(set(y_arr[te].tolist())) < 2:
            skipped += 1
            continue
        for n in names:
            x = feature_sets[n]
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x[tr])
            xte = scaler.transform(x[te])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(xtr, y_arr[tr])
            pred = clf.predict(xte)
            proba = clf.predict_proba(xte)[:, 1]
            c = (pred == y_arr[te]).astype(float)
            acc[n].append(float(np.mean(c)))
            correct[n].extend(c.tolist())
            try:
                auc[n].append(float(roc_auc_score(y_arr[te], proba)))
            except ValueError:
                auc[n].append(float("nan"))
    out: dict = {
        "n": len(y),
        "labels": list(labels),
        "pos_rate": float(np.mean(y_arr)),
        "skipped_folds": skipped,
        "per_set": {},
    }
    base_c = np.array(correct["base"]) if "base" in correct and correct["base"] else None
    for n in names:
        d: dict = {
            "acc_mean": float(np.mean(acc[n])) if acc[n] else float("nan"),
            "auc_mean": float(np.nanmean(auc[n])) if auc[n] else float("nan"),
            "acc_folds": acc[n],
        }
        if base_c is not None and n != "base" and correct[n]:
            cn = np.array(correct[n])
            diff = cn - base_c
            t_stat, p_val = ttest_rel(cn, base_c)
            d["vs_base"] = {
                "mean_diff": float(np.mean(diff)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "win_rate": float(np.mean(cn > base_c)),
            }
        out["per_set"][n] = d
    return out


def feature_importance(x: np.ndarray, y: list[int], feature_names: list[str]) -> list[dict]:
    """Standardized logistic coefficients fit on all data (descriptive)."""
    y_arr = np.array(y)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(xs, y_arr)
    coefs = clf.coef_[0]
    order = np.argsort(-np.abs(coefs))
    return [{"feature": feature_names[i], "coef": float(coefs[i])} for i in order]


# ---------------------------------------------------------------------------
# exp46a: BoC rule-surprise prediction (primary)
# ---------------------------------------------------------------------------


def run_boc_surprise(quick: bool = False) -> dict:
    print("[exp46] BoC surprise: loading meetings...", flush=True)
    records = r43.load_boc()
    if quick:
        records = records[-40:]
    meetings = [date.fromisoformat(m["meeting_date"]) for m in records]
    cov = coverage_report(meetings)
    print(f"[exp46] BoC coverage: {cov}", flush=True)

    y = [1 if m["policies"]["rule"] != m["actual"] else 0 for m in records]
    print(
        f"[exp46] BoC: {len(records)} meetings, surprise rate {sum(y)}/{len(y)}={np.mean(y):.3f}",
        flush=True,
    )
    x_base = np.array([m["features"] for m in records])
    fsets: dict[str, np.ndarray] = {"base": x_base}
    # news variants
    variants = {
        "news_all": (None, None),
        "news_boc": ("perplexity_boc", None),
        "news_rate": (None, RATE_KW),
    }
    for vname, (src, kw) in variants.items():
        xn = []
        for md in meetings:
            start = md - timedelta(days=NEWS_WINDOW_DAYS)
            titles = query_titles_filtered(start, md, source=src, require_kw=kw)
            nf = news_features_boc_from_titles(titles)
            xn.append(list(x_base[meetings.index(md)]) + nf)
        fsets[vname] = np.array(xn)
    # news-only (boc source) to test standalone signal
    xn_only = []
    for md in meetings:
        start = md - timedelta(days=NEWS_WINDOW_DAYS)
        titles = query_titles_filtered(start, md, source="perplexity_boc")
        xn_only.append(news_features_boc_from_titles(titles))
    fsets["news_only"] = np.array(xn_only)

    folds = r43.blocked_folds(len(records), 5)
    cmp = compare_classifiers_v2(fsets, y, folds, labels=("no_surprise", "surprise"))

    # feature importance for base+news_boc
    feat_names = ["rate_momentum", "inflation_gap", "unemployment_momentum", "yield_spread"]
    imp = feature_importance(fsets["news_boc"], y, feat_names + NEWS_BOC_NAMES)
    return {
        "dataset": "boc",
        "target": "rule_surprise (rule != actual)",
        "coverage": cov,
        "base_features": feat_names,
        "news_features": NEWS_BOC_NAMES,
        "classification": cmp,
        "feature_importance_news_boc": imp,
    }


# ---------------------------------------------------------------------------
# exp46b: BoC EASY/HARD replication (exp44 with dense news)
# ---------------------------------------------------------------------------


def run_boc_regime(quick: bool = False) -> dict:
    print("[exp46] BoC regime replication...", flush=True)
    records = r43.load_boc()
    if quick:
        records = records[-40:]
    meetings = [date.fromisoformat(m["meeting_date"]) for m in records]
    y = [1 if m["regime"] == "HARD" else 0 for m in records]
    x_base = np.array([m["features"] for m in records])
    fsets: dict[str, np.ndarray] = {"base": x_base}
    for vname, (src, kw) in {
        "news_all": (None, None),
        "news_boc": ("perplexity_boc", None),
    }.items():
        xn = []
        for md in meetings:
            start = md - timedelta(days=NEWS_WINDOW_DAYS)
            titles = query_titles_filtered(start, md, source=src, require_kw=kw)
            xn.append(list(x_base[meetings.index(md)]) + news_features_boc_from_titles(titles))
        fsets[vname] = np.array(xn)
    folds = r43.blocked_folds(len(records), 5)
    cmp = compare_classifiers_v2(fsets, y, folds, labels=("EASY", "HARD"))
    return {
        "dataset": "boc",
        "target": "regime (EASY/HARD) -- exp44 replication with dense news",
        "classification": cmp,
    }


# ---------------------------------------------------------------------------
# exp46c/d: SMM forecast difficulty + regime replication
# ---------------------------------------------------------------------------


def _smm_origin_date(origin_idx: int) -> date:
    y = 2021 + (origin_idx // 12)
    m = 1 + (origin_idx % 12)
    return date(y, m, 1)


def run_smm(quick: bool = False) -> dict:
    print("[exp46] SMM: generating records...", flush=True)
    series = f42.load_smm()
    n_origins = 12 if quick else 50
    records = f42.generate_records(series, 12, 1, n_origins, "smm")
    print(f"[exp46] SMM: {len(records)} records", flush=True)
    origins = [_smm_origin_date(int(r["origin"])) for r in records]
    cov = coverage_report(origins)
    print(f"[exp46] SMM origin coverage: {cov}", flush=True)

    # forecast difficulty: top-tercile |ETS q50 error|
    ets_err = []
    for r in records:
        q = np.asarray(r["quantiles"]["ets"], dtype=float)  # (9, H)
        actual = np.asarray(r["actual"], dtype=float)
        ets_err.append(float(abs(q[4, 0] - actual[0])))
    ets_err = np.array(ets_err)
    thresh = float(np.quantile(ets_err, 2.0 / 3.0))
    y_hard = [1 if e >= thresh else 0 for e in ets_err]

    x_base = np.array([r["features"] for r in records])
    fsets: dict[str, np.ndarray] = {"base": x_base}
    for vname, (src, kw) in {
        "news_all": (None, None),
        "news_rate": (None, RATE_KW),
    }.items():
        xn = []
        for od in origins:
            start = od - timedelta(days=NEWS_WINDOW_DAYS)
            titles = query_titles_filtered(start, od, source=src, require_kw=kw)
            nf = news_features_smm_from_titles(titles)
            xn.append(list(x_base[origins.index(od)]) + nf)
        fsets[vname] = np.array(xn)

    folds = r43.blocked_folds(len(records), 5)
    cmp_hard = compare_classifiers_v2(fsets, y_hard, folds, labels=("easy", "hard"))

    # regime replication (calm/stressed, exp44-style)
    recs_reg = r43.attach_regimes([dict(r) for r in records], "smm")
    y_reg = [1 if r["regime"] == "stressed" else 0 for r in recs_reg]
    cmp_reg = compare_classifiers_v2(fsets, y_reg, folds, labels=("calm", "stressed"))

    imp = feature_importance(fsets["news_all"], y_hard, f42.FEATURE_NAMES + NEWS_SMM_NAMES)
    return {
        "dataset": "smm",
        "coverage": cov,
        "ets_error_tercile_threshold": thresh,
        "hard_rate": float(np.mean(y_hard)),
        "target_hard": "top-tercile |ETS q50 error|",
        "classification_hard": cmp_hard,
        "target_regime": "calm/stressed -- exp44 replication",
        "classification_regime": cmp_reg,
        "feature_importance_hard_news_all": imp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run for smoke test")
    args = ap.parse_args()
    t0 = time.time()
    out: dict = {
        "experiment": "exp46_news_dense",
        "date": datetime.now(UTC).isoformat(),
        "news_db": str(NEWS_DB),
        "news_window_days": NEWS_WINDOW_DAYS,
        "pit_rule": "articles with pub_date in [D-30d, D) only; strictly before origin",
        "method": (
            "News-as-selector with DENSE news (123/143 BoC meetings covered). "
            "Primary BoC target is rule-surprise (rule != actual), not EASY/HARD. "
            "News variants handle the temporal confound (all vs boc-only vs "
            "rate-keyword-filtered). SMM adds a forecast-difficulty target."
        ),
    }
    if not NEWS_DB.exists():
        print(f"[exp46] ERROR: news DB not found at {NEWS_DB}", flush=True)
        return 1

    out["boc_surprise"] = run_boc_surprise(args.quick)
    out["boc_regime"] = run_boc_regime(args.quick)
    out["smm"] = run_smm(args.quick)
    out["elapsed_s"] = round(time.time() - t0, 1)

    out_path = ROOT / "experiments" / "exp46_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[exp46] wrote {out_path} ({out['elapsed_s']}s)", flush=True)

    # console summary
    for sec in ("boc_surprise", "boc_regime"):
        c = out[sec]["classification"]["per_set"]
        print(f"[exp46] {sec}:")
        for n, d in c.items():
            vs = d.get("vs_base", {})
            print(
                f"    {n:10s} acc={d['acc_mean']:.3f} auc={d['auc_mean']:.3f}"
                + (
                    f"  vs_base: d={vs.get('mean_diff', float('nan')):+.3f} "
                    f"p={vs.get('p_value', float('nan')):.3f}"
                    if vs
                    else ""
                )
            )
    for tgt in ("classification_hard", "classification_regime"):
        c = out["smm"][tgt]["per_set"]
        print(f"[exp46] smm/{tgt}:")
        for n, d in c.items():
            vs = d.get("vs_base", {})
            print(
                f"    {n:10s} acc={d['acc_mean']:.3f} auc={d['auc_mean']:.3f}"
                + (
                    f"  vs_base: d={vs.get('mean_diff', float('nan')):+.3f} "
                    f"p={vs.get('p_value', float('nan')):.3f}"
                    if vs
                    else ""
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
