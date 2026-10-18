"""exp44: News-as-selector -- use news to choose which expert to trust.

Motivation: exp19/21/25 proved news fails as *features* (it does not improve
the forecast value). The literature says news works as *selection signals*:
use news to decide WHICH model to trust, not what the value should be.

This experiment takes exp43's regime-probability router and asks: do
FORWARD-looking news signals improve regime classification over
backward-looking features alone? The target is regime accuracy, not CRPS
directly -- if news cannot even predict regimes, it is useless for monthly
forecasting.

PIT safety: for origin date D, only articles with pub_date in [D-30d, D)
are used -- strictly before D, so same-day decision reporting cannot leak.

Datasets: BoC (EASY/HARD, real news), SMM (calm/stressed, real news),
synthetic (event simulator + synthetic news feed, true latent state known).
M4 has no news and is skipped.

Success criterion: news-augmented classifier beats backward-only classifier
on regime accuracy (paired test, p < 0.05) on at least one dataset.

Usage:
    cd td-one && .venv/bin/python experiments/exp44_news_selector.py [--quick]
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp42_fforma as f42  # noqa: E402
import exp43_regime_router as r43  # noqa: E402

from td_one.data_gen.events import (  # noqa: E402
    active_effect_shift,
    generate_event_timeline,
)
from td_one.data_gen.news_synthesis import generate_news_feed  # noqa: E402

warnings.filterwarnings("ignore")

NEWS_DB = ROOT / "data" / "news" / "news_2026.db"
NEWS_WINDOW_DAYS = 30

# ---------------------------------------------------------------------------
# Keyword sets (simple counts, no LLM -- deterministic and PIT-safe)
# ---------------------------------------------------------------------------

HAWK_KW = (
    "hike",
    "hikes",
    "hiked",
    "hiking",
    "raise",
    "raised",
    "raising",
    "raises",
    "tighten",
    "tightening",
    "tightens",
    "increase",
    "increases",
    "increasing",
    "hawkish",
    "restrictive",
)
DOVE_KW = (
    "cut",
    "cuts",
    "cutting",
    "lower",
    "lowered",
    "lowering",
    "lowers",
    "ease",
    "easing",
    "eases",
    "decrease",
    "decreases",
    "decreasing",
    "dovish",
    "hold",
    "holds",
    "holding",
    "pause",
    "pauses",
)
RATE_KW = (
    "rate",
    "rates",
    "mortgage",
    "mortgages",
    "refinance",
    "bank of canada",
    "federal reserve",
    "fed",
)
STRESS_KW = (
    "surge",
    "surging",
    "plunge",
    "plunging",
    "volatile",
    "volatility",
    "crisis",
    "shock",
    "shocks",
    "turmoil",
    "spike",
    "spiking",
    "collapse",
    "uncertain",
    "uncertainty",
)
EVENT_KW = (  # synthetic news templates use these direction words
    "tightens",
    "tightening",
    "eases",
    "easing",
    "hike",
    "hiked",
    "cut",
    "fall",
    "falling",
    "rise",
    "rising",
    "push",
    "pushing",
)


def _count_kw(titles: list[str], kws: tuple[str, ...]) -> int:
    return sum(1 for t in titles if any(k in t for k in kws))


def query_titles(start: date, end: date) -> list[str]:
    """Return article titles with start <= pub_date < end (PIT-safe)."""
    con = sqlite3.connect(NEWS_DB)
    try:
        rows = con.execute(
            "SELECT title FROM articles WHERE pub_date >= ? AND pub_date < ?",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        con.close()
    return [r[0].lower() for r in rows]


def news_features_boc(meeting: date) -> list[float]:
    """News features for a BoC meeting: 30d window strictly before meeting."""
    start = meeting - timedelta(days=NEWS_WINDOW_DAYS)
    titles = query_titles(start, meeting)
    n = len(titles)
    hawk = _count_kw(titles, HAWK_KW)
    dove = _count_kw(titles, DOVE_KW)
    return [
        float(n),
        float(hawk),
        float(dove),
        1.0 if (hawk > 0 and dove > 0) else 0.0,  # disagreement
        float(hawk - dove) / float(n + 1),  # net tone
    ]


NEWS_BOC_NAMES = ["news_n", "news_hawk", "news_dove", "news_disagree", "news_net"]


def news_features_smm(origin_day: date) -> list[float]:
    """News features for an SMM origin: 30d window strictly before origin."""
    start = origin_day - timedelta(days=NEWS_WINDOW_DAYS)
    titles = query_titles(start, origin_day)
    n = len(titles)
    rate = _count_kw(titles, RATE_KW)
    stress = _count_kw(titles, STRESS_KW)
    return [
        float(n),
        float(rate),
        float(stress),
        float(stress) / float(n + 1),  # stress share
    ]


NEWS_SMM_NAMES = ["news_n", "news_rate", "news_stress", "news_stress_share"]


# ---------------------------------------------------------------------------
# Classifier comparison helper
# ---------------------------------------------------------------------------


def compare_classifiers(
    x_base: np.ndarray,
    x_news: np.ndarray,
    y: list[str],
    regimes: list[str],
    folds: list[tuple[list[int], list[int]]],
) -> dict:
    """Train LogisticRegression on base vs base+news features; compare accuracy.

    Returns per-fold accuracies and a paired t-test on correctness indicators.
    """
    y_idx = np.array([regimes.index(v) for v in y])
    acc_base_folds, acc_news_folds = [], []
    correct_base_all, correct_news_all = [], []
    skipped_folds = 0
    for tr, te in folds:
        if len(set(y_idx[tr].tolist())) < 2:
            # Cannot train a classifier on a single class; skip fold.
            skipped_folds += 1
            continue
        for x_mat, acc_folds, correct_all in (
            (x_base, acc_base_folds, correct_base_all),
            (x_news, acc_news_folds, correct_news_all),
        ):
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_mat[tr])
            xte = scaler.transform(x_mat[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(xtr, y_idx[tr])
            pred = clf.predict(xte)
            correct = (pred == y_idx[te]).astype(float)
            acc_folds.append(float(np.mean(correct)))
            correct_all.extend(correct.tolist())
    cb = np.array(correct_base_all)
    cn = np.array(correct_news_all)
    if len(cb) == 0:
        return {
            "n": len(y),
            "acc_base_mean": float("nan"),
            "acc_news_mean": float("nan"),
            "acc_base_folds": acc_base_folds,
            "acc_news_folds": acc_news_folds,
            "mean_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "win_rate": float("nan"),
            "skipped_folds": skipped_folds,
            "note": "all folds skipped (single-class train)",
        }
    diff = cn - cb
    t_stat, p_val = ttest_rel(cn, cb)
    return {
        "n": len(y),
        "acc_base_mean": float(np.mean(acc_base_folds)),
        "acc_news_mean": float(np.mean(acc_news_folds)),
        "acc_base_folds": acc_base_folds,
        "acc_news_folds": acc_news_folds,
        "mean_diff": float(np.mean(diff)),  # positive = news better
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "win_rate": float(np.mean(cn > cb)),
        "skipped_folds": skipped_folds,
    }


# ---------------------------------------------------------------------------
# BoC: does news predict EASY/HARD?
# ---------------------------------------------------------------------------


def run_boc_news(quick: bool = False) -> dict:
    print("[exp44] BoC: loading meetings + news features...", flush=True)
    records = r43.load_boc()
    if quick:
        records = records[-30:]
    print(f"[exp44] BoC: {len(records)} meetings", flush=True)
    x_base, x_news, y = [], [], []
    news_nz = 0
    for m in records:
        md = date.fromisoformat(m["meeting_date"])
        nf = news_features_boc(md)
        if nf[0] > 0:
            news_nz += 1
        x_base.append(m["features"])
        x_news.append(list(m["features"]) + nf)
        y.append(m["regime"])
    print(
        f"[exp44] BoC: {news_nz}/{len(records)} meetings have >0 news in window",
        flush=True,
    )
    x_base = np.array(x_base)
    x_news = np.array(x_news)
    folds = r43.blocked_folds(len(records), 5)
    cmp = compare_classifiers(x_base, x_news, y, r43.BOC_REGIMES, folds)

    # Also: does news predict the actual decision surprise? (HARD meetings are
    # where the rule fails; news disagreement ~ uncertainty.)
    # Secondary: news-augmented router accuracy vs macro-only router.
    return {
        "dataset": "boc",
        "regimes": r43.BOC_REGIMES,
        "base_features": [
            "rate_momentum",
            "inflation_gap",
            "unemployment_momentum",
            "yield_spread",
        ],
        "news_features": NEWS_BOC_NAMES,
        "meetings_with_news": news_nz,
        "regime_classification": cmp,
    }


# ---------------------------------------------------------------------------
# SMM: does news predict calm/stressed?
# ---------------------------------------------------------------------------


def _smm_origin_date(origin_idx: int) -> date:
    """Map series index -> first day of that month. Series starts 2021-01."""
    y = 2021 + (origin_idx // 12)
    m = 1 + (origin_idx % 12)
    return date(y, m, 1)


def run_smm_news(quick: bool = False) -> dict:
    print("[exp44] SMM: generating records + news features...", flush=True)
    series = f42.load_smm()
    n_origins = 12 if quick else 50
    records = r43.attach_regimes(f42.generate_records(series, 12, 1, n_origins, "smm"), "smm")
    print(f"[exp44] SMM: {len(records)} records", flush=True)
    x_base, x_news, y = [], [], []
    news_nz = 0
    for r in records:
        oday = _smm_origin_date(int(r["origin"]))
        nf = news_features_smm(oday)
        if nf[0] > 0:
            news_nz += 1
        x_base.append(r["features"])
        x_news.append(list(r["features"]) + nf)
        y.append(r["regime"])
    print(
        f"[exp44] SMM: {news_nz}/{len(records)} origins have >0 news in window",
        flush=True,
    )
    x_base = np.array(x_base)
    x_news = np.array(x_news)
    folds = r43.blocked_folds(len(records), 5)
    cmp = compare_classifiers(x_base, x_news, y, r43.SMM_REGIMES, folds)

    # Full router CRPS: does better regime prediction translate to better forecasts?
    router_cmp = _router_crps_comparison(records, x_base, x_news, y, r43.SMM_REGIMES, folds)
    return {
        "dataset": "smm",
        "regimes": r43.SMM_REGIMES,
        "base_features": f42.FEATURE_NAMES,
        "news_features": NEWS_SMM_NAMES,
        "origins_with_news": news_nz,
        "regime_classification": cmp,
        "router_crps": router_cmp,
    }


def _router_crps_comparison(
    records: list[dict],
    x_base: np.ndarray,
    x_news: np.ndarray,
    y: list[str],
    regimes: list[str],
    folds: list[tuple[list[int], list[int]]],
) -> dict:
    """Run exp43's soft router with base vs base+news classifier features.

    Experts are picked per regime on train records (same for both); only the
    classifier input differs. Compares test CRPS.
    """
    crps_base_all, crps_news_all = [], []
    skipped = 0
    for tr, te in folds:
        tr_recs = [records[i] for i in tr]
        te_recs = [records[i] for i in te]
        experts, _ = r43.pick_experts(tr_recs, regimes)
        y_idx_tr = np.array([regimes.index(v) for v in y])
        if len(set(y_idx_tr[tr].tolist())) < 2:
            skipped += 1
            continue
        for x_mat, store in ((x_base, crps_base_all), (x_news, crps_news_all)):
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_mat[tr])
            xte = scaler.transform(x_mat[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(xtr, y_idx_tr[tr])
            p_mat = r43.regime_proba(clf, xte, regimes)
            for i, r in enumerate(te_recs):
                actual = np.asarray(r["actual"], dtype=float)
                qm = {m: np.asarray(r["quantiles"][m], dtype=float) for m in f42.MODELS}
                q_soft = np.zeros((9, actual.shape[0]))
                for k, reg in enumerate(regimes):
                    q_soft += p_mat[i, k] * qm[experts[reg]]
                store.append(float(f42._crps_quantile(actual, q_soft)))
    cb = np.array(crps_base_all)
    cn = np.array(crps_news_all)
    if len(cb) == 0:
        return {
            "n": 0,
            "mean_crps_base": float("nan"),
            "mean_crps_news": float("nan"),
            "mean_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "win_rate": float("nan"),
            "skipped_folds": skipped,
            "note": "all folds skipped (single-class train)",
        }
    diff = cn - cb  # negative = news better
    t_stat, p_val = ttest_rel(cn, cb)
    return {
        "n": len(cb),
        "mean_crps_base": float(np.mean(cb)),
        "mean_crps_news": float(np.mean(cn)),
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "win_rate": float(np.mean(cn < cb)),
        "skipped_folds": skipped,
    }


# ---------------------------------------------------------------------------
# Synthetic: event simulator + synthetic news feed, true latent state known
# ---------------------------------------------------------------------------


def _month_start(year: int, month_idx0: int) -> date:
    y = year + month_idx0 // 12
    m = 1 + month_idx0 % 12
    return date(y, m, 1)


def run_synth_news(quick: bool = False) -> dict:
    print("[exp44] Synthetic: generating events + series + news feed...", flush=True)
    n_months = 36 if quick else 60
    start = date(2020, 1, 31)
    events = generate_event_timeline(start, n_months, seed=7, event_rate_per_year=2.0)
    articles = generate_news_feed(events, seed=8)
    # strip eval-only linkage
    articles = [
        {"article_date": a["article_date"], "headline": str(a["headline"])} for a in articles
    ]
    print(
        f"[exp44] Synthetic: {len(events)} events, {len(articles)} articles",
        flush=True,
    )

    # Series: AR(1) + event shifts (PIT-safe: shift at month t uses events
    # visible at t, i.e. event_date <= month-end t).
    rng = np.random.default_rng(11)
    x = np.zeros(n_months)
    x[0] = rng.normal(0, 1)
    effect_at = np.zeros(n_months)
    for t in range(n_months):
        # month-end asof for effect lookup
        asof_end = _month_start(2020, t + 1) - timedelta(days=1)
        vis = [e for e in events if e.event_date <= asof_end]
        effect_at[t] = active_effect_shift(vis, asof_end)
    for t in range(1, n_months):
        x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.3) + effect_at[t]

    # Origins: need 12-month train window; predict 1 month ahead.
    train_window, horizon = 12, 1
    origins = list(range(train_window, n_months - horizon + 1))
    x_base, x_news, y_back, y_true = [], [], [], []
    news_nz = 0
    for o in origins:
        train_raw = x[o - train_window : o]
        actual_raw = x[o : o + horizon]
        train, actual = f42._normalize(train_raw, actual_raw)
        feats = f42.extract_features(train)
        # backward-label regime (exp43 style): stressed if recent vol high
        fdict = dict(zip(f42.FEATURE_NAMES, feats, strict=True))
        y_back.append(
            "stressed" if (fdict["n_recent_cps"] >= 1 or fdict["vol_ratio"] > 1.0) else "calm"
        )
        # true latent state: did an event occur recently enough that news about
        # it could have been published before the origin? (event in last 90d).
        # This is the state news can plausibly signal; backward features may
        # or may not reflect it yet.
        oday = _month_start(2020, o)
        recent_ev = any(oday - timedelta(days=90) <= e.event_date < oday for e in events)
        y_true.append("stressed" if recent_ev else "calm")
        # news features: articles strictly before origin month start
        window_titles = [
            a["headline"].lower()
            for a in articles
            if oday - timedelta(days=NEWS_WINDOW_DAYS) <= a["article_date"] < oday
        ]
        n = len(window_titles)
        if n > 0:
            news_nz += 1
        ev = _count_kw(window_titles, EVENT_KW)
        nf = [float(n), float(ev), float(ev) / float(n + 1)]
        x_base.append(feats)
        x_news.append(list(feats) + nf)
    print(
        f"[exp44] Synthetic: {len(origins)} origins, "
        f"{news_nz} with >0 news; true stressed share="
        f"{sum(1 for v in y_true if v == 'stressed')}/{len(y_true)}",
        flush=True,
    )
    x_base = np.array(x_base)
    x_news = np.array(x_news)
    folds = r43.blocked_folds(len(origins), 5)
    cmp_back = compare_classifiers(x_base, x_news, y_back, ["calm", "stressed"], folds)
    cmp_true = compare_classifiers(x_base, x_news, y_true, ["calm", "stressed"], folds)
    return {
        "dataset": "synthetic",
        "regimes": ["calm", "stressed"],
        "base_features": f42.FEATURE_NAMES,
        "news_features": ["news_n", "news_event_kw", "news_event_share"],
        "n_events": len(events),
        "n_articles": len(articles),
        "origins_with_news": news_nz,
        "regime_classification_backward_labels": cmp_back,
        "regime_classification_true_labels": cmp_true,
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
        "experiment": "exp44_news_selector",
        "date": datetime.now(UTC).isoformat(),
        "news_db": str(NEWS_DB),
        "news_window_days": NEWS_WINDOW_DAYS,
        "pit_rule": "articles with pub_date in [D-30d, D) only; strictly before origin",
        "method": (
            "News-as-selector: LogisticRegression maps backward features vs "
            "backward+news features -> P(regime). Compares regime classification "
            "accuracy (primary) and, for SMM, full router CRPS (secondary). "
            "News never touches the forecast value -- only the regime weights."
        ),
    }
    if not NEWS_DB.exists():
        print(f"[exp44] ERROR: news DB not found at {NEWS_DB}", flush=True)
        return 1

    out["boc"] = run_boc_news(args.quick)
    out["smm"] = run_smm_news(args.quick)
    out["synthetic"] = run_synth_news(args.quick)

    out["elapsed_s"] = round(time.time() - t0, 1)
    out_path = ROOT / "experiments" / "exp44_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[exp44] wrote {out_path} ({out['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
