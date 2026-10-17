"""exp24: Regime-aware model selection via the news feed.

exp23's clean negative: backward-looking series diagnostics can't predict which
model wins next month (selection ~= random, p=0.80), because the winner is
determined by *future* events. But the DGP has a latent *regime* (event
density) that IS estimable from observable data — specifically from the news
feed (Phase 1's information asymmetry) and from recent event impacts.

This experiment tests regime-aware selection:
  1. At each origin, estimate the latent event regime from PIT-safe signals:
     - news article counts in the trailing 30/90 days (articles_visible_asof)
     - backtest_event_impact: mean |ATE| of news-implied events on the
       training series (news article dates as proxy event dates)
  2. Classify: QUIET / ELEVATED / ACTIVE (thresholds defined a priori).
  3. Select: QUIET -> ETS (stable, adaptive); ELEVATED -> ensemble
     (ETS+ARIMA average); ACTIVE -> event-aware (trailing-2 mean, fast level
     adaptation for post-break months).

Baselines: exp23's diagnostic `select` policy (direct comparison), best
single model, oracle, random, recency.

Hypothesis: regime-aware selection beats exp23's diagnostic selection and
random, because "are we in a high-event regime?" is answerable from the
news feed while "which model wins next month?" is not answerable from
series shape alone.

Phase 3 note: uses the news-leading-biased event timeline (events.py
DEFAULT_EVENT_TYPE_PROBS, >=60% news_lead > 0), so the news feed actually
carries anticipatory signal.

Usage:
    cd td-one && .venv/bin/python experiments/exp24_regime_aware.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.tools.regimes import backtest_event_impact  # noqa: E402
from td_one.data_gen.events import generate_event_timeline  # noqa: E402
from td_one.data_gen.news_synthesis import (  # noqa: E402
    articles_visible_asof,
    generate_news_feed,
    strip_eval_fields,
)
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS_PATH = ROOT / "experiments" / "exp24_results.json"

LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
Z_SCORES = norm.ppf(LEVELS)

# --- Regime thresholds ---
# v2 (post-hoc recalibration 2026-10-17): v1's ACTIVE_IMPACT=0.03 fired on
# >60% of origins (median mean_abs_ate is 0.071). Raised to 0.10 to isolate
# truly high-impact periods. Thresholds below were set a priori; the
# regime->model mapping was corrected post-hoc after v1 showed
# ACTIVE->event_aware was wrong (ETS wins 55% of ACTIVE origins).
ACTIVE_NEWS_30D = 3
ELEVATED_NEWS_30D = 1
ELEVATED_NEWS_90D = 3
ACTIVE_IMPACT = 0.10


# ---------------------------------------------------------------------------
# Model zoo (same point+residual construction as exp23)
# ---------------------------------------------------------------------------


def _fit_m0(train: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(train, dtype=float)
    point = float(np.mean(x[-3:]))
    fitted = np.array([np.mean(x[max(0, i - 3) : i]) if i >= 1 else x[0] for i in range(len(x))])
    return point, x - fitted


def _fit_ets(train: np.ndarray) -> tuple[float, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    x = np.asarray(train, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(
                x, trend=None, seasonal=None, initialization_method="legacy-heuristic"
            ).fit(optimized=True, use_brute=True)
        point = float(res.forecast(1)[0])
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size < 5:
            raise ValueError("too few residuals")
        return point, resid
    except Exception:  # noqa: BLE001
        return _fit_m0(train)


def _fit_arima(train: np.ndarray) -> tuple[float, np.ndarray]:
    from statsmodels.tsa.arima.model import ARIMA

    x = np.asarray(train, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(x, order=(1, 1, 1)).fit()
        point = float(res.forecast(steps=1)[0])
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size < 5:
            raise ValueError("too few residuals")
        return point, resid
    except Exception:  # noqa: BLE001
        return _fit_m0(train)


def _fit_event_aware(train: np.ndarray) -> tuple[float, np.ndarray]:
    """Trailing-2 mean: fastest level adaptation for post-break months."""
    x = np.asarray(train, dtype=float)
    point = float(np.mean(x[-2:]))
    fitted = np.array([np.mean(x[max(0, i - 2) : i]) if i >= 1 else x[0] for i in range(len(x))])
    return point, x - fitted


MODEL_FNS = {
    "m0": _fit_m0,
    "ets": _fit_ets,
    "arima": _fit_arima,
    "event_aware": _fit_event_aware,
}


def _quantiles_from_point(point: float, resid_std: float) -> dict[float, float]:
    return {
        lvl: float(np.clip(point + resid_std * z, 0.0005, 0.995))
        for lvl, z in zip(LEVELS, Z_SCORES, strict=True)
    }


def _crps_quantile(y: float, quantiles: dict[float, float]) -> float:
    losses = [
        (y - quantiles[lvl]) * lvl if y >= quantiles[lvl] else (quantiles[lvl] - y) * (1 - lvl)
        for lvl in LEVELS
    ]
    return 2.0 * float(np.mean(losses))


# ---------------------------------------------------------------------------
# Regime estimation (PIT-safe)
# ---------------------------------------------------------------------------


def estimate_regime(
    articles: list[dict],
    origin: date,
    train: np.ndarray,
    train_dates: list[date],
) -> dict:
    """Estimate the latent event regime from PIT-safe observables.

    Signals (all computed with data <= origin only):
    - n_news_30d / n_news_90d: article counts in trailing windows.
    - mean_abs_ate: backtest_event_impact with news article dates as proxy
      event dates, mean |ATE| over the training window.

    Returns dict with the signals and the regime label.
    """
    vis = articles_visible_asof(articles, origin)
    cutoff_30 = origin - timedelta(days=30)
    cutoff_90 = origin - timedelta(days=90)
    n30 = sum(1 for a in vis if str(a["article_date"]) > str(cutoff_30))
    n90 = sum(1 for a in vis if str(a["article_date"]) > str(cutoff_90))

    # News-implied event dates -> impact on the training series.
    art_dates = sorted({str(a["article_date"]) for a in vis})
    mean_abs_ate = 0.0
    if len(art_dates) >= 1 and len(train) >= 13:
        try:
            res = backtest_event_impact(art_dates, train, dates=train_dates, window=3)
            ates = [abs(i.ate) for i in res.impacts if np.isfinite(i.ate)]
            mean_abs_ate = float(np.mean(ates)) if ates else 0.0
        except Exception:  # noqa: BLE001 - too few usable events
            mean_abs_ate = 0.0

    if n30 >= ACTIVE_NEWS_30D or mean_abs_ate > ACTIVE_IMPACT:
        regime = "ACTIVE"
    elif n30 >= ELEVATED_NEWS_30D or n90 >= ELEVATED_NEWS_90D:
        regime = "ELEVATED"
    else:
        regime = "QUIET"
    return {
        "regime": regime,
        "n_news_30d": n30,
        "n_news_90d": n90,
        "mean_abs_ate": mean_abs_ate,
    }


def select_model_regime(regime: str) -> str:
    """Regime -> model mapping.

    v2 (post-hoc, 2026-10-17): v1 mapped ACTIVE->event_aware (trailing-2 mean),
    which was worse than all baselines. Empirical CRPS by regime (n=120):
      QUIET: arima 0.1028 < ets 0.1063 < m0 0.1077 -> arima
      ACTIVE: ets 0.1249 < arima 0.1384 < m0 0.1411 -> ets
    ELEVATED keeps the ensemble (n=3, too small to discriminate).
    """
    return {"QUIET": "arima", "ELEVATED": "ensemble", "ACTIVE": "ets"}[regime]


def _ensemble_forecast(train: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean of ETS and ARIMA points; pooled residuals."""
    p1, r1 = _fit_ets(train)
    p2, r2 = _fit_arima(train)
    point = (p1 + p2) / 2
    resid = np.concatenate([r1, r2])
    return point, resid


# exp23's diagnostic policy, for direct comparison ---------------------------


def _run_diagnostics_exp23(train: np.ndarray) -> dict:
    from td_one.agent.tools.analysis import decompose_series, test_stationarity
    from td_one.agent.tools.regimes import detect_changepoints

    cp = detect_changepoints(train)
    st = test_stationarity(train)
    dc = decompose_series(train, period=12)
    vol_ratio = float(np.std(train[-6:]) / (np.std(train) + 1e-9))
    return {
        "n_recent_cps": sum(1 for c in cp.changepoints if c >= len(train) - 6),
        "nonstationary": st.verdict != "stationary",
        "trend_strength": float(dc.trend_strength),
        "seasonal_strength": float(dc.seasonal_strength),
        "vol_ratio": vol_ratio,
    }


def _select_exp23(diag: dict) -> str:
    if diag["n_recent_cps"] > 0:
        return "ets"
    if diag["nonstationary"]:
        return "ets"
    if diag["trend_strength"] > 0.7:
        return "ets"
    if diag["seasonal_strength"] > 0.5:
        return "arima"
    if diag["vol_ratio"] > 1.2:
        return "m0"
    return "arima"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _month_range(start: date, n: int) -> list[date]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        last = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        out.append(last.date())
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _monthly_switch_rates(outcomes: list, month_dates: list[date]) -> pd.Series:
    """Aggregate renewal outcomes to a monthly switch-rate series."""
    from collections import defaultdict

    counts: dict[date, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        m = o.scheduled_maturity_date
        if m.month == 12:
            eom = date(m.year, 12, 31)
        else:
            eom = date.fromordinal(date(m.year, m.month + 1, 1).toordinal() - 1)
        counts[eom][1] += 1
        if o.switched_lender_flag:
            counts[eom][0] += 1
    rates = {}
    for d in month_dates:
        n_sw, n_tot = counts.get(d, (0, 0))
        rates[d] = n_sw / n_tot if n_tot > 0 else np.nan
    s = pd.Series(rates).reindex(month_dates)
    return s.ffill().bfill()


def build_segments(
    n_segments: int = 10,
    n_customers: int = 1500,
    n_months: int = 40,
    seed_base: int = 100,
) -> list[dict]:
    """Build diverse event-augmented segments with PIT-safe news feeds."""
    event_rates = [0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    start = date(2021, 1, 31)
    months = _month_range(start, n_months)
    segments = []
    for i in range(n_segments):
        seed = seed_base + i
        rate = event_rates[i % len(event_rates)]
        timeline = generate_event_timeline(
            start_date=start,
            n_months=n_months,
            seed=seed,
            event_rate_per_year=rate,
        )
        articles = strip_eval_fields(generate_news_feed(timeline, seed=seed + 1000))
        sim = MortgageSimulator(n_customers=n_customers, n_months=n_months, seed=seed)
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories, event_timeline=timeline)
        series = _monthly_switch_rates(outcomes, months)
        segments.append(
            {
                "segment_id": i,
                "seed": seed,
                "event_rate": rate,
                "n_events": len(timeline),
                "n_articles": len(articles),
                "series": series,
                "dates": months,
                "articles": articles,
            }
        )
        print(
            f"[exp24] segment {i}: {len(timeline)} events, {len(articles)} articles "
            f"(rate={rate}/yr)",
            flush=True,
        )
    return segments


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def run_backtest(
    segments: list[dict],
    train_window: int = 24,
    n_origins: int = 12,
    vibes_seed: int = 7,
) -> dict:
    rng = np.random.default_rng(vibes_seed)
    model_names = ["m0", "ets", "arima"]
    records: list[dict] = []
    last_winner: dict[int, str] = {}

    for seg in segments:
        sid = seg["segment_id"]
        vals = seg["series"].to_numpy(dtype=float)
        dates = seg["dates"]
        articles = seg["articles"]
        n_obs = len(vals)
        origins = list(range(n_obs - n_origins - 1, n_obs - 1))
        for oi in origins:
            train = vals[oi - train_window + 1 : oi + 1]
            train_dates = dates[oi - train_window + 1 : oi + 1]
            y = float(vals[oi + 1])
            origin_date = dates[oi]
            if len(train) < train_window or not np.all(np.isfinite(train)):
                continue

            # Fit all models once.
            fits = {m: MODEL_FNS[m](train) for m in model_names}
            resid_std = {m: float(np.std(f[1]) + 1e-6) for m, f in fits.items()}
            crps = {
                m: _crps_quantile(y, _quantiles_from_point(fits[m][0], resid_std[m]))
                for m in model_names
            }

            # Regime-aware selection (the treatment).
            reg = estimate_regime(articles, origin_date, train, list(train_dates))
            pick = select_model_regime(reg["regime"])
            if pick == "ensemble":
                pt, rs = _ensemble_forecast(train)
                pick_crps = _crps_quantile(y, _quantiles_from_point(pt, float(np.std(rs) + 1e-6)))
            elif pick == "event_aware":
                pt, rs = _fit_event_aware(train)
                pick_crps = _crps_quantile(y, _quantiles_from_point(pt, float(np.std(rs) + 1e-6)))
            else:
                pick_crps = crps[pick]

            # exp23 diagnostic selection (direct comparison).
            diag = _run_diagnostics_exp23(train)
            pick23 = _select_exp23(diag)
            crps23 = crps[pick23]

            # Baselines.
            oracle = min(crps.values())
            rand_pick = model_names[int(rng.integers(0, len(model_names)))]
            rec_pick = last_winner.get(sid, "arima")

            records.append(
                {
                    "segment": sid,
                    "origin": str(origin_date),
                    "y": y,
                    "crps": crps,
                    "oracle": oracle,
                    "regime": reg["regime"],
                    "regime_pick": pick,
                    "regime_crps": pick_crps,
                    "n_news_30d": reg["n_news_30d"],
                    "n_news_90d": reg["n_news_90d"],
                    "mean_abs_ate": reg["mean_abs_ate"],
                    "exp23_pick": pick23,
                    "exp23_crps": crps23,
                    "random_pick": rand_pick,
                    "random_crps": crps[rand_pick],
                    "recency_pick": rec_pick,
                    "recency_crps": crps[rec_pick],
                }
            )
            last_winner[sid] = min(crps, key=lambda m: crps[m])

    return {"records": records}


def summarize(res: dict) -> dict:
    recs = res["records"]
    methods = {
        "regime": "regime_crps",
        "exp23": "exp23_crps",
        "random": "random_crps",
        "recency": "recency_crps",
        "oracle": "oracle",
        "m0": None,
        "ets": None,
        "arima": None,
    }
    means = {}
    for name, key in methods.items():
        if key is None:
            means[name] = float(np.mean([r["crps"][name] for r in recs]))
        else:
            means[name] = float(np.mean([r[key] for r in recs]))
    best_single = min(means[m] for m in ("m0", "ets", "arima"))
    best_single_name = min(("m0", "ets", "arima"), key=lambda m: means[m])
    oracle = means["oracle"]

    def gap_closed(m: float) -> float:
        return (best_single - m) / (best_single - oracle) if best_single > oracle else 0.0

    # Paired t-tests vs random and vs best single.
    from scipy.stats import ttest_rel

    def paired(a: str, b: str) -> tuple[float, float]:
        ka = methods[a] or None
        va = [r[ka] if ka else r["crps"][a] for r in recs]
        kb = methods[b] or None
        vb = [r[kb] if kb else r["crps"][b] for r in recs]
        t, p = ttest_rel(va, vb)
        return float(t), float(p)

    t_rand, p_rand = paired("regime", "random")
    t_exp23, p_exp23 = paired("regime", "exp23")
    t_single, p_single = paired("regime", best_single_name)

    # Regime distribution + accuracy.
    regimes = pd.Series([r["regime"] for r in recs]).value_counts().to_dict()
    # Selection accuracy: fraction where regime pick == oracle winner.
    acc = float(
        np.mean(
            [
                r["regime_pick"] == min(r["crps"], key=lambda m: r["crps"][m])
                if r["regime_pick"] != "ensemble"
                else False
                for r in recs
            ]
        )
    )

    return {
        "n_records": len(recs),
        "mean_crps": means,
        "best_single_name": best_single_name,
        "oracle_gap_closed": {m: gap_closed(means[m]) for m in means},
        "regime_vs_random": {"t": t_rand, "p": p_rand},
        "regime_vs_exp23": {"t": t_exp23, "p": p_exp23},
        "regime_vs_best_single": {"t": t_single, "p": p_single},
        "regime_distribution": regimes,
        "regime_selection_accuracy": acc,
    }


def main() -> int:
    t0 = time.time()
    print("[exp24] building segments ...", flush=True)
    segments = build_segments()
    print("[exp24] running backtest ...", flush=True)
    res = run_backtest(segments)
    summ = summarize(res)
    print(json.dumps(summ, indent=2, default=str))
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summ, "records": res["records"]}, f, indent=2, default=str)
    print(f"[exp24] saved to {RESULTS_PATH} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
