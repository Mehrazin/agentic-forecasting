"""exp25: Does the news edge appear at longer horizons? (h=3).

exp21's diagnosis: at h=1, ETS adapts to level shifts within 1-2 months, so
news features (even news-leading ones) can't beat adaptive smoothing. But at
h=3, ETS *cannot* have adapted to shifts it hasn't seen yet — anticipation
should matter more.

Design:
- Same Phase-3 news-leading-biased DGP as exp24 (>=60% news_lead > 0).
- h=3: forecast the switch rate 3 months ahead.
- Models:
  - ETS(h=3): fit on training, forecast 3 steps.
  - ARIMA(h=3): same.
  - NEWS: OLS regression of y[t+3] on [y[t], y[t-1], y[t-2]] + news features
    (news_rate_shift_3m, news_policy_shock_3m, news_total_3m — the features
    exp21's discovery found sensible, PIT-safe with 1-month lag).
- Metric: CRPS at h=3, Diebold-Mariano tests.

Hypothesis: NEWS beats ETS at h=3 (reversing exp21's h=1 result), because
the 1-2 month news lead now falls inside the forecast horizon.

Usage:
    cd td-one && .venv/bin/python experiments/exp25_h3_horizon.py
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
from scipy.stats import norm, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.data_gen.events import generate_event_timeline  # noqa: E402
from td_one.data_gen.news_synthesis import (  # noqa: E402
    articles_visible_asof,
    generate_news_feed,
    strip_eval_fields,
)
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS_PATH = ROOT / "experiments" / "exp25_results.json"

LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
Z_SCORES = norm.ppf(LEVELS)
HORIZON = 3

# News features exp21's discovery ranked highest (all PIT-safe, 1-month lag).
NEWS_FEATURES = ["news_rate_shift_3m", "news_policy_shock_3m", "news_total_3m"]


def _month_range(start: date, n: int) -> list[date]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        last = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).date()
        out.append(last)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _monthly_switch_rates(outcomes: list, month_dates: list[date]) -> pd.Series:
    from collections import defaultdict

    counts: dict[date, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        m = o.scheduled_maturity_date
        eom = (pd.Timestamp(m.year, m.month, 1) + pd.offsets.MonthEnd(0)).date()
        counts[eom][1] += 1
        if o.switched_lender_flag:
            counts[eom][0] += 1
    rates = {d: (counts[d][0] / counts[d][1] if counts[d][1] > 0 else np.nan) for d in month_dates}
    return pd.Series(rates).reindex(month_dates).ffill().bfill()


def _news_features_asof(
    articles: list[dict], month_dates: list[date], upto_idx: int
) -> pd.DataFrame:
    """PIT-safe news features: trailing-3m article counts per type, lagged 1m.

    For origin at month_dates[upto_idx], features use articles with
    article_date <= month_dates[upto_idx - 1] (1-month lag, matching exp21).
    """
    asof = month_dates[upto_idx - 1] if upto_idx >= 1 else month_dates[0]
    vis = articles_visible_asof(articles, asof)
    # Count per event type over trailing 3 months (relative to asof).
    cutoff = asof - timedelta(days=90)
    counts: dict[str, int] = {}
    total = 0
    for a in vis:
        if str(a["article_date"]) > str(cutoff):
            et = str(a.get("event_type", "unknown"))
            counts[et] = counts.get(et, 0) + 1
            total += 1
    return pd.DataFrame(
        [
            {
                "news_rate_shift_3m": counts.get("rate_shift", 0),
                "news_policy_shock_3m": counts.get("policy_shock", 0),
                "news_total_3m": total,
            }
        ]
    )


def _fit_ets_h(train: np.ndarray, h: int) -> tuple[float, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    x = np.asarray(train, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(
                x, trend=None, seasonal=None, initialization_method="legacy-heuristic"
            ).fit(optimized=True, use_brute=True)
        fc = np.asarray(res.forecast(h), dtype=float)
        point = float(fc[-1])
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size < 5:
            raise ValueError("too few residuals")
        return point, resid
    except Exception:  # noqa: BLE001
        point = float(np.mean(x[-3:]))
        return point, x - np.mean(x)


def _fit_arima_h(train: np.ndarray, h: int) -> tuple[float, np.ndarray]:
    from statsmodels.tsa.arima.model import ARIMA

    x = np.asarray(train, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(x, order=(1, 1, 1)).fit()
        fc = np.asarray(res.forecast(steps=h), dtype=float)
        point = float(fc[-1])
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size < 5:
            raise ValueError("too few residuals")
        return point, resid
    except Exception:  # noqa: BLE001
        point = float(np.mean(x[-3:]))
        return point, x - np.mean(x)


def _fit_news_ols(
    train_y: np.ndarray, train_x: pd.DataFrame, x_now: pd.DataFrame
) -> tuple[float, np.ndarray]:
    """OLS: y[t+H] ~ y[t],y[t-1],y[t-2] + news features (all PIT-safe)."""
    # Build aligned design: for each t in training, target = y[t+H].
    n = len(train_y)
    rows, tgts = [], []
    for t in range(2, n - HORIZON):
        rows.append(
            [train_y[t], train_y[t - 1], train_y[t - 2]]
            + [float(train_x.iloc[t][c]) for c in NEWS_FEATURES]
        )
        tgts.append(train_y[t + HORIZON])
    if len(rows) < 10:
        # Fallback: trailing mean.
        point = float(np.mean(train_y[-3:]))
        return point, train_y - np.mean(train_y)
    x_design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
    yv = np.asarray(tgts)
    try:
        beta, *_ = np.linalg.lstsq(x_design, yv, rcond=None)
    except Exception:  # noqa: BLE001
        point = float(np.mean(train_y[-3:]))
        return point, train_y - np.mean(train_y)
    xrow = np.array(
        [1.0, train_y[-1], train_y[-2], train_y[-3]]
        + [float(x_now.iloc[0][c]) for c in NEWS_FEATURES]
    )
    point = float(xrow @ beta)
    resid = yv - x_design @ beta
    return point, resid


def _quantiles(point: float, resid_std: float) -> dict[float, float]:
    return {
        lvl: float(np.clip(point + resid_std * z, 0.0005, 0.995))
        for lvl, z in zip(LEVELS, Z_SCORES, strict=True)
    }


def _crps(y: float, qs: dict[float, float]) -> float:
    losses = [(y - qs[lvl]) * lvl if y >= qs[lvl] else (qs[lvl] - y) * (1 - lvl) for lvl in LEVELS]
    return 2.0 * float(np.mean(losses))


def build_segments(
    n_segments: int = 10,
    n_customers: int = 1500,
    n_months: int = 48,
    seed_base: int = 200,
) -> list[dict]:
    """Longer series (48mo) to allow h=3 training + evaluation."""
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
                "series": series,
                "dates": months,
                "articles": articles,
            }
        )
        print(f"[exp25] segment {i}: {len(timeline)} events, {len(articles)} articles", flush=True)
    return segments


def run_backtest(segments: list[dict], train_window: int = 30, n_origins: int = 12) -> dict:
    records: list[dict] = []
    for seg in segments:
        vals = seg["series"].to_numpy(dtype=float)
        dates = seg["dates"]
        articles = seg["articles"]
        n_obs = len(vals)
        # Origins leave HORIZON months for the target.
        origins = list(range(n_obs - n_origins - HORIZON, n_obs - HORIZON))
        for oi in origins:
            train = vals[oi - train_window + 1 : oi + 1]
            y = float(vals[oi + HORIZON])
            if len(train) < train_window or not np.all(np.isfinite(train)):
                continue
            # News features need per-month history; build incrementally.
            # For simplicity: feature frame aligned to train indices.
            feat_rows = []
            for j in range(len(train)):
                f = _news_features_asof(articles, dates, oi - train_window + 1 + j)
                feat_rows.append(f.iloc[0])
            train_x = pd.DataFrame(feat_rows).reset_index(drop=True)
            x_now = _news_features_asof(articles, dates, oi + 1)

            p_ets, r_ets = _fit_ets_h(train, HORIZON)
            p_ari, r_ari = _fit_arima_h(train, HORIZON)
            p_new, r_new = _fit_news_ols(train, train_x, x_now)

            crps = {
                "ets": _crps(y, _quantiles(p_ets, float(np.std(r_ets) + 1e-6))),
                "arima": _crps(y, _quantiles(p_ari, float(np.std(r_ari) + 1e-6))),
                "news": _crps(y, _quantiles(p_new, float(np.std(r_new) + 1e-6))),
            }
            records.append(
                {
                    "segment": seg["segment_id"],
                    "origin": str(dates[oi]),
                    "y": y,
                    "crps": crps,
                    "oracle": min(crps.values()),
                }
            )
    return {"records": records}


def main() -> int:
    t0 = time.time()
    print("[exp25] building segments ...", flush=True)
    segments = build_segments()
    print("[exp25] running h=3 backtest ...", flush=True)
    res = run_backtest(segments)
    recs = res["records"]
    means = {m: float(np.mean([r["crps"][m] for r in recs])) for m in ("ets", "arima", "news")}
    oracle = float(np.mean([r["oracle"] for r in recs]))
    best_single = min(means["ets"], means["arima"])
    t_en, p_en = ttest_rel([r["crps"]["ets"] for r in recs], [r["crps"]["news"] for r in recs])
    t_an, p_an = ttest_rel([r["crps"]["arima"] for r in recs], [r["crps"]["news"] for r in recs])
    summ = {
        "experiment": "exp25_h3_horizon",
        "horizon": HORIZON,
        "n_records": len(recs),
        "mean_crps": means,
        "oracle": oracle,
        "news_vs_ets": {
            "t": float(t_en),
            "p": float(p_en),
            "rel": (means["ets"] - means["news"]) / means["ets"],
        },
        "news_vs_arima": {
            "t": float(t_an),
            "p": float(p_an),
            "rel": (means["arima"] - means["news"]) / means["arima"],
        },
        "oracle_gap_closed_news": (
            (best_single - means["news"]) / (best_single - oracle) if best_single > oracle else 0.0
        ),
    }
    print(json.dumps(summ, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summ, "records": recs}, f, indent=2, default=str)
    print(f"[exp25] saved ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
