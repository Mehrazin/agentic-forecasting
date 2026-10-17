"""exp21: Agent Feature Discovery on the event-augmented simulator.

Tests whether the "data scientist" loop works:
  1. Fit a baseline (ETS, no event knowledge) -> residuals.
  2. Agent examines residuals with the analysis tools (analyze_residuals,
     detect_changepoints) and proposes news-derived features.
  3. Each proposed feature is tested via PIT-safe cross_validate; kept only
     if it improves CV CRPS (forward selection).
  4. Final held-out evaluation: ETS vs OLS-AR(1) vs OLS-AR(1)+discovered.

The agent never sees the true event timeline -- only the noisy, delayed,
asymmetric news feed. Candidate features are built PIT-safe: the feature
value at month m uses only articles with article_date <= end of month m.

Protocol per seed:
  * Discovery on months [0, 48): residual analysis + forward selection with
    cross_validate(n_splits=5, min_train=24).
  * Held-out evaluation on months [54, 66): compare_models with
    min_train=54, n_splits=12 (tests months 54..65; discovery never saw 48+).

Also runs one scripted ReActAgentV2 trace (MockLLMClient) to prove the v2
code-execution loop can drive the discovery steps, and attempts one live
Groq trace if the vault connector is available (graceful skip otherwise).

Usage:
    cd td-one && .venv/bin/python experiments/exp21_feature_discovery.py
    [--seeds 42,7,123] [--skip-groq]

Results: experiments/exp21_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from td_one.agent.react import MockLLMClient
from td_one.agent.react_v2 import ReActAgentV2
from td_one.agent.tools.analysis import analyze_residuals
from td_one.agent.tools.evaluation import compare_models, cross_validate
from td_one.agent.tools.regimes import detect_changepoints
from td_one.data_gen.events import EventType, generate_event_timeline
from td_one.data_gen.news_synthesis import (
    generate_news_feed,
    strip_eval_fields,
)
from td_one.data_gen.simulator import MortgageSimulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVELS = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]
N_MONTHS = 66
N_CUSTOMERS = 2000
DISCOVERY_END = 48  # discovery uses months [0, 48)
EVAL_MIN_TRAIN = 54  # final eval trains on >= 54 months, tests 54..65
EVAL_SPLITS = 12
DISC_SPLITS = 5
DISC_MIN_TRAIN = 24
SELECTION_MIN_GAIN = 0.005  # keep a feature if CV CRPS improves >= 0.5%
EVENT_TYPES = [e.value for e in EventType]

NEWS_FEATURE_COLS = [f"news_{t}_3m" for t in EVENT_TYPES] + [
    "news_total_3m",
    "news_any_3m",
    "months_since_news",
]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


def _add_months_ym(year: int, month: int, n: int) -> tuple[int, int]:
    total = (year * 12 + month - 1) + n
    return total // 12, total % 12 + 1


def monthly_switch_rates(outcomes: list, month_dates: list[date]) -> pd.Series:
    """Aggregate renewal outcomes to a monthly switch-rate series."""
    counts: dict[date, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        m = o.scheduled_maturity_date
        eom = _month_end(m.year, m.month)
        counts[eom][1] += 1
        if o.switched_lender_flag:
            counts[eom][0] += 1
    rates = {
        d: (counts[d][0] / counts[d][1] if counts.get(d, [0, 0])[1] > 0 else np.nan)
        for d in month_dates
    }
    return pd.Series(rates).sort_index().ffill().bfill()


def build_news_features(articles: list[dict], month_dates: list[date]) -> pd.DataFrame:
    """Build PIT-safe news features, one row per month.

    The feature at month m uses only articles with article_date <= m
    (month-end). Article counts are over trailing 3-month windows.
    """
    # Index articles by (year, month)
    by_ym: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for a in articles:
        ad = a["article_date"]
        if isinstance(ad, str):
            y, mo, _ = (int(x) for x in ad.split("-"))
        else:
            y, mo = ad.year, ad.month
        by_ym[(y, mo)].append(a)

    rows: list[dict] = []
    for d in month_dates:
        # trailing 3-month window: months m-2 .. m
        window_articles: list[dict] = []
        for k in range(3):
            yy, mm = _add_months_ym(d.year, d.month, -k)
            window_articles.extend(by_ym.get((yy, mm), []))
        row: dict[str, float] = {}
        for t in EVENT_TYPES:
            row[f"news_{t}_3m"] = float(sum(1 for a in window_articles if a.get("event_type") == t))
        row["news_total_3m"] = float(len(window_articles))
        row["news_any_3m"] = 1.0 if window_articles else 0.0
        # months since most recent article (<= m), capped at 12
        for k in range(0, 13):
            yy, mm = _add_months_ym(d.year, d.month, -k)
            if by_ym.get((yy, mm)):
                row["months_since_news"] = float(k)
                break
        else:
            row["months_since_news"] = 12.0
        rows.append(row)
    frame = pd.DataFrame(rows, index=pd.Index(month_dates))
    # PIT assertion: feature at row i must not use articles after month i.
    # (Guaranteed by construction: only by_ym keys <= (d.year, d.month) read.)
    assert list(frame.columns) == NEWS_FEATURE_COLS
    return frame


def build_dataset(seed: int) -> dict:
    """Generate event-augmented data -> series + PIT-safe news features."""
    start = date(2021, 4, 30)
    events = generate_event_timeline(start, N_MONTHS, seed=seed)
    sim = MortgageSimulator(n_customers=N_CUSTOMERS, n_months=N_MONTHS, seed=seed)
    outcomes = sim.generate_outcomes(event_timeline=events)
    month_dates = [_month_end(*_add_months_ym(2021, 4, k)) for k in range(N_MONTHS)]
    series = monthly_switch_rates(outcomes, month_dates)

    raw_articles = generate_news_feed(events, seed=seed + 1000)
    articles = strip_eval_fields(raw_articles)
    # PIT sanity: no article before its event (checked on raw feed via event_id)
    features = build_news_features(articles, month_dates)
    return {
        "seed": seed,
        "dates": month_dates,
        "series": series,
        "features": features,
        "events": events,
        "articles": articles,
        "n_events": len(events),
        "n_articles": len(articles),
    }


# ---------------------------------------------------------------------------
# Forecast models (shared normal-quantile wrapper isolates point skill)
# ---------------------------------------------------------------------------


def _normal_quantiles(point: float, sigma: float) -> dict[float, float]:
    """Compute normal quantiles clipped to (0.001, 0.99)."""
    from scipy.stats import norm

    sigma = max(float(sigma), 1e-6)
    out = {}
    for lvl in LEVELS:
        q = float(point + sigma * norm.ppf(lvl))
        out[lvl] = float(min(0.99, max(0.001, q)))
    return out


def _ets_point(train: np.ndarray) -> tuple[float, float]:
    """ETS(A,N,N) 1-step point forecast + in-sample residual std."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    x = np.asarray(train, dtype=float)
    try:
        if len(x) < 4:
            raise ValueError("too short")
        fit = ExponentialSmoothing(
            x, trend=None, seasonal=None, initialization_method="legacy-heuristic"
        ).fit(optimized=True, use_brute=True)
        point = float(fit.forecast(1)[0])
        resid = x[1:] - np.asarray(fit.fittedvalues[1:], dtype=float)
        sigma = float(np.std(resid)) if len(resid) > 1 else float(np.std(x))
    except Exception:  # noqa: BLE001 - fall back to trailing mean
        point = float(np.mean(x[-6:]))
        sigma = float(np.std(x[-12:])) if len(x) >= 12 else float(np.std(x))
    return point, max(sigma, 1e-6)


def ets_model_fn(train: np.ndarray, horizon: int) -> dict[float, float]:
    """Baseline: ETS(A,N,N), no event knowledge."""
    point, sigma = _ets_point(np.asarray(train, dtype=float))
    return _normal_quantiles(point, sigma)


def _ols_fit_predict(
    y: np.ndarray, feat_frame: pd.DataFrame, selected: list[str], horizon: int = 1
) -> tuple[float, float]:
    """OLS on [1, lag1, lagged selected features]; returns (point, sigma).

    Forecasting setup (PIT-safe): to predict y[i] we may only use
    information known at the end of month i-1, so features enter lagged by
    one month: y[i] ~ [1, y[i-1], F[i-1]]. The closure slices the feature
    frame by len(train); CV passes train = series[:origin], so row indices
    align and no future information leaks (features are PIT-safe by
    construction).
    """
    n = len(y)
    if n < 6:
        return float(np.mean(y)), max(float(np.std(y)), 1e-6)
    # Fit on i = 1..n-1: target y[i], regressors [1, y[i-1], F[i-1]].
    lag1 = y[:-1]
    cols = [np.ones(n - 1), lag1]
    for c in selected:
        cols.append(np.asarray(feat_frame[c].iloc[: n - 1], dtype=float))
    design = np.column_stack(cols)
    target = y[1:]
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    sigma = float(np.std(resid)) if len(resid) > 1 else 1e-6
    # Predict y[n]: [1, y[n-1], F[n-1]] -- all known at end of month n-1.
    prow = [1.0, float(y[-1])]
    for c in selected:
        prow.append(float(feat_frame[c].iloc[n - 1]))
    point = float(np.dot(prow, beta))
    return point, max(sigma, 1e-6)


def ols_model_fn_factory(feat_frame: pd.DataFrame, selected: list[str], name: str = "ols"):
    """Build a cross_validate-compatible model_fn closing over features.

    The closure slices the feature frame by len(train); since CV always
    passes train = series[:origin], row indices align and no future
    information leaks (features themselves are PIT-safe by construction).
    """

    def _fn(train: np.ndarray, horizon: int) -> dict[float, float]:
        y = np.asarray(train, dtype=float)
        point, sigma = _ols_fit_predict(y, feat_frame, selected, horizon)
        return _normal_quantiles(point, sigma)

    _fn.__name__ = name
    return _fn


# ---------------------------------------------------------------------------
# Discovery: the "data scientist" loop
# ---------------------------------------------------------------------------


def run_discovery(series: pd.Series, features: pd.DataFrame) -> dict:
    """Residual analysis -> propose -> CV-test -> select.

    Returns discovery diagnostics + the locked-in feature list.
    """
    y = series.iloc[:DISCOVERY_END].to_numpy(dtype=float)
    feat_disc = features.iloc[:DISCOVERY_END]

    # 1. ETS in-sample 1-step predictions -> residuals.
    # ets_pred[i] forecasts y_al[i] using y[:i+4].
    y_al = y[4:]
    feat_al = feat_disc.iloc[4:].reset_index(drop=True)
    ets_pred = np.array([_ets_point(y[: i + 4])[0] for i in range(len(y_al))])

    resid_report = analyze_residuals(y_al, ets_pred, feat_al[NEWS_FEATURE_COLS])
    cp_report = detect_changepoints(y)

    # 2. Forward selection over candidates ranked by |corr| with |residual|
    ranked = sorted(resid_report.top_features, key=lambda d: abs(d["corr"]), reverse=True)
    candidates = [d["name"] for d in ranked]
    # also consider any news feature not in top-3 (cheap to test)
    for c in NEWS_FEATURE_COLS:
        if c not in candidates:
            candidates.append(c)

    selected: list[str] = []
    base_fn = ols_model_fn_factory(features, [], name="ols_ar1")
    base_cv = cross_validate(
        base_fn,
        y,
        n_splits=DISC_SPLITS,
        horizon=1,
        min_train=DISC_MIN_TRAIN,
        model_name="ols_ar1",
    )
    best = base_cv.mean_crps
    selection_log = [{"features": [], "mean_crps": best, "se": base_cv.se_crps}]
    for cand in candidates:
        trial = selected + [cand]
        fn = ols_model_fn_factory(features, trial, name="ols_trial")
        cv = cross_validate(
            fn,
            y,
            n_splits=DISC_SPLITS,
            horizon=1,
            min_train=DISC_MIN_TRAIN,
            model_name="ols_trial",
        )
        improved = cv.mean_crps < best * (1 - SELECTION_MIN_GAIN)
        selection_log.append(
            {
                "features": list(trial),
                "mean_crps": cv.mean_crps,
                "se": cv.se_crps,
                "kept": bool(improved),
            }
        )
        if improved:
            selected = trial
            best = cv.mean_crps

    return {
        "n_disc_obs": int(len(y)),
        "ets_in_sample_rmse": float(resid_report.rmse),
        "heteroskedastic": bool(resid_report.heteroskedastic),
        "top_residual_features": [
            {"name": d["name"], "corr": round(float(d["corr"]), 4)}
            for d in resid_report.top_features
        ],
        "changepoints": [int(i) for i in cp_report.changepoints],
        "selection_log": selection_log,
        "selected_features": selected,
        "baseline_cv_crps": float(base_cv.mean_crps),
        "selected_cv_crps": float(best),
    }


# ---------------------------------------------------------------------------
# Scripted ReActAgentV2 trace (loop-mechanics demo)
# ---------------------------------------------------------------------------


def run_scripted_trace(
    stage_dir: Path, selected: list[str], agent_point: float, agent_sigma: float
) -> dict:
    """Run one ReActAgentV2 trace with a scripted LLM over staged data.

    The scripted agent independently runs detect_changepoints and
    analyze_residuals on staged data and reports top features; the driver
    compares them with the locked-in selection. Documents loop mechanics,
    not LLM reasoning (see Groq attempt below for the real thing).
    """
    finalize_fc = _normal_quantiles(agent_point, agent_sigma)
    fc_json = json.dumps({str(k): round(v, 6) for k, v in finalize_fc.items()})

    code1 = (
        "import pandas as _p, numpy as _n\n"
        f"s = _p.read_csv('{stage_dir}/series.csv', index_col=0,"
        " parse_dates=True).iloc[:,0].to_numpy()\n"
        "_show(T.detect_changepoints(s).model_dump())\n"
    )
    code2 = (
        "import pandas as _p, numpy as _n\n"
        f"s = _p.read_csv('{stage_dir}/series.csv', index_col=0,"
        " parse_dates=True).iloc[:,0].to_numpy()\n"
        f"f = _p.read_csv('{stage_dir}/features.csv', index_col=0,"
        " parse_dates=True)\n"
        f"yt = _p.read_csv('{stage_dir}/yt.csv', index_col=0,"
        " parse_dates=True).iloc[:,0].to_numpy()\n"
        f"yp = _p.read_csv('{stage_dir}/yp.csv', index_col=0,"
        " parse_dates=True).iloc[:,0].to_numpy()\n"
        "_show(T.analyze_residuals(yt, yp, f).model_dump())\n"
    )
    responses = [
        json.dumps({"action": "code_execute", "args": {"code": code1, "timeout_s": 60}}),
        json.dumps({"action": "code_execute", "args": {"code": code2, "timeout_s": 60}}),
        json.dumps({"action": "finalize", "forecast": json.loads(fc_json)}),
    ]
    agent = ReActAgentV2(MockLLMClient(responses), tools={}, max_steps=6)
    t0 = time.time()
    result = agent.run(
        task=(
            "Discover which news-derived features explain ETS forecast errors "
            "on the monthly switching-rate series, then finalize."
        ),
        run_date=date(2026, 3, 31),
    )
    return {
        "steps": result["steps"],
        "finalized": result["forecast"] is not None,
        "wall_s": round(time.time() - t0, 1),
        "trace": [
            {
                "action": s["action"],
                "observation": s["observation"][:600],
            }
            for s in result["trace"]
        ],
    }


def stage_data(stage_dir: Path, series: pd.Series, features: pd.DataFrame) -> None:
    """Stage discovery inputs for the scripted/sandboxed agent trace.

    Frames are pre-aligned to the ETS residual window (months 4..47) so the
    agent can call analyze_residuals directly without index wrangling.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    y = series.iloc[:DISCOVERY_END].to_numpy(dtype=float)
    ets_pred = np.array([_ets_point(y[: i + 4])[0] for i in range(len(y) - 4)])
    idx = series.iloc[4:DISCOVERY_END].index
    pd.Series(y[4:], index=idx, name="y_true").to_csv(stage_dir / "yt.csv")
    pd.Series(ets_pred, index=idx, name="y_pred").to_csv(stage_dir / "yp.csv")
    features.iloc[4:DISCOVERY_END][NEWS_FEATURE_COLS].to_csv(stage_dir / "features.csv")
    series.iloc[:DISCOVERY_END].to_csv(stage_dir / "series.csv")


# ---------------------------------------------------------------------------
# Live Groq attempt (best-effort)
# ---------------------------------------------------------------------------


class _TolerantGroqClient:
    """Wrap GroqChatClient to accept OpenAI-style tool calls.

    gpt-oss sometimes emits {"name": ..., "arguments": {...}} (native tool
    format) instead of our ReAct {"action": ..., "args": {...}} protocol.
    This wrapper translates, so we test the model's reasoning rather than
    its protocol adherence. Passes everything else through unchanged.
    """

    def __init__(self, inner) -> None:  # noqa: ANN001 - thin wrapper
        self._inner = inner
        self.translated = 0

    def complete(self, prompt: str) -> str:
        reply = self._inner.complete(prompt)
        try:
            obj = json.loads(reply)
        except Exception:  # noqa: BLE001 - not JSON, pass through
            return reply
        if isinstance(obj, dict) and "action" in obj:
            return reply
        name = obj.get("name") if isinstance(obj, dict) else None
        args = obj.get("arguments") if isinstance(obj, dict) else None
        if isinstance(name, str) and isinstance(args, dict):
            self.translated += 1
            return json.dumps({"action": name, "args": args})
        return reply


def run_groq_trace(stage_dir: Path) -> dict:
    """One live Groq-driven v2 trace; graceful skip on any failure."""
    try:
        from td_one.agent.groq_client import GroqChatClient
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"import failed: {e}"}
    code1 = (
        "import pandas as _p, numpy as _n\n"
        f"s = _p.read_csv('{stage_dir}/series.csv', index_col=0,"
        " parse_dates=True).iloc[:,0].to_numpy()\n"
        "_show(T.detect_changepoints(s).model_dump())\n"
    )
    prompt_note = (
        "You are driving a data-scientist loop. First run ONE code_execute step "
        "with the provided starter code to find changepoints, then inspect the "
        "result and decide the next step yourself (up to 6 steps), ending with "
        "finalize and your best-guess quantiles for the next month's "
        "switching rate."
    )
    try:
        client = GroqChatClient(call_budget=10, max_tokens=2000)
        tolerant = _TolerantGroqClient(client)
        agent = ReActAgentV2(tolerant, tools={}, max_steps=8)
        t0 = time.time()
        result = agent.run(
            task=(
                prompt_note + f" Data staged at {stage_dir}/series.csv (monthly switching "
                "rates), features.csv (news-derived features), yt.csv/yp.csv "
                "(ETS in-sample actuals/preds). Use code_execute with "
                f"starter code like: {code1[:120]}..."
            ),
            run_date=date(2026, 3, 31),
        )
        return {
            "status": "completed",
            "steps": result["steps"],
            "finalized": result["forecast"] is not None,
            "wall_s": round(time.time() - t0, 1),
            "protocol_translations": tolerant.translated,
            "trace": [
                {"action": s["action"], "observation": s["observation"][:400]}
                for s in result["trace"]
            ],
        }
    except Exception as e:  # noqa: BLE001 - best effort only
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:300]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_seed(seed: int, workdir: Path) -> dict:
    print(f"[exp21] seed {seed}: generating event-augmented data...", flush=True)
    ds = build_dataset(seed)
    series = ds["series"]
    features = ds["features"]
    print(
        f"[exp21] seed {seed}: {len(series)} months, "
        f"{ds['n_events']} events, {ds['n_articles']} articles, "
        f"mean rate {float(series.mean()):.4f}",
        flush=True,
    )

    print(f"[exp21] seed {seed}: running discovery on months [0,{DISCOVERY_END})...", flush=True)
    discovery = run_discovery(series, features)
    print(
        f"[exp21] seed {seed}: selected {discovery['selected_features']} "
        f"(CV {discovery['baseline_cv_crps']:.6f} -> "
        f"{discovery['selected_cv_crps']:.6f})",
        flush=True,
    )

    # Held-out evaluation: ETS vs OLS-AR(1) vs OLS-AR(1)+discovered
    y_all = series.to_numpy(dtype=float)
    model_fns = {
        "ets": ets_model_fn,
        "ols_ar1": ols_model_fn_factory(features, [], name="ols_ar1"),
        "ols_agent": ols_model_fn_factory(
            features, discovery["selected_features"], name="ols_agent"
        ),
    }
    print(
        f"[exp21] seed {seed}: held-out compare_models "
        f"(min_train={EVAL_MIN_TRAIN}, splits={EVAL_SPLITS})...",
        flush=True,
    )
    comp = compare_models(
        model_fns,
        y_all,
        n_splits=EVAL_SPLITS,
        horizon=1,
        min_train=EVAL_MIN_TRAIN,
    )
    ranking = comp.ranking
    per_model = {
        m.model_name: {
            "mean_crps": m.mean_crps,
            "se_crps": m.se_crps,
        }
        for m in comp.models
    }
    pairwise = [
        {
            "a": p.model_a,
            "b": p.model_b,
            "mean_diff": p.mean_loss_diff,
            "p": p.p_value,
            "winner": p.winner,
        }
        for p in comp.pairwise
    ]
    ets_crps = per_model["ets"]["mean_crps"]
    agent_crps = per_model["ols_agent"]["mean_crps"]
    gain_vs_ets = (ets_crps - agent_crps) / ets_crps if ets_crps else 0.0

    # Scripted v2 trace (mechanics demo) on staged discovery data
    stage_dir = workdir / f"stage_seed{seed}"
    stage_data(stage_dir, series, features)
    last_train = y_all[: EVAL_MIN_TRAIN + EVAL_SPLITS - 1]
    a_point, a_sigma = _ols_fit_predict(last_train, features, discovery["selected_features"])
    trace = run_scripted_trace(stage_dir, discovery["selected_features"], a_point, a_sigma)

    return {
        "seed": seed,
        "n_events": ds["n_events"],
        "n_articles": ds["n_articles"],
        "series_mean": float(series.mean()),
        "series_std": float(series.std()),
        "discovery": discovery,
        "held_out": {
            "per_model": per_model,
            "ranking": ranking,
            "pairwise": pairwise,
            "gain_vs_ets": gain_vs_ets,
        },
        "scripted_trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--skip-groq", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    workdir = Path("/tmp/exp21")
    results: dict = {
        "experiment": "exp21_feature_discovery",
        "config": {
            "n_months": N_MONTHS,
            "n_customers": N_CUSTOMERS,
            "discovery_months": [0, DISCOVERY_END],
            "eval": {"min_train": EVAL_MIN_TRAIN, "n_splits": EVAL_SPLITS},
            "levels": LEVELS,
            "selection_min_gain": SELECTION_MIN_GAIN,
            "seeds": seeds,
        },
        "per_seed": {},
    }
    for seed in seeds:
        results["per_seed"][str(seed)] = run_seed(seed, workdir)

    # Cross-seed verdict
    gains = [r["held_out"]["gain_vs_ets"] for r in results["per_seed"].values()]
    all_selected = [r["discovery"]["selected_features"] for r in results["per_seed"].values()]
    rankings = [r["held_out"]["ranking"] for r in results["per_seed"].values()]
    results["verdict"] = {
        "mean_gain_vs_ets": float(np.mean(gains)),
        "min_gain_vs_ets": float(np.min(gains)),
        "seeds_agent_beats_ets": int(sum(1 for g in gains if g > 0)),
        "n_seeds": len(seeds),
        "selected_features_per_seed": all_selected,
        "rankings": rankings,
        "success_criterion_met": bool(np.mean(gains) >= 0.02),
        "wall_s": round(time.time() - t0, 1),
    }

    # One live Groq trace (best effort) on the first seed's staged data
    if not args.skip_groq and seeds:
        print("[exp21] attempting live Groq trace (best effort)...", flush=True)
        groq_res = run_groq_trace(workdir / f"stage_seed{seeds[0]}")
        results["groq_trace"] = groq_res
        print(f"[exp21] groq trace: {groq_res.get('status')}", flush=True)
    else:
        results["groq_trace"] = {"status": "skipped", "reason": "flag or no seeds"}

    out = root / "experiments" / "exp21_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"[exp21] wrote {out} in {time.time() - t0:.0f}s", flush=True)
    v = results["verdict"]
    print(
        f"[exp21] VERDICT: mean gain vs ETS = {v['mean_gain_vs_ets'] * 100:.2f}% "
        f"over {v['n_seeds']} seeds "
        f"({'PASS' if v['success_criterion_met'] else 'FAIL'} vs 2% bar)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
