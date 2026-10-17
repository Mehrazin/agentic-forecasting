"""exp23: Diagnostic-based model selection vs single-best, oracle, and vibes.

M5 (LLM router) failed because it was vibes-based — the LLM guessed weights
with no diagnostic information. This experiment tests whether *diagnostic-
driven* selection beats the alternatives:

- Selection policy (rule-based v1, defined a priori from model properties):
    1. Recent changepoint (last 6 obs)      -> ETS   (fast level-shift adaptation)
    2. Non-stationary (ADF fails)            -> ETS   (stochastic trend handling)
    3. Strong trend (trend_strength > 0.7)   -> ETS
    4. Strong seasonality (> 0.5)            -> ARIMA (differencing; our ETS is simple)
    5. High noise (vol_ratio > 1.2)         -> M0    (robust, don't overfit)
    6. Default                              -> ARIMA

- Baselines:
    (a) best_single: lowest mean-CRPS model overall (chosen in hindsight on test)
    (b) oracle:      per-origin argmin CRPS (unachievable upper bound)
    (c) random:      uniform random pick per origin (M5-vibes proxy: no information)
    (d) recency:     last origin's winner for that segment (adaptive heuristic)

Data: event-augmented simulator (Phase 1). Segments vary in event density
(0 to 8 events/year) so different models genuinely win in different regimes.

Metric: mean CRPS per method + "oracle gap closed"
    = (best_single_crps - method_crps) / (best_single_crps - oracle_crps)

Hypothesis: diagnostic selection > random and > recency, and closes a
meaningful fraction of the oracle gap. May or may not beat best_single —
if one model dominates everywhere, selection can't help (and that's a finding).

Usage:
    cd td-one && .venv/bin/python experiments/exp23_model_selection.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.tools.analysis import decompose_series, test_stationarity  # noqa: E402
from td_one.agent.tools.regimes import detect_changepoints  # noqa: E402
from td_one.data_gen.events import generate_event_timeline  # noqa: E402
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS_PATH = ROOT / "experiments" / "exp23_results.json"

LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
Z_SCORES = norm.ppf(LEVELS)

# ---------------------------------------------------------------------------
# Model zoo (point forecasts + residual-based quantiles)
# ---------------------------------------------------------------------------


def _fit_m0(train: np.ndarray) -> tuple[float, np.ndarray]:
    """Trailing-3 mean: point forecast + in-sample one-step residuals."""
    x = np.asarray(train, dtype=float)
    point = float(np.mean(x[-3:]))
    # In-sample one-step residuals: trailing-3 mean as the "fitted" value.
    fitted = np.array([np.mean(x[max(0, i - 3) : i]) if i >= 1 else x[0] for i in range(len(x))])
    resid = x - fitted
    return point, resid


def _fit_ets(train: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit simple exponential smoothing ETS(A,N,N)."""
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
    except Exception:  # noqa: BLE001 - fallback to M0 on fit failure
        return _fit_m0(train)


def _fit_arima(train: np.ndarray) -> tuple[float, np.ndarray]:
    """ARIMA(1,1,1) one-step forecast."""
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
    except Exception:  # noqa: BLE001 - fallback to M0 on fit failure
        return _fit_m0(train)


MODEL_FNS = {
    "m0": _fit_m0,
    "ets": _fit_ets,
    "arima": _fit_arima,
}


def _quantiles_from_point(point: float, resid_std: float) -> dict[float, float]:
    """Build normal quantiles around the point forecast, clipped to (0, 1)."""
    qs = {}
    for lvl, z in zip(LEVELS, Z_SCORES, strict=True):
        q = point + resid_std * z
        qs[lvl] = float(np.clip(q, 0.0005, 0.995))
    return qs


def _crps_quantile(y: float, quantiles: dict[float, float]) -> float:
    """CRPS via 2 * mean pinball loss (standard quantile approximation)."""
    losses = []
    for lvl in LEVELS:
        q = quantiles[lvl]
        losses.append((y - q) * lvl if y >= q else (q - y) * (1 - lvl))
    return 2.0 * float(np.mean(losses))


# ---------------------------------------------------------------------------
# Diagnostics + selection policy (defined a priori)
# ---------------------------------------------------------------------------


def run_diagnostics(train: np.ndarray) -> dict:
    """Run the diagnostic battery on the training window (PIT-safe: train only)."""
    x = np.asarray(train, dtype=float)
    diag: dict = {}

    # Changepoints: count + recency.
    try:
        cp = detect_changepoints(x, min_distance=6)
        diag["n_changepoints"] = cp.n_segments - 1
        # Recent = changepoint in the last 6 observations.
        diag["n_recent_cps"] = sum(1 for c in cp.changepoints if c >= len(x) - 6)
    except Exception:  # noqa: BLE001
        diag["n_changepoints"] = 0
        diag["n_recent_cps"] = 0

    # Stationarity.
    try:
        st = test_stationarity(x)
        diag["stationarity"] = st.verdict
    except Exception:  # noqa: BLE001
        diag["stationarity"] = "inconclusive"

    # Trend / seasonal strength (needs >= 24 obs; we use 24-month windows).
    try:
        dc = decompose_series(x, period=12)
        diag["trend_strength"] = dc.trend_strength
        diag["seasonal_strength"] = dc.seasonal_strength
    except Exception:  # noqa: BLE001
        diag["trend_strength"] = 0.0
        diag["seasonal_strength"] = 0.0

    # Noise ratio: std(diff) / std(level). White noise ~1.41; smooth trend << 1.
    sx = float(np.std(x))
    diag["vol_ratio"] = float(np.std(np.diff(x)) / sx) if sx > 1e-9 else 0.0

    return diag


def select_model(diag: dict) -> str:
    """Rule-based selection policy, defined a priori from model properties.

    ETS adapts fastest to level shifts and handles trends; ARIMA's
    differencing suits seasonality and stable autocorrelation; M0 is the
    robust choice when noise dominates signal.
    """
    if diag["n_recent_cps"] > 0:
        return "ets"
    if diag["stationarity"] == "non_stationary":
        return "ets"
    if diag["trend_strength"] > 0.7:
        return "ets"
    if diag["seasonal_strength"] > 0.5:
        return "arima"
    if diag["vol_ratio"] > 1.2:
        return "m0"
    return "arima"


# ---------------------------------------------------------------------------
# Data: event-augmented segments with diverse event density
# ---------------------------------------------------------------------------


def _monthly_switch_rates(outcomes: list, month_dates: list[date]) -> pd.Series:
    """Aggregate renewal outcomes to a monthly switch-rate series."""
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
    s = pd.Series(rates).sort_index()
    return s.ffill().bfill()


def _month_range(start: date, n: int) -> list[date]:
    out = []
    y, m = start.year, start.month
    for _ in range(n):
        if m == 12:
            out.append(date(y, 12, 31))
        else:
            out.append(date.fromordinal(date(y, m + 1, 1).toordinal() - 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def build_segments(
    n_segments: int = 10,
    n_customers: int = 1500,
    n_months: int = 40,
    seed_base: int = 100,
) -> list[dict]:
    """Build diverse event-augmented segments.

    Event density varies 0 -> 8/year so regimes differ across segments:
    quiet segments favor stable models, eventful ones favor adaptive models.
    """
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
        sim = MortgageSimulator(
            n_customers=n_customers,
            n_months=n_months,
            seed=seed,
        )
        histories = sim.generate_history()
        outcomes = sim.generate_outcomes(histories, event_timeline=timeline)
        series = _monthly_switch_rates(outcomes, months)
        segments.append(
            {
                "segment_id": i,
                "seed": seed,
                "event_rate": rate,
                "n_events": len(timeline),
                "series": series,
                "dates": months,
            }
        )
        print(
            f"[exp23] segment {i}: {len(timeline)} events "
            f"(rate={rate}/yr), series len={len(series)}",
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
    """Rolling-origin backtest with diagnostic selection vs baselines."""
    rng = np.random.default_rng(vibes_seed)
    model_names = list(MODEL_FNS)

    # Per-origin records.
    records: list[dict] = []
    # For recency baseline: last winner per segment.
    last_winner: dict[int, str] = {}

    for seg in segments:
        sid = seg["segment_id"]
        vals = seg["series"].to_numpy(dtype=float)
        dates = seg["dates"]
        n_obs = len(vals)
        # Origins: last n_origins months that leave 1 month for the target.
        origins = list(range(n_obs - n_origins - 1, n_obs - 1))
        for oi in origins:
            train = vals[oi - train_window + 1 : oi + 1]
            y = float(vals[oi + 1])
            origin_date = dates[oi]
            if len(train) < train_window or not np.all(np.isfinite(train)):
                continue

            # Fit every model, score CRPS.
            crps: dict[str, float] = {}
            for name, fn in MODEL_FNS.items():
                point, resid = fn(train)
                rstd = float(np.std(resid)) if resid.size else float(np.std(train))
                rstd = max(rstd, 1e-6)
                qs = _quantiles_from_point(point, rstd)
                crps[name] = _crps_quantile(y, qs)

            # Diagnostics + selection.
            diag = run_diagnostics(train)
            chosen = select_model(diag)

            # Baselines for this origin.
            oracle_choice = min(crps, key=lambda k: crps[k])  # type: ignore[arg-type]
            random_choice = str(rng.choice(model_names))
            recency_choice = last_winner.get(sid, "arima")

            records.append(
                {
                    "segment": sid,
                    "origin": origin_date.isoformat(),
                    "chosen": chosen,
                    "crps": crps,
                    "oracle_choice": oracle_choice,
                    "random_choice": random_choice,
                    "recency_choice": recency_choice,
                    "diag": diag,
                }
            )
            last_winner[sid] = oracle_choice  # recency uses realized winner

    # Aggregate.
    methods = ["select", "oracle", "random", "recency"] + [f"single_{m}" for m in model_names]
    totals: dict[str, list[float]] = {m: [] for m in methods}
    choice_counts: dict[str, int] = defaultdict(int)
    for r in records:
        totals["select"].append(r["crps"][r["chosen"]])
        totals["oracle"].append(r["crps"][r["oracle_choice"]])
        totals["random"].append(r["crps"][r["random_choice"]])
        totals["recency"].append(r["crps"][r["recency_choice"]])
        for m in model_names:
            totals[f"single_{m}"].append(r["crps"][m])
        choice_counts[r["chosen"]] += 1

    mean_crps = {m: float(np.mean(v)) for m, v in totals.items() if v}
    # Best single model (chosen in hindsight on the test set — favors the baseline).
    single_means = {m: mean_crps[f"single_{m}"] for m in model_names}
    best_single = min(single_means, key=lambda k: single_means[k])  # type: ignore[arg-type]
    best_single_crps = single_means[best_single]
    oracle_crps = mean_crps["oracle"]
    select_crps = mean_crps["select"]
    denom = best_single_crps - oracle_crps
    gap_closed = (best_single_crps - select_crps) / denom if denom > 1e-12 else 0.0

    return {
        "n_records": len(records),
        "n_segments": len(segments),
        "mean_crps": mean_crps,
        "best_single_model": best_single,
        "best_single_crps": best_single_crps,
        "oracle_crps": oracle_crps,
        "selection_crps": select_crps,
        "oracle_gap_closed": float(gap_closed),
        "choice_counts": dict(choice_counts),
        "selection_policy": (
            "recent_cp->ets; non_stationary->ets; trend>0.7->ets; "
            "seasonal>0.5->arima; vol_ratio>1.2->m0; default arima"
        ),
        "records": records,
    }


def main() -> int:
    t0 = time.time()
    print("[exp23] building event-augmented segments...", flush=True)
    segments = build_segments()
    print("[exp23] running backtest...", flush=True)
    summary = run_backtest(segments)
    summary["elapsed_s"] = round(time.time() - t0, 1)

    print("\n[exp23] results:")
    for m, v in sorted(summary["mean_crps"].items(), key=lambda kv: kv[1]):
        print(f"  {m:15s} CRPS {v:.6f}")
    print(f"  oracle gap closed by selection: {summary['oracle_gap_closed']:.1%}")
    print(f"  selection choices: {summary['choice_counts']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp23] wrote {RESULTS_PATH} ({summary['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
