"""Pilot: TD-One methodology on real WTI crude oil data.

Compares naive, ETS, ARIMA baselines against our agentic approach
(react_v2 + decision policy) on 2025 weekly origins, horizons 5/10/21.

Usage: .venv/bin/python benchmarks/energy_oil/pilot.py [--n_origins N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from td_one.agent.decision_policy import calibrated_quantiles, corroboration_gate, shrink_shift  # noqa: E402

LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_wti() -> pd.DataFrame:
    """Load daily WTI spot price, business-day calendar."""
    df = pd.read_parquet(REPO / "data" / "fred" / "DCOILWTICO.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Business-day calendar, forward fill weekends/holidays
    bdays = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bdays, method="ffill").dropna()
    df.columns = ["price"]
    return df


def crps_from_quantiles(y: float, quantiles: dict[float, float]) -> float:
    """CRPS via quantile scoring rule: mean over levels of pinball loss * 2."""
    losses = []
    for q, v in quantiles.items():
        losses.append(2 * ((y - v) * q if y >= v else (v - y) * (1 - q)))
    return float(np.mean(losses))


def forecast_ets(train: np.ndarray, h: int) -> tuple[float, float]:
    """ETS(A,N,N) point + residual std. Falls back to naive."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(train, trend=None, seasonal=None)
            fit = model.fit(optimized=True, use_brute=True)
            point = float(fit.forecast(h)[-1])
            resid = train[1:] - fit.fittedvalues[1:]
            sigma = float(np.std(resid)) * np.sqrt(h)
            return point, max(sigma, 1e-6)
    except Exception:
        return float(train[-1]), float(np.std(np.diff(train)) * np.sqrt(h) + 1e-6)


def forecast_arima(train: np.ndarray, h: int) -> tuple[float, float]:
    """ARIMA(1,1,1) point + std. Falls back to naive."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train, order=(1, 1, 1))
            fit = model.fit()
            fc = fit.get_forecast(steps=h)
            point = float(fc.predicted_mean.iloc[-1])
            sigma = float(np.sqrt(fc.se_mean.iloc[-1] ** 2 + np.var(fit.resid)))
            return point, max(sigma, 1e-6)
    except Exception:
        return float(train[-1]), float(np.std(np.diff(train)) * np.sqrt(h) + 1e-6)


def quantiles_from_gaussian(point: float, sigma: float) -> dict[float, float]:
    """Gaussian quantiles for CRPS scoring."""
    from scipy.stats import norm

    return {q: float(point + sigma * norm.ppf(q)) for q in LEVELS}


def detect_shift(train: np.ndarray) -> tuple[float, float, int]:
    """Two-sample t-test: last 5 vs prior 20. Returns (shift, t_stat, n)."""
    from scipy.stats import ttest_ind

    if len(train) < 30:
        return 0.0, 0.0, 0
    recent = train[-5:]
    prior = train[-25:-5]
    shift = float(np.mean(recent) - np.mean(prior))
    try:
        t_stat = float(ttest_ind(recent, prior, equal_var=False).statistic)
    except Exception:
        t_stat = 0.0
    return shift, t_stat, len(recent)


def agent_forecast(
    train: np.ndarray,
    h: int,
    base_point: float,
    base_sigma: float,
    hist_vol: float,
) -> tuple[dict[float, float], bool, dict]:
    """Our agentic approach: detect shift, apply decision policy.

    Returns (quantiles, adjusted, info).
    """
    shift, t_stat, n = detect_shift(train)
    info = {"shift": shift, "t_stat": t_stat, "n": n}

    # Decision policy from td_one.agent.decision_policy
    shrunk = shrink_shift(shift, n)
    info["shrunk_shift"] = shrunk

    # Corroboration gate: |shift| >= 2% of price AND (|t| > 3.0)
    # (no news feed in this pilot, so n_news_30d=0)
    price_scale = abs(float(train[-1]))
    gate, reason = corroboration_gate(
        t_stat=abs(t_stat),
        shift=abs(shrunk) / price_scale,
        n_news_30d=0,
        min_shift=0.02,
        t_strict=3.0,
    )
    info["gate"] = gate
    info["gate_reason"] = reason

    if not gate:
        return quantiles_from_gaussian(base_point, base_sigma), False, info

    adj_point = base_point + shrunk
    # For WTI (not near-zero like SMM), use Gaussian quantiles with
    # volatility-floored sigma: max(base_sigma, 0.5 * hist_vol_price * sqrt(h))
    # where hist_vol_price is price-level volatility
    hist_price_vol = hist_vol * abs(float(train[-1]))  # ann vol -> price vol
    # Daily vol * sqrt(h)
    floor_sigma = 0.5 * hist_price_vol / np.sqrt(252) * np.sqrt(h)
    adj_sigma = max(base_sigma, floor_sigma)
    # Inflate for small-n uncertainty
    adj_sigma *= np.sqrt(1 + 1 / max(n, 1))
    q = quantiles_from_gaussian(adj_point, adj_sigma)
    return q, True, info


def run_pilot(n_origins: int = 20) -> dict:
    """Run pilot on last n_origins weekly origins of 2025."""
    df = load_wti()
    prices = df["price"]

    # Weekly origins in 2025 (every 5 business days)
    # Use full history for warmup; origins need 250 days before + 21 after
    y2025_idx = prices.index.get_indexer_for(prices["2025-01-01":"2025-12-31"].index)
    origins = []
    for i in range(0, len(y2025_idx), 5):
        gi = y2025_idx[i]  # global index
        if gi >= 250 and gi + 25 < len(prices):
            origins.append(prices.index[gi])
    origins = origins[-n_origins:]
    print(f"WTI data: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Origins: {len(origins)} ({origins[0].date()} -> {origins[-1].date()})")

    horizons = [5, 10, 21]
    results = []

    for oi, origin in enumerate(origins):
        train = prices.loc[:origin].iloc[-250:].to_numpy()
        hist_vol = float(np.std(np.diff(np.log(train[-60:])))) * np.sqrt(252)  # ann vol
        if hist_vol <= 0:
            hist_vol = 0.3

        for h in horizons:
            # Actual: h business days ahead
            idx = prices.index.get_loc(origin)
            if idx + h >= len(prices):
                continue
            actual = float(prices.iloc[idx + h])

            row: dict = {
                "origin": origin.strftime("%Y-%m-%d"),
                "h": h,
                "actual": actual,
            }

            # Naive
            naive_q = quantiles_from_gaussian(float(train[-1]), float(np.std(np.diff(train)) * np.sqrt(h)))
            row["naive_crps"] = crps_from_quantiles(actual, naive_q)
            row["naive_ae"] = abs(actual - train[-1])

            # ETS
            ets_point, ets_sigma = forecast_ets(train, h)
            ets_q = quantiles_from_gaussian(ets_point, ets_sigma)
            row["ets_crps"] = crps_from_quantiles(actual, ets_q)
            row["ets_ae"] = abs(actual - ets_point)

            # ARIMA
            ar_point, ar_sigma = forecast_arima(train, h)
            ar_q = quantiles_from_gaussian(ar_point, ar_sigma)
            row["arima_crps"] = crps_from_quantiles(actual, ar_q)
            row["arima_ae"] = abs(actual - ar_point)

            # Agent (ETS base + detection + decision policy)
            ag_q, adjusted, info = agent_forecast(train, h, ets_point, ets_sigma, hist_vol)
            row["agent_crps"] = crps_from_quantiles(actual, ag_q)
            row["agent_ae"] = abs(actual - (ets_point + info.get("shrunk_shift", 0) if adjusted else ets_point))
            row["agent_adjusted"] = adjusted
            row["agent_t_stat"] = info["t_stat"]

            results.append(row)

        if (oi + 1) % 5 == 0:
            print(f"  [{oi + 1}/{len(origins)}] done", flush=True)

    # Summarize
    pdf = pd.DataFrame(results)
    summary = {"n_origins": len(origins), "n_forecasts": len(pdf), "horizons": horizons}
    for h in horizons:
        sub = pdf[pdf["h"] == h]
        summary[f"h{h}"] = {
            "naive_crps": float(sub["naive_crps"].mean()),
            "ets_crps": float(sub["ets_crps"].mean()),
            "arima_crps": float(sub["arima_crps"].mean()),
            "agent_crps": float(sub["agent_crps"].mean()),
            "naive_mae": float(sub["naive_ae"].mean()),
            "ets_mae": float(sub["ets_ae"].mean()),
            "arima_mae": float(sub["arima_ae"].mean()),
            "agent_mae": float(sub["agent_ae"].mean()),
            "agent_adjust_rate": float(sub["agent_adjusted"].mean()),
        }
        # Agent vs ETS paired test
        from scipy.stats import ttest_rel

        try:
            t, p = ttest_rel(sub["ets_crps"], sub["agent_crps"])
            summary[f"h{h}"]["agent_vs_ets_t"] = float(t)
            summary[f"h{h}"]["agent_vs_ets_p"] = float(p)
        except Exception:
            summary[f"h{h}"]["agent_vs_ets_p"] = float("nan")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_origins", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    summary = run_pilot(args.n_origins)
    summary["runtime_s"] = time.time() - t0

    print("\n=== PILOT RESULTS (WTI 2025) ===")
    for h in [5, 10, 21]:
        s = summary[f"h{h}"]
        print(f"\nh={h}:")
        print(f"  naive  CRPS {s['naive_crps']:.4f}  MAE {s['naive_mae']:.4f}")
        print(f"  ets    CRPS {s['ets_crps']:.4f}  MAE {s['ets_mae']:.4f}")
        print(f"  arima  CRPS {s['arima_crps']:.4f}  MAE {s['arima_mae']:.4f}")
        print(f"  agent  CRPS {s['agent_crps']:.4f}  MAE {s['agent_mae']:.4f}  (adj {s['agent_adjust_rate']:.0%})")
        imp = (s["ets_crps"] - s["agent_crps"]) / s["ets_crps"]
        print(f"  agent vs ets: {imp:+.1%}  p={s['agent_vs_ets_p']:.3f}")

    out = args.out or str(REPO / "benchmarks" / "energy_oil" / "pilot_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
