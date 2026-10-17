"""Experiment 16: Freddie Mac real-data SMM forecasting (proof of concept).

First real-data forecasting experiment on Freddie Mac loan-level data.
Forecasts monthly SMM (single-month mortality / prepayment rate) for
cohorts defined by origination vintage x FICO band.

DESIGN (honest limitations):
- Only 3 months of performance data (2026-01 to 2026-03). Train on Jan+Feb,
  forecast Mar. This is EXTREMELY thin — 2 training points per cohort.
- This is a proof-of-concept for the real-data pipeline, NOT a definitive
  model comparison. The signal may not be forecastable with 2 points.

PIT DISCIPLINE:
- Cohort SMM for month m uses only perf rows with period == m.
- Backtest at origin 2026-02-28: train on SMM[2026-01], SMM[2026-02];
  test on SMM[2026-03]. No future perf rows touch the models.
- See td_one.freddie.monthly_prepay_rate for the SMM computation.

Models (forecast next month's SMM):
- m0: trailing mean of available SMM history (point forecast).
- m0_conformal: trailing mean +/- empirical residual quantile (interval).
- chronos2: real Chronos-2 TSFM on the SMM series (21 quantiles).

Scoring: CRPS via pinball-loss estimator (crps_from_quantiles);
for point forecasts, CRPS = MAE. Also report MAE and 80% coverage.

Usage:
    cd td-one && uv run python experiments/exp16_freddie_smm.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.evaluation.real_benchmark import crps_from_quantiles  # noqa: E402
from td_one.freddie import load_vintage, monthly_prepay_rate  # noqa: E402
from td_one.models.tsfm import Chronos2Wrapper  # noqa: E402

DATA_DIR = ROOT / "data" / "freddie" / "raw"
VINTAGE = "2026Q1"
FORECAST_MONTH = pd.Timestamp("2026-03-01")

# Cohort definition: vintage (first_payment_date) x FICO band.
VINTAGES = ["202602", "202603"]
FICO_BANDS = [
    ("lt700", 0, 700),
    ("700_750", 700, 750),
    ("gte750", 750, 9000),
]

#: Minimum loans at risk for a cohort-month to be usable. January 2026
#: has degenerate samples for the 202603 vintage (n=1..3) since those
#: loans barely existed yet. Months below this are dropped.
MIN_AT_RISK = 100


def _fico_band(score: float) -> str | None:
    """Map a FICO score to a band label. Returns None for missing/invalid."""
    if not np.isfinite(score) or score < 300 or score >= 9000:
        return None
    for label, lo, hi in FICO_BANDS:
        if lo <= score < hi:
            return label
    return None


def build_cohort_smm(data_dir: Path, vintage: str) -> dict[str, pd.DataFrame]:
    """Build per-cohort monthly SMM panels.

    Returns {cohort_key: DataFrame(period, n_at_risk, n_prepay, smm)} where
    cohort_key = f"{first_payment_date}_{fico_band}".
    Each cohort's SMM uses only its own loans' perf rows (PIT-safe by
    construction: monthly_prepay_rate groups by period, no cross-month leak).
    """
    data = load_vintage(data_dir, vintage)
    orig = data.orig.copy()
    perf = data.perf.copy()

    orig["fico_band"] = orig["credit_score"].apply(_fico_band)
    # Keep only the two vintages with meaningful performance history.
    orig = orig[orig["first_payment_date"].isin(VINTAGES)]
    orig = orig[orig["fico_band"].notna()]
    if orig.empty:
        raise ValueError("no loans after vintage/FICO filtering")

    # Map loan -> cohort, then filter perf to those loans.
    loan_cohort = orig.set_index("loan_seq")[["first_payment_date", "fico_band"]]
    perf = perf[perf["loan_seq"].isin(loan_cohort.index)]
    perf = perf.join(loan_cohort, on="loan_seq")
    perf["cohort"] = perf["first_payment_date"] + "_" + perf["fico_band"]

    cohorts: dict[str, pd.DataFrame] = {}
    for key, grp in perf.groupby("cohort"):
        smm = monthly_prepay_rate(grp)
        # Drop degenerate months (tiny n_at_risk, e.g. Jan 2026 for the
        # 202603 vintage). Document the drop in the results.
        smm = smm[smm["n_at_risk"] >= MIN_AT_RISK].reset_index(drop=True)
        cohorts[str(key)] = smm
    return cohorts


def m0_forecast(train_smm: np.ndarray) -> float:
    """Trailing-mean point forecast."""
    return float(np.mean(train_smm))


def m0_conformal_quantiles(
    train_smm: np.ndarray, levels: np.ndarray, alpha: float = 0.2
) -> np.ndarray:
    """Trailing mean +/- empirical residual quantile.

    With n training points, residuals are |smm_i - mean|; the interval
    half-width is the (1-alpha) empirical quantile of residuals.
    Quantiles are symmetric around the mean, clipped to [0, 1].
    """
    mu = float(np.mean(train_smm))
    resid = np.abs(train_smm - mu)
    # Empirical (1-alpha) quantile; with 2 points this is max(resid).
    width = float(np.quantile(resid, 1 - alpha)) if len(resid) > 0 else 0.0
    lo = np.clip(mu - width, 0.0, 1.0)
    hi = np.clip(mu + width, 0.0, 1.0)
    # Map levels to quantiles: linear interpolation between lo/mu/hi.
    # levels < 0.5 -> [lo, mu], levels >= 0.5 -> [mu, hi].
    q = np.where(
        levels < 0.5,
        lo + (mu - lo) * (levels / 0.5),
        mu + (hi - mu) * ((levels - 0.5) / 0.5),
    )
    return np.clip(q, 0.0, 1.0)


def chronos2_forecast(
    wrapper: Chronos2Wrapper, train_smm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Chronos-2 one-step-ahead quantile forecast on the SMM series."""
    levels, values = wrapper.forecast_quantiles(train_smm, horizon=1)
    return np.asarray(levels, dtype=float), np.asarray(values[:, 0], dtype=float)


def main() -> int:
    t0 = time.time()
    print("[exp16] Loading Freddie Mac 2026Q1 ...", flush=True)
    cohorts = build_cohort_smm(DATA_DIR, VINTAGE)
    print(f"[exp16] {len(cohorts)} cohorts", flush=True)
    for key, smm in sorted(cohorts.items()):
        print(
            f"[exp16]   {key}: n_at_risk={list(smm['n_at_risk'])}, "
            f"smm={[round(v, 5) for v in smm['smm']]}",
            flush=True,
        )

    # PIT split: train on months < 2026-03, test on 2026-03.
    # After MIN_AT_RISK filtering, some cohorts have 1 train month.

    results: dict[str, dict[str, dict[str, float]]] = {}
    dropped: dict[str, list[str]] = {}
    wrapper = Chronos2Wrapper(use_mock=False, weights_dir="models/weights/chronos2")
    try:
        for key in sorted(cohorts):
            df = cohorts[key]
            kept = {p.strftime("%Y-%m") for p in df["period"].dt.to_period("M")}
            dropped[key] = [m for m in ("2026-01", "2026-02", "2026-03") if m not in kept]
            smm = df.set_index("period")["smm"]
            train = smm[smm.index < FORECAST_MONTH].to_numpy(dtype=float)
            if len(train) < 1:
                print(f"[exp16] {key}: no train months after filtering, skipping", flush=True)
                continue
            if FORECAST_MONTH not in smm.index:
                print(f"[exp16] {key}: no 2026-03 observation, skipping", flush=True)
                continue
            actual = float(smm.loc[FORECAST_MONTH])

            row: dict[str, dict[str, float]] = {}

            # m0: point forecast -> CRPS = MAE.
            p0 = m0_forecast(train)
            row["m0"] = {
                "crps": abs(p0 - actual),
                "mae": abs(p0 - actual),
                "pred": p0,
                "actual": actual,
            }

            # m0_conformal: quantile forecast.
            clevels = np.linspace(0.05, 0.95, 19)
            cq = m0_conformal_quantiles(train, clevels)
            row["m0_conformal"] = {
                "crps": crps_from_quantiles(clevels, cq, actual),
                "mae": abs(float(np.mean(train)) - actual),
                "coverage_80": float(cq[1] <= actual <= cq[-2]),
                "pred_median": float(cq[9]),
                "actual": actual,
            }

            # chronos2: real TSFM quantiles.
            try:
                c2_levels, c2_q = chronos2_forecast(wrapper, train)
                # Sort by level (Chronos2 may emit out-of-order).
                order = np.argsort(c2_levels, kind="stable")
                c2_levels = c2_levels[order]
                c2_q = np.sort(c2_q[order])  # monotone quantiles
                row["chronos2"] = {
                    "crps": crps_from_quantiles(c2_levels, c2_q, actual),
                    "mae": abs(float(c2_q[len(c2_q) // 2]) - actual),
                    "coverage_80": float(
                        (c2_q[2] <= actual <= c2_q[-3]) if len(c2_q) >= 6 else 0.0
                    ),
                    "pred_median": float(c2_q[len(c2_q) // 2]),
                    "actual": actual,
                }
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"[exp16] {key}: chronos2 failed ({exc})", flush=True)
                row["chronos2"] = {"crps": float("nan"), "mae": float("nan")}

            results[key] = row
            print(
                f"[exp16] {key}: m0_crps={row['m0']['crps']:.6f} "
                f"conf_crps={row['m0_conformal']['crps']:.6f} "
                f"c2_crps={row['chronos2'].get('crps', float('nan')):.6f}",
                flush=True,
            )
    finally:
        wrapper.unload()

    # Aggregate: mean CRPS/MAE across cohorts (NaN-safe).
    summary: dict[str, dict[str, float]] = {}
    for model in ("m0", "m0_conformal", "chronos2"):
        crps_vals = [
            r[model]["crps"]
            for r in results.values()
            if np.isfinite(r[model].get("crps", float("nan")))
        ]
        mae_vals = [
            r[model]["mae"]
            for r in results.values()
            if np.isfinite(r[model].get("mae", float("nan")))
        ]
        summary[model] = {
            "mean_crps": float(np.mean(crps_vals)) if crps_vals else float("nan"),
            "mean_mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
            "n_cohorts": len(crps_vals),
        }

    out = {
        "experiment": "exp16_freddie_smm",
        "config": {
            "vintage": VINTAGE,
            "cohorts": sorted(results.keys()),
            "train_months": ["2026-01", "2026-02"],
            "forecast_month": "2026-03",
            "min_at_risk": MIN_AT_RISK,
            "dropped_months": dropped,
            "pit_note": "train uses perf period < 2026-03-01 only; "
            "cohort SMM from cohort's own loans only",
            "limitation": "1-2 training points per cohort after filtering; "
            "proof of concept only, not a model comparison",
        },
        "per_cohort": results,
        "summary": summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp16_freddie_smm_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[exp16] wrote {out_path}", flush=True)
    print("[exp16] summary:", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
