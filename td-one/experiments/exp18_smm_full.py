"""Experiment 18: Full 63-month Freddie Mac SMM benchmark (calibrated TSFMs).

Rolling-origin backtest on the Freddie Mac monthly prepayment (SMM)
series, Jan 2021 - Mar 2026 (63 months,
``data/freddie/smm_monthly_2021_2026.csv``).

This is the real benchmark exp16 pointed at: 50 rolling origins
(2022-01 .. 2026-02), train on months <= origin, forecast origin+1.

Models:
- m0_w3 / m0_w6 / m0_w12: trailing-mean point forecasts.
- m0_conformal: trailing-12 mean +/- empirical residual quantiles
  (same construction as exp16).
- chronos2_cal: Chronos-2 (real weights) + LogScaleCalibrator.
- timesfm25_cal: TimesFM-2.5 (real weights) + LogScaleCalibrator.

The LogScaleCalibrator is REQUIRED here: raw TSFM spreads are
calibrated in absolute terms and blow up 29x on near-zero SMM
(exp16). Log-space forecasting makes spreads relative.

Scoring: CRPS via the pinball-loss estimator (``crps_from_quantiles``,
same as exp16 — the sample-based ``crps_counts`` in
``evaluation/metrics.py`` needs predictive draws, not quantiles);
MAE on the median/point; 80% interval coverage.

PIT discipline: the SMM CSV is precomputed per calendar month;
slicing train to months <= origin guarantees no lookahead. Each
month's SMM uses only that month's perf rows.

Usage:
    cd td-one && uv run python experiments/exp18_smm_full.py
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
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper  # noqa: E402
from td_one.models.tsfm_calibration import LogScaleCalibrator  # noqa: E402

SMM_CSV = ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv"

#: First origin needs >=12 months of history: origins are 2022-01..2026-02.
N_ORIGINS = 50
MIN_HISTORY = 12

#: Quantile grid for the conformal model (must include 0.1/0.9/0.5).
CONF_LEVELS = np.linspace(0.05, 0.95, 19)


def load_smm() -> pd.DataFrame:
    """Load the monthly SMM series, sorted ascending, with a datetime index."""
    df = pd.read_csv(SMM_CSV)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    df = df.sort_values("date").reset_index(drop=True)
    assert len(df) == 63, f"expected 63 months, got {len(df)}"
    assert df["date"].iloc[0] == pd.Timestamp("2021-01-01")
    assert df["date"].iloc[-1] == pd.Timestamp("2026-03-01")
    return df


def m0_point(train: np.ndarray, window: int) -> float:
    """Trailing-window mean point forecast."""
    return float(np.mean(train[-window:]))


def conformal_quantiles(
    train: np.ndarray, levels: np.ndarray, window: int = 12, alpha: float = 0.2
) -> np.ndarray:
    """Trailing-window mean +/- empirical residual quantiles (exp16 method)."""
    hist = train[-window:]
    mu = float(np.mean(hist))
    resid = np.abs(hist - mu)
    width = float(np.quantile(resid, 1 - alpha)) if len(resid) > 0 else 0.0
    lo = max(mu - width, 0.0)
    hi = mu + width
    q = np.where(
        levels < 0.5,
        lo + (mu - lo) * (levels / 0.5),
        mu + (hi - mu) * ((levels - 0.5) / 0.5),
    )
    return np.clip(q, 0.0, None)


def interval_from_quantiles(
    levels: np.ndarray, qvals: np.ndarray, lo: float = 0.1, hi: float = 0.9
) -> tuple[float, float]:
    """Interpolate (lo, hi) interval bounds from a quantile lattice."""
    order = np.argsort(levels, kind="stable")
    lv = np.asarray(levels, dtype=float)[order]
    qv = np.asarray(qvals, dtype=float)[order]
    # Ensure monotone for interpolation.
    qv = np.maximum.accumulate(qv)
    return float(np.interp(lo, lv, qv)), float(np.interp(hi, lv, qv))


def median_from_quantiles(levels: np.ndarray, qvals: np.ndarray) -> float:
    """Interpolated median from a quantile lattice."""
    order = np.argsort(levels, kind="stable")
    lv = np.asarray(levels, dtype=float)[order]
    qv = np.asarray(qvals, dtype=float)[order]
    qv = np.maximum.accumulate(qv)
    return float(np.interp(0.5, lv, qv))


def score_quantiles(levels: np.ndarray, qvals: np.ndarray, actual: float) -> dict[str, float]:
    """CRPS / MAE / 80% coverage from a quantile forecast."""
    crps = float(crps_from_quantiles(np.asarray(levels), np.asarray(qvals), actual))
    med = median_from_quantiles(levels, qvals)
    q10, q90 = interval_from_quantiles(levels, qvals)
    return {
        "crps": crps,
        "mae": abs(med - actual),
        "coverage_80": float(q10 <= actual <= q90),
        "pred_median": med,
        "q10": q10,
        "q90": q90,
    }


def main() -> int:
    t0 = time.time()
    df = load_smm()
    smm = df["smm"].to_numpy(dtype=float)
    dates = df["date"]
    print(f"[exp18] {len(smm)} months: {dates.iloc[0].date()}..{dates.iloc[-1].date()}", flush=True)

    # Origins: indices MIN_HISTORY .. len-2  (forecast idx+1 <= len-1).
    origin_idx = list(range(MIN_HISTORY, len(smm) - 1))
    assert len(origin_idx) == N_ORIGINS, f"expected {N_ORIGINS} origins, got {len(origin_idx)}"
    print(
        f"[exp18] {N_ORIGINS} origins: "
        f"{dates.iloc[origin_idx[0]].strftime('%Y-%m')}..{dates.iloc[origin_idx[-1]].strftime('%Y-%m')}",
        flush=True,
    )

    print("[exp18] loading Chronos-2 ...", flush=True)
    c2 = Chronos2Wrapper(use_mock=False, weights_dir=str(ROOT / "models" / "weights" / "chronos2"))
    cal2 = LogScaleCalibrator(c2)
    print("[exp18] loading TimesFM-2.5 ...", flush=True)
    tfm = TimesFMWrapper(use_mock=False, weights_dir=str(ROOT / "models" / "weights" / "timesfm25"))
    calt = LogScaleCalibrator(tfm)

    per_origin: list[dict[str, object]] = []
    # Resume: skip origins already in a partial results file.
    partial_path = ROOT / "experiments" / "exp18_smm_full_partial.json"
    done_origins: set[str] = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            per_origin = prev.get("per_origin", [])
            done_origins = {r["origin"] for r in per_origin}
            print(f"[exp18] resuming: {len(done_origins)} origins already done", flush=True)
        except Exception:  # noqa: BLE001 - corrupt partial file: start fresh
            pass
    try:
        for oi, idx in enumerate(origin_idx):
            origin_label = dates.iloc[idx].strftime("%Y-%m")
            if origin_label in done_origins:
                continue
            train = smm[: idx + 1]
            actual = float(smm[idx + 1])
            row: dict[str, dict[str, float]] = {}

            # Classical trailing means (point -> CRPS = MAE).
            for w in (3, 6, 12):
                p = m0_point(train, w)
                row[f"m0_w{w}"] = {"crps": abs(p - actual), "mae": abs(p - actual), "pred": p}

            # Conformalized.
            cq = conformal_quantiles(train, CONF_LEVELS)
            row["m0_conformal"] = score_quantiles(CONF_LEVELS, cq, actual)

            # Calibrated TSFMs.
            for name, cal in (("chronos2_cal", cal2), ("timesfm25_cal", calt)):
                try:
                    lv, vals = cal.forecast_quantiles(train, horizon=1)
                    row[name] = score_quantiles(lv, vals[:, 0], actual)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    print(f"[exp18] {origin_label}: {name} failed ({exc})", flush=True)
                    row[name] = {
                        "crps": float("nan"),
                        "mae": float("nan"),
                        "coverage_80": float("nan"),
                    }

            per_origin.append({"origin": origin_label, "actual": actual, "models": row})
            # Incremental save every 5 origins (survive crashes).
            if len(per_origin) % 5 == 0:
                partial_path.write_text(json.dumps({"per_origin": per_origin}, default=float))
            if (oi + 1) % 5 == 0:
                print(
                    f"[exp18] {oi + 1}/{N_ORIGINS} origins done ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    finally:
        c2.unload()
        tfm.unload()

    # Summary: mean CRPS / MAE / coverage per model (NaN-safe).
    # Sort by origin (resume may have appended out of order).
    per_origin.sort(key=lambda r: r["origin"])
    model_names = ["m0_w3", "m0_w6", "m0_w12", "m0_conformal", "chronos2_cal", "timesfm25_cal"]
    summary: dict[str, dict[str, float]] = {}
    for m in model_names:
        crps_v = [r["models"][m]["crps"] for r in per_origin if np.isfinite(r["models"][m]["crps"])]
        mae_v = [r["models"][m]["mae"] for r in per_origin if np.isfinite(r["models"][m]["mae"])]
        cov_v = [
            r["models"][m].get("coverage_80", float("nan"))
            for r in per_origin
            if np.isfinite(r["models"][m].get("coverage_80", float("nan")))
        ]
        summary[m] = {
            "mean_crps": float(np.mean(crps_v)),
            "mean_mae": float(np.mean(mae_v)),
            "coverage_80": float(np.mean(cov_v)) if cov_v else float("nan"),
            "n_origins": len(crps_v),
        }

    # Win rate: fraction of origins where each model has the lowest CRPS.
    wins: dict[str, int] = dict.fromkeys(model_names, 0)
    scored = 0
    for r in per_origin:
        crpss = {m: r["models"][m]["crps"] for m in model_names}
        if all(np.isfinite(v) for v in crpss.values()):
            best = min(crpss, key=lambda k: crpss[k])  # type: ignore[arg-type]
            wins[best] += 1
            scored += 1
    win_rate = {m: round(wins[m] / scored, 3) if scored else 0.0 for m in model_names}

    # Regime breakdown: yearly mean CRPS per model.
    by_year: dict[str, dict[str, float]] = {}
    for r in per_origin:
        yr = r["origin"][:4]
        by_year.setdefault(yr, {}).setdefault("rows", []).append(r)
    yearly: dict[str, dict[str, float]] = {}
    for yr, blob in sorted(by_year.items()):
        rows = blob["rows"]
        yearly[yr] = {}
        for m in model_names:
            vals = [x["models"][m]["crps"] for x in rows if np.isfinite(x["models"][m]["crps"])]
            yearly[yr][m] = round(float(np.mean(vals)), 6) if vals else float("nan")

    out = {
        "experiment": "exp18_smm_full",
        "config": {
            "series": "Freddie Mac monthly SMM, 2021-01..2026-03 (63 months)",
            "origins": N_ORIGINS,
            "origin_range": [per_origin[0]["origin"], per_origin[-1]["origin"]],
            "min_history_months": MIN_HISTORY,
            "models": model_names,
            "crps_method": "pinball-loss estimator crps_from_quantiles (same as exp16); "
            "point forecasts scored as CRPS=MAE",
            "pit_note": "train uses months <= origin only; SMM CSV precomputed per-month",
            "calibration_note": "TSFMs wrapped in LogScaleCalibrator (log-space for "
            "max<0.01); required tsfm.py fix so validate_input=False also skips "
            "the [0,1] OUTPUT clip on the real path",
        },
        "summary": summary,
        "win_rate": win_rate,
        "yearly_crps": yearly,
        "per_origin": per_origin,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp18_smm_full_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[exp18] wrote {out_path}", flush=True)
    print("[exp18] summary:", json.dumps(summary, indent=2), flush=True)
    print("[exp18] win_rate:", json.dumps(win_rate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
