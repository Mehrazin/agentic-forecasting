"""Experiment 18b: Fair probabilistic ablation on Freddie Mac SMM.

Follow-up to exp18 addressing Argon's scoring-asymmetry critique: exp18
scored m0_w3/w6/w12 as point forecasts (CRPS = MAE) against TSFM quantile
distributions, inflating the "30% win" — on point MAE the TSFM edge is only
10-13%. exp18b applies uncertainty wrappers UNIFORMLY so every
(forecaster x wrapper) cell is comparable.

Matrix:
    forecasters: m0_w3, m0_w6, ewm_hl2, chronos2, timesfm25
    wrappers:
      - point:      deterministic point forecast (MAE only)
      - resid:      point +/- empirical residual quantiles, original space
                    (conformal-style; same construction as exp18's
                    conformal_quantiles but centered on each forecaster)
      - logcal:     LogScaleCalibrator around a quantile adapter
                    (log-space residual spreads -> multiplicative)

All quantile forecasts are interpolated onto a COMMON 19-level grid
(0.05..0.95) before CRPS, so the pinball estimator sees the same grid
for every cell.

Key question: does m0_w3 + logcal close the gap to timesfm25_cal?
(Argon predicts classical logcal CRPS ~0.00055-0.00057 vs 0.000515.)

PIT discipline: same as exp18 — train uses months <= origin only.

Usage:
    cd td-one && .venv/bin/python experiments/exp18b_fair_smm.py
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
from td_one.models.ewm import decay_weights  # noqa: E402
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper  # noqa: E402
from td_one.models.tsfm_calibration import LogScaleCalibrator  # noqa: E402

SMM_CSV = ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv"

N_ORIGINS = 50
MIN_HISTORY = 12
TSFM_SEED = 42

#: Common quantile grid for ALL cells (fair pinball CRPS).
COMMON_LEVELS = np.linspace(0.05, 0.95, 19)

#: Residual window for the resid wrapper.
RESID_WINDOW = 12

#: EWM lookback / half-life.
EWM_LOOKBACK = 12
EWM_HALF_LIFE = 2.0


def load_smm() -> pd.DataFrame:
    """Load the monthly SMM series, sorted ascending, with a datetime index."""
    df = pd.read_csv(SMM_CSV)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    df = df.sort_values("date").reset_index(drop=True)
    assert len(df) == 63, f"expected 63 months, got {len(df)}"
    return df


# ---------------------------------------------------------------------------
# Point forecasters (operate on the raw SMM rate series).
# ---------------------------------------------------------------------------

def m0_point(train: np.ndarray, window: int) -> float:
    """Trailing-window mean point forecast."""
    return float(np.mean(train[-window:]))


def ewm_point(train: np.ndarray, half_life: float = EWM_HALF_LIFE,
              lookback: int = EWM_LOOKBACK) -> float:
    """Exponentially-weighted trailing mean point forecast."""
    hist = train[-lookback:]
    w = decay_weights(len(hist), half_life)
    return float(np.average(hist, weights=w))


POINT_FNS = {
    "m0_w3": lambda tr: m0_point(tr, 3),
    "m0_w6": lambda tr: m0_point(tr, 6),
    "ewm_hl2": ewm_point,
}


# ---------------------------------------------------------------------------
# Quantile adapter: point forecaster -> forecast_quantiles via residuals.
# Works in whatever space the input series is in (original or log), so
# LogScaleCalibrator can wrap it for multiplicative spreads.
# ---------------------------------------------------------------------------

class ResidualQuantileAdapter:
    """Turn a point forecaster into a quantile forecaster.

    Quantiles = point +/- empirical quantiles of trailing absolute
    residuals. Accepts log-space (negative) inputs when
    ``validate_input=False`` so LogScaleCalibrator can wrap it.
    """

    def __init__(self, point_fn, resid_window: int = RESID_WINDOW,
                 levels: np.ndarray = COMMON_LEVELS) -> None:
        self.point_fn = point_fn
        self.resid_window = resid_window
        self.levels = np.asarray(levels, dtype=float)

    def forecast_quantiles(
        self,
        series: np.ndarray,
        horizon: int | None = None,
        validate_input: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            raise ValueError("series must be non-empty")
        if bool(np.isnan(arr).any()):
            raise ValueError("series must not contain NaN")
        if validate_input and bool((arr < 0.0).any()):
            raise ValueError("series must be non-negative")
        h = horizon or 1
        mu = float(self.point_fn(arr))
        hist = arr[-self.resid_window:]
        resid = np.abs(hist - self.point_fn(hist))
        # Empirical residual quantiles -> symmetric spread lattice.
        spreads = np.quantile(resid, np.abs(self.levels - 0.5) * 2.0)
        q = mu + np.sign(self.levels - 0.5) * spreads
        vals = np.tile(q[:, None], (1, h))
        return self.levels.copy(), vals


def interp_to_common(levels: np.ndarray, qvals: np.ndarray) -> np.ndarray:
    """Interpolate a quantile lattice onto COMMON_LEVELS (monotone)."""
    order = np.argsort(levels, kind="stable")
    lv = np.asarray(levels, dtype=float)[order]
    qv = np.maximum.accumulate(np.asarray(qvals, dtype=float)[order])
    return np.interp(COMMON_LEVELS, lv, qv)


def score_common(qvals_common: np.ndarray, actual: float) -> dict[str, float]:
    """CRPS / MAE / 80% coverage from common-grid quantiles."""
    crps = float(crps_from_quantiles(COMMON_LEVELS, qvals_common, actual))
    med = float(np.interp(0.5, COMMON_LEVELS, qvals_common))
    q10 = float(np.interp(0.1, COMMON_LEVELS, qvals_common))
    q90 = float(np.interp(0.9, COMMON_LEVELS, qvals_common))
    return {
        "crps": crps,
        "mae": abs(med - actual),
        "coverage_80": float(q10 <= actual <= q90),
        "pred_median": med,
    }


def main() -> int:
    t0 = time.time()
    df = load_smm()
    smm = df["smm"].to_numpy(dtype=float)
    dates = df["date"]
    print(f"[exp18b] {len(smm)} months", flush=True)

    origin_idx = list(range(MIN_HISTORY, len(smm) - 1))
    assert len(origin_idx) == N_ORIGINS

    print("[exp18b] loading Chronos-2 ...", flush=True)
    c2 = Chronos2Wrapper(use_mock=False, weights_dir=str(ROOT / "models" / "weights" / "chronos2"))
    cal2 = LogScaleCalibrator(c2)
    print("[exp18b] loading TimesFM-2.5 ...", flush=True)
    tfm = TimesFMWrapper(use_mock=False, weights_dir=str(ROOT / "models" / "weights" / "timesfm25"))
    calt = LogScaleCalibrator(tfm)

    # Classical adapters (quantile-capable, wrappable by LogScaleCalibrator).
    adapters = {name: ResidualQuantileAdapter(fn) for name, fn in POINT_FNS.items()}
    cal_adapters = {name: LogScaleCalibrator(ad) for name, ad in adapters.items()}

    cells = (
        [f"{m}__point" for m in POINT_FNS]
        + [f"{m}__resid" for m in POINT_FNS]
        + [f"{m}__logcal" for m in POINT_FNS]
        + ["chronos2__logcal", "timesfm25__logcal"]
    )

    per_origin: list[dict[str, object]] = []
    partial_path = ROOT / "experiments" / "exp18b_fair_smm_partial.json"
    done_origins: set[str] = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            per_origin = prev.get("per_origin", [])
            done_origins = {r["origin"] for r in per_origin}
            print(f"[exp18b] resuming: {len(done_origins)} origins done", flush=True)
        except Exception:  # noqa: BLE001 - corrupt partial: start fresh
            pass
    try:
        for oi, idx in enumerate(origin_idx):
            origin_label = dates.iloc[idx].strftime("%Y-%m")
            if origin_label in done_origins:
                continue
            train = smm[: idx + 1]
            actual = float(smm[idx + 1])
            row: dict[str, dict[str, float]] = {}

            # Classical: point / resid / logcal.
            for name, fn in POINT_FNS.items():
                p = float(fn(train))
                row[f"{name}__point"] = {"crps": abs(p - actual), "mae": abs(p - actual)}
                lv, vals = adapters[name].forecast_quantiles(train, horizon=1)
                row[f"{name}__resid"] = score_common(interp_to_common(lv, vals[:, 0]), actual)
                lv2, vals2 = cal_adapters[name].forecast_quantiles(train, horizon=1)
                row[f"{name}__logcal"] = score_common(interp_to_common(lv2, vals2[:, 0]), actual)

            # TSFMs: calibrated only (native uncalibrated blows up ~29x, exp16).
            for cname, cal in (("chronos2__logcal", cal2), ("timesfm25__logcal", calt)):
                try:
                    lv, vals = cal.forecast_quantiles(train, horizon=1, seed=TSFM_SEED)
                    row[cname] = score_common(interp_to_common(lv, vals[:, 0]), actual)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    print(f"[exp18b] {origin_label}: {cname} failed ({exc})", flush=True)
                    row[cname] = {"crps": float("nan"), "mae": float("nan"),
                                  "coverage_80": float("nan")}

            per_origin.append({"origin": origin_label, "actual": actual, "models": row})
            if len(per_origin) % 5 == 0:
                partial_path.write_text(json.dumps({"per_origin": per_origin}, default=float))
            if (oi + 1) % 5 == 0:
                print(f"[exp18b] {oi + 1}/{N_ORIGINS} done ({time.time() - t0:.0f}s)", flush=True)
    finally:
        c2.unload()
        tfm.unload()

    per_origin.sort(key=lambda r: r["origin"])
    summary: dict[str, dict[str, float]] = {}
    for m in cells:
        crps_v = [r["models"][m]["crps"] for r in per_origin if np.isfinite(r["models"][m]["crps"])]
        mae_v = [r["models"][m]["mae"] for r in per_origin if np.isfinite(r["models"][m]["mae"])]
        cov_v = [r["models"][m].get("coverage_80", float("nan")) for r in per_origin
                 if np.isfinite(r["models"][m].get("coverage_80", float("nan")))]
        summary[m] = {
            "mean_crps": float(np.mean(crps_v)),
            "mean_mae": float(np.mean(mae_v)),
            "coverage_80": float(np.mean(cov_v)) if cov_v else float("nan"),
            "n_origins": len(crps_v),
        }

    # Win rate on CRPS across fully-scored origins.
    wins: dict[str, int] = dict.fromkeys(cells, 0)
    scored = 0
    for r in per_origin:
        crpss = {m: r["models"][m]["crps"] for m in cells}
        if all(np.isfinite(v) for v in crpss.values()):
            best = min(crpss, key=lambda k: crpss[k])  # type: ignore[arg-type]
            wins[best] += 1
            scored += 1
    win_rate = {m: round(wins[m] / scored, 3) if scored else 0.0 for m in cells}

    # Headline: fair CRPS comparison.
    fair = {m: summary[m]["mean_crps"] for m in cells if "logcal" in m or "resid" in m}

    out = {
        "experiment": "exp18b_fair_smm",
        "config": {
            "series": "Freddie Mac monthly SMM, 2021-01..2026-03 (63 months)",
            "origins": N_ORIGINS,
            "origin_range": [per_origin[0]["origin"], per_origin[-1]["origin"]],
            "common_grid": "19 levels 0.05..0.95 for ALL cells (fair pinball CRPS)",
            "cells": cells,
            "tsfm_seed": TSFM_SEED,
            "wrappers": {
                "point": "deterministic point forecast; CRPS=MAE",
                "resid": "point +/- empirical trailing-12 residual quantiles, original space",
                "logcal": "LogScaleCalibrator(residual-quantile adapter): log-space "
                          "residual spreads -> multiplicative; applied to classical too",
            },
            "pit_note": "train uses months <= origin only",
            "limitations": "no ETS/ARIMA (not in repo); no native TSFM cell "
                           "(uncalibrated blows up ~29x, exp16); single TSFM seed",
        },
        "summary": summary,
        "win_rate": win_rate,
        "fair_crps_ranking": dict(sorted(fair.items(), key=lambda kv: kv[1])),
        "per_origin": per_origin,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp18b_fair_smm_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[exp18b] wrote {out_path}", flush=True)
    print("[exp18b] fair CRPS ranking:", json.dumps(out["fair_crps_ranking"], indent=2), flush=True)
    print("[exp18b] win_rate:", json.dumps(win_rate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
