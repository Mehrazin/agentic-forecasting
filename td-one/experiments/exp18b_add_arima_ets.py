"""exp18b extension: add ARIMA/ETS baselines, merge into existing results.

Adds arima / ets base forecasters with the same three wrappers
(point / resid / logcal) as the classical cells in exp18b, then merges
the new cells into experiments/exp18b_fair_smm_results.json without
re-running the TSFM cells.

Usage:
    cd td-one && .venv/bin/python experiments/exp18b_add_arima_ets.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Import helpers from the exp18b script itself.
_spec = importlib.util.spec_from_file_location(
    "exp18b", str(ROOT / "experiments" / "exp18b_fair_smm.py")
)
assert _spec is not None and _spec.loader is not None
_exp18b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp18b)

from td_one.models.tsfm_calibration import LogScaleCalibrator  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS_PATH = ROOT / "experiments" / "exp18b_fair_smm_results.json"


def arima_point(train: np.ndarray) -> float:
    """ARIMA(1,1,1) one-step point forecast. Falls back to trailing mean."""
    from statsmodels.tsa.arima.model import ARIMA

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(np.asarray(train, dtype=float), order=(1, 1, 1)).fit()
        return float(res.forecast(steps=1)[0])
    except Exception:  # noqa: BLE001 - fallback to trailing mean on fit failure
        return float(np.mean(train[-3:]))


def ets_point(train: np.ndarray) -> float:
    """Forecast one step with simple exponential smoothing (ETS(A,N,N))."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(np.asarray(train, dtype=float)).fit(
                optimized=True, use_brute=True
            )
        return float(res.forecast(1)[0])
    except Exception:  # noqa: BLE001 - fallback to trailing mean on fit failure
        return float(np.mean(train[-3:]))


NEW_POINT_FNS = {
    "arima": arima_point,
    "ets": ets_point,
}


def main() -> int:
    t0 = time.time()
    data = json.loads(RESULTS_PATH.read_text())
    per_origin: list[dict] = data["per_origin"]
    print(f"[ext] loaded {len(per_origin)} origins", flush=True)

    new_adapters = {
        name: _exp18b.ResidualQuantileAdapter(fn) for name, fn in NEW_POINT_FNS.items()
    }
    new_cal = {name: LogScaleCalibrator(ad) for name, ad in new_adapters.items()}

    new_cells = (
        [f"{m}__point" for m in NEW_POINT_FNS]
        + [f"{m}__resid" for m in NEW_POINT_FNS]
        + [f"{m}__logcal" for m in NEW_POINT_FNS]
    )

    df = _exp18b.load_smm()
    smm = df["smm"].to_numpy(dtype=float)
    dates = df["date"]
    origin_to_train = {
        dates.iloc[i].strftime("%Y-%m"): smm[: i + 1]
        for i in range(_exp18b.MIN_HISTORY, len(smm) - 1)
    }

    for oi, row in enumerate(per_origin):
        origin = row["origin"]
        actual = float(row["actual"])
        train = origin_to_train[origin]
        models = row["models"]
        for name, fn in NEW_POINT_FNS.items():
            p = float(fn(train))
            models[f"{name}__point"] = {"crps": abs(p - actual), "mae": abs(p - actual)}
            lv, vals = new_adapters[name].forecast_quantiles(train, horizon=1)
            models[f"{name}__resid"] = _exp18b.score_common(
                _exp18b.interp_to_common(lv, vals[:, 0]), actual
            )
            lv2, vals2 = new_cal[name].forecast_quantiles(train, horizon=1)
            models[f"{name}__logcal"] = _exp18b.score_common(
                _exp18b.interp_to_common(lv2, vals2[:, 0]), actual
            )
        if (oi + 1) % 10 == 0:
            print(f"[ext] {oi + 1}/{len(per_origin)} origins ({time.time() - t0:.0f}s)",
                  flush=True)

    # Recompute summary / win_rate / fair ranking over all cells.
    cells = data["config"]["cells"] + new_cells
    summary: dict[str, dict[str, float]] = {}
    for m in cells:
        crps_v = [r["models"][m]["crps"] for r in per_origin
                  if np.isfinite(r["models"][m]["crps"])]
        mae_v = [r["models"][m]["mae"] for r in per_origin
                 if np.isfinite(r["models"][m]["mae"])]
        cov_v = [r["models"][m].get("coverage_80", float("nan")) for r in per_origin
                 if np.isfinite(r["models"][m].get("coverage_80", float("nan")))]
        summary[m] = {
            "mean_crps": float(np.mean(crps_v)),
            "mean_mae": float(np.mean(mae_v)),
            "coverage_80": float(np.mean(cov_v)) if cov_v else float("nan"),
            "n_origins": len(crps_v),
        }

    wins: dict[str, int] = dict.fromkeys(cells, 0)
    scored = 0
    for r in per_origin:
        crpss = {m: r["models"][m]["crps"] for m in cells}
        if all(np.isfinite(v) for v in crpss.values()):
            best = min(crpss, key=lambda k: crpss[k])  # type: ignore[arg-type]
            wins[best] += 1
            scored += 1
    win_rate = {m: round(wins[m] / scored, 3) if scored else 0.0 for m in cells}
    fair = {m: summary[m]["mean_crps"] for m in cells if "logcal" in m or "resid" in m}

    data["config"]["cells"] = cells
    data["config"]["limitations"] = (
        data["config"].get("limitations", "")
        + "; ARIMA(1,1,1) + ETS(A,N,N) added via statsmodels 0.15.0, "
        "fallback to trailing mean on fit failure"
    )
    data["summary"] = summary
    data["win_rate"] = win_rate
    data["fair_crps_ranking"] = dict(sorted(fair.items(), key=lambda kv: kv[1]))
    data["per_origin"] = per_origin

    RESULTS_PATH.write_text(json.dumps(data, indent=2, default=float))
    print("[ext] updated", RESULTS_PATH, flush=True)
    print("[ext] fair CRPS ranking:",
          json.dumps(data["fair_crps_ranking"], indent=2), flush=True)
    print("[ext] elapsed", round(time.time() - t0, 1), "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
