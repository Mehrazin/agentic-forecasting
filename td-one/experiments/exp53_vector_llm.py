"""exp53: Vector's QuantileGridLLMPredictor vs our ProposedForecaster, head-to-head.

Runs Vector's actual LLM predictor class (not a reimplementation) against
Groq's gpt-oss-120b via a local auth proxy, on the SAME 59 Food CPI origins
and SAME CRPS scoring as exp52.

Adaptations from Vector defaults (documented in exp53_report.md):
- model: gpt-oss-120b (their default is a Vector-proxy Gemini; unavailable here)
- max_tokens: 2048 (their 16384 default risks Groq HTTP 413)
- reasoning_effort: "low" (exp48 evidence; their default None = provider default)
Everything else (temperature=1.0, full history, JSON-schema elicitation) is theirs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AIENG = Path(__file__).resolve().parents[2] / "aieng-forecasting"
sys.path.insert(0, str(AIENG))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieng.forecasting.data import DataService, SeriesMetadata  # noqa: E402
from aieng.forecasting.data.adapters.base import BaseAdapter  # noqa: E402
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES  # noqa: E402
from aieng.forecasting.evaluation.task import ForecastingTask  # noqa: E402
from aieng.forecasting.methods.llm_processes.quantile_grid import (  # noqa: E402
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
)

DATA_CSV = Path(__file__).resolve().parents[1] / "benchmarks" / "food_cpi" / "data.csv"
RESULTS_JSON = Path(__file__).resolve().parent / "exp53_results.json"

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# indices of our 9 levels inside Vector's 11-level STANDARD_QUANTILES
SUBSET_IDX = [STANDARD_QUANTILES.index(float(lv)) for lv in LEVELS]

PROXY_URL = "http://127.0.0.1:8765/v1"


def crps_quantile(quantiles: np.ndarray, actual: float, levels: np.ndarray) -> float:
    return float(np.mean([
        2.0 * ((actual - q) * lvl - (actual - q) * (actual < q))
        for q, lvl in zip(quantiles, levels, strict=True)
    ]))


class _InMemoryAdapter(BaseAdapter):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        return self._df.copy()


def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    values = df["value"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"])
    n = len(values)
    print(f"Data: {n} obs, {dates.min()} to {dates.max()}", flush=True)

    # Same origins as exp52: origin `orig` -> train values[:orig], actual values[orig]
    origins = list(range(n - 60, n - 1))
    print(f"Origins: {len(origins)}", flush=True)

    # Build Vector DataService once (context(as_of) enforces PIT cutoff per origin)
    series_df = pd.DataFrame({"timestamp": dates, "value": values})
    svc = DataService()
    svc.register(
        "food_cpi",
        _InMemoryAdapter(series_df),
        SeriesMetadata(
            series_id="food_cpi",
            description="Canada Food CPI (headline), monthly index",
            source="StatCan",
            units="index",
            frequency="MS",
        ),
    )
    task = ForecastingTask(
        task_id="food_cpi_1m",
        target_series_id="food_cpi",
        horizons=[1],
        frequency="MS",
        description="Canada Food CPI 1-month-ahead forecast.",
    )

    cfg = QuantileGridLLMPredictorConfig(
        model="gpt-oss-120b",
        openai_base_url=PROXY_URL,
        openai_api_key="dummy",  # real auth injected by the proxy
        reasoning_effort="low",
        max_tokens=2048,
        variant_tag="groq120b",
    )
    predictor = QuantileGridLLMPredictor(cfg)
    print(f"Predictor: {predictor.predictor_id}", flush=True)

    # Resume from checkpoint if present
    per_origin: list[dict] = []
    done_origins: set[int] = set()
    if RESULTS_JSON.exists():
        try:
            prev = json.loads(RESULTS_JSON.read_text())
            per_origin = prev.get("per_origin", [])
            done_origins = {r["origin"] for r in per_origin}
            print(f"Resuming: {len(done_origins)} origins already done", flush=True)
        except Exception:  # noqa: BLE001  # checkpoint best-effort: start fresh on corrupt file
            pass

    for i, orig in enumerate(origins):
        if orig in done_origins:
            continue
        as_of = dates.iloc[orig - 1].to_pydatetime()
        actual = float(values[orig])
        rec: dict = {"origin": orig, "as_of": str(dates.iloc[orig - 1].date()), "actual": actual}
        try:
            preds = predictor.predict(task, svc.context(as_of))
            qdict = preds[0].payload.quantiles
            grid11 = np.array([qdict[float(q)] for q in STANDARD_QUANTILES])
            grid9 = grid11[SUBSET_IDX]
            rec["crps"] = crps_quantile(grid9, actual, LEVELS)
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        per_origin.append(rec)
        done_origins.add(orig)
        ok = sum(1 for r in per_origin if r.get("ok"))
        print(f"[{i+1}/{len(origins)}] origin={orig} ok={rec['ok']} "
              f"crps={rec.get('crps', 'NA')} ({ok} ok so far)", flush=True)
        RESULTS_JSON.write_text(json.dumps({"per_origin": per_origin}, indent=1))
        time.sleep(2.0)  # respect Groq rate limits

    ok_recs = [r for r in per_origin if r.get("ok")]
    crps = np.array([r["crps"] for r in ok_recs])
    out = {
        "n_origins": len(origins),
        "n_ok": len(ok_recs),
        "llm_mean_crps": float(np.mean(crps)) if len(crps) else None,
        "llm_std_crps": float(np.std(crps)) if len(crps) else None,
        "per_origin": per_origin,
        "config": {
            "model": "openai/gpt-oss-120b (via Groq)",
            "temperature": 1.0,
            "reasoning_effort": "low",
            "max_tokens": 2048,
            "history_window": "full (Vector default)",
        },
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_origin"}, indent=1))


if __name__ == "__main__":
    main()
