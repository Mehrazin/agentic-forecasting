"""Shared Food CPI classical-record builder for the LLM ladder (exp56-58).

Verbatim extraction of the record pipeline from
``experiments/exp55_rung2_llm_expert.py`` (raw values, 36m train window,
``series_features(period=12)``, experts naive/snaive/ets/arima — the
exp52/exp53 classical pipeline), so every rung evaluates on identical
classical records. LLM-grid injection delegates to
``td_one.llm_rungs.combine.build_records_with_llm`` with a caller-chosen
expert name, so the hybrid (exp58) can hold several prompt variants as
distinct experts.

Nothing here calls an LLM; all inputs are the audited caches and
``benchmarks/food_cpi/data.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp52_vector_baselines as e52  # noqa: E402

from td_one.agent.tools.data import naive_forecasts, series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps, load_grid_cache  # noqa: E402
from td_one.llm_rungs.combine import build_records_with_llm  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402

DATA_CSV = ROOT / "benchmarks" / "food_cpi" / "data.csv"
EXP_DIR = ROOT / "experiments"

CLASSICAL_MODELS = ["naive", "snaive", "ets", "arima"]
Z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])


def load_food_cpi_values() -> np.ndarray:
    """Raw Food CPI values from ``benchmarks/food_cpi/data.csv``."""
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    return df["value"].to_numpy(dtype=float)


def origins_59(values: np.ndarray) -> list[int]:
    """Return the 59 origins used by exp52/exp53 and the ladder caches."""
    return list(range(len(values) - 60, len(values) - 1))


def feature_names_for(values: np.ndarray) -> list[str]:
    """Sorted ``series_features`` keys — the exp55/exp53 feature order."""
    feats = series_features(values[:50], period=12)
    fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
    return sorted(fd.keys())


def origin_quantiles(train: np.ndarray) -> dict[str, np.ndarray]:
    """Classical 9-quantile grids for one origin (exp52/exp53 pipeline)."""
    qm: dict[str, np.ndarray] = {}
    nq = naive_forecasts(train, 1, period=12)
    q = np.asarray(nq.naive)
    qm["naive"] = q.T if q.shape == (1, 9) else q
    sp = train[-12] if len(train) >= 12 else train[-1]
    resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    qm["snaive"] = (sp + Z * sigma).reshape(9, 1)
    qm["ets"] = e52.fit_ets_like(train, 1)
    qm["arima"] = e52.fit_arima_like(train, 1)
    return qm


def build_classical_records(
    values: np.ndarray, origins: list[int], feature_names: list[str]
) -> list[dict[str, Any]]:
    """One classical record per origin (raw values, H=1)."""
    records: list[dict[str, Any]] = []
    for o in origins:
        train = values[o - 36 : o]
        actual = float(values[o])
        if len(train) < 36:
            continue
        feats = series_features(train, period=12)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        qm = origin_quantiles(train)
        records.append(
            {
                "origin": int(o),
                "features": feat_vec,
                "quantiles": {m: q.tolist() for m, q in qm.items()},
                "crps": {m: grid_to_crps(q[:, 0], LEVELS_9, actual) for m, q in qm.items()},
                "actual": [actual],
            }
        )
    return records


def inject_expert(
    records: list[dict[str, Any]],
    cache_path: str | Path,
    expert_name: str,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inject a validated grid cache into classical records as one expert.

    Returns ``(records_with_expert, cache)``. Origins without a usable
    (``ok=True``) cache entry are dropped.
    """
    cache = load_grid_cache(cache_path)
    out = build_records_with_llm(records, cache, feature_names, expert_name=expert_name)
    return out, cache


def aligned_classical(
    classical: list[dict[str, Any]], injected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Classical records restricted to the origins kept by an injection."""
    keep = {r["origin"] for r in injected}
    return [r for r in classical if r["origin"] in keep]
