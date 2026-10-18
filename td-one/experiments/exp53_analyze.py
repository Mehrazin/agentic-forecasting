"""exp53 analysis: paired comparison of Vector LLM predictor vs our proposed method.

Reads exp53_results.json (LLM per-origin CRPS) and compares against our
proposed-method CRPS on the same 59 origins. Our per-origin CRPS values are
recomputed here with the identical procedure as exp52 (same folds/records),
to guarantee apples-to-apples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.agent.proposed_method import ProposedForecaster, m4_regime_spec
from td_one.agent.tools.data import naive_forecasts, series_features

DATA_CSV = Path(__file__).resolve().parents[1] / "benchmarks" / "food_cpi" / "data.csv"
LLM_JSON = Path(__file__).resolve().parent / "exp53_results.json"
OUT_JSON = Path(__file__).resolve().parent / "exp53_comparison.json"

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MODELS = ["naive", "snaive", "ets", "arima"]


def crps_quantile(quantiles, actual, levels):
    return float(np.mean([
        2.0 * ((actual - q) * lv - (actual - q) * (actual < q))
        for q, lv in zip(quantiles, levels, strict=True)
    ]))


def fit_ets_like(train, h):
    alpha, beta = 0.3, 0.1
    level, trend = train[0], 0.0
    for y in train[1:]:
        prev = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    point = level + trend * h
    resid = train[1:] - train[:-1]
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
    return (point + z * sigma * np.sqrt(h)).reshape(9, 1)


def fit_arima_like(train, h):
    if len(train) < 2:
        return fit_ets_like(train, h)
    y, x = train[1:], train[:-1]
    denom = np.sum(x ** 2)
    phi = np.clip(np.sum(x * y) / denom if denom > 1e-12 else 0.0, -0.99, 0.99)
    point = train[-1] * (phi ** h) if abs(phi) < 1 else train[-1]
    sigma = np.std(y - phi * x) if len(y) > 1 else 1.0
    z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
    return (point + z * sigma * np.sqrt(h)).reshape(9, 1)


def build_records(series, feature_names):
    n = len(series)
    window = 36
    records = []
    for start in range(0, n - window - 1 + 1, 3):
        train = series[start:start + window]
        actual = np.array([series[start + window]])
        try:
            feats = series_features(train, period=12)
            fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        except Exception:  # noqa: BLE001
            continue
        quantiles, crps = {}, {}
        try:
            nq = naive_forecasts(train, 1, period=12)
            q = np.asarray(nq.naive)
            if q.shape == (1, 9):
                q = q.T
            quantiles["naive"] = q
            crps["naive"] = crps_quantile(q[:, 0], actual[0], LEVELS)
        except Exception:
            continue
        try:
            sp = train[-12] if len(train) >= 12 else train[-1]
            resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
            sigma = np.std(resid) if len(resid) > 1 else 1.0
            z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
            q = (sp + z * sigma).reshape(9, 1)
            quantiles["snaive"] = q
            crps["snaive"] = crps_quantile(q[:, 0], actual[0], LEVELS)
        except Exception:
            pass
        for name, fn in (("ets", fit_ets_like), ("arima", fit_arima_like)):
            try:
                q = fn(train, 1)
                quantiles[name] = q
                crps[name] = crps_quantile(q[:, 0], actual[0], LEVELS)
            except Exception:
                pass
        if len(quantiles) < 2:
            continue
        feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        records.append({"features": feat_vec, "quantiles": quantiles, "crps": crps, "actual": actual})
    return records


def main():
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    values = df["value"].to_numpy(dtype=float)

    llm_data = json.loads(LLM_JSON.read_text())
    llm_recs = {r["origin"]: r for r in llm_data["per_origin"] if r.get("ok")}

    tmp = series_features(values[:50], period=12)
    fd = tmp.model_dump() if hasattr(tmp, "model_dump") else dict(tmp)
    feature_names = sorted(fd.keys())

    test_start = len(values) - 60
    origins = list(range(test_start, len(values) - 1))
    n_folds = 5
    chunk = len(origins) // n_folds

    prop_scores, llm_scores, matched_origins = [], [], []
    for f in range(n_folds):
        te_origins = origins[f * chunk:(f + 1) * chunk if f < n_folds - 1 else len(origins)]
        tr_end = te_origins[0]
        records = build_records(values[:tr_end], feature_names)
        if len(records) < 10:
            continue
        fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec(), models=MODELS)
        fc.fit(records, feature_names)
        for orig in te_origins:
            if orig not in llm_recs:
                continue
            train = values[:orig]
            actual = values[orig]
            if len(train) < 36:
                continue
            try:
                feats = series_features(train[-36:], period=12)
                fd2 = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
                feat_vec = [float(fd2.get(fn, 0.0)) for fn in feature_names]
                qm = {}
                nq = naive_forecasts(train[-36:], 1, period=12)
                q = np.asarray(nq.naive)
                qm["naive"] = q.T if q.shape == (1, 9) else q
                sp = train[-12] if len(train) >= 12 else train[-1]
                resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
                sigma = np.std(resid) if len(resid) > 1 else 1.0
                z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
                qm["snaive"] = (sp + z * sigma).reshape(9, 1)
                qm["ets"] = fit_ets_like(train[-36:], 1)
                qm["arima"] = fit_arima_like(train[-36:], 1)
                out = fc.predict(
                    train=train[-36:],
                    model_quantiles=qm,
                    feats={fn: float(fd2.get(fn, 0.0)) for fn in feature_names},
                    feature_names=feature_names,
                )
                pq = np.asarray(out.quantiles)
                if pq.shape == (1, 9):
                    pq = pq.T
                prop_scores.append(crps_quantile(pq[:, 0], actual, LEVELS))
                llm_scores.append(llm_recs[orig]["crps"])
                matched_origins.append(orig)
            except Exception:
                continue

    prop = np.array(prop_scores)
    llm = np.array(llm_scores)
    diff = llm - prop  # positive => we win
    t, p = stats.ttest_rel(llm, prop)
    result = {
        "n_matched_origins": len(prop),
        "llm_origins_attempted": len(llm_data["per_origin"]),
        "llm_origins_ok": sum(1 for r in llm_data["per_origin"] if r.get("ok")),
        "proposed_mean_crps": float(np.mean(prop)),
        "llm_mean_crps": float(np.mean(llm)),
        "diff_llm_minus_proposed": float(np.mean(diff)),
        "t_stat_llm_vs_proposed": float(t),
        "p_value": float(p),
        "proposed_wins": int(np.sum(diff > 0)),
        "llm_wins": int(np.sum(diff < 0)),
        "matched_origins": matched_origins,
        "llm_config": llm_data.get("config", {}),
        "per_origin_llm": llm_data["per_origin"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_origin_llm", "matched_origins")}, indent=1))


if __name__ == "__main__":
    main()
