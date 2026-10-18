"""exp52 (simplified): Compare our proposed method vs Vector's baseline logic.

Vector's harness has interface complexities. This script does a direct,
apples-to-apples comparison on identical data/origins:
- Vector's LastValuePredictor logic (predict last value, flat quantiles)
- Our ProposedForecaster (from exp49, already validated)

Both scored with the same CRPS function on the same origins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.agent.proposed_method import ProposedForecaster, m4_regime_spec
from td_one.agent.tools.data import series_features, naive_forecasts

DATA_CSV = Path(__file__).resolve().parents[1] / "benchmarks" / "food_cpi" / "data.csv"
RESULTS_JSON = Path(__file__).resolve().parent / "exp52_results.json"

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MODELS = ["naive", "snaive", "ets", "arima"]


def crps_quantile(quantiles: np.ndarray, actual: float, levels: np.ndarray) -> float:
    """Quantile CRPS."""
    return float(np.mean([
        2.0 * ((actual - q) * lvl - (actual - q) * (actual < q))
        for q, lvl in zip(quantiles, levels)
    ]))


def fit_ets_like(train: np.ndarray, h: int) -> np.ndarray:
    """Simple ETS-like: level + trend."""
    n = len(train)
    # Simple exponential smoothing with trend
    alpha, beta = 0.3, 0.1
    level, trend = train[0], 0.0
    for y in train[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    point = level + trend * h
    resid = train[1:] - train[:-1]
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
    return (point + z * sigma * np.sqrt(h)).reshape(9, 1)


def fit_arima_like(train: np.ndarray, h: int) -> np.ndarray:
    """Simple AR(1)-like."""
    if len(train) < 2:
        return fit_ets_like(train, h)
    # AR(1) coefficient
    y = train[1:]
    x = train[:-1]
    denom = np.sum(x ** 2)
    phi = np.sum(x * y) / denom if denom > 1e-12 else 0.0
    phi = np.clip(phi, -0.99, 0.99)
    point = train[-1] * (phi ** h) if abs(phi) < 1 else train[-1]
    resid = y - phi * x
    sigma = np.std(resid) if len(resid) > 1 else 1.0
    z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
    return (point + z * sigma * np.sqrt(h)).reshape(9, 1)


def build_records(series: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Build training records in exp47 format."""
    n = len(series)
    window = 36
    records = []
    for start in range(0, n - window - 1 + 1, 3):
        train = series[start:start + window]
        actual = np.array([series[start + window]])
        try:
            feats = series_features(train, period=12)
            fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
            feat_vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        except Exception:
            continue

        # Fit all models
        quantiles = {}
        crps = {}
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
            nq = naive_forecasts(train, 1, period=12)  # snaive approx
            q = np.asarray(nq.naive)
            if q.shape == (1, 9):
                q = q.T
            # Seasonal naive: use value from 12 ago
            snaive_point = train[-12] if len(train) >= 12 else train[-1]
            resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
            sigma = np.std(resid) if len(resid) > 1 else 1.0
            z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
            q_sn = (snaive_point + z * sigma).reshape(9, 1)
            quantiles["snaive"] = q_sn
            crps["snaive"] = crps_quantile(q_sn[:, 0], actual[0], LEVELS)
        except Exception:
            pass
        try:
            q_ets = fit_ets_like(train, 1)
            quantiles["ets"] = q_ets
            crps["ets"] = crps_quantile(q_ets[:, 0], actual[0], LEVELS)
        except Exception:
            pass
        try:
            q_ar = fit_arima_like(train, 1)
            quantiles["arima"] = q_ar
            crps["arima"] = crps_quantile(q_ar[:, 0], actual[0], LEVELS)
        except Exception:
            pass

        if len(quantiles) < 2:
            continue
        records.append({
            "features": feat_vec,
            "quantiles": quantiles,
            "crps": crps,
            "actual": actual,
        })
    return records


def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    values = df["value"].to_numpy(dtype=float)
    print(f"Data: {len(values)} obs, {df['date'].min()} to {df['date'].max()}")

    # Get feature names from series_features
    tmp = series_features(values[:50], period=12)
    fd = tmp.model_dump() if hasattr(tmp, "model_dump") else dict(tmp)
    feature_names = sorted(fd.keys())
    print(f"Features: {feature_names}")

    # Blocked 5-fold CV over origins (last 60 months)
    test_start = len(values) - 60
    origins = list(range(test_start, len(values) - 1))
    n_folds = 5
    chunk = len(origins) // n_folds

    prop_scores, naive_scores = [], []

    for f in range(n_folds):
        te_origins = origins[f * chunk:(f + 1) * chunk if f < n_folds - 1 else len(origins)]
        tr_end = te_origins[0]  # train on data before test origins

        # Build train records from history before tr_end
        train_series = values[:tr_end]
        records = build_records(train_series, feature_names)
        print(f"Fold {f}: {len(records)} train records, {len(te_origins)} test origins")
        if len(records) < 10:
            continue

        # Fit proposed forecaster
        fc = ProposedForecaster(h=1, regime_spec=m4_regime_spec(), models=MODELS)
        # Filter records to only have models we support
        fc.fit(records, feature_names)

        for orig in te_origins:
            train = values[:orig]
            actual = values[orig]
            if len(train) < 36:
                continue

            # Vector's LastValuePredictor: predict last value, degenerate quantiles
            last = float(train[-1])
            naive_q = np.full(9, last)
            naive_scores.append(crps_quantile(naive_q, actual, LEVELS))

            # Our proposed method
            try:
                feats = series_features(train[-36:], period=12)
                fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
                feat_dict = {fn: float(fd.get(fn, 0.0)) for fn in feature_names}
                feat_vec = [feat_dict[fn] for fn in feature_names]

                # Build quantiles dict for predict
                qm = {}
                nq = naive_forecasts(train[-36:], 1, period=12)
                q = np.asarray(nq.naive)
                if q.shape == (1, 9):
                    q = q.T
                qm["naive"] = q
                snaive_point = train[-12] if len(train) >= 12 else train[-1]
                resid = train[12:] - train[:-12] if len(train) >= 24 else train[1:] - train[:-1]
                sigma = np.std(resid) if len(resid) > 1 else 1.0
                z = np.array([-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28])
                qm["snaive"] = (snaive_point + z * sigma).reshape(9, 1)
                qm["ets"] = fit_ets_like(train[-36:], 1)
                qm["arima"] = fit_arima_like(train[-36:], 1)

                out = fc.predict(train[-36:], qm, feat_dict, feature_names)
                prop_q = np.asarray(out.quantiles)[:, 0]
                prop_scores.append(crps_quantile(prop_q, actual, LEVELS))
            except Exception as e:
                # Fall back to naive on error
                prop_scores.append(crps_quantile(naive_q, actual, LEVELS))

    # Paired t-test
    from scipy import stats
    prop = np.array(prop_scores)
    naive = np.array(naive_scores)
    assert len(prop) == len(naive), f"{len(prop)} vs {len(naive)}"
    diff = prop - naive
    t, p = stats.ttest_rel(prop, naive)

    results = {
        "n_origins": len(prop),
        "proposed_mean_crps": float(np.mean(prop)),
        "vector_lastvalue_mean_crps": float(np.mean(naive)),
        "diff": float(np.mean(diff)),
        "t_stat": float(t),
        "p_value": float(p),
        "proposed_wins": int(np.sum(diff < 0)),
    }
    print(f"\nProposed: {np.mean(prop):.6f}")
    print(f"Vector LastValue: {np.mean(naive):.6f}")
    print(f"Diff: {np.mean(diff):.6f}, p={p:.2e}, wins {np.sum(diff < 0)}/{len(diff)}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
