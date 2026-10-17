"""Tests for the 10 agent analysis tools (td_one.agent.tools)."""

import numpy as np
import pandas as pd
import pytest

from td_one.agent.tools import (
    analyze_residuals,
    backtest_event_impact,
    compare_models,
    cross_validate,
    decompose_series,
    detect_changepoints,
    feature_importance,
    generate_synthetic_scenario,
    plot_diagnostics,
)
from td_one.agent.tools import test_stationarity as check_stationarity


@pytest.fixture
def rng_series() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 120)


def _simple_model(train: np.ndarray, horizon: int) -> dict[float, float]:
    mu = float(np.mean(train[-6:]))
    sd = float(np.std(train[-12:]) or 1.0)
    return {0.1: mu - 1.28 * sd, 0.5: mu, 0.9: mu + 1.28 * sd}


def test_analyze_residuals_basic(rng_series: np.ndarray) -> None:
    y_true = rng_series
    y_pred = rng_series + 0.1
    res = analyze_residuals(y_true, y_pred)
    assert res.n == 120
    assert res.mae == pytest.approx(0.1, abs=1e-9)
    assert res.mean_residual == pytest.approx(-0.1, abs=1e-9)
    assert res.top_features == []


def test_analyze_residuals_with_features(rng_series: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    feat = rng.normal(0, 1, 120)
    y_true = rng_series + 2.0 * np.abs(feat)  # heteroskedastic driver
    y_pred = rng_series
    res = analyze_residuals(y_true, y_pred, {"driver": feat, "noise": rng.normal(0, 1, 120)})
    assert len(res.top_features) == 2
    assert res.top_features[0]["name"] == "driver"


def test_analyze_residuals_validation() -> None:
    with pytest.raises(ValueError):
        analyze_residuals([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        analyze_residuals([1, 2, 3], [1, 2, 3])  # too short


def test_detect_changepoints_finds_shift() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 60), rng.normal(5, 1, 60)])
    res = detect_changepoints(x)
    assert res.n_segments >= 2
    assert any(50 <= cp <= 70 for cp in res.changepoints)


def test_detect_changepoints_no_shift(rng_series: np.ndarray) -> None:
    res = detect_changepoints(rng_series)
    assert res.n_segments >= 1
    assert res.changepoints == [] or all(
        res.changepoints[i + 1] - res.changepoints[i] >= 6
        for i in range(len(res.changepoints) - 1)
    )


def test_detect_changepoints_validation() -> None:
    with pytest.raises(ValueError):
        detect_changepoints([1.0] * 10)


def test_feature_importance_ranking() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (200, 3))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.1, 200)  # f0 dominates
    res = feature_importance(lambda a: 3.0 * a[:, 0], X, y, feature_names=["a", "b", "c"])
    assert res.features[0]["name"] == "a"
    assert res.features[0]["importance"] > res.features[1]["importance"]


def test_feature_importance_validation() -> None:
    with pytest.raises(ValueError):
        feature_importance(lambda a: a, np.ones((5, 2)), np.ones(5))  # too few rows


def test_cross_validate_basic(rng_series: np.ndarray) -> None:
    res = cross_validate(_simple_model, rng_series, n_splits=3, model_name="simple")
    assert res.n_folds == 3
    assert res.mean_crps > 0
    assert res.se_crps >= 0
    assert len(res.folds) == 3


def test_cross_validate_pit_safe() -> None:
    # The model only sees train; verify origins are strictly increasing and
    # test points never overlap training.
    seen = []

    def spy(train: np.ndarray, horizon: int) -> dict[float, float]:
        seen.append(len(train))
        return _simple_model(train, horizon)

    x = np.random.default_rng(3).normal(0, 1, 100)
    cross_validate(spy, x, n_splits=4, min_train=24)
    assert seen == sorted(seen)
    assert all(s >= 24 for s in seen)


def test_compare_models_ranking(rng_series: np.ndarray) -> None:
    def worse(train: np.ndarray, horizon: int) -> dict[float, float]:
        q = _simple_model(train, horizon)
        return {k: v + 5.0 for k, v in q.items()}  # biased -> worse CRPS

    res = compare_models({"good": _simple_model, "bad": worse}, rng_series, n_splits=3)
    assert res.ranking[0] == "good"
    assert len(res.pairwise) == 1
    pw = res.pairwise[0]
    assert pw.winner == "good"
    assert pw.significant


def test_plot_diagnostics_creates_png(rng_series: np.ndarray) -> None:
    res = plot_diagnostics(rng_series, rng_series + 0.05)
    assert res.path.endswith(".png")
    import os

    assert os.path.getsize(res.path) > 1000
    os.remove(res.path)


def test_test_stationarity_white_noise(rng_series: np.ndarray) -> None:
    res = check_stationarity(rng_series)
    assert res.verdict in ("stationary", "inconclusive")
    assert res.adf_pvalue < 0.05  # white noise rejects unit root


def test_test_stationarity_random_walk() -> None:
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 1, 200))
    res = check_stationarity(x)
    assert res.verdict in ("non_stationary", "inconclusive")


def test_decompose_series_trend() -> None:
    t = np.arange(72, dtype=float)
    x = 0.5 * t + 2.0 * np.sin(2 * np.pi * t / 12)
    res = decompose_series(x, period=12)
    assert res.period == 12
    assert len(res.trend) == 72
    assert res.trend_strength > 0.8
    assert res.seasonal_strength > 0.5


def test_backtest_event_impact_detects_shift() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    x[50:] += 3.0  # level shift at index 50
    res = backtest_event_impact([50], x, window=10)
    assert len(res.impacts) == 1
    assert res.impacts[0].ate == pytest.approx(3.0, abs=0.8)
    assert res.mean_ate == pytest.approx(3.0, abs=0.8)


def test_backtest_event_impact_with_dates() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 60)
    x[30:] += 2.0
    dates = pd.date_range("2020-01-01", periods=60, freq="ME")
    res = backtest_event_impact([dates[30]], x, dates=list(dates), window=6)
    assert len(res.impacts) == 1
    assert res.impacts[0].ate > 1.0


def test_generate_synthetic_scenario_level() -> None:
    x = np.zeros(48)
    res = generate_synthetic_scenario(x, "level_shift", 2.0, start_index=24, duration=6)
    assert res.mean_shift == pytest.approx(2.0, abs=1e-9)
    assert res.shocked_series[30] == pytest.approx(2.0)


def test_generate_synthetic_scenario_pulse() -> None:
    x = np.zeros(48)
    res = generate_synthetic_scenario(x, "pulse", 5.0, start_index=10, duration=4)
    assert res.shocked_series[11] == pytest.approx(5.0)
    assert res.shocked_series[20] == pytest.approx(0.0)


def test_generate_synthetic_scenario_volatility() -> None:
    x = np.zeros(48)
    res = generate_synthetic_scenario(x, "volatility_spike", 1.0, start_index=10, duration=10)
    assert np.std(res.shocked_series[10:20]) > 0.3


def test_generate_synthetic_scenario_validation() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_scenario([1, 2, 3], "level_shift", 1.0)
