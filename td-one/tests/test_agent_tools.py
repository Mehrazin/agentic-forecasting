"""Tests for the agent analysis tools (td_one.agent.tools): 12 in the namespace."""

import numpy as np
import pandas as pd
import pytest

from td_one.agent.tools import (
    analyze_residuals,
    backtest_event_impact,
    combine_quantiles,
    compare_models,
    cross_validate,
    decompose_series,
    detect_changepoints,
    feature_importance,
    load_series,
    naive_forecasts,
    regime_probabilities,
    series_features,
)
from td_one.agent.tools import test_stationarity as check_stationarity

# Retired from the agent namespace but modules kept for manual use.
from td_one.agent.tools.plotting import plot_diagnostics
from td_one.agent.tools.scenarios import generate_synthetic_scenario


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
    x_mat = rng.normal(0, 1, (200, 3))
    y = 3.0 * x_mat[:, 0] + rng.normal(0, 0.1, 200)  # f0 dominates
    res = feature_importance(
        lambda a: 3.0 * a[:, 0], x_mat, y, feature_names=["a", "b", "c"]
    )
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


# ---------------------------------------------------------------------------
# New tools (2026-10-18 tool optimization)
# ---------------------------------------------------------------------------


def test_load_series_basic(tmp_path) -> None:
    import pandas as pd

    dates = pd.date_range("2021-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"date": dates, "smm": np.linspace(0.001, 0.002, 12)})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    res = load_series(str(p), value_col="smm")
    assert res.n == 12
    assert res.values[0] == pytest.approx(0.001)
    assert res.dates[0] == "2021-01-31"
    assert res.origin_applied is None


def test_load_series_pit_filter(tmp_path) -> None:
    import pandas as pd

    dates = pd.date_range("2021-01-01", periods=12, freq="ME")
    df = pd.DataFrame({"date": dates, "smm": np.linspace(0.001, 0.002, 12)})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    res = load_series(str(p), value_col="smm", origin="2021-06-15")
    assert res.n == 5  # Jan..May month-ends only (Jun-30 is after the cutoff)
    assert res.origin_applied == "2021-06-15"
    assert all(d <= "2021-06-15" for d in res.dates)


def test_load_series_index_layout(tmp_path) -> None:
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=10, freq="ME")
    df = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=idx)
    p = tmp_path / "s.csv"
    df.to_csv(p)
    res = load_series(str(p))
    assert res.n == 10
    assert res.value_col == "x"


def test_naive_forecasts_basic(rng_series: np.ndarray) -> None:
    res = naive_forecasts(rng_series, horizon=3)
    assert res.horizon == 3
    assert res.levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert len(res.naive) == 3 and len(res.naive[0]) == 9
    # median of naive == last value
    assert res.naive[0][4] == pytest.approx(float(rng_series[-1]))
    # intervals widen with horizon
    w1 = res.naive[0][8] - res.naive[0][0]
    w3 = res.naive[2][8] - res.naive[2][0]
    assert w3 > w1
    assert res.sigma > 0


def test_naive_forecasts_seasonal(rng_series: np.ndarray) -> None:
    res = naive_forecasts(rng_series, horizon=1, period=12)
    assert res.seasonal_naive[0][4] == pytest.approx(float(rng_series[-12]))


def test_naive_forecasts_validation() -> None:
    with pytest.raises(ValueError):
        naive_forecasts([1.0], horizon=1)  # too short
    with pytest.raises(ValueError):
        naive_forecasts(np.ones(10), horizon=0)


def test_series_features_keys(rng_series: np.ndarray) -> None:
    res = series_features(rng_series)
    d = res.as_dict()
    expected = {
        "n_changepoints",
        "n_recent_cps",
        "trend_strength",
        "seasonal_strength",
        "vol_ratio",
        "is_stationary",
        "is_nonstationary",
        "acf1",
        "spectral_entropy",
        "hurst",
    }
    assert set(d) == expected
    assert 0.0 <= d["trend_strength"] <= 1.0
    assert 0.0 <= d["seasonal_strength"] <= 1.0


def test_series_features_seasonal() -> None:
    t = np.arange(60)
    x = 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(0).normal(0, 0.1, 60)
    res = series_features(x)
    assert res.seasonal_strength > 0.5


def test_series_features_short_series() -> None:
    # graceful degradation, no raise
    res = series_features(np.random.default_rng(1).normal(0, 1, 10))
    assert res.as_dict()["seasonal_strength"] == 0.0


def test_regime_probabilities_m4() -> None:
    res = regime_probabilities(
        {"seasonal_strength": 0.8, "trend_strength": 0.2}, spec="m4"
    )
    assert res.label == "seasonal"
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-9
    assert res.probabilities["seasonal"] > 0.5


def test_regime_probabilities_generic() -> None:
    res = regime_probabilities({"n_recent_cps": 2.0, "vol_ratio": 0.5}, spec="generic")
    assert res.label == "stressed"
    res2 = regime_probabilities({"n_recent_cps": 0.0, "vol_ratio": 0.5}, spec="generic")
    assert res2.label == "calm"


def test_regime_probabilities_boc() -> None:
    res = regime_probabilities(
        {"rate_momentum": 0.5, "inflation_gap": 1.5, "yield_spread": 0.8}, spec="boc"
    )
    assert res.label == "EASY"  # all three imply hike
    res2 = regime_probabilities(
        {"rate_momentum": 0.0, "inflation_gap": 1.5, "yield_spread": 0.0}, spec="boc"
    )
    assert res2.label == "HARD"


def test_regime_probabilities_bad_spec() -> None:
    with pytest.raises(ValueError):
        regime_probabilities({}, spec="nope")


def test_combine_quantiles_equal() -> None:
    mq = {"a": {0.1: 1.0, 0.5: 2.0, 0.9: 3.0}, "b": {0.1: 3.0, 0.5: 4.0, 0.9: 5.0}}
    res = combine_quantiles(mq)
    assert res.levels == [0.1, 0.5, 0.9]
    assert res.values == pytest.approx([2.0, 3.0, 4.0])
    assert res.weights_used == {"a": 0.5, "b": 0.5}


def test_combine_quantiles_weighted() -> None:
    mq = {"a": {0.5: 10.0}, "b": {0.5: 20.0}}
    res = combine_quantiles(mq, weights={"a": 0.75, "b": 0.25})
    assert res.values == pytest.approx([12.5])


def test_combine_quantiles_list_input() -> None:
    mq = {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]}
    res = combine_quantiles(mq, levels=[0.1, 0.5, 0.9])
    assert res.values == pytest.approx([2.0, 3.0, 4.0])


def test_combine_quantiles_mismatch() -> None:
    mq = {"a": {0.1: 1.0}, "b": {0.5: 2.0}}
    with pytest.raises(ValueError):
        combine_quantiles(mq)


def test_stationarity_fast(rng_series: np.ndarray) -> None:
    res = check_stationarity(rng_series, fast=True)
    assert res.verdict in ("stationary", "non_stationary")
    assert np.isnan(res.kpss_pvalue)
