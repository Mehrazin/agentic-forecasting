"""Tests for td_one.agent.group_forecaster."""

from datetime import date

import numpy as np
import pytest

from td_one.agent.group_forecaster import GroupForecaster
from td_one.data_gen.group_series import GroupSeries


def _gs(gid: str, n_risk: list[int], n_sw: list[int]) -> GroupSeries:
    months = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)][: len(n_risk)]
    return GroupSeries(group_id=gid, months=months, n_at_risk=n_risk, n_switched=n_sw)


def _stub_fn(series: np.ndarray) -> np.ndarray:
    # Returns the series mean tiled across 5 quantile levels, tagged so we
    # can tell group vs global forecasts apart.
    m = float(np.mean(series)) if len(series) else 0.0
    return np.full(5, m)


def _groups() -> dict[str, GroupSeries]:
    return {
        "big": _gs("big", [300, 300, 300], [30, 33, 27]),
        "small": _gs("small", [10, 10, 10], [1, 2, 1]),
        "__portfolio__": _gs("__portfolio__", [310, 310, 310], [31, 35, 28]),
    }


def test_big_group_uses_own_forecast() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    out = gf.forecast(_groups())
    assert not out["big"].used_fallback
    assert out["big"].quantiles == pytest.approx(np.full(5, 0.1))  # 90/900


def test_small_group_falls_back() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    out = gf.forecast(_groups())
    assert out["small"].used_fallback
    # Global rate: 94/930
    assert out["small"].quantiles == pytest.approx(np.full(5, 94 / 930), abs=1e-6)


def test_missing_portfolio_raises() -> None:
    gf = GroupForecaster(_stub_fn)
    with pytest.raises(ValueError, match="__portfolio__"):
        gf.forecast({"g": _gs("g", [1], [0])})


def test_bad_min_group_n_raises() -> None:
    with pytest.raises(ValueError, match="min_group_n"):
        GroupForecaster(_stub_fn, min_group_n=0)


def test_train_months_window() -> None:
    gf = GroupForecaster(_stub_fn, min_group_n=200)
    # Only last month counts: 300 >= 200 -> no fallback.
    out = gf.forecast(_groups(), train_months=1)
    assert not out["big"].used_fallback
    # small: 10 < 200 -> fallback even on 1 month.
    assert out["small"].used_fallback
