"""Tests for the conformalized M0.

Focus: the widening actually happens when M0 misses (non-obvious
calibration logic), the small-history fallback is explicit, and the
lattice stays monotone.
"""

from __future__ import annotations

import numpy as np
import pytest

from td_one.models.baseline import HistoricalBaseline
from td_one.models.conformal import ConformalM0
from td_one.models.tsfm import TIMESFM_LEVELS

LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)


def _volatile_history(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Regime shift mid-history: M0's trailing window will miss the new level.
    rng = np.random.default_rng(seed)
    n = np.full(14, 200.0)
    rates = np.concatenate([np.full(7, 0.10), np.full(7, 0.35)])
    a = rng.binomial(n.astype(int), rates).astype(float)
    return n, a


def test_conformal_widens_when_m0_misses() -> None:
    n, a = _volatile_history()
    cm = ConformalM0(trailing_window=6, cal_months=4)
    cm.fit(n, a)
    assert cm.n_cal_ > 0
    assert cm.qhat_ >= 0.0
    raw = HistoricalBaseline(trailing_window=6)
    raw.fit(n, a)
    raw_lat = raw.predict_rate_quantiles(200, LEVELS)
    conf_lat = cm.predict_rate_quantiles(200)
    # Widened interval must contain the raw interval.
    assert conf_lat[0.1] <= raw_lat[0.1]
    assert conf_lat[0.9] >= raw_lat[0.9]
    # Interior quantiles untouched.
    assert conf_lat[0.5] == pytest.approx(raw_lat[0.5])


def test_small_history_falls_back_to_plain_m0() -> None:
    n = np.array([100.0, 120.0, 110.0])
    a = np.array([10.0, 12.0, 11.0])
    cm = ConformalM0(trailing_window=6, cal_months=4)
    cm.fit(n, a)
    assert cm.qhat_ == 0.0  # explicit fallback, not silent
    raw = HistoricalBaseline(trailing_window=6)
    raw.fit(n, a)
    assert cm.predict_rate_quantiles(110) == raw.predict_rate_quantiles(110, LEVELS)


def test_lattice_monotone_and_bounded() -> None:
    n, a = _volatile_history()
    cm = ConformalM0()
    cm.fit(n, a)
    lat = cm.predict_rate_quantiles(200)
    vals = [lat[float(k)] for k in LEVELS]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_unfitted_and_bad_inputs_rejected() -> None:
    cm = ConformalM0()
    with pytest.raises(RuntimeError):
        cm.predict_rate_quantiles(100)
    with pytest.raises(ValueError):
        ConformalM0(alpha=1.5)
    n = np.array([100.0, 100.0])
    a = np.array([10.0, 200.0])  # A > N
    with pytest.raises(ValueError):
        ConformalM0().fit(n, a)
