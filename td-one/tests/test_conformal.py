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


def _stable_history(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Constant rate: no regime shift, calibration should mostly cover.
    rng = np.random.default_rng(seed)
    n = np.full(14, 200.0)
    a = rng.binomial(n.astype(int), 0.20).astype(float)
    return n, a


def _late_shift_history(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Shift in the last 6 months: inside the 8-month calibration window,
    # so ACI sees the misses sequentially.
    rng = np.random.default_rng(seed)
    n = np.full(20, 200.0)
    rates = np.concatenate([np.full(14, 0.10), np.full(6, 0.35)])
    a = rng.binomial(n.astype(int), rates).astype(float)
    return n, a


def test_aci_adapts_down_after_misses() -> None:
    # Regime shift inside the calibration window: recent months miss ->
    # alpha should drop below the target, widening vs static CQR.
    from td_one.models.conformal import AdaptiveConformalM0

    n, a = _late_shift_history()
    aci = AdaptiveConformalM0(trailing_window=6, cal_months=8, gamma=0.05)
    aci.fit(n, a)
    assert aci.n_cal_ > 0
    assert len(aci.alpha_path_) == aci.n_cal_ + 1
    # At least one miss should have pushed alpha below target.
    assert min(aci.alpha_path_) < 0.2, aci.alpha_path_
    # ACI interval must be at least as wide as static on this shift.
    static = ConformalM0(trailing_window=6, cal_months=8)
    static.fit(n, a)
    assert aci.qhat_ >= static.qhat_ - 1e-12


def test_aci_stays_near_target_on_stable_data() -> None:
    from td_one.models.conformal import AdaptiveConformalM0

    n, a = _stable_history()
    aci = AdaptiveConformalM0(trailing_window=6, cal_months=4, gamma=0.05)
    aci.fit(n, a)
    # Stable: alpha should not collapse to the floor.
    assert min(aci.alpha_path_) > 0.05, aci.alpha_path_
    # And qhat should be finite and non-negative.
    assert np.isfinite(aci.qhat_) and aci.qhat_ >= 0.0


def test_aci_gamma_validation() -> None:
    from td_one.models.conformal import AdaptiveConformalM0

    with pytest.raises(ValueError, match="gamma"):
        AdaptiveConformalM0(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        AdaptiveConformalM0(gamma=-0.1)


def test_aci_falls_back_on_short_history() -> None:
    from td_one.models.conformal import AdaptiveConformalM0

    n = np.array([100.0, 120.0, 110.0])
    a = np.array([10.0, 12.0, 11.0])
    aci = AdaptiveConformalM0(trailing_window=6, cal_months=4, gamma=0.05)
    aci.fit(n, a)
    assert aci.qhat_ == 0.0
    assert aci.alpha_path_ == []
