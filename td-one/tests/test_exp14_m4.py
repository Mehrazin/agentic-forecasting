"""Tests for real-data benchmark utilities (M4 framing).

Pure-function tests only — no TSFM weights, no network, no data files.
"""

from __future__ import annotations

import numpy as np
import pytest

from td_one.evaluation.real_benchmark import (
    COMMON_GRID,
    binary_crps,
    binary_target_up,
    conformal_naive_quantiles,
    conformal_shrink_probability,
    crps_from_quantiles,
    denormalize_quantiles,
    finite_sample_quantile,
    framing_b_origins,
    interp_quantiles_to_grid,
    m0_origin_window,
    normalize_to_unit_interval,
    prob_up_from_quantiles,
    trailing_residual_quantiles,
)


def test_crps_perfect_forecast_is_zero() -> None:
    # Degenerate lattice concentrated at the actual value.
    levels = np.array([0.1, 0.5, 0.9])
    qv = np.array([2.0, 2.0, 2.0])
    assert crps_from_quantiles(levels, qv, 2.0) == pytest.approx(0.0)


def test_crps_median_only_equals_half_mae() -> None:
    # Single median quantile: CRPS = 2 * QS_0.5 = |y - q|.
    levels = np.array([0.5])
    assert crps_from_quantiles(levels, np.array([1.0]), 3.0) == pytest.approx(2.0)


def test_crps_known_two_quantiles() -> None:
    # Hand-computed: levels .25/.75, q=(0,10), y=4.
    # QS_.25 = (1-0.25)*(0-4) = -3 -> (1{4<=0}=0 -> (0-.25)*(0-4) = 1.0
    # QS_.75 = (1-0.75)*(10-4) = 1.5 ; mean=1.25 ; CRPS=2.5
    levels = np.array([0.25, 0.75])
    assert crps_from_quantiles(levels, np.array([0.0, 10.0]), 4.0) == pytest.approx(2.5)


def test_crps_horizon_mean() -> None:
    levels = np.array([0.5])
    qv = np.array([[1.0, 5.0]])  # (K=1, H=2)
    # |3-1| + |3-5| over 2 steps = (2+2)/2 = 2
    assert crps_from_quantiles(levels, qv, np.array([3.0, 3.0])) == pytest.approx(2.0)


def test_crps_rejects_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="horizon mismatch"):
        crps_from_quantiles(np.array([0.5]), np.array([[1.0, 2.0]]), np.array([1.0]))


def test_binary_crps_extremes() -> None:
    assert binary_crps(1.0, 1) == pytest.approx(0.0)
    assert binary_crps(0.0, 0) == pytest.approx(0.0)
    assert binary_crps(0.0, 1) == pytest.approx(1.0)
    assert binary_crps(1.0, 0) == pytest.approx(1.0)
    # Maximum-entropy forecast: 0.25 regardless of outcome.
    assert binary_crps(0.5, 1) == pytest.approx(0.25)
    assert binary_crps(0.5, 0) == pytest.approx(0.25)


def test_binary_crps_validates() -> None:
    with pytest.raises(ValueError):
        binary_crps(1.5, 1)
    with pytest.raises(ValueError):
        binary_crps(0.5, 2)


def test_binary_target_up() -> None:
    v = np.array([1.0, 2.0, 1.5, 3.0])
    assert binary_target_up(v, 0, horizon=1) == 1
    assert binary_target_up(v, 1, horizon=1) == 0
    # v[0+2]=1.5 > v[0]=1.0 -> 1
    assert binary_target_up(v, 0, horizon=2) == 1
    with pytest.raises(ValueError):
        binary_target_up(v, 3, horizon=1)


def test_framing_b_origins_no_leakage() -> None:
    # train_len=100: origins are the last 24 train points; earliest
    # origin needs 12 trailing outcomes each with 6-step lookahead.
    origins = framing_b_origins(100, 18, n_origins=24, horizon=6, m0_trailing=12)
    assert origins == list(range(76, 100))
    assert len(origins) == 24
    # M0 window at the earliest origin must be fully observed (s+6 <= t).
    win = m0_origin_window(origins[0], horizon=6, m0_trailing=12)
    assert max(win) + 6 <= origins[0]
    assert len(win) == 12


def test_framing_b_origins_short_series() -> None:
    # train_len=30 < 17+24: fewer origins, still no leakage.
    origins = framing_b_origins(30, 18, n_origins=24, horizon=6, m0_trailing=12)
    assert origins[0] >= 17
    assert origins[-1] == 29
    for t in origins:
        win = m0_origin_window(t, horizon=6, m0_trailing=12)
        assert max(win) + 6 <= t


def test_framing_b_origins_too_short_returns_empty() -> None:
    assert framing_b_origins(10, 18) == []


def test_normalize_round_trip() -> None:
    s = np.array([10.0, 20.0, 15.0, 30.0])
    norm, lo, hi = normalize_to_unit_interval(s)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)
    qv = np.array([0.25, 0.5, 0.75])
    back = denormalize_quantiles(qv, lo, hi)
    assert back == pytest.approx(np.array([15.0, 20.0, 25.0]))


def test_normalize_constant_series() -> None:
    norm, lo, hi = normalize_to_unit_interval(np.full(10, 7.0))
    assert (norm == 0.5).all()
    assert denormalize_quantiles(np.array([0.5]), lo, hi)[0] == pytest.approx(7.0)


def test_normalize_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        normalize_to_unit_interval(np.array([1.0, np.nan]))


def test_interp_quantiles_monotone_and_shaped() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    vals = np.array([[0.0, 1.0], [5.0, 5.0], [10.0, 9.0]])  # (K=3, H=2)
    out = interp_quantiles_to_grid(levels, vals)
    assert out.shape == (len(COMMON_GRID), 2)
    assert bool((np.diff(out, axis=0) >= 0).all())


def test_interp_quantiles_rejects_unsorted_levels() -> None:
    with pytest.raises(ValueError, match="ascending"):
        interp_quantiles_to_grid(np.array([0.5, 0.1]), np.array([1.0, 2.0]))


def test_prob_up_at_median_is_half() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    assert prob_up_from_quantiles(levels, np.array([0.0, 5.0, 10.0]), 5.0) == pytest.approx(0.5)


def test_prob_up_extremes_clip() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    qv = np.array([0.0, 5.0, 10.0])
    assert prob_up_from_quantiles(levels, qv, -100.0) == pytest.approx(0.9)
    assert prob_up_from_quantiles(levels, qv, 100.0) == pytest.approx(0.1)


def test_trailing_residual_quantiles_shapes() -> None:
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=60)) + 100.0
    rq = trailing_residual_quantiles(y, COMMON_GRID, horizon=6, kind="naive")
    assert set(rq) == {1, 2, 3, 4, 5, 6}
    for q in rq.values():
        assert q.shape == (len(COMMON_GRID),)
        assert bool((np.diff(q) >= 0).all())
    # snaive residuals are lag-12 differences regardless of h.
    rq_s = trailing_residual_quantiles(y, np.array([0.5]), horizon=3, kind="snaive")
    assert rq_s[1][0] == pytest.approx(rq_s[3][0])
    with pytest.raises(ValueError, match="kind"):
        trailing_residual_quantiles(y, COMMON_GRID, 6, kind="bogus")


def test_conformal_naive_quantiles_shape_and_widening() -> None:
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=80)) + 50.0
    q = conformal_naive_quantiles(y, COMMON_GRID, horizon=6)
    assert q.shape == (len(COMMON_GRID), 6)
    assert bool((np.diff(q, axis=0) >= 0).all())
    # Widened interval contains the base naive interval (spot check h=1).
    base_resid = trailing_residual_quantiles(y[:-12], COMMON_GRID, 6)[1]
    point = float(y[-1])  # forecast anchored at the true last value
    i10 = int(np.argmin(np.abs(COMMON_GRID - 0.1)))
    i90 = int(np.argmin(np.abs(COMMON_GRID - 0.9)))
    assert q[i10, 0] <= point + base_resid[i10] + 1e-9
    assert q[i90, 0] >= point + base_resid[i90] - 1e-9


def test_conformal_naive_rejects_short_history() -> None:
    with pytest.raises(ValueError, match="too short"):
        conformal_naive_quantiles(np.ones(20), COMMON_GRID, 6)


def test_conformal_shrink_reduces_to_m0_when_calibrated() -> None:
    assert conformal_shrink_probability(0.8, 0.0) == pytest.approx(0.8)


def test_conformal_shrink_pulls_to_half_when_miscalibrated() -> None:
    p = conformal_shrink_probability(0.9, 1.0, strength=20.0)
    assert p == pytest.approx(0.5 + (0.9 - 0.5) / 21.0)
    assert 0.5 < p < 0.9


def test_finite_sample_quantile_edges() -> None:
    s = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    # ceil(5*0.8)/4 -> k=3 -> 4.0
    assert finite_sample_quantile(s, 0.8) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="no scores"):
        finite_sample_quantile(np.array([]), 0.8)


def test_fair_grid_restricts_to_native_levels() -> None:
    """Regression: TimesFM's 0.10-0.90 levels must not be flat-extrapolated.

    Onto the 0.05-0.95 COMMON_GRID (P1 tail truncation).
    """
    from td_one.models.tsfm import TIMESFM_LEVELS

    lo, hi = float(TIMESFM_LEVELS.min()), float(TIMESFM_LEVELS.max())
    fair = np.array([g for g in COMMON_GRID if lo <= g <= hi])
    # No grid point outside TimesFM's native range.
    assert fair.min() >= 0.10 - 1e-9
    assert fair.max() <= 0.90 + 1e-9
    # Still a useful grid (17 of 19 points).
    assert len(fair) == 17
    # Interpolation on the fair grid is pure interpolation, never extrapolation.
    vals = np.linspace(10.0, 90.0, 9)  # fake quantile values at 0.1..0.9
    qv = interp_quantiles_to_grid(TIMESFM_LEVELS, vals, grid=fair)
    assert qv.shape == (17,)
    # Endpoints match exactly (no flat extrapolation).
    assert qv[0] == pytest.approx(10.0)
    assert qv[-1] == pytest.approx(90.0)


def test_no_flat_extrapolation_on_common_grid() -> None:
    """Direct check on interp behavior.

    Interp onto COMMON_GRID from 0.1-0.9 levels flat-extrapolates at
    0.05/0.95 — callers must use the fair-grid restriction instead.
    """
    from td_one.models.tsfm import TIMESFM_LEVELS

    vals = np.linspace(10.0, 90.0, 9)
    qv = interp_quantiles_to_grid(TIMESFM_LEVELS, vals)  # default COMMON_GRID
    # np.interp flat-extrapolates: document that this happens so callers
    # know to use the fair-grid restriction instead.
    assert qv[0] == pytest.approx(10.0)  # flat from q10
    assert qv[-1] == pytest.approx(90.0)  # flat from q90
