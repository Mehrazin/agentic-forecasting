"""Tests for the macro-adjusted M0.

Focus: the honest null (no macro structure -> beta ~ 0, reduces to M0),
recovery of a planted macro sensitivity, and the missing-z fallback.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from td_one.models.baseline import HistoricalBaseline
from td_one.models.macro_m0 import MacroM0, load_fred_z
from td_one.models.tsfm import TIMESFM_LEVELS

LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)


def _months(n: int, year: int = 2021, month: int = 4) -> list[date]:
    # Month-end dates, n consecutive months starting at (year, month).
    out: list[date] = []
    y, m = year, month
    for _ in range(n):
        last = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
        out.append(last.date())
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _macro_panel(
    beta: float, seed: int = 0, n_months: int = 36
) -> tuple[np.ndarray, np.ndarray, list[date], dict[date, float]]:
    """Synthetic panel with logit(rate[i]) = logit(0.2) + beta * z[i-1].

    Lagged z (predictive regression): the rate in month i depends on the
    macro state at i-1, which is knowable at a forecast origin. z is a
    fixed AR(1)-like wave (not random) so the test is deterministic
    given the seed; binomial noise comes from the seed.
    """
    rng = np.random.default_rng(seed)
    months = _months(n_months)
    t = np.arange(n_months)
    z = np.sin(2 * np.pi * t / 12.0) + 0.5 * np.sin(2 * np.pi * t / 5.0)
    z = (z - z.mean()) / z.std()
    base = np.log(0.2 / 0.8)
    n = np.full(n_months, 500.0)
    z_lag = np.concatenate([[0.0], z[:-1]])  # z[i-1]; first month uses 0
    p = 1.0 / (1.0 + np.exp(-(base + beta * z_lag)))
    a = rng.binomial(n.astype(int), p).astype(float)
    return n, a, months, {d: float(v) for d, v in zip(months, z, strict=True)}


def test_null_no_macro_structure_gives_zero_beta() -> None:
    # Pure noise rates: OLS should find ~no macro sensitivity.
    n, a, months, z = _macro_panel(beta=0.0, seed=1)
    mm = MacroM0(trailing_window=6, fred_z=z)
    mm.fit(n, a, months)
    assert mm.n_est_ == 24  # est_months caps the OLS window
    assert abs(mm.beta_hat_) < 0.15, mm.beta_hat_
    # And the forecast must be negligibly different from plain M0.
    m0 = HistoricalBaseline(trailing_window=6)
    m0.fit(n, a)
    got = mm.predict_rate_quantiles(500, months[-1])
    want = m0.predict_rate_quantiles(500, LEVELS)
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=0.01)


def test_recovers_planted_beta() -> None:
    n, a, months, z = _macro_panel(beta=0.6, seed=2)
    mm = MacroM0(trailing_window=6, fred_z=z)
    mm.fit(n, a, months)
    # Binomial noise + trailing-window rate noise; allow generous tolerance.
    assert mm.beta_hat_ == pytest.approx(0.6, abs=0.25), mm.beta_hat_


def test_macro_shift_moves_forecast_in_z_direction() -> None:
    n, a, months, z = _macro_panel(beta=0.6, seed=3)
    mm = MacroM0(trailing_window=6, fred_z=z)
    mm.fit(n, a, months)
    assert mm.beta_hat_ > 0.2
    # Positive z at the origin (last training month) must raise the
    # forecast for the next month (PIT-safe lagged shift).
    fz = dict(z)
    fz[months[-1]] = 2.0
    mm2 = MacroM0(trailing_window=6, fred_z=fz)
    mm2.fit(n, a, months)
    next_month = (pd.Timestamp(months[-1]) + pd.offsets.MonthEnd(1)).date()
    up = mm2.predict_rate_quantiles(500, next_month)
    base = mm.predict_rate_quantiles(500, next_month)
    for k in base:
        assert up[k] > base[k]
    # Monotonicity preserved.
    vals = [up[float(k)] for k in LEVELS]
    assert all(b >= a_ for a_, b in zip(vals, vals[1:], strict=False))


def test_missing_z_falls_back_to_m0() -> None:
    n, a, months, z = _macro_panel(beta=0.6, seed=4)
    mm = MacroM0(trailing_window=6, fred_z=z)
    mm.fit(n, a, months)
    # Forecast month absent from fred_z -> shift is 0 -> M0 lattice.
    other = date(1999, 1, 31)
    got = mm.predict_rate_quantiles(500, other)
    m0 = HistoricalBaseline(trailing_window=6)
    m0.fit(n, a)
    want = m0.predict_rate_quantiles(500, LEVELS)
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=1e-12)


def test_too_little_history_gives_zero_beta() -> None:
    n, a, months, z = _macro_panel(beta=0.6, seed=5, n_months=8)
    mm = MacroM0(trailing_window=6, fred_z=z)
    mm.fit(n, a, months)
    # 8 months -> 7 usable (first has no lagged z); still < 12 minimum.
    assert mm.n_est_ == 7
    assert mm.beta_hat_ == 0.0


def test_input_validation() -> None:
    mm = MacroM0()
    n = np.array([100.0, 120.0])
    a = np.array([10.0, 200.0])  # A > N
    with pytest.raises(ValueError, match="0 <= A <= N"):
        mm.fit(n, a, _months(2))
    with pytest.raises(ValueError, match="align"):
        mm.fit(np.array([100.0]), np.array([10.0]), _months(2))
    with pytest.raises(RuntimeError, match="fit"):
        _ = MacroM0().beta_hat_
    with pytest.raises(RuntimeError, match="fit"):
        _ = MacroM0().n_est_
    with pytest.raises(RuntimeError, match="fit"):
        _ = MacroM0().predict_rate_quantiles(100, date(2021, 4, 30))
    with pytest.raises(ValueError, match="trailing_window"):
        MacroM0(trailing_window=0)


def test_load_fred_z_synthetic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # 8 months of linearly rising values -> positive 3m changes.
    dates = pd.date_range("2021-01-01", periods=8, freq="ME")
    df = pd.DataFrame({"date": dates, "value": np.linspace(100.0, 114.0, 8)})
    df.to_parquet(tmp_path / "TEST.parquet")
    z = load_fred_z("TEST", data_dir=tmp_path)
    assert len(z) == 5  # 8 months minus 3 leading NaNs
    assert all(isinstance(d, date) for d in z)
    vals = np.array(list(z.values()))
    assert abs(vals.mean()) < 1e-9  # z-scored
    assert abs(vals.std(ddof=1) - 1.0) < 1e-9
    # Rising series -> all 3m changes positive -> z-scores have same sign
    # pattern as the (positive) changes; just check finiteness and keys.
    assert min(z) == date(2021, 4, 30)
    with pytest.raises(FileNotFoundError):
        load_fred_z("NOPE", data_dir=tmp_path)


def test_load_fred_z_asof_no_future_leak(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression (P0-1): asof must restrict normalization stats to date <= asof.

    Without asof, z-scores use full-history mean/std (future leak).
    With asof, a later structural break must not affect earlier z-scores.
    """
    # 16 months: 10 months of noisy growth, then 6 months of flat-at-high
    # (structural break in variance, not just level).
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=16, freq="ME")
    values = list(100 + np.cumsum(rng.normal(1.0, 0.5, 10))) + [115.0] * 6
    df = pd.DataFrame({"date": dates, "value": values})
    df.to_parquet(tmp_path / "BREAK.parquet")

    z_full = load_fred_z("BREAK", data_dir=tmp_path)
    z_asof = load_fred_z("BREAK", data_dir=tmp_path, asof=date(2021, 10, 31))

    # asof restricts the returned months.
    assert max(z_asof) <= date(2021, 10, 31)
    assert max(z_full) == date(2022, 4, 30)
    # The pre-break z-scores differ: full-history stats are contaminated
    # by the break, asof stats are not.
    assert z_asof[date(2021, 6, 30)] != z_full[date(2021, 6, 30)]
