"""Tests for MacroConditionedTSFM (residual macro conditioning).

Uses mock wrappers and a recording stub — no network, no weights.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from td_one.models.tsfm import TIMESFM_LEVELS, TimesFMWrapper
from td_one.models.tsfm_macro import MacroConditionedTSFM


def _months(n: int) -> list[date]:
    """N consecutive month-ends starting 2021-04-30."""
    y, m = 2021, 4
    out: list[date] = []
    for _ in range(n):
        last = 30 if m in (4, 6, 9, 11) else 28 if m == 2 else 31
        out.append(date(y, m, last))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


class RecordingWrapper:
    """Stub wrapper capturing the series it was asked to forecast."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def forecast_quantiles(
        self, series: np.ndarray, horizon: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=float)
        self.seen.append(arr.copy())
        h = horizon or 1
        # Echo the last value at every level (monotone by construction).
        vals = np.tile(arr[-1], (len(TIMESFM_LEVELS), h))
        return TIMESFM_LEVELS.copy(), vals


def _const_panels(n: int = 24, rate: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    n_arr = np.full(n, 500.0)
    a_arr = np.full(n, 500.0 * rate)
    return n_arr, a_arr


def test_beta_zero_matches_plain_wrapper() -> None:
    """Empty fred_z -> beta_hat 0 -> identical to the plain wrapper output."""
    n, a = _const_panels()
    months = _months(len(n))
    wrapper = TimesFMWrapper(use_mock=True)
    mc = MacroConditionedTSFM(wrapper, fred_z={})
    got = mc.forecast_quantiles(n, a, months, months[-1])
    assert mc.beta_hat_ == 0.0
    # Plain path: wrapper on raw rates, horizon 1, sorted.
    rates = np.clip(a / n, 1e-6, 1 - 1e-6)
    levels, values = wrapper.forecast_quantiles(rates, horizon=1)
    raw = np.sort(np.asarray(values[:, 0], dtype=float))
    lvls = np.asarray(levels, dtype=float)
    order = np.argsort(lvls, kind="stable")
    expected = {float(lvls[i]): float(raw[j]) for j, i in enumerate(order)}
    assert got.keys() == expected.keys()
    for k in got:
        assert got[k] == pytest.approx(expected[k])


def test_output_monotone_after_shift() -> None:
    """Macro shift preserves quantile monotonicity (sigmoid is monotone)."""
    n, a = _const_panels()
    months = _months(len(n))
    # Strong trend in z so beta_hat != 0.
    fred_z = {d: float(i - len(months) / 2) for i, d in enumerate(months)}
    # Inject a macro-correlated component into rates.
    a = a * (1.0 + 0.05 * np.array([fred_z[d] for d in months]))
    mc = MacroConditionedTSFM(TimesFMWrapper(use_mock=True), fred_z=fred_z)
    fmonth = date(2021, 5, 31)
    fred_z[fmonth] = 2.0
    got = mc.forecast_quantiles(n, a, months, fmonth)
    vals = [got[float(lvl)] for lvl in sorted(got)]
    assert all(b >= a_ for a_, b in zip(vals, vals[1:], strict=False))


def test_residualization_removes_macro() -> None:
    """The wrapper sees macro-residualized rates, not raw rates."""
    n = 24
    months = _months(n)
    base = 0.2
    beta_true = 0.6
    fred_z = {d: float(np.sin(i / 3.0)) for i, d in enumerate(months)}
    # DGP: logit(r[i]) = logit(base) + beta_true * z[i-1] (lagged).
    logit_base = float(np.log(base / (1 - base)))
    z_list = [fred_z[d] for d in months]
    rates = np.array(
        [
            1.0 / (1.0 + np.exp(-(logit_base + beta_true * (z_list[i - 1] if i > 0 else 0.0))))
            for i in range(n)
        ]
    )
    n_arr = np.full(n, 2000.0)
    a_arr = rates * n_arr

    rec: RecordingWrapper = RecordingWrapper()
    mc = MacroConditionedTSFM(rec, fred_z=fred_z)  # type: ignore[arg-type]
    mc.forecast_quantiles(n_arr, a_arr, months, months[-1])

    assert len(rec.seen) == 1
    seen = rec.seen[0]
    # Residualized rates should be ~flat at base (macro removed).
    assert np.allclose(seen, base, atol=0.02), seen[:5]
    # And beta_hat should recover something close to beta_true.
    assert abs(mc.beta_hat_ - beta_true) < 0.15, mc.beta_hat_


def test_missing_z_months_graceful() -> None:
    """Months absent from fred_z are treated as z=0 (no crash)."""
    n, a = _const_panels()
    months = _months(len(n))
    fred_z = {months[0]: 1.0, months[-1]: -1.0}  # sparse
    mc = MacroConditionedTSFM(TimesFMWrapper(use_mock=True), fred_z=fred_z)
    got = mc.forecast_quantiles(n, a, months, date(2030, 1, 31))
    assert len(got) == len(TIMESFM_LEVELS)
    assert all(0.0 <= v <= 1.0 for v in got.values())


def test_misaligned_inputs_raise() -> None:
    """Misaligned n/a/months raise ValueError."""
    n, a = _const_panels()
    months = _months(len(n))
    mc = MacroConditionedTSFM(TimesFMWrapper(use_mock=True), fred_z={})
    with pytest.raises(ValueError, match="aligned"):
        mc.forecast_quantiles(n, a[:10], months, months[-1])
    with pytest.raises(ValueError, match="align"):
        mc.forecast_quantiles(n, a, months[:10], months[-1])
