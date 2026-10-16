"""Tests for the exponentially-weighted M0.

Focus: the decay math (the non-obvious part) -- weight shape, the
uniform limit, the last-observation limit -- plus defensive contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from td_one.models.baseline import HistoricalBaseline
from td_one.models.ewm import EWMHistoricalBaseline, decay_weights


def test_weights_shape_and_halving() -> None:
    w = decay_weights(6, half_life=2.0)
    assert w[-1] == pytest.approx(1.0)  # most recent has full weight
    # Weight halves every half_life periods going back.
    assert w[-1] / w[-3] == pytest.approx(2.0)
    assert bool((w[:-1] < w[1:]).all())  # strictly increasing recency


def test_invalid_half_life_rejected() -> None:
    with pytest.raises(ValueError):
        decay_weights(4, half_life=0.0)
    with pytest.raises(ValueError):
        EWMHistoricalBaseline(half_life=-1.0)


def test_huge_half_life_approximates_uniform_window() -> None:
    n = np.array([100.0, 120.0, 110.0, 90.0])
    a = np.array([10.0, 24.0, 11.0, 27.0])
    ewm = EWMHistoricalBaseline(half_life=1e9)
    ewm.fit(n, a)
    plain = HistoricalBaseline(trailing_window=4)
    plain.fit(n, a)
    assert ewm.expected_rate == pytest.approx(plain.expected_rate, rel=1e-6)


def test_tiny_half_life_uses_only_last_month() -> None:
    n = np.array([100.0, 100.0, 100.0, 100.0])
    a = np.array([5.0, 5.0, 5.0, 80.0])  # shock in the last month
    ewm = EWMHistoricalBaseline(half_life=1e-9)
    ewm.fit(n, a)
    # Effective counts ~= last month only: (0.5 + 80) / (1 + 100).
    assert ewm.expected_rate == pytest.approx(80.5 / 101.0, rel=1e-6)


def test_recency_weighting_direction() -> None:
    # Rising rates: EWM (short half-life) must exceed the uniform average.
    n = np.full(6, 200.0)
    a = np.array([10.0, 12.0, 14.0, 30.0, 40.0, 60.0])
    ewm = EWMHistoricalBaseline(half_life=1.0)
    ewm.fit(n, a)
    plain = HistoricalBaseline(trailing_window=6)
    plain.fit(n, a)
    assert ewm.expected_rate > plain.expected_rate


def test_effective_periods_sane() -> None:
    ewm = EWMHistoricalBaseline(half_life=2.0)
    ewm.fit(np.full(12, 100.0), np.full(12, 10.0))
    # Infinite-horizon sum would be 1/(1-2^-0.5) ~= 3.41; 12 months ~ same.
    assert ewm.effective_periods == pytest.approx(3.41, rel=0.05)
    with pytest.raises(RuntimeError):
        _ = EWMHistoricalBaseline().effective_periods


def test_misaligned_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        EWMHistoricalBaseline().fit(np.array([1.0]), np.array([1.0, 2.0]))
