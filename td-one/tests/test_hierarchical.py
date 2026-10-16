"""Tests for the hierarchical (partial-pooling) baseline.

Focus: shrinkage direction (small segments borrow strength, large ones
don't), the no-heterogeneity fallback, and defensive contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from td_one.models.hierarchical import HierarchicalBaseline
from td_one.models.tsfm import TIMESFM_LEVELS

LEVELS = np.asarray(TIMESFM_LEVELS, dtype=float)


def _seg(n_total: int, rate: float, months: int = 6) -> tuple[np.ndarray, np.ndarray]:
    n = np.full(months, n_total / months)
    a = n * rate
    return n, a


def test_small_segment_shrinks_to_pooled_mean() -> None:
    # One tiny segment at rate 0.9, nine big segments at 0.1.
    segs = {"tiny": _seg(60, 0.9)}
    for i in range(9):
        segs[f"big{i}"] = _seg(6000, 0.1)
    m = HierarchicalBaseline()
    m.fit(segs)
    raw_tiny = 0.9
    shrunk = m.expected_rate("tiny")
    # Pooled mean is near 0.1; the tiny segment must move toward it.
    assert shrunk < raw_tiny
    assert abs(shrunk - m.pooled_rate) < abs(raw_tiny - m.pooled_rate)
    # A big segment is barely moved.
    assert abs(m.expected_rate("big0") - 0.1) < 0.02


def test_large_segment_dominates_own_posterior() -> None:
    segs = {f"s{i}": _seg(20000, 0.2 + 0.01 * i) for i in range(5)}
    m = HierarchicalBaseline()
    m.fit(segs)
    for i in range(5):
        assert m.shrinkage(f"s{i}") < 0.05  # prior contributes <5%


def test_no_heterogeneity_falls_back_to_jeffreys() -> None:
    # Identical rates -> no detectable between-segment variance.
    segs = {f"s{i}": _seg(1000, 0.3) for i in range(6)}
    m = HierarchicalBaseline()
    m.fit(segs)
    assert m.prior_params == (0.5, 0.5)


def test_unknown_segment_raises() -> None:
    m = HierarchicalBaseline()
    m.fit({"a": _seg(100, 0.2)})
    with pytest.raises(KeyError):
        m.expected_rate("nope")


def test_empty_segments_raises() -> None:
    with pytest.raises(ValueError):
        HierarchicalBaseline().fit({})


def test_lattice_monotone_and_bounded() -> None:
    segs = {"a": _seg(500, 0.25), "b": _seg(800, 0.15)}
    m = HierarchicalBaseline()
    m.fit(segs)
    lat = m.predict_rate_quantiles("a", 100, LEVELS)
    vals = [lat[float(k)] for k in LEVELS]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_bad_levels_rejected() -> None:
    m = HierarchicalBaseline()
    m.fit({"a": _seg(500, 0.25)})
    with pytest.raises(ValueError):
        m.predict_rate_quantiles("a", 100, np.array([0.5, 1.5]))
