"""Tests for M3 shrinkage and regime gating.

Covers the non-obvious math (shrinkage in logit space, z-test gating)
and the defensive contracts. No network; no LLM.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logit

from td_one.models.hybrid import HybridReadout
from td_one.models.shrinkage import (
    ShrunkHybridReadout,
    gate_delta,
    regime_z_stat,
)


def _flat_panel(
    months: int = 12, n_per: int = 50, rate: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    n = np.full(months, n_per, dtype=float)
    a = np.full(months, n_per * rate, dtype=float)
    return n, a


def _step_panel(
    months: int = 12,
    n_per: int = 200,
    base_rate: float = 0.2,
    shift_rate: float = 0.5,
    shift_months: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.full(months, n_per, dtype=float)
    a = np.full(months, n_per * base_rate, dtype=float)
    a[-shift_months:] = n_per * shift_rate
    return n, a


class TestShrunkHybridReadout:
    def test_k_zero_is_pure_m0(self) -> None:
        r = ShrunkHybridReadout(shrinkage=0.0)
        assert r.apply_adjustment(0.3, 0.75) == pytest.approx(0.3)
        assert r.apply_adjustment(0.3, -0.75) == pytest.approx(0.3)

    def test_k_one_matches_parent(self) -> None:
        shrunk = ShrunkHybridReadout(shrinkage=1.0)
        parent = HybridReadout()
        for base, delta in [(0.3, 0.5), (0.1, -0.4), (0.6, 2.0)]:
            assert shrunk.apply_adjustment(base, delta) == pytest.approx(
                parent.apply_adjustment(base, delta)
            )

    def test_k_half_is_halfway_in_logit_space(self) -> None:
        r = ShrunkHybridReadout(shrinkage=0.5)
        base, delta = 0.3, 0.6
        adjusted = r.apply_adjustment(base, delta)
        # logit(adjusted) == logit(base) + 0.5 * delta
        assert logit(adjusted) == pytest.approx(logit(base) + 0.5 * delta)

    def test_shrinkage_still_respects_bound(self) -> None:
        # delta=2.0, k=0.5 -> effective 1.0 -> clipped to +0.75
        r = ShrunkHybridReadout(shrinkage=0.5)
        parent = HybridReadout()
        assert r.apply_adjustment(0.3, 2.0) == pytest.approx(parent.apply_adjustment(0.3, 0.75))

    def test_monotonicity_preserved(self) -> None:
        r = ShrunkHybridReadout(shrinkage=0.4)
        rates = [r.apply_adjustment(0.3, d) for d in (-0.75, -0.1, 0.0, 0.2, 0.75)]
        assert rates == sorted(rates)

    @pytest.mark.parametrize("bad_k", [-0.1, 1.1, 2.0, float("nan")])
    def test_invalid_shrinkage_raises(self, bad_k: float) -> None:
        with pytest.raises(ValueError, match="shrinkage"):
            ShrunkHybridReadout(shrinkage=bad_k)

    def test_default_is_one(self) -> None:
        assert ShrunkHybridReadout().shrinkage == 1.0


class TestRegimeZStat:
    def test_flat_panel_z_near_zero(self) -> None:
        n, a = _flat_panel()
        assert abs(regime_z_stat(n, a)) < 0.5

    def test_step_up_positive_z(self) -> None:
        n, a = _step_panel()
        assert regime_z_stat(n, a) > 3.0

    def test_step_down_negative_z(self) -> None:
        n, a = _step_panel(base_rate=0.5, shift_rate=0.2)
        assert regime_z_stat(n, a) < -3.0

    def test_too_short_panel_returns_zero(self) -> None:
        n = np.array([10.0, 12.0])
        a = np.array([3.0, 9.0])
        assert regime_z_stat(n, a, recent_months=3) == 0.0

    def test_zero_renewals_returns_zero(self) -> None:
        n = np.zeros(12)
        a = np.zeros(12)
        assert regime_z_stat(n, a) == 0.0


class TestGateDelta:
    def test_flat_panel_gate_closed(self) -> None:
        n, a = _flat_panel()
        assert gate_delta(0.30, n, a, threshold=2.0) == 0.0

    def test_step_panel_gate_open(self) -> None:
        n, a = _step_panel()
        assert gate_delta(0.30, n, a, threshold=2.0) == pytest.approx(0.30)

    def test_threshold_zero_always_open(self) -> None:
        n, a = _flat_panel()
        assert gate_delta(0.30, n, a, threshold=0.0) == pytest.approx(0.30)

    def test_gate_is_label_blind(self) -> None:
        # Same panel, different deltas: the gate decision must not depend
        # on the delta's sign or magnitude, only on the panel.
        n, a = _flat_panel()
        assert gate_delta(0.75, n, a) == 0.0
        assert gate_delta(-0.75, n, a) == 0.0
        n2, a2 = _step_panel()
        assert gate_delta(0.75, n2, a2) == pytest.approx(0.75)
        assert gate_delta(-0.75, n2, a2) == pytest.approx(-0.75)

    def test_short_panel_gate_closed(self) -> None:
        n = np.array([10.0, 12.0])
        a = np.array([3.0, 9.0])
        assert gate_delta(0.5, n, a) == 0.0
