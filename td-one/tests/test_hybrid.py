"""Tests for the hybrid readout (TSFM + LLM adjustment)."""

import numpy as np

from td_one.models.hybrid import HybridReadout


def test_hybrid_no_adjustment() -> None:
    """Zero adjustment -> same as base."""
    readout = HybridReadout()
    base_rate = 0.20
    # Delta = 0 means no change
    adjusted = readout.apply_adjustment(base_rate, delta_logit=0.0)
    assert abs(adjusted - base_rate) < 0.001


def test_hybrid_positive_adjustment() -> None:
    """Positive delta increases rate."""
    readout = HybridReadout()
    adjusted = readout.apply_adjustment(0.20, delta_logit=0.5)
    assert adjusted > 0.20
    assert adjusted < 1.0  # Bounded


def test_hybrid_negative_adjustment() -> None:
    """Negative delta decreases rate."""
    readout = HybridReadout()
    adjusted = readout.apply_adjustment(0.20, delta_logit=-0.5)
    assert adjusted < 0.20
    assert adjusted > 0.0  # Bounded


def test_hybrid_delta_clipped() -> None:
    """Delta clipped to [-0.75, +0.75] (plan §4.3)."""
    readout = HybridReadout()
    # Extreme delta should be clipped
    adj1 = readout.apply_adjustment(0.20, delta_logit=5.0)
    adj2 = readout.apply_adjustment(0.20, delta_logit=0.75)
    assert abs(adj1 - adj2) < 0.001


def test_hybrid_count_coherence() -> None:
    """Counts stay in [0, N]."""
    readout = HybridReadout()
    counts = readout.predict_counts(
        base_rate=0.20,
        delta_logit=0.3,
        n_upcoming=100,
        n_samples=500,
        seed=42,
    )
    assert (counts >= 0).all()
    assert (counts <= 100).all()
