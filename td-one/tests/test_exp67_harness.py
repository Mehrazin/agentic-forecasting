"""Regression tests for the exp67 red-team harness helpers.

Covers the naive-baseline horizon-tiling bug (2026-10-20): _naive_quantiles
returned shape (19, 1) instead of (19, h), which made every naive forecast
raise "horizon mismatch" inside crps_from_quantiles.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "exp67_tsfm_redteam.py"


def _load_exp67():
    spec = importlib.util.spec_from_file_location("exp67_tsfm_redteam", EXP)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp67_tsfm_redteam"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def exp67():
    return _load_exp67()


def test_naive_quantiles_shape_matches_horizon(exp67) -> None:
    ctx = np.linspace(0.02, 0.05, 40)
    levels, vals = exp67._naive_quantiles(ctx, 6)
    assert levels.shape == (19,)
    assert vals.shape == (19, 6)


def test_naive_quantiles_constant_across_horizon(exp67) -> None:
    ctx = np.linspace(0.02, 0.05, 40)
    _, vals = exp67._naive_quantiles(ctx, 6)
    assert np.allclose(vals[:, 0], vals[:, -1])


def test_naive_quantiles_monotone_in_level(exp67) -> None:
    ctx = np.linspace(0.02, 0.05, 40)
    _, vals = exp67._naive_quantiles(ctx, 6)
    assert bool(np.all(np.diff(vals, axis=0) >= -1e-12))
