"""Tests for the lattice ensemble.

Focus: weight normalization, grid-mismatch defense, monotonicity
preservation, and the inverse-CRPS weighting direction.
"""

from __future__ import annotations

import pytest

from td_one.models.ensemble import LatticeEnsemble, average_lattices, crps_weights

GRID = [0.1, 0.5, 0.9]


def _lat(vals: list[float]) -> dict[float, float]:
    return {float(k): float(v) for k, v in zip(GRID, vals, strict=True)}


def test_equal_weight_average_is_exact() -> None:
    out = average_lattices([_lat([0.1, 0.2, 0.3]), _lat([0.3, 0.4, 0.5])])
    assert out[0.1] == pytest.approx(0.2)
    assert out[0.5] == pytest.approx(0.3)
    assert out[0.9] == pytest.approx(0.4)


def test_weights_are_normalized_not_taken_raw() -> None:
    a = average_lattices([_lat([0.0, 0.0, 0.0]), _lat([0.3, 0.6, 0.9])], weights=[2.0, 1.0])
    b = average_lattices([_lat([0.0, 0.0, 0.0]), _lat([0.3, 0.6, 0.9])], weights=[4.0, 2.0])
    for k, v in ((0.1, 0.1), (0.5, 0.2), (0.9, 0.3)):
        assert a[k] == pytest.approx(v)
        assert b[k] == pytest.approx(v)


def test_mismatched_grids_rejected() -> None:
    with pytest.raises(ValueError):
        average_lattices([_lat([0.1, 0.2, 0.3]), {0.2: 0.2, 0.5: 0.5}])
    with pytest.raises(ValueError):
        average_lattices([])


def test_monotonicity_preserved() -> None:
    out = average_lattices([_lat([0.05, 0.5, 0.95]), _lat([0.1, 0.4, 0.8]), _lat([0.0, 0.6, 1.0])])
    vals = [out[k] for k in GRID]
    assert vals == sorted(vals)


def test_crps_weights_favor_better_component() -> None:
    w = crps_weights([[1.0, 1.2], [4.0, 5.0], [2.0, 2.2]])
    assert abs(sum(w) - 1.0) < 1e-12
    assert w[0] > w[2] > w[1]  # lower CRPS -> higher weight
    with pytest.raises(ValueError):
        crps_weights([[], [1.0]])


def test_ensemble_class_weight_validation() -> None:
    ens = LatticeEnsemble()
    assert ens.weights is None
    out = ens.combine([_lat([0.2, 0.2, 0.2]), _lat([0.4, 0.4, 0.4])])
    assert out[0.5] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        LatticeEnsemble(weights=[-1.0, 2.0])
    ens.set_weights([3.0, 1.0])
    assert ens.weights == [3.0, 1.0]
    with pytest.raises(ValueError):
        ens.combine([_lat([0.1, 0.1, 0.1])])  # 2 weights, 1 lattice
