"""Tests for cohort recovery metrics."""

import pytest

from td_one.evaluation.cohort_recovery import adjusted_rand_index, best_match_per_cohort


def test_ari_perfect_match():
    true = ["a", "a", "b", "b", "c"]
    assert adjusted_rand_index(true, true) == pytest.approx(1.0)


def test_ari_relabel_invariant():
    true = ["a", "a", "b", "b"]
    pred = ["x", "x", "y", "y"]
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0)


def test_ari_chance_level():
    # One group for everything: no information -> ARI ~ 0.
    true = ["a", "a", "b", "b"]
    pred = ["g", "g", "g", "g"]
    assert abs(adjusted_rand_index(true, pred)) < 0.05


def test_ari_length_mismatch():
    with pytest.raises(ValueError):
        adjusted_rand_index(["a"], ["a", "b"])


def test_ari_empty():
    with pytest.raises(ValueError):
        adjusted_rand_index([], [])


def test_best_match_exact():
    true = ["a", "a", "b", "b"]
    pred = ["g0", "g0", "g1", "g1"]
    m = best_match_per_cohort(true, pred)
    assert m["a"]["group_id"] == "g0"
    assert m["a"]["precision"] == pytest.approx(1.0)
    assert m["a"]["recall"] == pytest.approx(1.0)
    assert m["b"]["group_id"] == "g1"


def test_best_match_split_cohort():
    # Cohort 'a' split across g0 (3/4) and g1 (1/4); g0 is the best match.
    true = ["a", "a", "a", "a", "b"]
    pred = ["g0", "g0", "g0", "g1", "g1"]
    m = best_match_per_cohort(true, pred)
    assert m["a"]["group_id"] == "g0"
    assert m["a"]["recall"] == pytest.approx(0.75)
    assert m["a"]["precision"] == pytest.approx(1.0)
    assert m["a"]["f1"] == pytest.approx(2 * 1.0 * 0.75 / 1.75)


def test_best_match_impure_group():
    # g0 = 2 a's + 1 b: precision for 'a' is 2/3.
    true = ["a", "a", "b"]
    pred = ["g0", "g0", "g0"]
    m = best_match_per_cohort(true, pred)
    assert m["a"]["group_id"] == "g0"
    assert m["a"]["precision"] == pytest.approx(2 / 3)
    assert m["a"]["recall"] == pytest.approx(1.0)
