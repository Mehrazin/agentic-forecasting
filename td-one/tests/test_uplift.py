"""Tests for T-learner uplift."""

import numpy as np
import pandas as pd
import pytest

from td_one.evaluation import uplift as up


def _heterogeneous_dgp(n: int = 1000, seed: int = 0):
    # Treatment effect is large when x1 > 0, zero otherwise.
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    base = -1.0 + 0.5 * x1
    effect = 2.0 * (x1 > 0) * t
    p = 1 / (1 + np.exp(-(base + effect)))
    y = (rng.random(n) < p).astype(int)
    x = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n)})
    return x, t, y, x1


def test_t_learner_recovers_heterogeneity():
    x, t, y, x1 = _heterogeneous_dgp()
    fit = up.fit_t_learner(x, t, y, seed=0)
    scores = up.uplift_scores(fit, x)
    high = scores[x1 > 0.5].mean()
    low = scores[x1 < -0.5].mean()
    assert high > low + 0.1, f"uplift should separate high/low responders: {high} vs {low}"


def test_t_learner_rejects_tiny_arms():
    x = pd.DataFrame({"x1": np.arange(60, dtype=float)})
    t = np.array([1] * 10 + [0] * 50)
    y = np.zeros(60, dtype=int)
    with pytest.raises(ValueError, match=">= 50 rows per arm"):
        up.fit_t_learner(x, t, y)


def test_uplift_deciles_deterministic_and_cover():
    scores = np.linspace(0, 1, 100)
    d1 = up.uplift_deciles(scores, n_deciles=10)
    d2 = up.uplift_deciles(scores, n_deciles=10)
    assert (d1 == d2).all()
    assert set(d1) == set(range(10))
    assert (d1[:10] == 0).all()  # lowest scores -> decile 0


def test_variable_rate_treatment():
    df = pd.DataFrame({"rate_type": ["variable", "fixed", "variable"]})
    assert list(up.variable_rate_treatment(df)) == [1, 0, 1]


def test_decile_cohort_table_shape():
    deciles = np.array([0, 0, 1, 1, 9])
    cohorts = ["a", "b", "a", "b", "a"]
    tab = up.decile_cohort_table(deciles, cohorts)
    assert tab.loc[0, "a"] == 1
    assert tab.loc[9, "a"] == 1
