"""Tests for agent.deterministic_grouper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from td_one.agent.deterministic_grouper import fit_deterministic_groups
from td_one.agent.grouping import OUTCOME_COLUMNS, validate_rule


@pytest.fixture()
def meta_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 600
    provinces = rng.choice(["ON", "BC", "AB", "QC"], size=n)
    return pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(n)],
            "province": provinces,
            "rate_type": rng.choice(["fixed", "variable"], size=n),
            "primary_borrower_age": rng.integers(25, 70, size=n),
            "household_income_cad": rng.lognormal(11.2, 0.5, size=n),
            "mortgage_rate_pct": rng.uniform(2.0, 7.0, size=n),
            "credit_utilization_pct": rng.uniform(0.0, 0.9, size=n),
            "has_advisor": rng.choice([True, False], size=n),
            "insured_flag": rng.choice([True, False], size=n),
            "switched_lender_flag": rng.choice([0, 1], size=n),  # outcome: must be ignored
        }
    )


def test_reproducible_same_seed(meta_df: pd.DataFrame) -> None:
    g1 = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    g2 = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    assert [g.rule for g in g1.groups] == [g.rule for g in g2.groups]
    assert [g.group_id for g in g1.groups] == [g.group_id for g in g2.groups]
    pd.testing.assert_series_equal(g1.assignment, g2.assignment)


def test_rules_valid_and_sized(meta_df: pd.DataFrame) -> None:
    g = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    for gd in g.groups:
        assert (
            validate_rule(gd.rule, set(meta_df.columns), min_group_n=50, df=meta_df) is not None
        ), f"rule failed validation: {gd.rule}"
        n = len(meta_df.query(gd.rule))
        assert n >= 50


def test_full_coverage_exactly_once(meta_df: pd.DataFrame) -> None:
    g = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    assert len(g.assignment) == len(meta_df)
    assert set(g.assignment.unique()) == {gd.group_id for gd in g.groups}
    matched_total = sum(len(meta_df.query(gd.rule)) for gd in g.groups)
    assert matched_total == len(meta_df)


def test_no_outcome_columns_in_rules(meta_df: pd.DataFrame) -> None:
    g = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    import re

    for gd in g.groups:
        lowered = gd.rule.lower()
        for oc in OUTCOME_COLUMNS:
            assert not re.search(rf"(?<![a-z0-9_]){re.escape(oc)}(?![a-z0-9_])", lowered), (
                f"outcome column leaked into rule: {gd.rule}"
            )


def test_rule_application_matches_assignment(meta_df: pd.DataFrame) -> None:
    """Applying each rule via DataFrame.query gives the tree's assignment."""
    g = fit_deterministic_groups(meta_df, n_clusters=4, seed=11, min_group_n=50)
    for gd in g.groups:
        members = set(meta_df.query(gd.rule).index)
        assigned = set(g.assignment[g.assignment == gd.group_id].index)
        assert members == assigned


def test_different_seeds_still_valid(meta_df: pd.DataFrame) -> None:
    for seed in (1, 2, 3):
        g = fit_deterministic_groups(meta_df, n_clusters=3, seed=seed, min_group_n=50)
        assert len(g.groups) >= 2
        assert len(g.assignment) == len(meta_df)


def test_rejects_too_few_clusters(meta_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_deterministic_groups(meta_df, n_clusters=1)
