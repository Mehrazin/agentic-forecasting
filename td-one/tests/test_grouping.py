"""Tests for td_one.agent.grouping."""

import pandas as pd
import pytest

from td_one.agent.grouping import (
    GroupDefinition,
    apply_groups,
    discover_candidate_groups,
    jaccard,
    resolve_overlaps,
    validate_rule,
)


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(10)],
            "province": ["ON"] * 6 + ["BC"] * 4,
            "household_income_cad": [50000, 60000, 70000, 80000, 90000, 100000,
                                     120000, 130000, 140000, 150000],
            "credit_utilization_pct": [70, 65, 30, 20, 80, 75, 10, 15, 25, 5],
            "switched": [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        }
    )


def test_validate_rule_ok(df) -> None:
    gd = validate_rule(
        "province == 'ON' and household_income_cad < 75000",
        set(df.columns), min_group_n=2, df=df,
    )
    assert gd is not None
    assert gd.rule == "province == 'ON' and household_income_cad < 75000"


def test_validate_rule_rejects_outcome_column(df) -> None:
    assert validate_rule("switched == 1", set(df.columns), df=df) is None
    assert validate_rule(
        "province == 'ON' and switched_lender_flag == True",
        set(df.columns), df=df,
    ) is None


def test_validate_rule_rejects_unknown_column(df) -> None:
    assert validate_rule("foobar > 3", set(df.columns), df=df) is None


def test_validate_rule_rejects_bad_syntax(df) -> None:
    assert validate_rule("province ===", set(df.columns), df=df) is None


def test_validate_rule_rejects_tiny_group(df) -> None:
    gd = validate_rule(
        "household_income_cad > 149000", set(df.columns), min_group_n=2, df=df
    )
    assert gd is None


def test_validate_rule_empty(df) -> None:
    assert validate_rule("", set(df.columns)) is None
    assert validate_rule("   ", set(df.columns)) is None


def test_apply_groups_partition(df) -> None:
    groups = [
        GroupDefinition("on", "province == 'ON'"),
        GroupDefinition("bc", "province == 'BC'"),
    ]
    a = apply_groups(df, groups)
    assert len(a) == 10
    assert a["m0"] == "on"
    assert a["m6"] == "bc"


def test_apply_groups_unmatched_raises(df) -> None:
    groups = [GroupDefinition("on", "province == 'ON'")]
    with pytest.raises(ValueError, match="matches no group"):
        apply_groups(df, groups)


def test_apply_groups_overlap_raises(df) -> None:
    groups = [
        GroupDefinition("a", "province == 'ON'"),
        GroupDefinition("b", "household_income_cad < 100000"),
    ]
    with pytest.raises(ValueError, match="multiple groups"):
        apply_groups(df, groups)


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, {"a"}) == 1.0


def test_resolve_overlaps_keeps_better_crps(df) -> None:
    groups = [
        GroupDefinition("wide", "province == 'ON'"),  # 6 members
        GroupDefinition("narrow", "province == 'ON' and household_income_cad < 90000"),  # 5
    ]
    kept = resolve_overlaps(
        df, groups, {"wide": 0.5, "narrow": 0.3}, max_overlap=0.5
    )
    assert [g.group_id for g in kept] == ["narrow"]


def test_resolve_overlaps_keeps_disjoint(df) -> None:
    groups = [
        GroupDefinition("on", "province == 'ON'"),
        GroupDefinition("bc", "province == 'BC'"),
    ]
    kept = resolve_overlaps(df, groups, {"on": 0.5, "bc": 0.4})
    assert len(kept) == 2


def test_discover_candidate_groups_filters_bad(df) -> None:
    proposals = [
        {"group_id": "good", "rule": "province == 'ON'", "rationale": "ok"},
        {"group_id": "bad_outcome", "rule": "switched == 1", "rationale": "leak"},
        {"group_id": "bad_col", "rule": "nope > 1", "rationale": "bad"},
        {"group_id": "tiny", "rule": "household_income_cad > 149000", "rationale": "tiny"},
    ]
    out = discover_candidate_groups(df, proposals, min_group_n=2)
    ids = [g.group_id for g in out]
    assert "good" in ids
    assert "bad_outcome" not in ids
    assert "bad_col" not in ids
    assert "tiny" not in ids
    # Partition guarantee: catch-all covers the BC rows.
    assert "__catch_all__" in ids
    good = next(g for g in out if g.group_id == "good")
    assert good.rationale == "ok"
