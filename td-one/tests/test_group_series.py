"""Tests for td_one.data_gen.group_series."""

from datetime import date

import pandas as pd
import pytest

from td_one.data_gen.group_series import (
    GroupSeries,
    build_group_series,
    portfolio_totals,
)


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mortgage_id": ["a", "b", "c", "d", "e", "f"],
            "outcome_observed_at": [
                date(2024, 1, 15),
                date(2024, 1, 20),
                date(2024, 2, 10),
                date(2024, 2, 12),
                date(2024, 2, 15),
                date(2024, 3, 5),
            ],
            "switched_lender_flag": [True, False, True, True, False, False],
        }
    )


def test_build_group_series_basic() -> None:
    group_of = {"a": "g1", "b": "g1", "c": "g2", "d": "g2", "e": "g2", "f": "g1"}
    out = build_group_series(_outcomes(), group_of)
    assert set(out) == {"g1", "g2"}
    g1 = out["g1"]
    assert g1.months == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    assert g1.n_at_risk == [2, 0, 1]
    assert g1.n_switched == [1, 0, 0]
    g2 = out["g2"]
    assert g2.n_at_risk == [0, 3, 0]
    assert g2.n_switched == [0, 2, 0]


def test_aggregation_identity() -> None:
    """Group sums equal portfolio totals every month."""
    group_of = {"a": "g1", "b": "g1", "c": "g2", "d": "g2", "e": "g2", "f": "g1"}
    out = build_group_series(_outcomes(), group_of)
    total = portfolio_totals(out)
    assert total.n_at_risk == [2, 3, 1]
    assert total.n_switched == [1, 2, 0]


def test_attrition_rate_zero_safe() -> None:
    s = GroupSeries(
        group_id="g",
        months=[date(2024, 1, 1)],
        n_at_risk=[0],
        n_switched=[0],
    )
    assert s.attrition_rate == [0.0]


def test_unmapped_mortgage_raises() -> None:
    with pytest.raises(ValueError, match="no group assignment"):
        build_group_series(_outcomes(), {"a": "g1"})


def test_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_group_series(pd.DataFrame({"x": [1]}), {})


def test_month_bounds() -> None:
    group_of = dict.fromkeys("abcdef", "g")
    out = build_group_series(
        _outcomes(), group_of, min_month=date(2024, 2, 1), max_month=date(2024, 2, 1)
    )
    assert out["g"].months == [date(2024, 2, 1)]
    assert out["g"].n_at_risk == [3]


def test_portfolio_totals_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        portfolio_totals({})
