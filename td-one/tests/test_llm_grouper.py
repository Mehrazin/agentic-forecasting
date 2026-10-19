"""Tests for td_one.agent.llm_grouper (mock LLM; no API calls)."""

import json

import pandas as pd
import pytest

from td_one.agent.llm_grouper import (
    LLMGrouper,
    build_attribute_summaries,
    build_grouper_prompt,
    parse_group_proposals,
)


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mortgage_id": [f"m{i}" for i in range(300)],
            "province": ["ON"] * 180 + ["BC"] * 120,
            "household_income_cad": [60000.0] * 150 + [140000.0] * 150,
            "switched": [1] * 90 + [0] * 210,
        }
    )


def test_build_attribute_summaries(df) -> None:
    s = build_attribute_summaries(df, "switched", categorical=["province"])
    assert "province" in s
    assert "ON" in s["province"]
    assert "household_income_cad" in s  # numeric binned
    assert "mortgage_id" not in s  # id columns excluded
    assert "switched" not in s  # outcome excluded


def test_build_grouper_prompt_no_rows(df) -> None:
    s = build_attribute_summaries(df, "switched", categorical=["province"])
    from td_one.agent.llm_grouper import GrouperContext

    ctx = GrouperContext(attribute_summaries=s, news_summaries=["BoC holds rate"])
    system, user = build_grouper_prompt(ctx)
    assert "m0" not in user  # no individual rows leak
    assert "BoC holds rate" in user
    assert "pandas" in system


def test_parse_group_proposals_ok() -> None:
    text = json.dumps(
        {"groups": [{"group_id": "g1", "rule": "province == 'ON'", "rationale": "x"}]}
    )
    out = parse_group_proposals(text)
    assert out is not None and out[0]["group_id"] == "g1"


def test_parse_group_proposals_bad() -> None:
    assert parse_group_proposals("not json") is None
    assert parse_group_proposals(json.dumps({"groups": "x"})) is None
    assert parse_group_proposals(json.dumps({"groups": [{"rule": "x"}]})) is None


class GoodLLM:
    def complete(self, prompt: str, system=None, json_mode=False) -> str:
        return json.dumps(
            {
                "groups": [
                    {
                        "group_id": "on_low",
                        "rule": "province == 'ON'",
                        "rationale": "Ontario borrowers",
                    },
                    {
                        "group_id": "bc",
                        "rule": "province == 'BC'",
                        "rationale": "BC borrowers",
                    },
                ]
            }
        )


class BadLLM:
    def complete(self, prompt: str, system=None, json_mode=False) -> str:
        return "I cannot comply with JSON"


def test_llm_grouper_propose_ok(df) -> None:
    g = LLMGrouper(GoodLLM(), max_retries=0)
    out = g.propose(df, "switched", categorical=["province"], min_group_n=50)
    assert {x.group_id for x in out} == {"on_low", "bc"}


def test_llm_grouper_retries_then_empty(df) -> None:
    g = LLMGrouper(BadLLM(), max_retries=1)
    assert g.propose(df, "switched", min_group_n=50) == []


def test_llm_grouper_rejects_leaky_rule(df) -> None:
    class LeakyLLM:
        def complete(self, prompt: str, system=None, json_mode=False) -> str:
            return json.dumps(
                {"groups": [{"group_id": "x", "rule": "switched == 1", "rationale": "leak"}]}
            )

    g = LLMGrouper(LeakyLLM(), max_retries=0)
    assert g.propose(df, "switched", min_group_n=50) == []


def test_build_grouper_prompt_with_baseline(df) -> None:
    """Baseline CRPS appears in prompt when provided; hints present."""
    s = build_attribute_summaries(df, "switched", categorical=["province"])
    from td_one.agent.llm_grouper import GrouperContext

    ctx = GrouperContext(attribute_summaries=s)
    _, user = build_grouper_prompt(ctx, baseline_crps=0.01234, baseline_desc="test baseline")
    assert "0.01234" in user
    assert "test baseline" in user
    assert "INTERACTIONS" in user
    assert "10-15%" in user  # coverage guidance


def test_build_grouper_prompt_no_baseline(df) -> None:
    """No baseline section when baseline_crps is None."""
    s = build_attribute_summaries(df, "switched", categorical=["province"])
    from td_one.agent.llm_grouper import GrouperContext

    ctx = GrouperContext(attribute_summaries=s)
    _, user = build_grouper_prompt(ctx)
    assert "CRPS" not in user or "baseline" not in user.lower()


def test_fmt_interval_rounds() -> None:
    """Bin edges are rounded to human-friendly values."""
    import pandas as pd

    from td_one.agent.llm_grouper import _fmt_interval

    iv = pd.Interval(1661.546, 2950.686, closed="right")
    s = _fmt_interval(iv)
    assert "1661" not in s  # not over-precise
    assert "1700" in s and "3000" in s


def test_attribute_summaries_rounded(df) -> None:
    """Numeric summaries use rounded bin edges."""
    from td_one.agent.llm_grouper import build_attribute_summaries

    s = build_attribute_summaries(df, "switched", categorical=["province"])
    # household_income_cad has values 60000/140000; bins should be rounded
    assert "household_income_cad" in s
    # Bin edges (in parentheses) should not have 3+ decimal places.
    # Rate values like rate=0.300 are fine — only check parenthesized edges.
    import re

    edges = re.findall(r"\(([^)]+)\)", s["household_income_cad"])
    for e in edges:
        precise = re.findall(r"\d+\.\d{3,}", e)
        assert not precise, f"over-precise edge: {e}"
