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
