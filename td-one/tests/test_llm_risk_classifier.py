"""Tests for the LLM risk classifier (mock LLM)."""

import pytest

from td_one.agent.llm_risk_classifier import (
    classify_risk_llm,
    parse_risk_level,
)


class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt, system=None):
        self.calls += 1
        return self.responses.pop(0)


def _stats():
    return {"n_members": 209, "train_realized_attrition": 0.0833}


def test_parse_valid():
    assert parse_risk_level("RISK: HIGH\nJUSTIFICATION: x [stat]") == "HIGH"
    assert parse_risk_level("some text\nrisk: low\nmore") == "LOW"


def test_parse_invalid():
    assert parse_risk_level("RISK: EXTREME") is None
    assert parse_risk_level("no risk line here") is None


def test_classify_success():
    llm = MockLLM(["RISK: MEDIUM\nJUSTIFICATION: 209 members switched at 0.0833 [stat]."])
    out = classify_risk_llm(llm, "g2", "rule", _stats(), [])
    assert out.level == "MEDIUM"
    assert out.citations_valid is True


def test_classify_retries_on_garbage():
    llm = MockLLM(["hmm not sure", "RISK: HIGH\nJUSTIFICATION: risky [stat]."])
    out = classify_risk_llm(llm, "g0", "rule", _stats(), [], max_retries=1)
    assert out.level == "HIGH"
    assert llm.calls == 2


def test_classify_raises_after_exhausted_retries():
    llm = MockLLM(["nope", "still nope"])
    with pytest.raises(ValueError, match="parseable RISK"):
        classify_risk_llm(llm, "g0", "rule", _stats(), [], max_retries=1)
