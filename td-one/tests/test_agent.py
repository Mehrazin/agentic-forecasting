"""Tests for the TD-One agent controller (Groq-powered)."""

from unittest.mock import MagicMock

from td_one.agent.controller import ForecastAgent


def test_agent_initializes() -> None:
    """Agent creates with Groq provider."""
    mock_llm = MagicMock()
    agent = ForecastAgent(llm=mock_llm)
    assert agent is not None


def test_agent_builds_evidence_prompt() -> None:
    """Evidence prompt includes segment context and date guard."""
    mock_llm = MagicMock()
    agent = ForecastAgent(llm=mock_llm)
    prompt = agent._build_evidence_prompt(
        segment_id="low_rate",
        run_date="2026-10-15",
        segment_profile="Fixed rate <=3%, deposits < $5k",
    )
    assert "low_rate" in prompt
    assert "2026-10-15" in prompt
    # Date guard: must not request future data
    assert "before" in prompt.lower() or "as of" in prompt.lower()
