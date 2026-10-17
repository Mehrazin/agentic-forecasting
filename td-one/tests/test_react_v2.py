"""Tests for the code-executing ReAct loop (td_one.agent.react_v2)."""

import json
from datetime import date

from td_one.agent.react import MockLLMClient
from td_one.agent.react_v2 import ReActAgentV2, _run_code_tool


def test_code_execute_tool_basic() -> None:
    obs = _run_code_tool({"code": "print(2 * 21)"})
    assert "42" in obs
    assert "exit=0" in obs


def test_code_execute_tool_empty_code() -> None:
    obs = _run_code_tool({"code": "   "})
    assert "error" in obs.lower()


def test_code_execute_tool_bad_timeout() -> None:
    obs = _run_code_tool({"code": "print(1)", "timeout_s": 999})
    assert "error" in obs.lower()


def test_code_execute_tool_can_use_T_namespace() -> None:
    obs = _run_code_tool(
        {"code": "_show(T.detect_changepoints([0.0]*30 + [5.0]*30).n_segments)"}
    )
    assert "exit=0" in obs
    assert "2" in obs


def test_code_execute_timeout_observed() -> None:
    obs = _run_code_tool({"code": "import time; time.sleep(30)", "timeout_s": 2})
    assert "timed out" in obs.lower()


def test_v2_finalize_flow() -> None:
    responses = [
        json.dumps({"action": "code_execute", "args": {"code": "print('analysis done')"}}),
        json.dumps(
            {"action": "finalize", "forecast": {"0.1": 0.01, "0.5": 0.02, "0.9": 0.05}}
        ),
    ]
    agent = ReActAgentV2(MockLLMClient(responses), tools={})
    result = agent.run("forecast segment X", date(2026, 1, 31))
    assert result["forecast"] is not None
    assert result["forecast"][0.5] == 0.02
    assert result["steps"] == 2
    actions = [s["action"] for s in result["trace"]]
    assert actions == ["code_execute", "finalize"]


def test_v2_max_steps_exhausted() -> None:
    responses = [json.dumps({"action": "code_execute", "args": {"code": "print(1)"}})]
    agent = ReActAgentV2(MockLLMClient(responses), tools={}, max_steps=3)
    result = agent.run("never finalizes", date(2026, 1, 31))
    assert result["forecast"] is None
    assert result["steps"] == 3


def test_v2_v1_tools_still_work() -> None:
    def fake_data(args: dict) -> str:
        return "segment data"

    responses = [
        json.dumps({"action": "data_segment", "args": {"segment_id": "s1"}}),
        json.dumps(
            {"action": "finalize", "forecast": {"0.1": 0.01, "0.5": 0.02, "0.9": 0.05}}
        ),
    ]
    agent = ReActAgentV2(MockLLMClient(responses), tools={"data_segment": fake_data})
    result = agent.run("task", date(2026, 1, 31))
    assert result["forecast"] is not None
    assert result["trace"][0]["observation"] == "segment data"


def test_v2_unknown_action_recorded() -> None:
    responses = [
        json.dumps({"action": "teleport", "args": {}}),
        json.dumps(
            {"action": "finalize", "forecast": {"0.1": 0.01, "0.5": 0.02, "0.9": 0.05}}
        ),
    ]
    agent = ReActAgentV2(MockLLMClient(responses), tools={})
    result = agent.run("task", date(2026, 1, 31))
    assert result["trace"][0]["action"] == "unknown_action"
    assert result["forecast"] is not None
