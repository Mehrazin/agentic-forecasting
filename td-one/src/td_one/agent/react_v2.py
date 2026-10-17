"""Code-executing ReAct loop (M2 v2) for the TD-One forecast agent.

Extends the v1 loop (:mod:`td_one.agent.react`) with a ``code_execute``
action that runs Python in the sandbox (:mod:`td_one.agent.sandbox`). The
LLM is now a "data scientist": it can pull data, run the 10 analysis tools
in-process, plot, test hypotheses, and engineer features — not just call
the three v1 tools and guess.

Protocol per step: the LLM replies with exactly one JSON object:
- ``{"action": "data_segment"|"evidence"|"forecasting", "args": {...}}``
  dispatches to the injected tool and records its observation (v1 compat).
- ``{"action": "code_execute", "args": {"code": "<python>", "timeout_s": 30}}``
  runs the code in the sandbox; stdout/stderr become the observation.
- ``{"action": "finalize", "forecast": {"0.1": q10, "0.5": q50, "0.9": q90}}``
  emits the final rate-quantile forecast and ends the loop.

The sandbox preamble exposes the tool functions under the ``T`` namespace
(``T.analyze_residuals(...)`` etc.) so the agent doesn't reimplement them.
Malformed replies are recorded as failed steps and the loop continues;
if ``max_steps`` is exceeded without a finalize, the forecast is None.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date

from td_one.agent.react import (
    LLMClient,
    MockLLMClient,
    ReActResult,
    TraceStep,
    _parse_forecast,
)
from td_one.agent.sandbox import run_code

logger = logging.getLogger(__name__)

#: Tool actions the v2 loop will dispatch (v1 actions + code execution).
_TOOL_ACTIONS_V2 = ("data_segment", "evidence", "forecasting", "code_execute")

#: Names exposed to sandboxed agent code under the ``T`` namespace.
_TOOL_NAMESPACE = (
    "analyze_residuals",
    "detect_changepoints",
    "feature_importance",
    "cross_validate",
    "compare_models",
    "plot_diagnostics",
    "test_stationarity",
    "decompose_series",
    "backtest_event_impact",
    "generate_synthetic_scenario",
)

_CODE_PREAMBLE = """\
# Tool namespace: call T.<name>(...) — results print as JSON.
import json as _json
from td_one.agent import tools as T
import numpy as _np
import pandas as _pd
def _show(obj):
    try:
        print(_json.dumps(obj, default=str)[:8000])
    except Exception as _e:
        print(repr(obj)[:8000])
"""


def _run_code_tool(args: dict[str, object]) -> str:
    """Execute agent Python in the sandbox; return the observation string."""
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return "code_execute error: args.code must be a non-empty string"
    timeout = args.get("timeout_s", 30)
    timeout_s = int(timeout) if isinstance(timeout, (int, float, str)) else 30
    if not 1 <= timeout_s <= 120:
        return "code_execute error: timeout_s must be in [1, 120]"

    full_code = _CODE_PREAMBLE + "\n" + code
    result = run_code(full_code, timeout_s=timeout_s)
    parts: list[str] = []
    if result.timed_out:
        parts.append(f"[timed out after {timeout_s}s]")
    parts.append(f"[exit={result.returncode} wall_ms={result.wall_ms:.0f}]")
    if result.stdout:
        parts.append("--- stdout ---\n" + result.stdout[:6000])
    if result.stderr:
        parts.append("--- stderr ---\n" + result.stderr[:2000])
    if result.truncated:
        parts.append("[output truncated at byte cap]")
    return "\n".join(parts)


def _render_prompt_v2(task: str, run_date: date, trace: list[TraceStep]) -> str:
    """Build the per-step prompt for the code-executing loop."""
    lines = [
        "You are a mortgage-switching forecast agent with DATA-SCIENTIST tools.",
        f"Run date (PIT cutoff): {run_date.isoformat()}. NEVER use data after this date.",
        f"Task: {task}",
        "",
        "Reply with exactly ONE JSON object per step. Available actions:",
        '  {"action": "data_segment", "args": {...}} — pull segment history',
        '  {"action": "evidence", "args": {...}} — retrieve macro/policy evidence',
        '  {"action": "forecasting", "args": {...}} — run the numerical forecast',
        '  {"action": "code_execute", "args": {"code": "<python>", "timeout_s": 30}}',
        "    — run Python in a sandbox (30s, 1GB, no network). The tool",
        "    namespace T is pre-imported with:",
        "      T.analyze_residuals(y_true, y_pred, features) — residual diagnostics",
        "      T.detect_changepoints(series) — structural breaks",
        "      T.feature_importance(predict_fn, X, y) — permutation importance",
        "      T.cross_validate(model_fn, series) — PIT-safe rolling CV",
        "      T.compare_models(model_fns, series) — CV + Diebold-Mariano",
        "      T.plot_diagnostics(y_true, y_pred) — 4-panel PNG, prints path",
        "      T.test_stationarity(series) — ADF + KPSS",
        "      T.decompose_series(series) — STL trend/seasonal/resid",
        "      T.backtest_event_impact(event_dates, series, dates) — event ATEs",
        "      T.generate_synthetic_scenario(base, shock_type, magnitude) — what-if",
        "    numpy as _np, pandas as _pd. Use _show(obj) to print results.",
        "    Workflow: analyze first (code_execute), then forecast, then finalize.",
        '  {"action": "finalize", "forecast": {"0.1": q10, "0.5": q50, "0.9": q90}}',
        "    — emit the final switching-rate quantiles and stop.",
        "",
        "Trace so far:",
    ]
    if not trace:
        lines.append("  (no steps yet)")
    for i, step in enumerate(trace):
        lines.append(
            f"  [{i}] action={step['action']} args={json.dumps(step['args'])[:200]}"
            f" observation={step['observation'][:400]}"
        )
    lines.append("")
    lines.append("Next JSON action:")
    return "\n".join(lines)


class ReActAgentV2:
    """Code-executing Reason-Act loop (v1 tools + sandboxed Python)."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: dict[str, Callable[[dict[str, object]], str]] | None = None,
        max_steps: int = 10,
    ) -> None:
        """Initialize the v2 agent.

        Args:
            llm_client: LLM with a ``complete(prompt) -> str`` method.
            tools: Optional v1 tool mapping (data_segment/evidence/
                forecasting). ``code_execute`` is always available.
            max_steps: Hard cap on loop iterations (v2 default 10, since
                analysis takes more steps than v1's 6).

        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm_client = llm_client
        self.tools: dict[str, Callable[[dict[str, object]], str]] = dict(tools or {})
        self.tools["code_execute"] = _run_code_tool
        self.max_steps = max_steps

    def _record(
        self, trace: list[TraceStep], action: str, args: dict[str, object], obs: str
    ) -> None:
        trace.append(TraceStep(action=action, args=args, observation=obs))

    def run(self, task: str, run_date: date) -> ReActResult:
        """Run the v2 ReAct loop for a forecasting task.

        Args:
            task: Natural-language forecasting task.
            run_date: Point-in-time cutoff for all evidence and data pulls.

        Returns:
            ReActResult with the finalized rate-quantile forecast (None if
            the loop exhausted max_steps without a finalize), the full
            trace, and the step count.

        """
        # Import here so module import stays light without the tools package.
        from td_one.agent.react import ReActAgent

        trace: list[TraceStep] = []
        for step in range(self.max_steps):
            prompt = _render_prompt_v2(task, run_date, trace)
            try:
                reply = self.llm_client.complete(prompt)
            except Exception as e:  # noqa: BLE001 - record, don't crash the run
                self._record(trace, "llm_error", {}, f"LLM call failed: {e}")
                continue
            action_obj = ReActAgent._parse_action(reply)
            if action_obj is None:
                self._record(
                    trace,
                    "parse_error",
                    {},
                    f"Could not parse JSON action from reply: {reply[:400]!r}",
                )
                continue
            action = action_obj.get("action")
            if action == "finalize":
                forecast = _parse_forecast(action_obj.get("forecast"))
                if forecast is None:
                    self._record(
                        trace, "parse_error", {}, "finalize had an invalid forecast mapping"
                    )
                    continue
                self._record(trace, "finalize", {}, json.dumps(action_obj["forecast"]))
                return ReActResult(forecast=forecast, trace=trace, steps=step + 1)
            if action in _TOOL_ACTIONS_V2:
                tool = self.tools.get(str(action))
                if tool is None:
                    self._record(trace, str(action), {}, f"tool '{action}' is not wired")
                    continue
                args = action_obj.get("args")
                args_dict = args if isinstance(args, dict) else {}
                try:
                    observation = tool(args_dict)
                except Exception as e:  # noqa: BLE001 - record, don't crash
                    observation = f"tool '{action}' raised: {e}"
                self._record(trace, str(action), args_dict, observation)
                continue
            self._record(
                trace,
                "unknown_action",
                {},
                f"unknown action {action!r} in reply: {reply[:400]!r}",
            )
        return ReActResult(forecast=None, trace=trace, steps=self.max_steps)


__all__ = ["ReActAgentV2", "MockLLMClient", "_run_code_tool"]
