"""Tool call tracking for agent optimization.

Every tool invocation is logged with timing, inputs (hashed), outputs (truncated),
and outcome. This enables:
- Identifying slow tools that need optimization
- Finding tools that are never used (candidates for removal)
- Detecting tools that frequently fail (need better error handling)
- Measuring which tools correlate with successful outcomes

The tracker is a singleton; import and use `tracker.record()` from any tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """One recorded tool invocation."""

    tool_name: str
    timestamp: float
    duration_ms: float
    input_hash: str
    input_summary: str  # truncated repr, no PII
    output_truncated: str  # first 500 chars
    success: bool
    error_type: str | None = None


@dataclass
class ToolStats:
    """Aggregate stats for one tool."""

    calls: int = 0
    total_ms: float = 0.0
    failures: int = 0
    # outcome correlation: filled by analyze()

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


class ToolCallTracker:
    """Singleton tracker for all agent tool invocations."""

    _instance: ToolCallTracker | None = None

    def __new__(cls) -> ToolCallTracker:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._calls: list[ToolCall] = []
            cls._instance._stats: dict[str, ToolStats] = {}
        return cls._instance

    def record(
        self,
        tool_name: str,
        duration_ms: float,
        inputs: dict[str, Any],
        output: str,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record one tool call. Inputs are hashed; only a summary is stored."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
        # Truncate input repr to avoid PII/bloat
        input_summary = input_str[:200]

        call = ToolCall(
            tool_name=tool_name,
            timestamp=time.time(),
            duration_ms=duration_ms,
            input_hash=input_hash,
            input_summary=input_summary,
            output_truncated=output[:500],
            success=success,
            error_type=error_type,
        )
        self._calls.append(call)

        stats = self._stats.setdefault(tool_name, ToolStats())
        stats.calls += 1
        stats.total_ms += duration_ms
        if not success:
            stats.failures += 1

        logger.debug(
            "tool_call tool=%s duration_ms=%.1f success=%s",
            tool_name,
            duration_ms,
            success,
        )

    def get_stats(self) -> dict[str, ToolStats]:
        """Return aggregate stats per tool."""
        return dict(self._stats)

    def get_calls(self, tool_name: str | None = None) -> list[ToolCall]:
        """Return recorded calls, optionally filtered by tool."""
        if tool_name is None:
            return list(self._calls)
        return [c for c in self._calls if c.tool_name == tool_name]

    def summary_report(self) -> str:
        """Human-readable summary for optimization review."""
        lines = ["# Tool Call Tracking Report", ""]
        for name, stats in sorted(self._stats.items()):
            lines.append(
                f"- **{name}**: {stats.calls} calls, "
                f"avg {stats.avg_ms:.1f}ms, "
                f"failure rate {stats.failure_rate:.1%}"
            )
        lines.append("")
        lines.append(f"Total calls tracked: {len(self._calls)}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked data (for testing)."""
        self._calls.clear()
        self._stats.clear()


# Singleton instance
tracker = ToolCallTracker()
