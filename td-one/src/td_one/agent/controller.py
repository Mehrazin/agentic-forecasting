"""TD-One forecast agent controller (Groq-powered ReAct loop).

The agent orchestrates the 3 tools:
1. data_segment_tool: query segment history and profiles
2. evidence_tool: retrieve macro/policy evidence (frozen corpus)
3. forecasting_tool: run numerical backend + apply contextual adjustment

Uses Groq `openai/gpt-oss-120b` via the vault surrogate pattern.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Minimal LLM interface."""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        """Complete a prompt."""
        ...


class ForecastAgent:
    """ReAct controller for mortgage attrition forecasting."""

    def __init__(self, llm: LLMProvider) -> None:
        """Initialize with an LLM provider.

        Args:
            llm: Groq-backed provider (or mock for tests).
        """
        self.llm = llm

    def _build_evidence_prompt(
        self,
        segment_id: str,
        run_date: str,
        segment_profile: str,
    ) -> str:
        """Build evidence retrieval prompt with temporal guard.

        Args:
            segment_id: Segment identifier.
            run_date: Point-in-time cutoff (YYYY-MM-DD).
            segment_profile: Human-readable segment description.

        Returns:
            Prompt string.
        """
        return (
            f"Segment: {segment_id}\n"
            f"Profile: {segment_profile}\n"
            f"Run date: {run_date}\n\n"
            "Retrieve macro/policy evidence RELEVANT to this segment's "
            "switching risk. Only use documents published BEFORE "
            f"{run_date} (as of run date). "
            "Focus on: BoC rate decisions, OSFI policy changes, "
            "competitor rate offers, CPI/inflation trends.\n\n"
            "Output JSON: list of evidence cards with "
            "source_url, published_date, fact, directional_impact "
            "(increase_switching/decrease_switching/neutral)."
        )

    def estimate_adjustment(
        self,
        segment_id: str,
        run_date: str,
        segment_profile: str,
        base_rate: float,
        evidence: list[dict[str, Any]],
    ) -> float:
        """Estimate contextual log-odds adjustment from evidence.

        Args:
            segment_id: Segment identifier.
            run_date: PIT cutoff.
            segment_profile: Segment description.
            base_rate: Numerical backend's base rate.
            evidence: Retrieved evidence cards.

        Returns:
            Delta in [-0.75, +0.75] (bounded).
        """
        if not evidence:
            return 0.0

        evidence_str = "\n".join(
            f"- {e.get('fact', '')} ({e.get('directional_impact', 'neutral')})"
            for e in evidence
        )
        prompt = (
            f"Segment {segment_id} has base switching rate {base_rate:.3f}.\n"
            f"Profile: {segment_profile}\n\n"
            f"Evidence (as of {run_date}):\n{evidence_str}\n\n"
            "Estimate a log-odds adjustment delta in [-0.75, +0.75]. "
            "Positive = higher switching risk. "
            "Output ONLY the number, e.g. 0.25 or -0.10."
        )
        try:
            text = self.llm.complete(prompt, json_mode=False)
            # Extract first float from response
            import re

            m = re.search(r"-?\d+\.?\d*", text.strip())
            if m:
                delta = float(m.group(0))
                # Clip to bounds
                return float(max(-0.75, min(0.75, delta)))
        except Exception:
            pass
        return 0.0
