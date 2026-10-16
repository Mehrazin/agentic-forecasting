"""TD-One forecast agent helpers (Groq-powered).

Currently: evidence-prompt builder + bounded log-odds adjustment estimator.
The full ReAct orchestration loop (tool calls, iteration, trace) is not yet
implemented — see PLAN_SOLO.md build order.

Uses Groq `openai/gpt-oss-120b` via the vault surrogate pattern.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Protocol, TypedDict

logger = logging.getLogger(__name__)

#: Bound on the contextual log-odds adjustment (literature-derived).
MAX_DELTA = 0.75
#: Retry policy for the LLM call (Groq 429s are a known recurring issue).
_LLM_MAX_ATTEMPTS = 3
_LLM_BACKOFF_S = 2.0


class EvidenceCard(TypedDict, total=False):
    """A retrieved evidence card."""

    source_url: str
    published_date: str
    fact: str
    directional_impact: str


class LLMProvider(Protocol):
    """Minimal LLM interface."""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        timeout_s: float = 60.0,
    ) -> str:
        """Complete a prompt."""
        ...


class ForecastAgent:
    """Forecast-agent helpers for mortgage attrition forecasting."""

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

    def _complete_with_retry(self, prompt: str) -> str:
        """Call the LLM with exponential backoff on transient failures.

        Raises:
            The last exception if all attempts fail.

        """
        last_exc: Exception | None = None
        for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
            try:
                return self.llm.complete(prompt, json_mode=False, timeout_s=60.0)
            except Exception as e:  # noqa: BLE001 - retry any transient failure
                last_exc = e
                logger.warning(
                    "LLM attempt %d/%d failed: %s",
                    attempt,
                    _LLM_MAX_ATTEMPTS,
                    e,
                )
                if attempt < _LLM_MAX_ATTEMPTS:
                    time.sleep(_LLM_BACKOFF_S * 2 ** (attempt - 1))
        assert last_exc is not None
        raise last_exc

    def estimate_adjustment(
        self,
        segment_id: str,
        run_date: str,
        segment_profile: str,
        base_rate: float,
        evidence: list[EvidenceCard],
    ) -> float | None:
        """Estimate contextual log-odds adjustment from evidence.

        Args:
            segment_id: Segment identifier.
            run_date: PIT cutoff.
            segment_profile: Segment description.
            base_rate: Numerical backend's base rate.
            evidence: Retrieved evidence cards.

        Returns:
            Delta in [-0.75, +0.75] (bounded), 0.0 when there is no
            evidence, or None when the LLM call fails (distinct from a
            genuine "no adjustment" verdict so ablations stay honest).

        """
        if not evidence:
            return 0.0

        evidence_str = "\n".join(
            f"- {e.get('fact', '')} ({e.get('directional_impact', 'neutral')})" for e in evidence
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
            text = self._complete_with_retry(prompt)
        except Exception as e:  # noqa: BLE001 - any LLM failure -> None (logged)
            # Do NOT conflate outages with "no adjustment needed" — the
            # caller must be able to tell them apart for honest ablations.
            logger.warning("estimate_adjustment LLM call failed: %s", e)
            return None
        # Take the LAST float: models often echo context first
        # ("based on 3 sources, delta 0.25" -> 0.25, not 3).
        matches = re.findall(r"-?\d+\.?\d*", text.strip())
        if not matches:
            logger.warning("estimate_adjustment: no number in LLM response")
            return None
        try:
            delta = float(matches[-1])
        except ValueError:
            logger.warning("estimate_adjustment: unparseable number %r", matches[-1])
            return None
        return float(max(-MAX_DELTA, min(MAX_DELTA, delta)))
