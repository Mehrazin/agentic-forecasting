"""Group risk reporting: deterministic bins + grounded LLM explanations.

Risk levels are **computed, never LLM-generated**: each group's predicted
mean attrition is binned into LOW / MEDIUM / HIGH by training-period
tertiles, then downgraded one level when the conformal interval is wide
(uncertainty-aware binning).

Explanations are LLM-generated but **citation-enforced**: every factual
sentence must cite either a ``[stat]`` (a computed number passed in the
prompt) or a ``[url]`` (a retrieved news article). A post-hoc checker
rejects explanations with uncited factual claims. This is the
anti-hallucination gate — the most novel piece of Track 1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

# Sentences matching these patterns are structural, not factual, and do
# not require citations.
_NON_FACTUAL = re.compile(
    r"^\s*(in summary|overall|note that|to summarize)[,:]?\s", re.IGNORECASE
)

_CITATION = re.compile(r"\[(stat|url)[^\]]*\]")


@dataclass(frozen=True)
class RiskReport:
    group_id: str
    risk_level: str
    predicted_mean: float
    interval_low: float
    interval_high: float
    explanation: str
    citations_valid: bool


def compute_risk_bins(
    predicted_means: dict[str, float],
    train_means: list[float],
) -> dict[str, str]:
    """Bin predicted means into LOW/MEDIUM/HIGH by training tertiles."""
    if not train_means:
        raise ValueError("train_means must be non-empty")
    lo, hi = np.quantile(train_means, [1 / 3, 2 / 3])
    out = {}
    for gid, m in predicted_means.items():
        if m < lo:
            out[gid] = "LOW"
        elif m < hi:
            out[gid] = "MEDIUM"
        else:
            out[gid] = "HIGH"
    return out


def widen_for_uncertainty(
    bins: dict[str, str],
    intervals: dict[str, tuple[float, float]],
    width_threshold: float,
) -> dict[str, str]:
    """Downgrade one risk level when the interval is wide.

    A HIGH-mean but wide-interval group drops to MEDIUM, etc. LOW stays
    LOW (no lower level).
    """
    order = list(RISK_LEVELS)
    out = {}
    for gid, level in bins.items():
        lo, hi = intervals[gid]
        if hi - lo > width_threshold:
            idx = max(0, order.index(level) - 1)
            out[gid] = order[idx]
        else:
            out[gid] = level
    return out


def check_citations(explanation: str) -> bool:
    """Every factual sentence must carry a [stat] or [url] citation.

    Splits on sentence boundaries; skips structural openers. Returns
    True iff all factual sentences are cited.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation) if s.strip()]
    for s in sentences:
        if _NON_FACTUAL.match(s):
            continue
        if not _CITATION.search(s):
            return False
    return True


def build_explanation_prompt(
    group_id: str,
    rule: str,
    stats: dict[str, float | int | str],
    articles: list[dict],
) -> tuple[str, str]:
    """Build (system, user) prompt for the risk explanation.

    ``stats`` values become citable ``[stat]`` facts; each article becomes
    a citable ``[url]``. The system prompt states the citation contract.
    """
    system = (
        "You write a short risk explanation for a borrower cohort. "
        "Every factual sentence MUST end with a citation: [stat] for a "
        "number from the statistics below, [url:N] for news article N. "
        "Sentences without a factual claim need no citation. "
        "Never invent numbers or events not listed here."
    )
    lines = [
        f"Cohort {group_id}: {rule}",
        "",
        "Statistics:",
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: {v} [stat]")
    if articles:
        lines += ["", "News (published before the forecast origin):"]
        for i, a in enumerate(articles):
            lines.append(f"[{i}] {a.get('title', '')} ({a.get('url', '')}) [url:{i}]")
    lines += ["", "Write 3-5 sentences explaining this cohort's attrition risk."]
    return system, "\n".join(lines)


class RiskReporter:
    """Deterministic bins + citation-enforced LLM explanations."""

    def __init__(
        self,
        llm: object,  # .complete(prompt, system) -> str
        width_threshold: float = 0.05,
    ) -> None:
        self._llm = llm
        self.width_threshold = width_threshold

    def report(
        self,
        group_id: str,
        rule: str,
        predicted_mean: float,
        interval: tuple[float, float],
        risk_level: str,
        articles: list[dict] | None = None,
    ) -> RiskReport:
        """Generate a risk report with citation-checked explanation."""
        stats: dict[str, float | int | str] = {
            "predicted_mean_attrition": round(predicted_mean, 4),
            "interval_low": round(interval[0], 4),
            "interval_high": round(interval[1], 4),
            "risk_level": risk_level,
        }
        system, user = build_explanation_prompt(
            group_id, rule, stats, articles or []
        )
        text = self._llm.complete(user, system=system)  # type: ignore[union-attr]
        valid = check_citations(text)
        if not valid:
            logger.warning("risk_reporter: uncited claims in %s", group_id)
        return RiskReport(
            group_id=group_id,
            risk_level=risk_level,
            predicted_mean=predicted_mean,
            interval_low=interval[0],
            interval_high=interval[1],
            explanation=text,
            citations_valid=valid,
        )
