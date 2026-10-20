"""LLM as at-risk group classifier (ablation for exp66d).

The deterministic risk bins (training-tertile) were inverted on test
(exp66c honest negative): training-period group attrition does not
transfer. This module tests whether an LLM — given the same aggregate
stats PLUS PIT-safe news context the bins ignore — classifies group
risk better. The LLM outputs structure (a risk level), never numbers;
all numbers come from computed stats.

Honest framing: if the underlying signal does not transfer, the LLM
cannot fix that. The question is whether news/context improves
calibration, not whether the LLM is magic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from td_one.agent.risk_reporter import check_citations

RISK_LEVELS = ("HIGH", "MEDIUM", "LOW")

_RISK_LINE = re.compile(r"^RISK:\s*(HIGH|MEDIUM|LOW)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class RiskClassification:
    group_id: str
    level: str
    justification: str
    citations_valid: bool


def parse_risk_level(text: str) -> str | None:
    """Extract the ``RISK: <LEVEL>`` line; None if absent or invalid."""
    m = _RISK_LINE.search(text)
    return m.group(1).upper() if m else None


def build_risk_classification_prompt(
    group_id: str,
    rule: str,
    stats: dict,
    articles: list[dict],
    extra_sections: list[str] | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompt for group risk classification."""
    system = (
        "You classify borrower-cohort attrition risk for a lender's risk "
        "team. Reply with exactly two labeled lines:\n"
        "RISK: <HIGH, MEDIUM, or LOW>\n"
        "JUSTIFICATION: <1-2 sentences ending with [stat] or [url:N] "
        "citations>\n"
        "Base the level on the statistics and news below. Every factual "
        "sentence MUST end with a citation: [stat] for a number from the "
        "statistics, [url:N] for news article N. Never invent numbers."
    )
    lines = [
        f"Cohort {group_id}: {rule}",
        "",
        "Statistics (training period, known before the forecast origin):",
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: {v} [stat]")
    if articles:
        lines += ["", "News (published before the forecast origin):"]
        for i, a in enumerate(articles):
            lines.append(f"[{i}] {a.get('title', '')} ({a.get('pub_date', '')}) [url:{i}]")
    else:
        lines += ["", "News: none retrieved."]
    for section in extra_sections or []:
        lines += ["", section]
    return system, "\n".join(lines)


def classify_risk_llm(
    llm: object,
    group_id: str,
    rule: str,
    stats: dict,
    articles: list[dict],
    max_retries: int = 1,
    extra_sections: list[str] | None = None,
) -> RiskClassification:
    """Classify one group's risk via the LLM.

    Retries when the response has no parseable ``RISK:`` line. Raises
    ``ValueError`` if all attempts fail to parse.
    """
    system, user = build_risk_classification_prompt(
        group_id, rule, stats, articles, extra_sections=extra_sections
    )
    last_text = ""
    for _ in range(max_retries + 1):
        text = llm.complete(user, system=system)  # type: ignore[union-attr]
        last_text = text
        level = parse_risk_level(text)
        if level is not None:
            return RiskClassification(
                group_id=group_id,
                level=level,
                justification=text,
                citations_valid=check_citations(text),
            )
    raise ValueError(
        f"LLM did not return a parseable RISK line for {group_id}; "
        f"last response: {last_text[:200]!r}"
    )
