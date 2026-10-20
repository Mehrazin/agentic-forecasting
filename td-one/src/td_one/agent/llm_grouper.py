"""LLM group proposer: metadata distributions + news -> group rules.

The LLM sees **aggregate statistics only** — never individual borrower
rows. Prompt inputs (all computed on training-period data with
``observed_asof <= origin``):

- Per-attribute histograms (binned counts, not raw values).
- Training attrition rates by attribute level.
- Demographic-tagged news summaries retrieved via semantic RAG
  (``td_one.rag.semantic``), PIT-gated to the origin.

Output is strict JSON: a list of ``{"group_id", "rule", "rationale"}``
dicts where ``rule`` is an executable ``pandas.DataFrame.query()``
expression. A validation loop (``td_one.agent.grouping``) rejects bad
rules; the LLM is re-prompted at most ``max_retries`` times.

PIT rule: the prompt contains zero post-origin information. Group rules
are pure functions of borrower metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from td_one.agent.grouping import (
    GroupDefinition,
    discover_candidate_groups,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
DEFAULT_MAX_GROUPS = 5


@dataclass
class GrouperContext:
    """Everything the LLM prompt is built from (training-period only)."""

    attribute_summaries: dict[str, str]  # attr -> "level: n=..., attr_rate=..."
    news_summaries: list[str] = field(default_factory=list)
    tool_context: list[str] = field(default_factory=list)  # analyst-tool findings
    origin: str = ""
    max_groups: int = DEFAULT_MAX_GROUPS


def _fmt_interval(iv) -> str:
    """Format a pandas Interval with human-friendly rounded bounds.

    Prevents the LLM from copying over-precise bin edges (e.g. 1661.546)
    into rules that match too few rows. Rounds to 2 significant figures.
    """
    import math

    def _round(x: float) -> str:
        if x == 0 or not math.isfinite(x):
            return "0"
        mag = 10 ** math.floor(math.log10(abs(x)))
        # 2 significant figures
        r = round(x / mag, 1) * mag
        # Format without trailing .0
        if r == int(r):
            return str(int(r))
        return f"{r:.2f}".rstrip("0").rstrip(".")

    left = _round(iv.left)
    right = _round(iv.right)
    lbr = "(" if iv.closed in ("neither", "right") else "["
    rbr = "]" if iv.closed in ("neither", "left") else ")"
    # pandas qcut uses right-closed by default: (left, right]
    return f"{lbr}{left}, {right}{rbr}"


def build_attribute_summaries(
    df: pd.DataFrame,
    attrition_col: str,
    categorical: list[str] | None = None,
    numeric_bins: int = 5,
) -> dict[str, str]:
    """Summarize attributes: distribution + attrition rate per level.

    Categorical attributes get per-level rates; numeric attributes are
    binned into ``numeric_bins`` quantiles first. Bin edges are rounded
    to human-friendly values so the LLM doesn't overfit to precise
    thresholds. Never exposes rows.
    """
    out: dict[str, str] = {}
    cat = set(categorical or [])
    for col in df.columns:
        if col == attrition_col or col in ("mortgage_id", "customer_id"):
            continue
        s = df[col]
        if col in cat or s.dtype == object or str(s.dtype) == "category":
            grp = df.groupby(col, observed=True)[attrition_col].agg(["mean", "count"])
            parts = [
                f"{lvl}: rate={r['mean']:.3f} (n={int(r['count'])})" for lvl, r in grp.iterrows()
            ]
            out[col] = "; ".join(parts)
        elif pd.api.types.is_numeric_dtype(s):
            try:
                binned = pd.qcut(s, q=numeric_bins, duplicates="drop")
            except ValueError:
                continue
            grp = df.groupby(binned, observed=True)[attrition_col].agg(["mean", "count"])
            parts = [
                f"{_fmt_interval(iv)}: rate={r['mean']:.3f} (n={int(r['count'])})"
                for iv, r in grp.iterrows()
            ]
            out[col] = "; ".join(parts)
    return out


def build_grouper_prompt(
    ctx: GrouperContext,
    baseline_crps: float | None = None,
    baseline_desc: str = "",
) -> tuple[str, str]:
    """Build (system, user) prompt for group proposal.

    Parameters
    ----------
    ctx:
        Grouper context with attribute summaries, news, origin.
    baseline_crps:
        CRPS of a hand-written baseline grouping (if known). The LLM is
        told to beat it — gives the proposal a concrete target.
    baseline_desc:
        Human description of the baseline grouping.

    """
    system = (
        "You propose borrower cohorts for attrition forecasting. "
        "Output STRICT JSON only: "
        '{"groups": [{"group_id": "...", "rule": "...", "rationale": "..."}]}. '
        "Each rule must be a valid pandas DataFrame.query() expression over "
        "the attribute names shown. Rules may combine attributes with and/or. "
        "Do NOT reference outcome columns. Keep rationales to one sentence, "
        "no numbers. Propose at most "
        f"{ctx.max_groups} non-overlapping groups covering distinct risk profiles."
    )
    lines = [
        "Borrower attribute summaries (training period only).",
        "Format: attribute: level: attrition_rate (n).",
        "",
        "IMPORTANT: the strongest groups capture INTERACTIONS, not single",
        "attributes. For example, consider combinations like:",
        "- rate_type x income tier (variable-rate borrowers with low income",
        "  may behave very differently from variable-rate high-income ones)",
        "- credit utilization trend x province",
        "- digital channel x advisor status (self-serve vs advised)",
        "- life events x household size",
        "Single-attribute splits (e.g. just rate_type) are usually too coarse.",
        "",
        "BUT: each group must cover a substantial share of borrowers.",
        "Prefer 2-condition rules over 3-condition ones — a third condition",
        "often slices the group too thin to be useful. Aim for groups that",
        "each cover at least 10-15% of the borrower base.",
        "",
    ]
    for attr, summary in ctx.attribute_summaries.items():
        lines.append(f"- {attr}: {summary}")
    if baseline_crps is not None:
        lines += [
            "",
            f"A hand-written baseline grouping ({baseline_desc}) achieved "
            f"CRPS = {baseline_crps:.5f} on held-out data. Propose groups "
            "that you expect to BEAT this baseline by finding sharper "
            "risk differentiation.",
        ]
    if ctx.news_summaries:
        lines += ["", "Relevant news (published before the forecast origin):"]
        lines += [f"- {s}" for s in ctx.news_summaries]
    if ctx.tool_context:
        lines += [
            "",
            "Analyst-tool findings (from causal/explainability models fit on",
            "the same training data — use these to ground your groups in",
            "measured attrition drivers, not just raw attribute rates):",
        ]
        lines += [f"- {s}" for s in ctx.tool_context]
    lines += [
        "",
        "Propose borrower groups likely to have distinct attrition behavior.",
    ]
    return system, "\n".join(lines)


def parse_group_proposals(text: str) -> list[dict] | None:
    """Parse the LLM's JSON; None on any failure."""
    try:
        data = json.loads(text)
        groups = data.get("groups", [])
        if not isinstance(groups, list):
            return None
        out = []
        for g in groups:
            if not isinstance(g, dict):
                return None
            if "group_id" not in g or "rule" not in g:
                return None
            out.append(
                {
                    "group_id": str(g["group_id"]),
                    "rule": str(g["rule"]),
                    "rationale": str(g.get("rationale", "")),
                }
            )
        return out
    except (json.JSONDecodeError, AttributeError):
        return None


class LLMGrouper:
    """Propose borrower groups via LLM with validation loop."""

    def __init__(
        self,
        llm: object,  # .complete(prompt, system, json_mode) -> str
        max_retries: int = MAX_RETRIES,
        max_groups: int = DEFAULT_MAX_GROUPS,
    ) -> None:
        self._llm = llm
        self.max_retries = max_retries
        self.max_groups = max_groups

    def propose(
        self,
        df: pd.DataFrame,
        attrition_col: str,
        categorical: list[str] | None = None,
        news_summaries: list[str] | None = None,
        tool_context: list[str] | None = None,
        origin: str = "",
        min_group_n: int = 200,
        baseline_crps: float | None = None,
        baseline_desc: str = "",
    ) -> list[GroupDefinition]:
        """Propose, parse, and validate groups. Returns validated list."""
        summaries = build_attribute_summaries(df, attrition_col, categorical)
        ctx = GrouperContext(
            attribute_summaries=summaries,
            news_summaries=news_summaries or [],
            tool_context=tool_context or [],
            origin=origin,
            max_groups=self.max_groups,
        )
        system, user = build_grouper_prompt(
            ctx, baseline_crps=baseline_crps, baseline_desc=baseline_desc
        )
        for attempt in range(self.max_retries + 1):
            text = self._llm.complete(user, system=system, json_mode=True)  # type: ignore[union-attr]
            proposals = parse_group_proposals(text)
            if proposals is None:
                logger.warning("grouper: unparseable JSON (attempt %d)", attempt)
                continue
            groups = discover_candidate_groups(df, proposals, min_group_n=min_group_n)
            if groups:
                return groups
            logger.warning("grouper: no valid groups (attempt %d)", attempt)
        return []
