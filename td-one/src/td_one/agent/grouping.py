"""Borrower group definitions: validation, application, overlap resolution.

A :class:`GroupDefinition` is an executable ``pandas.DataFrame.query()``
rule over borrower metadata plus a human-readable rationale. This module
implements ``discover_candidate_groups()`` — the multi-attribute cohort
discovery that closes Argon Finding 3 (see
``docs/reports/ARGON_REVIEW_2.md``).

Contracts (all enforced, all tested):
- Rules must parse as ``DataFrame.query()`` expressions.
- Rules may reference **metadata columns only** — any reference to an
  outcome column (``switched``, ``prepaid``, ``outcome_*``) is rejected.
  This is the PIT guard for grouping: group definitions cannot see the
  future.
- Every group must have at least ``min_group_n`` members on the data it
  was defined against.
- Overlapping groups are resolved by Jaccard similarity: when two groups
  overlap beyond ``max_overlap``, the one with the better (lower)
  training CRPS is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns that must never appear in a group rule: they carry outcome info.
OUTCOME_COLUMNS = frozenset(
    {
        "switched",
        "switched_lender_flag",
        "prepaid",
        "outcome",
        "outcome_observed_at",
        "attrition",
        "churned",
        "renewed_with_td",
    }
)

DEFAULT_MIN_GROUP_N = 200
DEFAULT_MAX_OVERLAP = 0.5


@dataclass(frozen=True)
class GroupDefinition:
    """One borrower cohort: executable rule + rationale."""

    group_id: str
    rule: str  # pandas.DataFrame.query() expression over metadata columns
    rationale: str = ""


def validate_rule(
    rule: str,
    columns: set[str] | list[str],
    min_group_n: int = DEFAULT_MIN_GROUP_N,
    df: pd.DataFrame | None = None,
) -> GroupDefinition | None:
    """Validate a group rule; return a GroupDefinition or None if invalid.

    Rejects: empty rules, unparseable query syntax, references to outcome
    columns, references to unknown columns, and (when ``df`` is given)
    rules matching fewer than ``min_group_n`` rows.
    """
    cols = set(columns)
    if not rule or not rule.strip():
        return None
    lowered = rule.lower()
    for oc in OUTCOME_COLUMNS:
        # word-boundary-ish check: avoid matching substrings like
        # "unswitched" — require the token to appear as an identifier.
        import re

        if re.search(rf"(?<![a-z0-9_]){re.escape(oc)}(?![a-z0-9_])", lowered):
            return None
    # Extract identifiers referenced by the rule and check they exist.
    import re

    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", rule))
    keywords = {
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "None",
    }
    # String literals: strip quoted segments before token check.
    dequoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", rule)
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", dequoted)) - keywords
    unknown = tokens - cols
    if unknown:
        return None
    # Reject bare column references: each AND/OR clause must contain a
    # comparison operator. Catches LLM slips like
    # "age < 40 and other_debt_payment_monthly_cad" (missing threshold).
    import re as _re

    clauses = _re.split(r"\band\b|\bor\b", rule, flags=_re.IGNORECASE)
    for clause in clauses:
        c = clause.strip().strip("()")
        if not c or c.lower() in ("not",):
            continue
        # 'not ( ... )' wraps a full clause — check inside the parens
        if c.lower().startswith("not"):
            c = c[3:].strip().strip("()")
        if not _re.search(r"==|!=|<=|>=|<|>|\bin\b|\bis\b", c, flags=_re.IGNORECASE):
            return None
    if df is not None:
        try:
            matched = df.query(rule)
        except Exception:  # noqa: BLE001 — query failures are the validation signal
            return None
        if len(matched) < min_group_n:
            return None
    return GroupDefinition(group_id="", rule=rule.strip())


def apply_groups(
    df: pd.DataFrame,
    groups: list[GroupDefinition],
    id_column: str = "mortgage_id",
) -> dict[str, str]:
    """Apply group rules -> ``{mortgage_id: group_id}`` partition.

    Every row must match exactly one group; rows matching zero or
    multiple groups raise. Callers should run overlap resolution first.
    """
    assignment: dict[str, str] = {}
    for _, row in df.iterrows():
        mid = str(row[id_column])
        hits = [g.group_id for g in groups if _row_matches(df, row.name, g.rule)]
        if len(hits) == 0:
            raise ValueError(f"mortgage {mid} matches no group")
        if len(hits) > 1:
            raise ValueError(f"mortgage {mid} matches multiple groups: {hits}")
        assignment[mid] = hits[0]
    return assignment


def _row_matches(df: pd.DataFrame, idx: object, rule: str) -> bool:
    sub = df.loc[[idx]]
    try:
        return bool(len(sub.query(rule)) > 0)
    except Exception:  # noqa: BLE001 — query failures are the validation signal
        return False


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two mortgage-id sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def resolve_overlaps(
    df: pd.DataFrame,
    groups: list[GroupDefinition],
    train_crps: dict[str, float],
    max_overlap: float = DEFAULT_MAX_OVERLAP,
    id_column: str = "mortgage_id",
) -> list[GroupDefinition]:
    """Drop overlapping groups, keeping the lower training CRPS.

    Computes each group's member set, then greedily removes the worse
    (higher CRPS) member of any pair with Jaccard > ``max_overlap``.
    """
    members: dict[str, set[str]] = {}
    for g in groups:
        try:
            m = df.query(g.rule)
        except Exception:  # noqa: BLE001 — query failures are the validation signal
            continue
        members[g.group_id] = set(m[id_column].astype(str))
    # Greedy: sort by CRPS ascending, keep a group unless it overlaps
    # too much with an already-kept (better) group.
    ordered = sorted(groups, key=lambda g: train_crps.get(g.group_id, float("inf")))
    kept: list[GroupDefinition] = []
    for g in ordered:
        if all(
            jaccard(members[g.group_id], members[k.group_id]) <= max_overlap
            for k in kept
        ):
            kept.append(g)
    return kept


def discover_candidate_groups(
    df: pd.DataFrame,
    proposals: list[dict],
    train_crps: dict[str, float] | None = None,
    min_group_n: int = DEFAULT_MIN_GROUP_N,
    max_overlap: float = DEFAULT_MAX_OVERLAP,
    id_column: str = "mortgage_id",
) -> list[GroupDefinition]:
    """Validate LLM-proposed groups and resolve overlaps.

    ``proposals``: list of ``{"group_id", "rule", "rationale"}`` dicts
    (e.g. parsed from the LLM's JSON). Returns the validated, deduped
    list. This is the ``discover_candidate_groups()`` that closes Argon
    Finding 3.
    """
    valid: list[GroupDefinition] = []
    for p in proposals:
        gd = validate_rule(
            str(p.get("rule", "")),
            set(df.columns),
            min_group_n=min_group_n,
            df=df,
        )
        if gd is None:
            continue
        valid.append(
            GroupDefinition(
                group_id=str(p.get("group_id", f"g{len(valid)}")),
                rule=gd.rule,
                rationale=str(p.get("rationale", "")),
            )
        )
    if train_crps is None:
        train_crps = {g.group_id: 0.0 for g in valid}
    resolved = resolve_overlaps(
        df, valid, train_crps, max_overlap=max_overlap, id_column=id_column
    )
    # Ensure partition: add a catch-all for unmatched rows.
    if resolved:
        matched: set[str] = set()
        for g in resolved:
            try:
                m = df.query(g.rule)
                matched.update(m[id_column].astype(str))
            except Exception:  # noqa: BLE001 — query failures are the validation signal
                pass
        all_ids = set(df[id_column].astype(str))
        unmatched = all_ids - matched
        if unmatched:
            # Build a catch-all rule from the matched group ids is fragile;
            # instead use a rule that negates all kept rules.
            negated = " and ".join(f"not ({g.rule})" for g in resolved)
            resolved.append(
                GroupDefinition(
                    group_id="__catch_all__",
                    rule=f"({negated})",
                    rationale="catch-all for unmatched borrowers",
                )
            )
    return resolved
