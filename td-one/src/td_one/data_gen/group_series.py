"""Customer histories -> monthly per-group attrition series.

Aggregates renewal outcomes into monthly time series per borrower group.
A "group" is defined by a :class:`GroupDefinition`-style rule evaluated
against customer metadata (see ``td_one.agent.grouping``); this module
only needs the resulting assignment mapping.

Aggregation identity (tested): the sum of per-group ``n_switched`` and
``n_at_risk`` across groups equals the portfolio totals for every month.
Groups are a partition — every mortgage belongs to exactly one group.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class GroupSeries:
    """Monthly attrition series for one group."""

    group_id: str
    months: list[date]  # month start dates, sorted
    n_at_risk: list[int]
    n_switched: list[int]

    @property
    def attrition_rate(self) -> list[float]:
        """Monthly switched / at_risk (0.0 when at_risk == 0)."""
        return [
            s / r if r > 0 else 0.0 for s, r in zip(self.n_switched, self.n_at_risk, strict=True)
        ]


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def build_group_series(
    outcomes: pd.DataFrame,
    group_of: dict[str, str],
    min_month: date | None = None,
    max_month: date | None = None,
) -> dict[str, GroupSeries]:
    """Aggregate renewal outcomes into monthly per-group series.

    Parameters
    ----------
    outcomes:
        DataFrame with columns ``mortgage_id``, ``outcome_observed_at``
        (date), ``switched_lender_flag`` (bool). One row per renewal episode.
    group_of:
        Mapping ``mortgage_id -> group_id``. Every mortgage in ``outcomes``
        must appear; mortgages in the mapping but not in outcomes are
        ignored.
    min_month, max_month:
        Optional month-start bounds (inclusive). Defaults to the observed
        range.

    Returns
    -------
    Dict ``group_id -> GroupSeries`` covering the union month range.
    Months with no episodes get ``n_at_risk = 0, n_switched = 0``.

    Notes
    -----
    ``n_at_risk`` counts renewal episodes observed in the month;
    ``n_switched`` counts those with ``switched_lender_flag`` true.

    """
    required = {"mortgage_id", "outcome_observed_at", "switched_lender_flag"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"outcomes missing columns: {sorted(missing)}")

    df = outcomes.copy()
    df["month"] = pd.to_datetime(df["outcome_observed_at"]).dt.date.apply(_month_start)
    df["group_id"] = df["mortgage_id"].map(group_of)
    unmapped = df[df["group_id"].isna()]["mortgage_id"].unique()
    if len(unmapped) > 0:
        raise ValueError(
            f"{len(unmapped)} mortgages have no group assignment "
            f"(e.g. {unmapped[0]!r})"
        )

    months = sorted(df["month"].unique())
    if min_month is not None:
        months = [m for m in months if m >= min_month]
    if max_month is not None:
        months = [m for m in months if m <= max_month]
    if not months:
        raise ValueError("no months in range")

    out: dict[str, GroupSeries] = {}
    for gid, gdf in df.groupby("group_id"):
        per_month = gdf.groupby("month")
        n_risk = [int(per_month.size().get(m, 0)) for m in months]
        n_sw = [
            int(per_month["switched_lender_flag"].sum().get(m, 0)) for m in months
        ]
        out[str(gid)] = GroupSeries(
            group_id=str(gid),
            months=list(months),
            n_at_risk=n_risk,
            n_switched=n_sw,
        )
    return out


def portfolio_totals(series: dict[str, GroupSeries]) -> GroupSeries:
    """Sum per-group series into the portfolio total (aggregation check)."""
    if not series:
        raise ValueError("empty series dict")
    months = next(iter(series.values())).months
    for s in series.values():
        if s.months != months:
            raise ValueError(f"month mismatch in group {s.group_id}")
    n_risk = [0] * len(months)
    n_sw = [0] * len(months)
    for s in series.values():
        for i in range(len(months)):
            n_risk[i] += s.n_at_risk[i]
            n_sw[i] += s.n_switched[i]
    return GroupSeries(
        group_id="__portfolio__", months=list(months), n_at_risk=n_risk, n_switched=n_sw
    )
