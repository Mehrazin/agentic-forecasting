"""Segment discovery for mortgage renewal backtesting.

Partitions renewal events into interpretable segments and builds monthly
count panels per segment for the backtest runner.

A renewal is one mortgage-month at maturity: each
:class:`~td_one.schemas.RenewalOutcomeRecord` is linked to the history
record of the same mortgage in its maturity month, which supplies the
segment covariates.

Segment dimensions (flat key namespace ``{dim}_{value}``):

- ``payment_shock_q{1..5}``: quintile of the payment-to-income ratio at
  maturity. This mirrors the simulator's payment-shock DGP term
  (``max(0, payment/income - 0.32)``); the quintile is taken over the raw
  ratio so the cut points are data-driven.
- ``rate_type_{fixed,variable}``: rate type at maturity.
- ``region_{province}``: province code. The schema carries no postal code,
  so province is the geographic segment available.
- ``term_{short,medium,long}``: ``term_months <= 36`` / ``<= 60`` / else.

Each panel is a DataFrame with columns ``[month, n_renewals, n_switches]``;
``month`` is a month-end :class:`datetime.date`, rows sorted ascending.
Outcomes with an unresolved (``None``) ``switched_lender_flag`` are
excluded — they are renewals but unscorable.

Point-in-time masking: :func:`discover_segments` accepts ``observed_asof``.
For event months on/before ``observed_asof`` only outcomes with
``outcome_observed_at <= observed_asof`` are counted — anything observed
later was not yet known (leakage guard). Event months *after*
``observed_asof`` keep actuals: they are scoring targets for backtest
origins, never forecaster inputs (the backtest runner's cutoff check
enforces that the forecast function only sees ``month <= origin``).

All aggregation is pandas groupby; the only per-record pass is the
Pydantic ``model_dump`` extraction of the fields actually needed (Rust
speed, bounded memory at 660k rows).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from td_one.schemas import MortgageHistoryRecord, RenewalOutcomeRecord

#: Panel schema produced per segment.
PANEL_COLUMNS = ["month", "n_renewals", "n_switches"]

#: History fields needed to link renewals and compute segment covariates.
_HISTORY_FIELDS = {
    "mortgage_id",
    "snapshot_date",
    "maturity_date",
    "province",
    "rate_type",
    "term_months",
    "mortgage_payment_monthly_cad",
    "household_income_cad",
}

#: Outcome fields needed for panels.
_OUTCOME_FIELDS = {
    "mortgage_id",
    "scheduled_maturity_date",
    "switched_lender_flag",
    "outcome_observed_at",
}


def _month_end(dates: pd.Series) -> pd.Series:
    """Map dates to their month-end ``datetime.date`` (vectorized)."""
    return (pd.to_datetime(dates) + pd.offsets.MonthEnd(0)).dt.date


def _payment_shock_quintile(payment_to_income: pd.Series) -> pd.Series:
    """Quintile labels ``q1..qk`` for a payment-to-income ratio series."""
    codes = pd.qcut(payment_to_income, q=5, labels=False, duplicates="drop")
    return pd.Series(codes, index=payment_to_income.index).map(lambda c: f"q{int(c) + 1}")


def build_segment_panels(
    renewals: pd.DataFrame,
    dim_col: str,
    key_prefix: str,
) -> dict[str, pd.DataFrame]:
    """Group renewal events into monthly count panels per segment value.

    Args:
        renewals: One row per renewal event with columns ``month``
            (month-end date), ``switched_lender_flag`` (bool), and the
            segment dimension column ``dim_col``.
        dim_col: Column holding the segment value.
        key_prefix: Prefix for the returned keys (``f"{key_prefix}_{value}"``).

    Returns:
        Mapping of segment key to panel DataFrame with columns
        ``[month, n_renewals, n_switches]``, sorted by month.

    """
    if renewals.empty:
        raise ValueError("renewals is empty")
    if dim_col not in renewals.columns:
        raise ValueError(f"dim_col {dim_col!r} not in renewals columns")
    panels: dict[str, pd.DataFrame] = {}
    flags = renewals["switched_lender_flag"].astype(bool)
    for value, idx in renewals.groupby(dim_col, observed=True).groups.items():
        grp_flags = flags.loc[idx]
        months = renewals.loc[idx, "month"]
        agg = (
            pd.DataFrame({"month": months, "flag": grp_flags})
            .groupby("month")
            .agg(
                n_renewals=("flag", "size"),
                n_switches=("flag", "sum"),
            )
        )
        panel = agg.reset_index().sort_values("month")[PANEL_COLUMNS]
        panel["n_renewals"] = panel["n_renewals"].astype("int64")
        panel["n_switches"] = panel["n_switches"].astype("int64")
        panels[f"{key_prefix}_{value}"] = panel.reset_index(drop=True)
    return panels


def discover_segments(
    histories: list[MortgageHistoryRecord],
    outcomes: list[RenewalOutcomeRecord],
    observed_asof: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Discover segments and build monthly renewal-count panels.

    Args:
        histories: Validated mortgage history records (customers x months).
        outcomes: Validated renewal outcome records.
        observed_asof: Optional point-in-time bound for the *training* side
            of the panels (a backtest forecast origin). Outcomes whose
            event month is on/before ``observed_asof`` are kept only when
            ``outcome_observed_at <= observed_asof`` — later-observed
            outcomes were not knowable yet (leakage guard) and are treated
            as unresolved for that origin. Outcomes with an event month
            after ``observed_asof`` keep actuals: they are scoring targets,
            never forecaster inputs. ``None`` (default) keeps all resolved
            outcomes unmasked.

    Returns:
        Mapping of segment key (``{dim}_{value}``) to monthly panel
        DataFrame ``[month, n_renewals, n_switches]``.

    Raises:
        ValueError: On empty inputs, on outcomes with no history row in
            their maturity month, on unresolvable segment covariates, or
            when masking leaves no outcomes.

    """
    if not histories:
        raise ValueError("histories is empty")
    if not outcomes:
        raise ValueError("outcomes is empty")

    hist = pd.DataFrame([r.model_dump(include=_HISTORY_FIELDS) for r in histories])
    out = pd.DataFrame([r.model_dump(include=_OUTCOME_FIELDS) for r in outcomes])

    # Keep only resolved outcomes (unresolved ones are unscorable).
    out = out[out["switched_lender_flag"].notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("no resolved outcomes (all switched_lender_flag are None)")

    if observed_asof is not None:
        # PIT mask for the training side: a past outcome counts only if it
        # had been observed by the origin. Future event months keep actuals
        # (scoring targets; the runner never feeds them to forecast_fn).
        asof = pd.Timestamp(observed_asof)
        event_me = pd.to_datetime(out["scheduled_maturity_date"]) + pd.offsets.MonthEnd(0)
        past = event_me <= asof
        obs = pd.to_datetime(out["outcome_observed_at"])
        known = obs.notna() & (obs <= asof)
        out = out[~past | known].reset_index(drop=True)
        if out.empty:
            raise ValueError("no outcomes remain after observed_asof masking")

    # Link each outcome to the history row of the same mortgage in its
    # maturity month: one row per renewal event.
    out["renew_ym"] = pd.to_datetime(out["scheduled_maturity_date"]).dt.to_period("M").astype(str)
    hist["snap_ym"] = pd.to_datetime(hist["snapshot_date"]).dt.to_period("M").astype(str)
    renew = out.merge(
        hist,
        left_on=["mortgage_id", "renew_ym"],
        right_on=["mortgage_id", "snap_ym"],
        how="left",
    )
    missing = renew["snapshot_date"].isna().sum()
    if missing:
        raise ValueError(f"{missing} outcomes have no history row in their maturity month")

    renew["month"] = _month_end(renew["scheduled_maturity_date"])

    # Segment covariates from the maturity-month history row.
    income_mo = (renew["household_income_cad"] / 12).clip(lower=1.0)
    payment_to_income = renew["mortgage_payment_monthly_cad"] / income_mo
    # PIT-safe quintiles: fit boundaries on data at or before observed_asof,
    # apply via pd.cut to all months (no transductive leakage from future).
    if observed_asof is not None:
        asof_ts = pd.Timestamp(observed_asof)
        train_mask = pd.to_datetime(renew["month"]) <= asof_ts
        train_pti = payment_to_income[train_mask]
    else:
        train_pti = payment_to_income
    if len(train_pti) >= 5:
        _, bins = pd.qcut(train_pti, q=5, labels=False, duplicates="drop", retbins=True)
        # Ensure bins cover the full range for pd.cut.
        bins[0], bins[-1] = -np.inf, np.inf
        codes = pd.cut(payment_to_income, bins=bins, labels=False)
        renew["shock_q"] = pd.Series(codes, index=payment_to_income.index).map(
            lambda c: f"q{int(c) + 1}" if pd.notna(c) else "q1"
        )
    else:
        renew["shock_q"] = _payment_shock_quintile(payment_to_income)
    renew["term_bucket"] = pd.cut(
        renew["term_months"],
        bins=[0, 36, 60, 10**9],
        labels=["short", "medium", "long"],
    ).astype(str)

    segments: dict[str, pd.DataFrame] = {}
    segments.update(build_segment_panels(renew, "shock_q", "payment_shock"))
    segments.update(build_segment_panels(renew, "rate_type", "rate_type"))
    segments.update(build_segment_panels(renew, "province", "region"))
    segments.update(build_segment_panels(renew, "term_bucket", "term"))
    return segments
