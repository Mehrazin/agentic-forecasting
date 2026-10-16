"""Freddie Mac Single-Family Loan-Level Dataset loader.

Reads the pipe-delimited origination and monthly performance files,
validates schemas, and builds cohort panels for attrition modeling.

PIT SAFETY: performance files contain monthly observations. Any cohort
panel built for a forecast origin must only include performance rows
with reporting period < origin. The loader does NOT enforce this —
callers must filter. (See docs/DATA_AUDIT.md P0 findings.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Freddie Mac file layouts (pipe-delimited, no header).
# See https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
ORIG_COLUMNS = [
    "credit_score",  # FICO
    "first_payment_date",  # YYYYMM
    "first_time_homebuyer",  # Y/N
    "maturity_date",  # YYYYMM
    "msa",  # Metropolitan Statistical Area
    "mi_pct",  # Mortgage insurance %
    "num_units",
    "occupancy",  # P/S/I
    "oltv",  # Original combined LTV
    "dti",  # Original debt-to-income
    "orig_upb",
    "olTv_simple",  # Original LTV (duplicate name guard below)
    "orig_rate",
    "channel",  # R/B/C/T
    "ppm_flag",  # Prepayment penalty
    "product_type",  # FRM/ARM
    "state",
    "property_type",  # SF/CO/PU/MH/CP/LH
    "postal_code",
    "loan_seq",  # Loan sequence number (key)
    "loan_purpose",  # P/C/R
    "orig_term",
    "num_borrowers",
    "seller",
    "servicer",
    "super_conforming",
    "pre_harp_seq",
    "program_indicator",
    "harp_indicator",
    "valuation_method",
    "io_indicator",
]

# Fix the duplicate-name guard: column 12 is "oltv" (original LTV).
ORIG_COLUMNS[11] = "oltv_orig"

PERF_COLUMNS = [
    "loan_seq",
    "period",  # YYYYMM reporting period
    "actual_upb",
    "dq_status",  # XX=current, 00=30 days, etc.
    "loan_age",
    "rem_months",
    "defect_date",
    "mod_flag",
    "zb_code",  # Zero balance: 01=prepaid, 03/06/09=default
    "zb_date",  # YYYYMM
    "curr_rate",
    "deferred_upb",
    "dppi_date",  # Due date of last paid installment
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maint_costs",
    "tax_ins",
    "misc_expenses",
    "actual_loss",
    "mod_cost",
    "step_mod_flag",
    "deferred_plan",
    "est_ltv",
    "zb_upb",
    "delinq_interest",
    "disaster_flag",
    "assist_code",
    "curr_mod_cost",
    "interest_upb",
    # File has 3 additional trailing columns beyond the documented 32.
    # (Layout version drift; values observed: amount|code|OTHER|amount.)
    "extra_1",
    "extra_2",
    "extra_3",
]

# Zero-balance codes: 01 = voluntary payoff (prepayment event).
# 03/06/09 = credit events (default/foreclosure) — NOT prepayment.
PREPAY_ZB_CODES = {"01"}
DEFAULT_ZB_CODES = {"03", "06", "09"}


@dataclass(frozen=True)
class FreddieData:
    """Loaded origination + performance data."""

    orig: pd.DataFrame
    perf: pd.DataFrame


def load_orig(path: Path | str) -> pd.DataFrame:
    """Load an origination file with schema validation."""
    df = pd.read_csv(path, sep="|", header=None, names=ORIG_COLUMNS, dtype=str, na_filter=False)
    # Validate: loan_seq must be unique and non-empty.
    if df["loan_seq"].isna().any() or (df["loan_seq"] == "").any():
        raise ValueError("origination file has blank loan_seq")
    if df["loan_seq"].duplicated().any():
        raise ValueError("origination file has duplicate loan_seq")
    # Numeric conversions.
    for col in ("credit_score", "oltv_orig", "dti", "orig_upb", "orig_rate", "orig_term"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_perf(path: Path | str) -> pd.DataFrame:
    """Load a performance file with schema validation."""
    df = pd.read_csv(path, sep="|", header=None, names=PERF_COLUMNS, dtype=str, na_filter=False)
    if df["loan_seq"].isna().any() or (df["loan_seq"] == "").any():
        raise ValueError("performance file has blank loan_seq")
    for col in ("actual_upb", "loan_age", "curr_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["period"] = pd.to_datetime(df["period"], format="%Y%m", errors="coerce")
    return df


def load_vintage(data_dir: Path | str, vintage: str) -> FreddieData:
    """Load orig + perf for a vintage like '2026Q1'.

    Expects ``orig_<vintage>.txt`` and ``perf_<vintage>.txt`` in data_dir.
    """
    d = Path(data_dir)
    orig = load_orig(d / f"orig_{vintage}.txt")
    perf = load_perf(d / f"perf_{vintage}.txt")
    # Referential integrity: every perf loan must exist in orig.
    orphan = set(perf["loan_seq"].unique()) - set(orig["loan_seq"].unique())
    if orphan:
        raise ValueError(f"{len(orphan)} perf loans missing from origination")
    return FreddieData(orig=orig, perf=perf)


def monthly_prepay_rate(perf: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly SMM (single-month mortality) from performance data.

    SMM = voluntary payoffs / loans at risk at start of month.
    Voluntary payoff = zb_code == '01' (excludes defaults 03/06/09).

    Returns DataFrame with period, n_at_risk, n_prepay, smm.
    """
    df = perf.copy()
    # A loan is "at risk" if it has a performance row in that period
    # and hasn't already terminated in a prior period.
    df = df.sort_values(["loan_seq", "period"])
    # First termination per loan (any zb_code).
    term = df[df["zb_code"].isin(PREPAY_ZB_CODES | DEFAULT_ZB_CODES)]
    first_term = term.groupby("loan_seq")["period"].min().rename("term_period")
    df = df.join(first_term, on="loan_seq")
    # At risk: period <= term_period (or no termination yet).
    at_risk = df[df["term_period"].isna() | (df["period"] <= df["term_period"])]
    # Prepay event: zb_code == '01' in this period.
    at_risk["is_prepay"] = at_risk["zb_code"].isin(PREPAY_ZB_CODES)

    grouped = at_risk.groupby(at_risk["period"].dt.to_period("M"))
    out = pd.DataFrame(
        {
            "n_at_risk": grouped["loan_seq"].nunique(),
            "n_prepay": grouped["is_prepay"].sum(),
        }
    )
    out["smm"] = out["n_prepay"] / out["n_at_risk"]
    out = out.reset_index()
    out["period"] = out["period"].dt.to_timestamp()
    return out
