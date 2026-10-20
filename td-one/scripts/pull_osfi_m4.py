"""Pull OSFI M4 balance-sheet data (StatCan table 10100006) into local parquet.

Source: https://www150.statcan.gc.ca/n1/tbl/csv/10100006-eng.zip
(Open Government Licence - Canada). Monthly 2013-01 -> present.

Produces:
  data/osfi/m4_balances.parquet  (month, segment, outstanding_m, funds_advanced_m, rate_pct)
  data/osfi/m4_runoff.parquet    (month, segment, runoff_m) — implied portfolio runoff

Implied runoff identity (in millions of CAD):
    runoff[t] = outstanding[t-1] + funds_advanced[t] - outstanding[t]
i.e. the balance change not explained by new lending = prepayments +
maturities not renewed + defaults. This is our REAL Canadian
portfolio-level attrition target.

PIT discipline: parquet holds full history; filter ``month <= origin``
at use time. Publication lag is ~2 months (REF_DATE max is ~2 months
behind today); treat accordingly in experiments.

Usage:
    .venv/bin/python scripts/pull_osfi_m4.py [--refresh]

Never commit data/ (gitignored). Rebuild via this script.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STATCAN_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/10100006-eng.zip"
OUT_DIR = ROOT / "data" / "osfi"

#: Map raw StatCan component -> (segment, measure). Outstanding balances and
#: funds advanced are separate measures; runoff needs both for the SAME
#: segment. Segments without both measures are dropped.
OUTSTANDING_MAP: dict[str, str] = {
    "Total, outstanding balances, residential mortgages, insured": "insured",
    "Total, outstanding balances, residential mortgages, uninsured": "uninsured",
    "Variable rate, outstanding balances, residential mortgages, insured": "variable_insured",
    "Variable rate, outstanding balances, residential mortgages, uninsured": "variable_uninsured",
}
ADVANCES_MAP: dict[str, str] = {
    "Total, funds advanced, residential mortgages, insured": "insured",
    "Total, funds advanced, residential mortgages, uninsured": "uninsured",
    "Variable rate, funds advanced, residential mortgages, insured": "variable_insured",
    "Variable rate, funds advanced, residential mortgages, uninsured": "variable_uninsured",
}
#: StatCan publishes rates only for funds-advanced components (the rate on
#: NEW lending) — there are no rate rows for outstanding balances.
RATE_COMPONENTS: dict[str, str] = ADVANCES_MAP


def _download(refresh: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "10100006-eng.zip"
    csv_path = OUT_DIR / "10100006.csv"
    if refresh or not csv_path.exists():
        import urllib.request

        print(f"downloading {STATCAN_URL} ...")
        req = urllib.request.Request(
            STATCAN_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as r, open(
            zip_path, "wb"
        ) as f:
            f.write(r.read())
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(OUT_DIR)
        print(f"extracted {csv_path}")
    return csv_path


def pull(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download + parse; return (balances, runoff) dataframes."""
    csv_path = _download(refresh)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["month"] = pd.to_datetime(df["REF_DATE"], format="%Y-%m")

    dollars = df[df["Unit of measure"] == "Dollars"]
    # Dollars are in millions (SCALAR_FACTOR); rates in percent.
    bal = dollars[["month", "Components", "VALUE"]].copy()
    bal["segment"] = bal["Components"].map(OUTSTANDING_MAP)
    bal = (
        bal.dropna(subset=["segment"])
        .pivot_table(
            index="month", columns="segment", values="VALUE", aggfunc="first"
        )
        .sort_index()
    )
    adv = dollars[["month", "Components", "VALUE"]].copy()
    adv["segment"] = adv["Components"].map(ADVANCES_MAP)
    adv = (
        adv.dropna(subset=["segment"])
        .pivot_table(
            index="month", columns="segment", values="VALUE", aggfunc="first"
        )
        .sort_index()
    )
    rates = df[df["Unit of measure"] == "Interest rate"][["month", "Components", "VALUE"]].copy()
    rates["segment"] = rates["Components"].map(RATE_COMPONENTS)
    rates = (
        rates.dropna(subset=["segment"])
        .pivot_table(
            index="month", columns="segment", values="VALUE", aggfunc="first"
        )
        .sort_index()
    )

    bal_long = bal.reset_index().melt(
        id_vars="month", var_name="segment", value_name="outstanding_m"
    )
    adv_long = adv.reset_index().melt(
        id_vars="month", var_name="segment", value_name="funds_advanced_m"
    )
    rate_long = rates.reset_index().melt(
        id_vars="month", var_name="segment", value_name="rate_pct"
    )

    out = bal_long.merge(
        adv_long, on=["month", "segment"], how="left"
    ).merge(rate_long, on=["month", "segment"], how="left")
    out = out.sort_values(["segment", "month"]).reset_index(drop=True)

    # Implied runoff: runoff[t] = outstanding[t-1] + advanced[t] - outstanding[t]
    out["prev_outstanding_m"] = out.groupby("segment")["outstanding_m"].shift(1)
    out["runoff_m"] = (
        out["prev_outstanding_m"]
        + out["funds_advanced_m"].fillna(0)
        - out["outstanding_m"]
    )
    runoff = out[["month", "segment", "runoff_m"]].dropna(
        subset=["runoff_m"]
    ).reset_index(drop=True)

    return out, runoff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    balances, runoff = pull(refresh=args.refresh)
    bpath = OUT_DIR / "m4_balances.parquet"
    rpath = OUT_DIR / "m4_runoff.parquet"
    balances.to_parquet(bpath, index=False)
    runoff.to_parquet(rpath, index=False)
    n_months = balances["month"].nunique()
    print(
        f"wrote {bpath} ({len(balances)} rows, {n_months} months, "
        f"{balances['month'].min().date()} -> {balances['month'].max().date()})"
    )
    print(f"wrote {rpath} ({len(runoff)} rows)")


if __name__ == "__main__":
    main()
