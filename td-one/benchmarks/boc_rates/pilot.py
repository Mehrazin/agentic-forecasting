"""BoC Rate Decisions pilot: TD-One methodology on discrete data.

Tests whether our agent tools work on the BoC CUT/HOLD/HIKE task.
Baseline: logistic regression on 3 macro features (no yield_spread - missing 2yr).
Our method: react_v2-style analysis with our 10 tools.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_meetings() -> list[date]:
    """Load BoC meeting dates from the curated schedule."""
    sched_path = Path(__file__).resolve().parents[3] / "implementations" / "boc_rate_decisions" / "meeting_schedule.yaml"
    with open(sched_path) as f:
        data = yaml.safe_load(f)
    return sorted(data["announcement_dates"])


def load_rate_series() -> pd.DataFrame:
    """Load BoC policy rate (monthly FRED)."""
    fred_path = Path(__file__).resolve().parents[2] / "data" / "fred" / "IRSTCB01CAM156N.parquet"
    df = pd.read_parquet(fred_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def derive_decisions(meetings: list[date], rates: pd.DataFrame) -> pd.DataFrame:
    """Derive CUT/HOLD/HIKE for each meeting from rate changes.
    
    A decision is the change in the policy rate from the previous meeting's
    rate to this meeting's rate. We use the monthly rate series and find
    the rate in effect at each meeting date.
    """
    rates = rates.copy()
    rates["date"] = pd.to_datetime(rates["date"])
    
    rows = []
    prev_rate = None
    for m in meetings:
        m_ts = pd.Timestamp(m)
        # Rate in effect at meeting: last monthly observation <= meeting date
        visible = rates[rates["date"] <= m_ts]
        if visible.empty:
            continue
        curr_rate = float(visible.iloc[-1]["value"])
        if prev_rate is not None:
            change = curr_rate - prev_rate
            if change < -0.01:
                decision = -1  # cut
            elif change > 0.01:
                decision = 1   # hike
            else:
                decision = 0    # hold
        else:
            decision = None
        rows.append({"meeting_date": m, "rate": curr_rate, "decision": decision})
        prev_rate = curr_rate
    
    df = pd.DataFrame(rows)
    # Drop first row (no previous rate)
    return df[df["decision"].notna()].reset_index(drop=True)


def load_cpi() -> pd.DataFrame:
    """Load Canadian CPI."""
    fred_path = Path(__file__).resolve().parents[2] / "data" / "fred" / "CPALTT01CAM659N.parquet"
    df = pd.read_parquet(fred_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_unemployment() -> pd.DataFrame:
    """Load Canadian unemployment rate."""
    fred_path = Path(__file__).resolve().parents[2] / "data" / "fred" / "LRHUTTTTCAM156S.parquet"
    df = pd.read_parquet(fred_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(
    meeting_date: date,
    rates: pd.DataFrame,
    cpi: pd.DataFrame,
    unemp: pd.DataFrame,
    lead_days: int = 28,
) -> dict[str, float] | None:
    """Build 3 macro features as of (meeting_date - lead_days).
    
    Features (leak-safe):
    - rate_momentum: change in policy rate over trailing 90 days
    - inflation_gap: CPI YoY inflation minus 2% target (lagged 1 month for publication)
    - unemployment_momentum: 12-month change in unemployment rate (lagged 1 month)
    """
    origin = pd.Timestamp(meeting_date) - pd.Timedelta(days=lead_days)
    
    # Rate momentum: rate now vs 90 days ago
    rates["date"] = pd.to_datetime(rates["date"])
    r_now = rates[rates["date"] <= origin]
    r_90 = rates[rates["date"] <= origin - pd.Timedelta(days=90)]
    if r_now.empty or r_90.empty:
        return None
    rate_momentum = float(r_now.iloc[-1]["value"]) - float(r_90.iloc[-1]["value"])
    
    # Inflation gap: CPI YoY inflation minus 2% target
    # NOTE: CPALTT01CAM659N is already YoY % (not an index), so no YoY calc needed
    # Lagged 1 month for publication delay
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi_cutoff = origin - pd.DateOffset(months=1)
    c_now = cpi[cpi["date"] <= cpi_cutoff]
    if c_now.empty:
        return None
    inflation_yoy = float(c_now.iloc[-1]["value"])
    inflation_gap = inflation_yoy - 2.0
    
    # Unemployment momentum: 12m change, lagged 1 month
    unemp["date"] = pd.to_datetime(unemp["date"])
    u_cutoff = origin - pd.DateOffset(months=1)
    u_now = unemp[unemp["date"] <= u_cutoff]
    if len(u_now) < 13:
        return None
    unemp_momentum = float(u_now.iloc[-1]["value"]) - float(u_now.iloc[-13]["value"])
    
    return {
        "rate_momentum": rate_momentum,
        "inflation_gap": inflation_gap,
        "unemployment_momentum": unemp_momentum,
    }


def main() -> None:
    print("Loading data...")
    meetings = load_meetings()
    rates = load_rate_series()
    cpi = load_cpi()
    unemp = load_unemployment()

    # DATA CUTOFF (2026-10-19 fix): the FRED monthly rate series
    # (IRSTCB01CAM156N.parquet) ends in 2023. Meetings after 2023-12-31
    # would get stale forward-filled rates (decision=0, rate_momentum=0.0
    # for every 2024-2026 meeting — 7 real BoC cuts turned into fake holds).
    # Truncate to meetings with observable rates. See benchmarks/smm/
    # exp34_real_events.py for the 2024-2025 cut dates (not used here to
    # keep this pilot strictly on the FRED series).
    cutoff = date(2023, 12, 31)
    meetings = [m for m in meetings if m <= cutoff]
    print(f"Meetings (<= 2023-12-31): {len(meetings)}, Rate obs: {len(rates)}")
    
    decisions = derive_decisions(meetings, rates)
    print(f"Decisions derived: {len(decisions)}")
    print(decisions["decision"].value_counts().to_dict())
    
    # Build features for each meeting
    rows = []
    for _, row in decisions.iterrows():
        m = row["meeting_date"]
        feats = build_features(m, rates, cpi, unemp)
        if feats is not None:
            rows.append({
                "meeting_date": m.isoformat(),
                "decision": int(row["decision"]),
                **feats,
            })
    
    feat_df = pd.DataFrame(rows)
    print(f"\nFeature rows: {len(feat_df)}")
    print(feat_df[["decision"]].value_counts().to_dict())
    
    # Save
    out_path = Path(__file__).parent / "pilot_data.json"
    feat_df.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved to {out_path}")
    
    # Quick baseline: how well does rate_momentum alone predict?
    # (Cuts cluster in easing cycles; hikes cluster in tightening cycles)
    # Simple rule: if rate_momentum < -0.25 → cut; if > 0.25 → hike; else hold
    
    def rule_predict(row):
        rm = row["rate_momentum"]
        if rm < -0.25:
            return -1
        elif rm > 0.25:
            return 1
        else:
            return 0
    
    feat_df["rule_pred"] = feat_df.apply(rule_predict, axis=1)
    
    # Time-series split: test on last 30%
    n = len(feat_df)
    split = int(n * 0.7)
    test_df = feat_df.iloc[split:]
    
    rule_acc = (test_df["rule_pred"] == test_df["decision"]).mean()
    print(f"\nRule baseline (rate momentum) accuracy (last 30%): {rule_acc:.3f}")
    
    # Climatology baseline (always predict hold)
    clim_acc = (test_df["decision"] == 0).mean()
    print(f"Climatology (always hold) accuracy: {clim_acc:.3f}")
    
    # Confusion matrix for rule
    print("\nRule predictions vs actual (test set):")
    print(pd.crosstab(test_df["decision"], test_df["rule_pred"], 
                      rownames=["actual"], colnames=["pred"]))


if __name__ == "__main__":
    main()
