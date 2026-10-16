"""Download curated FRED macro series to local parquet for the TD-One models.

Each entry was verified against the live FRED API (2026-10-15). Series were
chosen for Canadian mortgage renewal/switching dynamics:

- rates: BoC policy proxy, Canada 3M/10Y, US 5Y/10Y, Fed funds, curve, US mortgage
- inflation: Canada CPI, US 5Y breakeven
- labor: Canada unemployment
- housing: Canada real house prices, US Case-Shiller, Canada permits
- fx: CAD/USD (BoC/Fed divergence)
- credit stress: HY spread, US mortgage delinquencies

Usage:
    uv run python td-one/scripts/fetch_fred.py            # fetch all (skip cached)
    uv run python td-one/scripts/fetch_fred.py --refresh  # re-download all
    uv run python td-one/scripts/fetch_fred.py --series DGS5 DEXCAUS

Output: td-one/data/fred/<series_id>.parquet with (date, value) columns,
plus fred_manifest.json recording titles, units, frequencies, and fetch time.

PIT discipline: files hold full history; filter ``date <= run_date`` at use
time (``td_one.fred.get_series_live`` takes start/end for the same purpose).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from td_one.fred.live import FredClient  # noqa: E402

#: series_id -> (category, why it matters for mortgage attrition)
SERIES: dict[str, tuple[str, str]] = {
    # --- Rates: the refinancing incentive lives here ---
    "IRSTCB01CAM156N": ("rates", "BoC policy proxy (immediate central-bank rate, Canada)"),
    "IR3TIB01CAM156N": ("rates", "Canada 3M interbank (variable-rate mortgage anchor)"),
    "IRLTLT01CAM156N": ("rates", "Canada 10Y gov bond (fixed-rate mortgage benchmark)"),
    "DGS5": ("rates", "US 5Y Treasury (cross-border rate pressure)"),
    "DGS10": ("rates", "US 10Y Treasury (global term-structure anchor)"),
    "FEDFUNDS": ("rates", "Fed funds effective (US policy stance)"),
    "T10Y2Y": ("rates", "US 10Y-2Y curve (recession/repricing signal)"),
    "MORTGAGE30US": ("rates", "US 30Y mortgage avg (refi incentive proxy)"),
    # --- Inflation: erodes real payments, drives BoC ---
    "CPALTT01CAM659N": ("inflation", "Canada CPI (BoC mandate variable)"),
    "T5YIE": ("inflation", "US 5Y breakeven inflation (expectations)"),
    # --- Labor: unemployment shocks drive defaults/switching ---
    "LRHUTTTTCAM156S": ("labor", "Canada harmonized unemployment rate"),
    # --- Housing: equity position drives refi/switch options ---
    "QCAR628BIS": ("housing", "BIS real residential property prices, Canada"),
    "CSUSHPISA": ("housing", "US Case-Shiller national home price index"),
    "CANPERMITMISMEI": ("housing", "Canada building permits (supply pipeline)"),
    # --- FX: CAD weakness constrains BoC easing ---
    "DEXCAUS": ("fx", "CAD per USD (policy-divergence gauge)"),
    # --- Credit stress: risk appetite for switching/refi ---
    "BAMLH0A0HYM2": ("credit", "US HY option-adjusted spread (risk appetite)"),
    "DRALACBN": ("credit", "US mortgage delinquency rate (stress coincident)"),
}

DATA_DIR = ROOT / "data" / "fred"


def fetch_all(series_ids: list[str], refresh: bool) -> dict[str, dict]:
    """Download each series to parquet; return the manifest entries."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = FredClient()
    manifest: dict[str, dict] = {}
    for sid in series_ids:
        out = DATA_DIR / f"{sid}.parquet"
        if out.exists() and not refresh:
            print(f"  skip {sid} (cached)")
            df = pd.read_parquet(out)
        else:
            print(f"  fetch {sid} ...", flush=True)
            df = client.observations(sid)
            df.to_parquet(out, index=False)
            print(f"        {len(df)} rows, {df['date'].min()}..{df['date'].max()}")
        info = client.series_info(sid)
        manifest[sid] = {
            "category": SERIES[sid][0],
            "rationale": SERIES[sid][1],
            "title": info.get("title"),
            "frequency": info.get("frequency"),
            "units": info.get("units"),
            "seasonal_adjustment": info.get("seasonal_adjustment"),
            "rows": int(len(df)),
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        }
    manifest["_meta"] = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "n_series": len(series_ids),
    }
    (DATA_DIR / "fred_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="*", default=None, help="Subset of series ids")
    parser.add_argument("--refresh", action="store_true", help="Re-download cached files")
    parser.add_argument("--list", action="store_true", help="Print the curated list and exit")
    args = parser.parse_args()

    if args.list:
        for sid, (cat, why) in SERIES.items():
            print(f"{sid:18s} [{cat:9s}] {why}")
        return 0

    ids = list(SERIES) if not args.series else args.series
    unknown = [s for s in ids if s not in SERIES]
    if unknown:
        print(f"unknown series ids: {unknown}")
        return 1
    print(f"fetching {len(ids)} FRED series -> {DATA_DIR}")
    fetch_all(ids, refresh=args.refresh)
    print("wrote fred_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
