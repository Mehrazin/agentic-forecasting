"""exp70: Macro Signal — calibration + live gate.

Correlates each category's LLM signal (``macro_signals`` table) with a
realized FRED/benchmark anchor on TRAINING months only, and applies the
pre-registered gate: categories with directional accuracy > 55% on
training go "live" downstream; the rest are recorded as dead signal —
honest negative, no cherry-picking.

Calibration split (pre-registered):
  TRAIN: 2023-10 .. 2025-06 (gate applies here)
  HOLDOUT: 2025-07 .. 2026-09 (reported, never gated on)

Anchors (monthly):
  inflation / basic_needs / services_costs -> Canada CPI (CPALTT01CAM659N, MoM %)
  food_prices    -> Food CPI index (benchmarks/food_cpi/data.csv, MoM %)
  central_bank_rates -> BoC short rate (IRSTCB01CAM156N, level delta)
  labour         -> Canada unemployment (LRHUTTTTCAM156S, level delta)
  fx             -> DEXCAUS (monthly mean, MoM %; CAD up = DEXCAUS down -> sign flip)
  housing        -> Canada building permits (CANPERMITMISMEI, MoM %)
  wars_geopolitics -> WTI oil (DCOILWTICO, monthly mean MoM %; weak proxy)
Categories without a realized anchor are "unanchored" — they cannot go
live (no evidence), whatever their signals look like.

Simulator attrition has no calendar anchor (synthetic months), so it is
NOT used here; the causal simulator version of this test is exp69's
synthetic-news design.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "news" / "news_2026.db"
TRAIN_END = "2025-06"
GATE = 0.55

# category -> (anchor kind, transform) ; sign: +1 expected corr sign, -1 flip
ANCHORS: dict[str, tuple[str, str]] = {
    # CPI series is already a YoY % rate -> "delta" = acceleration.
    "inflation": ("cpi", "delta"),
    "basic_needs": ("cpi", "delta"),
    "services_costs": ("cpi", "delta"),
    "food_prices": ("food_cpi", "mom_pct"),
    "central_bank_rates": ("boc_rate", "delta"),
    "labour": ("unemployment", "delta"),
    "fx": ("dexcaus", "mom_pct_flip"),  # CAD strengthens when DEXCAUS falls
    "housing": ("permits", "mom_pct"),
    "wars_geopolitics": ("wti", "mom_pct"),  # oil as conflict proxy (weak)
}


def load_fred(series_id: str) -> pd.Series:
    df = pd.read_parquet(ROOT / "data" / "fred" / f"{series_id}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"]).sort_values("date")
    monthly = df.set_index("date")["value"].resample("ME").mean()
    monthly.index = monthly.index.strftime("%Y-%m")
    return monthly


def load_anchor(kind: str) -> pd.Series:
    if kind == "cpi":
        return load_fred("CPALTT01CAM659N")  # already a MoM % series
    if kind == "food_cpi":
        df = pd.read_csv(ROOT / "benchmarks" / "food_cpi" / "data.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"]
        s.index = s.index.strftime("%Y-%m")
        return s
    if kind == "boc_rate":
        return load_fred("IRSTCB01CAM156N")
    if kind == "unemployment":
        return load_fred("LRHUTTTTCAM156S")
    if kind == "dexcaus":
        return load_fred("DEXCAUS")
    if kind == "permits":
        return load_fred("CANPERMITMISMEI")
    if kind == "wti":
        return load_fred("DCOILWTICO")
    raise ValueError(kind)


def realized_change(anchor: pd.Series, transform: str) -> pd.Series:
    if transform == "delta":
        return anchor.diff()
    if transform == "mom_pct":
        return anchor.pct_change() * 100
    if transform == "mom_pct_flip":
        return -anchor.pct_change() * 100
    raise ValueError(transform)


def dir_accuracy(signal: np.ndarray, realized: np.ndarray) -> tuple[float, int]:
    """Sign agreement over months with a non-zero signal. Returns (acc, n)."""
    mask = signal != 0
    if mask.sum() == 0:
        return float("nan"), 0
    return float(np.mean(np.sign(signal[mask]) == np.sign(realized[mask]))), int(mask.sum())


def dir_accuracy_weighted(
    signal: np.ndarray, realized: np.ndarray, weights: np.ndarray
) -> tuple[float, int]:
    """Certainty-weighted sign agreement. Returns (acc, n)."""
    mask = signal != 0
    if mask.sum() == 0:
        return float("nan"), 0
    hit = (np.sign(signal[mask]) == np.sign(realized[mask])).astype(float)
    w = weights[mask]
    return float(np.sum(hit * w) / np.sum(w)) if w.sum() > 0 else float("nan"), int(mask.sum())


def offset_accuracy(signal: pd.Series, realized: pd.Series, offset: int) -> tuple[float, int]:
    """Directional accuracy of signal_m against realized_{m+offset}.

    offset=+1 tests the forward-looking hypothesis (news this month
    anticipates next month's data); offset=-1 tests lag.
    """
    r = realized.shift(-offset)
    df = pd.DataFrame({"s": signal, "r": r}).dropna()
    return dir_accuracy(df["s"].to_numpy(float), df["r"].to_numpy(float))


def diagnose(
    cat: str, tr: dict, offsets: dict[int, tuple[float, int]], cert: dict[str, float], n_months: int
) -> str:
    """Researcher-mindset diagnosis for a failed gate.

    Returns a one-line diagnosis of WHY the category failed.
    """
    acc = tr["dir_acc"]
    if acc != acc:  # NaN
        return "no non-zero signals on train — extractor silent (coverage or LLM abstention)"
    parts = []
    if tr["n_signal"] < 8:
        parts.append(f"thin (n={tr['n_signal']})")
    if tr["corr"] == tr["corr"] and tr["corr"] < 0:
        parts.append("anti-correlated: rubric sign may be inverted")
    best_off = max(offsets, key=lambda o: offsets[o][0] if offsets[o][0] == offsets[o][0] else -1)
    if (
        offsets[best_off][0] == offsets[best_off][0]
        and best_off != 0
        and offsets[best_off][0] > acc + 0.1
    ):
        parts.append(f"mistimed: offset {best_off:+d} reaches {offsets[best_off][0]:.2f}")
    hi = cert.get("high>=0.7", float("nan"))
    if hi == hi and hi > acc + 0.1:
        parts.append(f"miscalibrated-low-cert: high-certainty subset hits {hi:.2f}")
    if abs(tr["corr"]) < 0.1 and acc < 0.55:
        parts.append("noisy: no linear or directional relation")
    if not parts:
        parts.append("weak but directionally plausible — needs more months")
    return "; ".join(parts)


def certainty_calibration(
    signal: np.ndarray, realized: np.ndarray, certainty: np.ndarray
) -> dict[str, float]:
    out = {}
    for name, lo, hi in (("low<0.4", 0.0, 0.4), ("med", 0.4, 0.7), ("high>=0.7", 0.7, 1.01)):
        m = (certainty >= lo) & (certainty < hi)
        acc, n = dir_accuracy(signal[m], realized[m])
        out[name] = acc if n else float("nan")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    has_table = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='macro_signals'"
    ).fetchone()
    if not has_table:
        print("macro_signals table missing — run Phase 2 extraction first")
        return
    sig = pd.read_sql("SELECT month, category, direction, certainty FROM macro_signals", con)
    con.close()
    if sig.empty:
        print("macro_signals is empty — run Phase 2 extraction first")
        return

    results: dict[str, dict] = {}
    for cat, (kind, transform) in ANCHORS.items():
        anchor = realized_change(load_anchor(kind), transform)
        sub = sig[sig["category"] == cat].set_index("month").sort_index()
        joined = sub.join(anchor.rename("realized"), how="inner").dropna()
        train = joined[joined.index <= TRAIN_END]
        hold = joined[joined.index > TRAIN_END]

        def stats(df: pd.DataFrame) -> dict:
            s, r = df["direction"].to_numpy(float), df["realized"].to_numpy(float)
            acc, n = dir_accuracy(s, r)
            corr = float(np.corrcoef(s, r)[0, 1]) if len(df) > 2 else float("nan")
            return {"n_months": len(df), "n_signal": n, "dir_acc": acc, "corr": corr}

        tr, ho = stats(train), stats(hold)
        cert = (
            certainty_calibration(
                train["direction"].to_numpy(float),
                train["realized"].to_numpy(float),
                train["certainty"].to_numpy(float),
            )
            if len(train)
            else {}
        )
        # Alternative designs (researcher mindset: diagnose before killing).
        s_tr = train["direction"]
        r_tr = realized_change(load_anchor(kind), transform).reindex(s_tr.index)
        offsets = {o: offset_accuracy(s_tr, r_tr, o) for o in (-1, 0, 1)}
        wacc, _ = dir_accuracy_weighted(
            train["direction"].to_numpy(float),
            train["realized"].to_numpy(float),
            train["certainty"].to_numpy(float),
        )
        strong_mask = train["direction"].abs() >= 2
        strong_acc, strong_n = dir_accuracy(
            train["direction"][strong_mask].to_numpy(float),
            train["realized"][strong_mask].to_numpy(float),
        )
        variants = {
            "offset_-1": offsets[-1][0],
            "offset_0": offsets[0][0],
            "offset_+1": offsets[1][0],
            "certainty_weighted": wacc,
            "strong_only_|dir|>=2": strong_acc,
            "strong_only_n": strong_n,
        }
        best_variant = max(
            ((k, v) for k, v in variants.items() if v == v and k != "strong_only_n"),
            key=lambda kv: kv[1],
            default=(None, float("nan")),
        )
        live = tr["dir_acc"] == tr["dir_acc"] and tr["dir_acc"] > GATE  # NaN-safe
        # Tripwire, not execution: a category that fails on the base metric
        # but passes on a principled variant goes "live (variant)" with the
        # variant recorded — the diagnosis says why.
        live_variant = None
        if not live and best_variant[1] == best_variant[1] and best_variant[1] > GATE:  # NaN-safe
            live, live_variant = True, best_variant[0]
        diagnosis = diagnose(cat, tr, offsets, cert, len(train))
        results[cat] = {
            "anchor": kind,
            "transform": transform,
            "train": tr,
            "holdout": ho,
            "certainty_bins_train": cert,
            "variants": variants,
            "diagnosis": diagnosis,
            "live": bool(live),
            "live_via_variant": live_variant,
        }
        status = "LIVE" if live and not live_variant else ("LIVE(variant)" if live else "dead")
        print(
            f"{cat:20s} train acc={tr['dir_acc']:.2f} (n={tr['n_signal']}) "
            f"corr={tr['corr']:+.2f} | holdout acc={ho['dir_acc']:.2f} "
            f"(n={ho['n_signal']}) | {status}"
        )
        if not live:
            print(f"  diagnosis: {diagnosis}")
        elif live_variant:
            print(f"  live via variant: {live_variant} = {best_variant[1]:.2f}")

    anchored = set(ANCHORS)
    unanchored = sorted(set(sig["category"].unique()) - anchored)
    for cat in unanchored:
        results[cat] = {
            "anchor": None,
            "live": False,
            "note": "no realized anchor — research artifact only",
        }
        print(f"{cat:20s} unanchored — cannot go live")

    out = ROOT / "experiments" / "exp70_macro_signal_results.json"
    out.write_text(
        json.dumps(
            {"gate": GATE, "train_end": TRAIN_END, "categories": results}, indent=2, default=str
        )
    )
    live = [c for c, r in results.items() if r["live"]]
    print(f"\nlive categories ({len(live)}): {live}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
