"""exp71: competitor spread as a real refi-pressure covariate (I2 pilot).

**Question:** does adding the real competitor spread (broker discount 5yr
rate - OSFI uninsured new-lending rate) to the router's features improve
total-runoff forecasts vs the standalone series? I.e. does SimulatorV2's
simulated refi-spread mechanism show up in real aggregate data?

**Data:** ``competitor_rates`` table (Perplexity, 8 months with broker
discounts 2024-12 -> 2026-04, forward-filled — competitor rates are
sticky) + OSFI M4 total runoff rate from exp70.

**Protocol:** origins within the spread window (2025-02 -> 2026-06,
~17 origins), h=1, total runoff rate. Arm A = v2 with standard
series features; Arm B = v2 with standard features + spread.
Same records/origins; 5-fold blocked time CV each.

**Honest framing:** UNDERPOWERED pilot (n~17). A null here does not kill
the mechanism; a positive would motivate expanding competitor coverage.
Pre-registered kill: no improvement at p < 0.10 either way -> record
as-is; no tuning.

**PIT:** spread for origin month M uses competitor rates as-of M-1
(forward-filled from the last pull <= M-1) and OSFI rates as-of M-1.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import exp70_osfi_benchmark as e70  # noqa: E402
import llm_food_cpi as lfc  # noqa: E402

from td_one.agent.proposed_method_v2 import ProposedForecasterV2  # noqa: E402
from td_one.agent.tools.data import series_features  # noqa: E402
from td_one.llm_rungs.cache import grid_to_crps  # noqa: E402
from td_one.llm_rungs.combine import blocked_cv_predict  # noqa: E402
from td_one.llm_rungs.prompts import LEVELS_9  # noqa: E402


def load_spread() -> pd.Series:
    """Monthly spread = broker discount 5yr - OSFI uninsured new rate.

    Forward-filled across gaps (rates sticky). Index = month start.
    Positive spread = market new-lending rate above broker best =
    refinancing incentive (borrowers can do better elsewhere).
    """
    con = sqlite3.connect(ROOT / "data" / "news" / "news_2026.db")
    cr = pd.read_sql(
        "SELECT month, product, rate_pct FROM competitor_rates", con
    )
    con.close()
    broker = (
        cr[cr["product"] == "broker_discount_5yr"]
        .groupby("month")["rate_pct"]
        .mean()
    )
    broker.index = pd.to_datetime(broker.index)
    bal = pd.read_parquet(ROOT / "data" / "osfi" / "m4_balances.parquet")
    osfi = (
        bal[bal["segment"] == "uninsured"]
        .set_index("month")
        .sort_index()["rate_pct"]
    )
    months = pd.date_range(
        broker.index.min(), bal["month"].max(), freq="MS"
    )
    broker = broker.reindex(months).ffill()
    osfi = osfi.reindex(months).ffill()
    spread = (osfi - broker).dropna()
    spread.name = "spread"
    return spread


def build_records_augmented(
    values: np.ndarray,
    months: pd.DatetimeIndex,
    spread: pd.Series,
    origins_idx: list[int],
) -> tuple[list[dict], list[str]]:
    """Build records for spread-window origins, with spread as extra feature."""
    feature_names = lfc.feature_names_for(values)
    base, aug = [], []
    for o in origins_idx:
        train = values[o - 36 : o]
        actual = float(values[o])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = series_features(train, period=12)
            qm = lfc.origin_quantiles(train)
        fd = feats.model_dump() if hasattr(feats, "model_dump") else dict(feats)
        vec = [float(fd.get(fn, 0.0)) for fn in feature_names]
        # PIT: spread as of origin month - 1
        origin_month = months[o]
        sp = float(spread.reindex([origin_month - pd.offsets.MonthBegin(1)]).iloc[0])
        rec = {
            "origin": int(o),
            "quantiles": {m: q.tolist() for m, q in qm.items()},
            "crps": {
                m: grid_to_crps(q[:, 0], LEVELS_9, actual)
                for m, q in qm.items()
            },
            "actual": [actual],
        }
        base.append({**rec, "features": vec})
        aug.append({**rec, "features": [*vec, sp]})
    return base, aug  # type: ignore[return-value]


def main() -> None:
    runoff = e70.load_runoff_rates()
    total = runoff["total"]
    months = total.index
    values = total.to_numpy(dtype=float)
    spread = load_spread()
    print(f"[exp71] spread months: {len(spread)}, "
          f"{spread.index.min().date()} -> {spread.index.max().date()}")

    # Origins: months strictly inside spread coverage, with 36m train.
    first_spread = spread.index.min()
    origins = [
        i
        for i in range(36, len(values))
        if months[i] > first_spread and months[i] <= spread.index.max()
    ]
    print(f"[exp71] origins in spread window: {len(origins)}")
    if len(origins) < 10:
        print("[exp71] ABORT: too few origins for a meaningful pilot")
        return

    base, aug = build_records_augmented(values, months, spread, origins)
    fn_base = lfc.feature_names_for(values)
    fn_aug = [*fn_base, "spread_lag1"]

    def factory() -> ProposedForecasterV2:
        fc = ProposedForecasterV2(h=1, use_llm_expert=False)
        fc.models = list(lfc.CLASSICAL_MODELS)
        return fc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = blocked_cv_predict(
            base, fn_base, factory, n_folds=5, baseline="best_single"
        )
        rb = blocked_cv_predict(
            aug, fn_aug, factory, n_folds=5, baseline="best_single"
        )
    a = np.asarray(ra["crps"])
    b = np.asarray(rb["crps"])
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    t, p = ttest_rel(a, b)
    result = {
        "n": int(len(a)),
        "armA_no_spread_crps": float(np.mean(a)),
        "armB_spread_crps": float(np.mean(b)),
        "diff_pct": float(100 * (np.mean(b) - np.mean(a)) / np.mean(a)),
        "p": float(p),
        "win_rate_B": float(np.mean(b < a)),
        "note": "UNDERPOWERED pilot: competitor coverage is 8 raw months.",
    }
    out_path = Path(__file__).parent / "exp71_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
