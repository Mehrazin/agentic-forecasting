"""Experiment 5, Phase 1: simulate and cache per-origin segment panels.

The 10k x 66 simulation is the memory peak of exp5. This script runs it
alone, builds the PIT-masked panels for each origin, pickles them, and
exits -- so Phase 2 (the model backtests) runs in a lean process even
when this VM is shared with sibling research tracks.

Usage:
    cd td-one && uv run python experiments/exp5_phase1.py \
        --out /tmp/exp5_panels.pkl
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scale_up import _default_origins  # noqa: E402

from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.segments.discovery import discover_segments  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="exp5 phase 1: cache panels")
    p.add_argument("--out", default="/tmp/exp5_panels.pkl")
    p.add_argument("--n-customers", type=int, default=10_000)
    p.add_argument("--n-months", type=int, default=66)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    print(
        f"[phase1] Simulating {args.n_customers} x {args.n_months} (seed={args.seed}) ...",
        flush=True,
    )
    sim = MortgageSimulator(n_customers=args.n_customers, n_months=args.n_months, seed=args.seed)
    histories = sim.generate_history()
    outcomes = sim.generate_outcomes(histories)
    del sim
    gc.collect()

    panels_full = discover_segments(histories, outcomes)
    origins = _default_origins(panels_full)
    print(f"[phase1] {len(panels_full)} segments, {len(origins)} origins", flush=True)

    panels_by_origin: dict[str, dict[str, object]] = {}
    kept_origins: list[str] = []
    for origin in origins:
        try:
            panels = discover_segments(histories, outcomes, observed_asof=origin)
        except ValueError:
            continue
        key = origin.isoformat()
        panels_by_origin[key] = panels
        kept_origins.append(key)
        del panels
        gc.collect()
        print(f"[phase1] origin {key} cached", flush=True)

    del histories, outcomes, panels_full
    gc.collect()

    payload = {
        "origins": kept_origins,
        "panels_by_origin": panels_by_origin,
        "config": {
            "n_customers": args.n_customers,
            "n_months": args.n_months,
            "seed": args.seed,
        },
    }
    out = Path(args.out)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[phase1] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
