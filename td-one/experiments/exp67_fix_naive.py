"""One-off repair for exp67: recompute the naive-baseline rows.

The initial real-TSFM grid ran with a buggy _naive_quantiles (shape (19, 1)
instead of (19, h)), so all naive rows recorded status="error". This script
reruns ONLY the naive model for every (shift, severity, seed) combo already in
the results files (series regeneration is seed-deterministic, so contexts are
identical) and replaces the naive rows in place.

Usage:
    .venv/bin/python experiments/exp67_fix_naive.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent


def _load_exp67():
    path = EXP_DIR / "exp67_tsfm_redteam.py"
    spec = importlib.util.spec_from_file_location("exp67_tsfm_redteam", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp67_tsfm_redteam"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    exp67 = _load_exp67()
    naive = {"naive": lambda ctx, h, seed: exp67._naive_quantiles(ctx, h)}
    oclasses = ("pre", "straddle", "post")

    for name in ("exp67_results_a.json", "exp67_results_b.json"):
        path = EXP_DIR / name
        if not path.exists():
            print(f"[skip] {name} missing")
            continue
        data = json.loads(path.read_text())
        rows = data["rows"]
        combos = sorted(
            {(str(r["shift"]), float(r["severity"]), int(r["seed"])) for r in rows}
        )
        print(f"{name}: repairing naive for {len(combos)} combos")
        fresh: list[dict] = []
        for sh, sev, sd in combos:
            fresh.extend(exp67.run_combo(sh, sev, sd, naive, origin_classes=oclasses))
        assert all(r["model"] == "naive" for r in fresh)
        assert all(r["status"] == "ok" for r in fresh), "naive still erroring"
        kept = [r for r in rows if r["model"] != "naive"]
        data["rows"] = kept + fresh
        data["summary"] = exp67.analyze(data["rows"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
        print(f"  wrote {name}: {len(data['rows'])} rows")


if __name__ == "__main__":
    main()
