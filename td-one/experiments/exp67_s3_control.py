"""exp67 S3 control: undiverged regional average as the causal baseline.

S3 (regional divergence) builds its aggregate from two regional series with
different RNG seeds than the 'none' placebo, so the placebo is not a valid
control for S3. This script generates, for every S3 (severity, seed) combo
already in exp67_results_b.json, the control series — the plain average of the
same two regional series WITHOUT divergence — runs all four models on it at
the identical origins, and appends the rows labeled shift="S3ctrl".

The analysis then compares S3 against S3ctrl (same seeds, same origins).

Usage:
    .venv/bin/python experiments/exp67_s3_control.py [--models chronos2,timesfm25,naive,m0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="chronos2,timesfm25,naive,m0")
    args = ap.parse_args()

    exp67 = _load_exp67()
    models = exp67.build_models("real", args.models.split(","))
    oclasses = ("pre", "straddle", "post")

    path = EXP_DIR / "exp67_results_a.json"
    data = json.loads(path.read_text())
    rows = data["rows"]
    combos = sorted(
        {(float(r["severity"]), int(r["seed"])) for r in rows if r["shift"] == "S3"}
    )
    # Drop any stale S3ctrl rows (e.g. from the average-of-ratios bug); regenerate.
    rows = [r for r in rows if r["shift"] != "S3ctrl"]
    print(f"S3ctrl: {len(combos)} combos to run")

    new_rows: list[dict] = []
    t_start = time.time()
    for i, (sev, sd) in enumerate(combos):
        ra = exp67.generate_base_series(exp67.N_MONTHS, seed=2000 + sd)
        rb = exp67.generate_base_series(exp67.N_MONTHS, seed=3000 + sd)
        # Control: plain average, no divergence applied. Must use the SAME
        # aggregation as apply_regional_divergence (ratio of weighted sums,
        # not average of ratios) to be bit-identical pre-onset.
        n_r = 0.5 * (
            np.asarray(ra["n_renewals"], dtype=float)
            + np.asarray(rb["n_renewals"], dtype=float)
        )
        n_s = 0.5 * (
            np.asarray(ra["n_switches"], dtype=float)
            + np.asarray(rb["n_switches"], dtype=float)
        )
        rate = n_s / np.maximum(n_r, 1)
        # PIT gate: control must be bit-identical to the diverged aggregate
        # pre-onset (divergence only touches post-onset months).
        _chk_rng = np.random.default_rng(1000 + sd)
        _div = exp67.apply_regional_divergence(
            ra, rb, exp67.ShiftSpec("S3", exp67.ONSET, sev), 0.5, _chk_rng
        )
        _pre_ctrl = rate[: exp67.ONSET]
        _pre_div = np.asarray(_div["attrition_rate"][: exp67.ONSET], dtype=float)
        assert np.array_equal(_pre_ctrl, _pre_div), (
            f"S3ctrl PIT violation at sev={sev} seed={sd}: "
            f"maxdiff={np.max(np.abs(_pre_ctrl - _pre_div))}"
        )
        for oclass in oclasses:
            for origin in exp67.ORIGIN_CLASSES[oclass]:
                if origin + exp67.HORIZON > exp67.N_MONTHS:
                    continue
                ctx = rate[:origin]
                actual = rate[origin : origin + exp67.HORIZON]
                for name, model in models.items():
                    t0 = time.time()
                    try:
                        if name == "m0":
                            levels, values = exp67._m0_quantiles(
                                n_r[:origin], n_s[:origin], exp67.HORIZON
                            )
                        elif name in ("chronos2", "timesfm25"):
                            levels, values = exp67._tsfm_quantiles(
                                model, ctx, exp67.HORIZON, seed=sd
                            )
                        else:
                            levels, values = model(ctx, exp67.HORIZON, sd)
                        crps = exp67.crps_from_quantiles(levels, values, actual)
                        cov = exp67._coverage80(levels, values, actual)
                        status, err = "ok", ""
                    except Exception as e:  # noqa: BLE001 — record, don't crash
                        crps, cov, status, err = (
                            float("nan"),
                            float("nan"),
                            "error",
                            str(e)[:200],
                        )
                    new_rows.append(
                        {
                            "shift": "S3ctrl",
                            "severity": sev,
                            "seed": sd,
                            "origin": origin,
                            "origin_class": oclass,
                            "model": name,
                            "crps": crps,
                            "coverage80": cov,
                            "status": status,
                            "error": err,
                            "elapsed_s": round(time.time() - t0, 2),
                        }
                    )
        print(f"  {i + 1}/{len(combos)} combos, {time.time() - t_start:.0f}s elapsed",
              flush=True)

    for model in models.values():
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()

    data["rows"] = rows + new_rows
    data["summary"] = exp67.analyze(data["rows"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)
    print(f"wrote {path}: {len(data['rows'])} rows")


if __name__ == "__main__":
    main()
