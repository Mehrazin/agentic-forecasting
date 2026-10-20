r"""exp67: Red-team Chronos-2 and TimesFM-2.5 on loan-attrition shifts.

For each (shift type, severity, seed, origin-class), every model forecasts
h=6 from a rolling origin. Metrics: CRPS on the common 19-level grid,
80% interval coverage. Paired t-test + win rate vs the model's own
pre-shift baseline and vs classical baselines on identical shifted data.

Placebo control: identical protocol with no shift must replicate the
known result (TSFM matches/beats classical on calm DGP).

Usage:
    .venv/bin/python experiments/exp67_tsfm_redteam.py --mode mock
    .venv/bin/python experiments/exp67_tsfm_redteam.py --mode real \\
        --shifts S1,S4,S6 --severities 1.0,4.0 --seeds 2 --models chronos2,timesfm25

Phase A of Track 2 (Track 2 plan: docs/LOAN_ATTRITION_PLAN.md). The agentic
guard (exp68) is built ONLY after these numbers exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.data_gen.shift_scenarios import (  # noqa: E402
    SHIFT_TYPES,
    ShiftSpec,
    apply_regional_divergence,
    apply_shift,
    generate_base_series,
)
from td_one.evaluation.real_benchmark import (  # noqa: E402
    COMMON_GRID,
    crps_from_quantiles,
    interp_quantiles_to_grid,
)
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.tsfm import (  # noqa: E402
    Chronos2Wrapper,
    TimesFMWrapper,
)

N_MONTHS = 48
ONSET = 30
HORIZON = 6
SEVERITIES = (0.5, 1.0, 2.0, 4.0)
ORIGIN_CLASSES = {
    "pre": (18, 21, 24),
    "straddle": (27, 30, 33),
    "post": (36, 39, 42),
}
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "weights"


def _naive_quantiles(context: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-mean + Gaussian residual quantiles (weak classical baseline)."""
    w = min(6, len(context))
    mu = float(np.mean(context[-w:]))
    resid = context[-min(24, len(context)) :] - np.mean(context[-min(24, len(context)) :])
    sd = float(np.std(resid)) if len(resid) > 1 else 0.01
    sd = max(sd, 1e-4)
    z = stats.norm.ppf(COMMON_GRID)
    vals = np.clip(mu + z[:, None] * sd, 0.0, 1.0)
    vals = np.repeat(vals, h, axis=1)  # constant quantiles across the horizon
    assert vals.shape == (len(COMMON_GRID), h), vals.shape
    return COMMON_GRID.copy(), vals


def _m0_quantiles(
    n_renewals: np.ndarray, n_switches: np.ndarray, h: int
) -> tuple[np.ndarray, np.ndarray]:
    """M0 (Jeffreys Beta-Binomial) rate quantiles."""
    m0 = HistoricalBaseline()
    m0.fit(n_renewals, n_switches)
    n_up = max(int(round(float(np.mean(n_renewals[-6:])))), 1)
    qmap = m0.predict_rate_quantiles(n_up, COMMON_GRID)
    vals = np.array([[qmap[float(lv)] for _ in range(h)] for lv in COMMON_GRID])
    return COMMON_GRID.copy(), np.clip(vals, 0.0, 1.0)


def _tsfm_quantiles(
    wrapper: Chronos2Wrapper | TimesFMWrapper,
    context: np.ndarray,
    h: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """TSFM forecast interpolated onto the common grid."""
    levels, values = wrapper.forecast_quantiles(context, horizon=h, seed=seed)
    grid_vals = interp_quantiles_to_grid(levels, values, COMMON_GRID)
    return COMMON_GRID.copy(), grid_vals


def _coverage80(levels: np.ndarray, values: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of actuals inside the [q10, q90] interval (h-averaged)."""
    lv = np.asarray(levels)
    i10 = int(np.argmin(np.abs(lv - 0.10)))
    i90 = int(np.argmin(np.abs(lv - 0.90)))
    lo = np.minimum(values[i10], values[i90])
    hi = np.maximum(values[i10], values[i90])
    return float(np.mean((actual >= lo) & (actual <= hi)))


def build_models(mode: str, model_names: list[str]):
    """Instantiate model callables. TSFM wrappers lazy-load on first real use."""
    use_mock = mode == "mock"
    models: dict[str, object] = {}
    if "naive" in model_names:
        models["naive"] = lambda ctx, h, seed: _naive_quantiles(ctx, h)
    if "m0" in model_names:
        models["m0"] = None  # needs (n_renewals, n_switches); handled inline
    if "chronos2" in model_names:
        models["chronos2"] = Chronos2Wrapper(
            prediction_length=HORIZON,
            use_mock=use_mock,
            weights_dir=str(WEIGHTS_DIR / "chronos2"),
        )
    if "timesfm25" in model_names:
        models["timesfm25"] = TimesFMWrapper(
            prediction_length=HORIZON,
            use_mock=use_mock,
            weights_dir=str(WEIGHTS_DIR / "timesfm25"),
        )
    return models


def run_combo(
    shift: str,
    severity: float,
    seed: int,
    models: dict,
    n_months: int = N_MONTHS,
    onset: int = ONSET,
    origin_classes: tuple[str, ...] = ("pre", "straddle", "post"),
) -> list[dict]:
    """Run all origins for one (shift, severity, seed). Returns result rows."""
    rng = np.random.default_rng(1000 + seed)
    if shift == "S3":
        ra = generate_base_series(n_months, seed=2000 + seed)
        rb = generate_base_series(n_months, seed=3000 + seed)
        series = apply_regional_divergence(ra, rb, ShiftSpec("S3", onset, severity), 0.5, rng)
    elif shift == "none":
        series = generate_base_series(n_months, seed=4000 + seed)
    else:
        base = generate_base_series(n_months, seed=4000 + seed)
        series = apply_shift(base, ShiftSpec(shift, onset, severity), rng)

    rate = np.asarray(series["attrition_rate"], dtype=float)
    n_r = np.asarray(series["n_renewals"], dtype=float)
    n_s = np.asarray(series["n_switches"], dtype=float)

    rows: list[dict] = []
    for oclass in origin_classes:
        origins = ORIGIN_CLASSES[oclass]
        for origin in origins:
            if origin + HORIZON > n_months:
                continue
            ctx = rate[:origin]
            actual = rate[origin : origin + HORIZON]
            for name, model in models.items():
                t0 = time.time()
                try:
                    if name == "m0":
                        levels, values = _m0_quantiles(n_r[:origin], n_s[:origin], HORIZON)
                    elif name in ("chronos2", "timesfm25"):
                        assert not isinstance(model, type(None))
                        levels, values = _tsfm_quantiles(
                            model,
                            ctx,
                            HORIZON,
                            seed=seed,  # type: ignore[arg-type]
                        )
                    else:
                        levels, values = model(ctx, HORIZON, seed)  # type: ignore[operator]
                    crps = crps_from_quantiles(levels, values, actual)
                    cov = _coverage80(levels, values, actual)
                    status, err = "ok", ""
                except Exception as e:  # noqa: BLE001 — record, don't crash the grid
                    crps, cov, status, err = float("nan"), float("nan"), "error", str(e)[:200]
                rows.append(
                    {
                        "shift": shift,
                        "severity": severity,
                        "seed": seed,
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
    return rows


def analyze(rows: list[dict]) -> dict:
    """Breaking-point analysis: per (shift, severity, model) vs pre-shift."""
    import pandas as pd

    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    summary: dict[str, dict] = {}
    for (shift, sev, model), g in df.groupby(["shift", "severity", "model"]):
        pre = g[g.origin_class == "pre"].groupby("seed").crps.mean()
        post = g[g.origin_class == "straddle"].groupby("seed").crps.mean()
        key = f"{shift}/sev={sev}/{model}"
        entry: dict[str, float | str] = {
            "n_seeds": int(len(pre)),
            "pre_crps": round(float(pre.mean()), 5),
            "straddle_crps": round(float(post.mean()), 5),
        }
        if len(pre) >= 2 and len(post) >= 2:
            idx = pre.index.intersection(post.index)
            if len(idx) >= 2:
                t, p = stats.ttest_rel(post.loc[idx], pre.loc[idx])
                entry["paired_t"] = round(float(t), 3)
                entry["paired_p"] = float(p)
                entry["win_rate"] = round(float((post.loc[idx] < pre.loc[idx]).mean()), 3)
                entry["degradation"] = round(
                    float((post.mean() - pre.mean()) / max(pre.mean(), 1e-9)), 3
                )
        # coverage on straddle origins
        cov = g[g.origin_class == "straddle"].coverage80.mean()
        entry["straddle_coverage80"] = round(float(cov), 3)
        summary[key] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--shifts", default=",".join(SHIFT_TYPES))
    ap.add_argument("--severities", default=",".join(map(str, SEVERITIES)))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", default="chronos2,timesfm25,naive,m0")
    ap.add_argument("--include-placebo", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap combos for smoke tests")
    ap.add_argument(
        "--origin-classes",
        default="pre,straddle,post",
        help="comma-separated subset of pre,straddle,post",
    )
    ap.add_argument("--out", default="experiments/exp67_results.json")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load existing --out and skip combos already present",
    )
    args = ap.parse_args()

    shifts = [s for s in args.shifts.split(",") if s in SHIFT_TYPES]
    if args.include_placebo:
        shifts = shifts + ["none"]
    severities = [float(s) for s in args.severities.split(",")]
    model_names = args.models.split(",")
    models = build_models(args.mode, model_names)
    oclasses = tuple(c for c in args.origin_classes.split(",") if c in ORIGIN_CLASSES)

    combos = [(sh, sev, sd) for sh in shifts for sev in severities for sd in range(args.seeds)]
    if args.limit:
        combos = combos[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(rows: list[dict]) -> None:
        """Atomic write of the results file (resume-safe)."""
        out = {
            "config": {
                "mode": args.mode,
                "shifts": shifts,
                "severities": severities,
                "seeds": args.seeds,
                "models": model_names,
                "n_months": N_MONTHS,
                "onset": ONSET,
                "horizon": HORIZON,
                "origin_classes": list(oclasses),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "summary": analyze(rows),
            "rows": rows,
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(out_path)

    rows: list[dict] = []
    done_keys: set[tuple[str, float, int]] = set()
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            rows = prev.get("rows", [])
            done_keys = {(str(r["shift"]), float(r["severity"]), int(r["seed"])) for r in rows}
            print(f"resuming: {len(done_keys)} combos already done")
        except Exception as e:  # noqa: BLE001 — corrupt file: start fresh
            print(f"resume file unreadable ({e}); starting fresh")
    print(f"exp67 [{args.mode}]: {len(combos)} combos x {len(model_names)} models")
    t_start = time.time()
    for i, (sh, sev, sd) in enumerate(combos):
        if (sh, sev, sd) in done_keys:
            continue
        rows.extend(run_combo(sh, sev, sd, models, origin_classes=oclasses))
        _save(rows)
        print(f"  {i + 1}/{len(combos)} combos, {time.time() - t_start:.0f}s elapsed", flush=True)
    print(f"done: {len(rows)} rows in {time.time() - t_start:.0f}s")

    # Release TSFM weights at the end (kept loaded across combos for speed;
    # combined footprint ~1.3GB is fine).
    for model in models.values():
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()

    _save(rows)  # final write with unload-safe state
    summary = analyze(rows)
    print(f"wrote {out_path}")

    # Console breaking-point table.
    print("\nBreaking-point summary (straddle vs pre, degradation = rel. CRPS change):")
    for key in sorted(summary):
        e = summary[key]
        deg = e.get("degradation", float("nan"))
        p = e.get("paired_p", float("nan"))
        print(
            f"  {key}: pre={e['pre_crps']} straddle={e['straddle_crps']} "
            f"degrad={deg:+.2%} p={p:.3g} cov80={e['straddle_coverage80']}"
        )


if __name__ == "__main__":
    main()
