"""exp68: agent+TSFM vs TSFM-alone on distributional shifts.

Pre-registered holdout design (locked in exp67_report.md BEFORE running):
- Dev shifts: S1, S4, S6. Eval (holdout) shifts: S2, S3, S5, S7.
- Disjoint RNG seeds from exp67 (exp67: 0-1; exp68: 100+).
- Dev magnitudes 1-2σ; holdout magnitudes 2-4σ.
- Claim under test: agent+TSFM beats TSFM-alone on post-shift origins at
  p<0.05 (paired t-test), with a do-no-harm gate (no significant degradation
  on pre-shift origins or on the no-break shifts S4/S5).
- If the claim fails: honest negative, no iteration.

The "agent" is the ShiftGuard (td_one/agent/shift_guard.py): CUSUM on a
policy-rate covariate (which shifts 2 months BEFORE the attrition target for
S1/S2/S6) + PIT-filtered announcements -> context truncation + conformal
quantile widening calibrated on DEV shift episodes only.

Usage:
    .venv/bin/python experiments/exp68_agent_vs_shift.py --mode mock   # smoke
    .venv/bin/python experiments/exp68_agent_vs_shift.py --mode real   # real
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

from td_one.agent.shift_guard import (
    GuardConfig,
    ShiftGuard,
    calibrate_widening,
    heuristic_classifier,
)
from td_one.data_gen.shift_scenarios import (
    ShiftSpec,
    apply_regional_divergence,
    apply_shift,
    generate_base_series,
)
from td_one.evaluation.real_benchmark import (
    COMMON_GRID,
    crps_from_quantiles,
    interp_quantiles_to_grid,
)

N_MONTHS = 48
ONSET = 30
HORIZON = 6
DEV_SHIFTS = ("S1", "S4", "S6")
EVAL_SHIFTS = ("S2", "S3", "S5", "S7")
DEV_SEVS = (1.0, 2.0)
EVAL_SEVS = (2.0, 4.0)
SEEDS = (100, 101)
POST_ORIGINS = (36, 38, 40)
PRE_ORIGINS = (18, 21, 24)
# Covariate leads the target by this many months for signal-carrying shifts.
COV_LEAD = 2
COV_SHIFTS = {"S1": 0.8, "S2": 0.8, "S6": 0.6}  # shift -> covariate jump (σ-ish)
ANNOUNCED = {"S1", "S2"}


def make_covariate(shift: str, onset: int, seed: int, n: int = N_MONTHS) -> np.ndarray:
    """Policy-rate covariate: stationary AR(1), level jump at onset-COV_LEAD.

    Stationary (not a random walk) so the CUSUM baseline is valid and the
    detector only fires on genuine jumps, not on drift.
    """
    rng = np.random.default_rng(9000 + seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + 0.1 * rng.standard_normal()
    x = 2.0 + x
    if shift in COV_SHIFTS:
        x[onset - COV_LEAD :] += COV_SHIFTS[shift]
    return x


def make_events(shift: str, onset: int) -> list[dict]:
    """Announcement timeline (PIT-filtered by the guard)."""
    if shift in ANNOUNCED:
        return [
            {
                "t": onset - COV_LEAD,
                "type": "RATE_SHIFT",
                "description": "Policy rate hike announced; attrition expected to react",
            }
        ]
    return []


def make_series(shift: str, severity: float, seed: int) -> dict:
    """Shifted attrition series (same constructors as exp67)."""
    rng = np.random.default_rng(1000 + seed)
    if shift == "S3":
        ra = generate_base_series(N_MONTHS, seed=2000 + seed)
        rb = generate_base_series(N_MONTHS, seed=3000 + seed)
        return apply_regional_divergence(ra, rb, ShiftSpec("S3", ONSET, severity), 0.5, rng)
    base = generate_base_series(N_MONTHS, seed=4000 + seed)
    return apply_shift(base, ShiftSpec(shift, ONSET, severity), rng)


def make_placebo(shift: str, seed: int) -> dict:
    """Unshifted control: same seeds as the shifted series."""
    if shift == "S3":
        ra = generate_base_series(N_MONTHS, seed=2000 + seed)
        rb = generate_base_series(N_MONTHS, seed=3000 + seed)
        n_r = 0.5 * (np.asarray(ra["n_renewals"]) + np.asarray(rb["n_renewals"]))
        n_s = 0.5 * (np.asarray(ra["n_switches"]) + np.asarray(rb["n_switches"]))
        rate = n_s / np.maximum(n_r, 1)
        return {"attrition_rate": rate, "n_renewals": n_r, "n_switches": n_s}
    return generate_base_series(N_MONTHS, seed=4000 + seed)


def _mock_tsfm(ctx: np.ndarray, h: int, seed: int):
    rng = np.random.default_rng(seed)
    mu = float(np.mean(ctx[-6:]))
    vals = mu + np.linspace(-0.02, 0.02, 19)[:, None] + 0.001 * rng.standard_normal((19, h))
    return COMMON_GRID.copy(), np.sort(vals, axis=0)


def _mock_classical(ctx: np.ndarray, h: int, seed: int):
    mu = float(np.mean(ctx[-6:]))
    vals = np.tile((mu + np.linspace(-0.03, 0.03, 19)[:, None]), (1, h))
    return COMMON_GRID.copy(), vals


def _real_tsfm(wrapper):
    def _f(ctx: np.ndarray, h: int, seed: int):
        levels, values = wrapper.forecast_quantiles(ctx, horizon=h, seed=seed)
        return COMMON_GRID.copy(), interp_quantiles_to_grid(levels, values, COMMON_GRID)

    return _f


def run_dev(tsfm_fn, classical_fn) -> float:
    """Calibrate the widening factor on DEV shifts. Returns q_hat."""
    guard = ShiftGuard(config=GuardConfig(), classifier=heuristic_classifier)
    residuals: list[float] = []
    halfwidths: list[float] = []
    for shift in DEV_SHIFTS:
        for sev in DEV_SEVS:
            for seed in SEEDS:
                series = make_series(shift, sev, seed)
                rate = np.asarray(series["attrition_rate"], dtype=float)
                cov = make_covariate(shift, ONSET, seed)
                events = make_events(shift, ONSET)
                for origin in POST_ORIGINS:
                    actual = rate[origin : origin + HORIZON]
                    res = guard.guard(
                        rate, cov, events, origin, tsfm_fn, classical_fn, HORIZON, seed
                    )
                    lv, vals = res.levels, res.values
                    i50 = int(np.argmin(np.abs(lv - 0.5)))
                    i10 = int(np.argmin(np.abs(lv - 0.10)))
                    i90 = int(np.argmin(np.abs(lv - 0.90)))
                    med = vals[i50]
                    hw = (vals[i90] - vals[i10]) / 2.0
                    for t in range(HORIZON):
                        residuals.append(abs(float(actual[t]) - float(med[t])))
                        halfwidths.append(float(hw[t]))
    q_hat = calibrate_widening(np.array(residuals), np.array(halfwidths), alpha=0.1)
    print(f"dev: calibrated widen_factor={q_hat:.3f} on {len(residuals)} residuals")
    return q_hat


def run_eval(
    tsfm_fn,
    classical_fn,
    widen_factor: float,
    shifts,
    sevs,
    origins,
    tag: str,
    out_path: Path,
    resume: bool = False,
) -> list[dict]:
    """Run guard+TSFM vs TSFM-alone; placebo-controlled rows.

    Results are checkpointed per combo (atomic write) so VM recycles lose
    nothing; ``resume=True`` skips combos already in the output file.
    Returns ONLY this call's rows (no file merging — the caller combines).
    """
    guard = ShiftGuard(
        config=GuardConfig(), classifier=heuristic_classifier, widen_factor=widen_factor
    )
    combos = [(sh, sv, sd) for sh in shifts for sv in sevs for sd in SEEDS]
    done: set[tuple[str, float, int, str]] = set()
    prior_rows: list[dict] = []
    if resume and out_path.exists():
        try:
            prior_rows = json.loads(out_path.read_text())["rows"]
            done = {
                (r["shift"], float(r["severity"]), int(r["seed"]), r["tag"]) for r in prior_rows
            }
            print(f"resuming {tag}: {len(done)} combos already done")
        except Exception as e:  # noqa: BLE001
            print(f"resume unreadable ({e}); starting fresh")

    def _save(new_rows: list[dict]) -> None:
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rows": prior_rows + new_rows}, indent=1))
        tmp.replace(out_path)

    rows: list[dict] = []
    t0 = time.time()
    for i, (shift, sev, seed) in enumerate(combos):
        if (shift, sev, seed, tag) in done:
            continue
        combo_rows: list[dict] = []
        series = make_series(shift, sev, seed)
        placebo = make_placebo(shift, seed)
        cov = make_covariate(shift, ONSET, seed)
        events = make_events(shift, ONSET)
        for origin in origins:
            for label, ser in (("shifted", series), ("placebo", placebo)):
                rate = np.asarray(ser["attrition_rate"], dtype=float)
                actual = rate[origin : origin + HORIZON]
                ctx = rate[: origin + 1]
                # Guard+TSFM.
                t1 = time.time()
                try:
                    g = guard.guard(ctx, cov, events, origin, tsfm_fn, classical_fn, HORIZON, seed)
                    crps_g = crps_from_quantiles(g.levels, g.values, actual)
                    status_g, err_g, tmpl = "ok", "", g.template.value
                except Exception as e:  # noqa: BLE001
                    crps_g, status_g, err_g, tmpl = float("nan"), "error", str(e)[:200], "error"
                # TSFM-alone.
                try:
                    lv_a, vals_a = tsfm_fn(ctx, HORIZON, seed)
                    crps_a = crps_from_quantiles(lv_a, vals_a, actual)
                    status_a, err_a = "ok", ""
                except Exception as e:  # noqa: BLE001
                    crps_a, status_a, err_a = float("nan"), "error", str(e)[:200]
                combo_rows.append(
                    {
                        "tag": tag,
                        "shift": shift,
                        "severity": sev,
                        "seed": seed,
                        "origin": origin,
                        "series": label,
                        "crps_guard": crps_g,
                        "crps_alone": crps_a,
                        "template": tmpl,
                        "status": status_g if status_g == "ok" else status_a,
                        "error": err_g or err_a,
                        "elapsed_s": round(time.time() - t1, 2),
                    }
                )
        rows.extend(combo_rows)
        _save(rows)
        print(
            f"  [{tag}] {i + 1}/{len(combos)} combos, {time.time() - t0:.0f}s elapsed", flush=True
        )
    return rows


def analyze(rows: list[dict]) -> dict:
    """Primary claim + do-no-harm gate. Returns a summary dict."""
    import pandas as pd

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    ok["diff"] = ok["crps_guard"] - ok["crps_alone"]  # negative = guard wins
    summary: dict = {}

    # Primary: post-shift origins on eval shifts, shifted series.
    post = ok[(ok["tag"] == "eval_post") & (ok["series"] == "shifted")]
    d = post["diff"].to_numpy()
    d = d[np.isfinite(d)]
    t, p = stats.ttest_1samp(d, 0.0)
    summary["primary"] = {
        "n": len(d),
        "mean_diff": float(np.mean(d)),
        "rel_improvement": float(-np.mean(d) / np.mean(post["crps_alone"])),
        "paired_t": float(t),
        "paired_p": float(p),
        "win_rate": float(np.mean(d < 0)),
        "verdict": "PASS" if p < 0.05 and np.mean(d) < 0 else "FAIL",
    }

    # Do-no-harm gate 1: pre-shift origins (guard must not hurt before shifts).
    pre = ok[(ok["tag"] == "eval_pre") & (ok["series"] == "shifted")]
    dp = pre["diff"].to_numpy()
    dp = dp[np.isfinite(dp)]
    tp, pp = stats.ttest_1samp(dp, 0.0)
    summary["do_no_harm_pre"] = {
        "n": len(dp),
        "mean_diff": float(np.mean(dp)),
        "paired_p": float(pp),
        "verdict": "PASS" if not (pp < 0.05 and np.mean(dp) > 0) else "FAIL",
    }

    # Do-no-harm gate 2: no-break shifts S4/S5 (guard must not fire harmfully).
    # S5 is in eval; S4 was dev-only, so check S5 here.
    s5 = ok[(ok["tag"] == "eval_post") & (ok["series"] == "shifted") & (ok["shift"] == "S5")]
    ds = s5["diff"].to_numpy()
    ds = ds[np.isfinite(ds)]
    ts, ps = stats.ttest_1samp(ds, 0.0)
    summary["do_no_harm_s5"] = {
        "n": len(ds),
        "mean_diff": float(np.mean(ds)),
        "paired_p": float(ps),
        "verdict": "PASS" if not (ps < 0.05 and np.mean(ds) > 0) else "FAIL",
    }

    # Per-shift breakdown (eval, post, shifted).
    for shift in EVAL_SHIFTS:
        g = ok[(ok["tag"] == "eval_post") & (ok["series"] == "shifted") & (ok["shift"] == shift)]
        dg = g["diff"].to_numpy()
        dg = dg[np.isfinite(dg)]
        if len(dg):
            tg, pg = stats.ttest_1samp(dg, 0.0)
            summary[f"per_shift_{shift}"] = {
                "n": len(dg),
                "mean_diff": float(np.mean(dg)),
                "rel_improvement": float(-np.mean(dg) / np.mean(g["crps_alone"])),
                "paired_p": float(pg),
                "templates": g["template"].value_counts().to_dict(),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--out", default="experiments/exp68_results.json")
    ap.add_argument("--resume", action="store_true", help="skip eval combos already in --out")
    ap.add_argument(
        "--skip-dev",
        action="store_true",
        help="reuse widen_factor from --out instead of recalibrating",
    )
    args = ap.parse_args()

    if args.mode == "real":
        from td_one.models.tsfm import Chronos2Wrapper  # noqa: PLC0415

        weights_dir = Path(__file__).resolve().parents[1] / "models" / "weights"
        wrapper = Chronos2Wrapper(
            prediction_length=HORIZON,
            weights_dir=str(weights_dir / "chronos2"),
        )
        tsfm_fn = _real_tsfm(wrapper)
    else:
        wrapper = None
        tsfm_fn = _mock_tsfm
    classical_fn = _mock_classical

    print(f"exp68 [{args.mode}]: pre-registered holdout (dev={DEV_SHIFTS}, eval={EVAL_SHIFTS})")
    t_start = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_dev and out_path.exists():
        try:
            widen_factor = float(json.loads(out_path.read_text())["config"]["widen_factor"])
            print(f"reusing widen_factor={widen_factor:.3f} from {out_path}")
        except (KeyError, ValueError):
            print("no widen_factor in checkpoint; recalibrating")
            widen_factor = run_dev(tsfm_fn, classical_fn)
    else:
        widen_factor = run_dev(tsfm_fn, classical_fn)
    rows = run_eval(
        tsfm_fn,
        classical_fn,
        widen_factor,
        EVAL_SHIFTS,
        EVAL_SEVS,
        POST_ORIGINS,
        "eval_post",
        out_path,
        args.resume,
    )
    rows = rows + run_eval(
        tsfm_fn,
        classical_fn,
        widen_factor,
        EVAL_SHIFTS,
        EVAL_SEVS,
        PRE_ORIGINS,
        "eval_pre",
        out_path,
        args.resume,
    )
    if wrapper is not None:
        unload = getattr(wrapper, "unload", None)
        if callable(unload):
            unload()

    summary = analyze(rows)
    out = {
        "config": {
            "mode": args.mode,
            "dev_shifts": list(DEV_SHIFTS),
            "eval_shifts": list(EVAL_SHIFTS),
            "dev_severities": list(DEV_SEVS),
            "eval_severities": list(EVAL_SEVS),
            "seeds": list(SEEDS),
            "widen_factor": widen_factor,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path} in {time.time() - t_start:.0f}s")

    print("\n=== exp68 verdict ===")
    for key in ("primary", "do_no_harm_pre", "do_no_harm_s5"):
        s = summary[key]
        print(
            f"  {key}: {s['verdict']} "
            f"(mean_diff={s['mean_diff']:+.5f}, p={s['paired_p']:.3g}, n={s['n']})"
        )
    for shift in EVAL_SHIFTS:
        s = summary[f"per_shift_{shift}"]
        print(
            f"  {shift}: rel_improvement={s['rel_improvement']:+.1%} "
            f"p={s['paired_p']:.3g} templates={s['templates']}"
        )


if __name__ == "__main__":
    main()
