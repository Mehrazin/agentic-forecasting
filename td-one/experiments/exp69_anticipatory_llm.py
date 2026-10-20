"""exp69: anticipatory LLM shift adjustment vs TSFM-alone.

exp68 (reactive guard: detect after onset, truncate context) was an honest
negative (+0.5%, p=0.824). This experiment tests the *anticipatory* variant
the user proposed: the LLM reads PIT-safe news *before* the shock, and a
calibrated judgmental overlay shifts the TSFM's forecast for the horizon
steps the shock will hit.

**Pre-registration (locked before running):**
- Shifts: S1 (rate shock), S2 (policy intervention). Severities 2σ/4σ.
- Seeds: calibration 200-202, test 300-302 (disjoint from exp67/exp68).
- Origins: STRADDLE=(27,28,29) primary (window covers onset, news visible
  from month 27); NONEWS=(24,25,26) do-no-harm (window covers onset but no
  news published yet — classifier must output none); POST=(31,33) do-no-harm
  (shock already in context — overlay must not fire by construction).
- Arms: A=TSFM alone; B=TSFM+LLM overlay; C=TSFM+oracle overlay (perfect
  classification — mechanism ceiling); B_noramp (ramp off); B_fixed (fixed
  Δ=0.05 instead of calibrated buckets).
- Calibration: (shift_type, bucket) -> median level change, measured on
  calibration seeds only. The LLM classifies; statistics quantify.
- Primary claim: B beats A on straddle origins (paired t-test on CRPS,
  p<0.05). B vs C isolates LLM error cost. NONEWS/POST are do-no-harm.
- Kill: if B doesn't beat A at p<0.05, dead — no iteration.

The LLM needs synthetic news: articles announcing the upcoming shock with
article_date < onset, wired through ``articles_visible_asof`` (the honest
way to test "the LLM knew from news"). Distractor articles included so the
classifier must actually find the signal.

Usage:
    .venv/bin/python experiments/exp69_anticipatory_llm.py --mode mock   # smoke
    .venv/bin/python experiments/exp69_anticipatory_llm.py --mode real   # real
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from td_one.agent.anticipatory_adjuster import (
    AnticipatoryClassifier,
    ShiftSignal,
    apply_overlay,
    calibrate_level_shifts,
    month_label,
)
from td_one.data_gen.shift_scenarios import (
    ShiftSpec,
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
SHIFTS = ("S1", "S2")
SEVS = (2.0, 4.0)
CAL_SEEDS = (200, 201, 202)
TEST_SEEDS = (300, 301, 302)
STRADDLE_ORIGINS = (27, 28, 29)
NONEWS_ORIGINS = (24, 25, 26)
POST_ORIGINS = (31, 33)

SHIFT_TYPE_OF = {"S1": "rate", "S2": "policy"}
BUCKET_OF = {2.0: "medium", 4.0: "large"}
FIXED_DELTA = 0.05
CACHE_PATH = Path(__file__).resolve().parent / "exp69_llm_cache.json"


def make_series(shift: str, severity: float, seed: int) -> dict:
    """Shifted attrition series (same constructors as exp67/exp68)."""
    rng = np.random.default_rng(1000 + seed)
    base = generate_base_series(N_MONTHS, seed=4000 + seed)
    return apply_shift(base, ShiftSpec(shift, ONSET, severity), rng)


def make_placebo(seed: int) -> dict:
    """Unshifted control: same seeds as the shifted series."""
    return generate_base_series(N_MONTHS, seed=4000 + seed)


def make_news(shift: str, severity: float) -> list[dict[str, object]]:
    """Synthetic news feed. Shock announcements at months 27-29 (all < ONSET=30).

    article_date uses zero-padded month labels so articles_visible_asof's
    string comparison stays chronological. Distractors carry no signal.
    """
    mag = (
        "50 basis points, a moderate tightening step"
        if severity == 2.0
        else ("100 basis points, an aggressive tightening move")
    )
    if shift == "S1":
        shock_articles = [
            (
                27,
                "Analysts see Bank of Canada hike on the horizon",
                f"Economists widely expect the Bank of Canada to raise its policy "
                f"rate by {mag}, with the decision taking effect in about 3 months. "
                f"Higher borrowing costs are expected to push mortgage attrition up.",
            ),
            (
                28,
                "BoC rate hike now seen as near-certain",
                f"Market pricing implies the Bank of Canada will lift rates by {mag} "
                f"taking effect in 2 months. Lenders are bracing for a wave of "
                f"borrowers shopping for better renewal terms.",
            ),
            (
                29,
                "BoC hike confirmed for next month",
                f"The Bank of Canada confirmed a rate increase of {mag} taking "
                f"effect next month. Attrition is expected to rise as payments reset.",
            ),
        ]
    else:  # S2
        mag = (
            "targeted tightening of mortgage qualification rules"
            if severity == 2.0
            else "sweeping overhaul of federal mortgage rules"
        )
        shock_articles = [
            (
                27,
                "Ottawa signals mortgage-rule changes ahead",
                f"The federal government signalled a {mag}, expected to take "
                f"effect in about 3 months. Stricter rules are expected to push "
                f"attrition up as borrowers seek alternatives.",
            ),
            (
                28,
                "Mortgage-rule tightening details emerge",
                f"Officials fleshed out the {mag}, taking effect in 2 months. "
                f"Industry groups warn of higher borrower churn.",
            ),
            (
                29,
                "New mortgage rules take effect next month",
                f"The {mag} will take effect next month. Lenders expect "
                f"attrition to rise as affected borrowers refinance elsewhere.",
            ),
        ]
    distractors = [
        (20, "Housing starts edge higher in Ontario", "Builders broke ground on more units."),
        (22, "Provincial budget includes buyer credit", "A new first-time buyer tax credit."),
        (24, "Oil prices steady as OPEC holds output", "Energy markets were calm."),
        (26, "Condo sales cool in Toronto", "Resale volumes dipped slightly."),
        (29, "Farm belt harvest outlook improves", "Crop forecasts were revised up."),
    ]
    articles: list[dict[str, object]] = []
    for m, title, body in shock_articles:
        articles.append(
            {"title": title, "body": body, "article_date": month_label(m), "source": "synth"}
        )
    for m, title, body in distractors:
        articles.append(
            {"title": title, "body": body, "article_date": month_label(m), "source": "synth"}
        )
    # PIT invariant: every article predates the shock.
    assert all(int(str(a["article_date"])[1:]) < ONSET for a in articles)
    return articles


class MockNewsClient:
    """Keyword mock for --mode mock.

    Anchors relative time phrases to the article's publication month,
    mirroring the real prompt's instructions.
    """

    def complete(self, prompt: str) -> str:
        import re

        def _none(reason: str) -> str:
            return json.dumps(
                {
                    "shift_type": "none",
                    "direction": "none",
                    "magnitude_bucket": "none",
                    "months_from_now": None,
                    "timing_confidence": "low",
                    "rationale": reason,
                }
            )

        m_origin = re.search(r"Today is month (M\d+)", prompt)
        origin = int(m_origin.group(1)[1:]) if m_origin else 0
        # Find shock articles: lines like "[M028] title: body".
        best = None  # (article_month, lag, bucket, stype)
        for m in re.finditer(r"\[(M\d+)\] ([^:]+): ([^\n]+)", prompt):
            amonth = int(m.group(1)[1:])
            body = m.group(3).lower()
            shock = "bank of canada" in body or "mortgage" in body
            if not shock:
                continue
            if "100 basis points" in body or "sweeping overhaul" in body:
                bucket = "large"
            elif "50 basis points" in body or "targeted" in body:
                bucket = "medium"
            else:
                continue
            stype = "rate" if "bank of canada" in body else "policy"
            if "3 months" in body:
                lag = 3
            elif "2 months" in body:
                lag = 2
            elif "next month" in body:
                lag = 1
            else:
                continue
            if best is None or amonth > best[0]:
                best = (amonth, lag, bucket, stype)
        if best is None:
            return _none("no shock announced")
        amonth, lag, bucket, stype = best
        onset = amonth + lag
        mfn = onset - origin
        if mfn < 0:
            return _none("announced shock already took effect")
        return json.dumps(
            {
                "shift_type": stype,
                "direction": "up",
                "magnitude_bucket": bucket,
                "months_from_now": mfn,
                "timing_confidence": "high",
                "rationale": "mock: shock announced",
            }
        )


class CachedClassifier:
    """AnticipatoryClassifier with a disk cache (one LLM call per unique prompt)."""

    def __init__(self, classifier: AnticipatoryClassifier, cache_path: Path):
        self._clf = classifier
        self._path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self.llm_calls = 0

    def _key(self, articles: list[dict[str, object]], origin_label: str) -> str:
        h = hashlib.sha256()
        h.update(origin_label.encode())
        for a in articles:
            h.update(str(a.get("article_date", "")).encode())
            h.update(str(a.get("title", "")).encode())
            h.update(str(a.get("body", "")).encode())
        return h.hexdigest()

    def __call__(
        self, articles: list[dict[str, object]], origin: int, origin_label: str
    ) -> ShiftSignal:
        key = self._key(articles, origin_label)
        if key in self._cache:
            d = self._cache[key]
            return ShiftSignal(
                d["shift_type"],
                d["direction"],
                d["magnitude_bucket"],
                d["months_from_now"],
                d["timing_confidence"],
                d.get("rationale", ""),
            )
        sig = self._clf(articles, origin, origin_label)
        self._cache[key] = {
            "shift_type": sig.shift_type,
            "direction": sig.direction,
            "magnitude_bucket": sig.magnitude_bucket,
            "months_from_now": sig.months_from_now,
            "timing_confidence": sig.timing_confidence,
            "rationale": sig.rationale,
        }
        self.llm_calls += 1
        with contextlib.suppress(OSError):
            self._path.write_text(json.dumps(self._cache, indent=1))
        return sig


def _mock_tsfm(ctx: np.ndarray, h: int, seed: int):
    rng = np.random.default_rng(seed)
    mu = float(np.mean(ctx[-6:]))
    vals = mu + np.linspace(-0.02, 0.02, 19)[:, None] + 0.001 * rng.standard_normal((19, h))
    return COMMON_GRID.copy(), np.sort(vals, axis=0)


def _real_tsfm(wrapper):
    def _f(ctx: np.ndarray, h: int, seed: int):
        levels, values = wrapper.forecast_quantiles(ctx, horizon=h, seed=seed)
        return COMMON_GRID.copy(), interp_quantiles_to_grid(levels, values, COMMON_GRID)

    return _f


def run_calibration() -> dict[tuple[str, str], float]:
    """Measure (shift_type, bucket) -> level change on calibration seeds."""
    episodes: list[tuple[str, str, np.ndarray, int]] = []
    for shift in SHIFTS:
        for sev in SEVS:
            for seed in CAL_SEEDS:
                rate = np.asarray(make_series(shift, sev, seed)["attrition_rate"], dtype=float)
                episodes.append((SHIFT_TYPE_OF[shift], BUCKET_OF[sev], rate, ONSET))
    cal = calibrate_level_shifts(episodes)
    print("calibration (level change in rate space):")
    for k in sorted(cal):
        print(f"  {k}: {cal[k]:+.4f}")
    return cal


def oracle_signal(shift: str, sev: float, origin: int) -> ShiftSignal:
    """Perfect classification: the mechanism ceiling (arm C)."""
    return ShiftSignal(SHIFT_TYPE_OF[shift], "up", BUCKET_OF[sev], ONSET - origin, "high", "oracle")


def run_eval(
    tsfm_fn,
    calibration: dict[tuple[str, str], float],
    classifier: CachedClassifier,
    out_path: Path,
    resume: bool = False,
) -> list[dict]:
    """Run arms A/B/C/B_noramp/B_fixed; checkpointed per combo."""
    combos = [(sh, sv, sd) for sh in SHIFTS for sv in SEVS for sd in TEST_SEEDS]
    done: set[tuple[str, float, int]] = set()
    prior_rows: list[dict] = []
    if resume and out_path.exists():
        try:
            prior_rows = json.loads(out_path.read_text())["rows"]
            done = {(r["shift"], float(r["severity"]), int(r["seed"])) for r in prior_rows}
            print(f"resuming: {len(done)} combos already done")
        except Exception as e:  # noqa: BLE001
            print(f"resume unreadable ({e}); starting fresh")

    def _save(new_rows: list[dict]) -> None:
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rows": prior_rows + new_rows}, indent=1))
        tmp.replace(out_path)

    rows: list[dict] = []
    t0 = time.time()
    fixed_cal = dict.fromkeys(calibration, FIXED_DELTA)
    for i, (shift, sev, seed) in enumerate(combos):
        if (shift, sev, seed) in done:
            continue
        combo_rows: list[dict] = []
        series = make_series(shift, sev, seed)
        placebo = make_placebo(seed)
        news = make_news(shift, sev)
        for origin_class, origins in (
            ("straddle", STRADDLE_ORIGINS),
            ("nonews", NONEWS_ORIGINS),
            ("post", POST_ORIGINS),
        ):
            for origin in origins:
                for label, ser in (("shifted", series), ("placebo", placebo)):
                    rate = np.asarray(ser["attrition_rate"], dtype=float)
                    actual = rate[origin : origin + HORIZON]
                    ctx = rate[: origin + 1]
                    olabel = month_label(origin)
                    # Baseline TSFM (shared by all arms).
                    try:
                        lv_a, vals_a = tsfm_fn(ctx, HORIZON, seed)
                        crps_a = crps_from_quantiles(lv_a, vals_a, actual)
                    except Exception as e:  # noqa: BLE001
                        combo_rows.append(
                            {
                                "shift": shift,
                                "severity": sev,
                                "seed": seed,
                                "origin": origin,
                                "origin_class": origin_class,
                                "series": label,
                                "status": "error",
                                "error": f"tsfm: {str(e)[:150]}",
                            }
                        )
                        continue
                    arms: dict[str, float] = {"A": crps_a}
                    sig_dict: dict[str, object] = {}
                    if label == "shifted":
                        # Arm B: LLM classification from PIT-safe news.
                        sig = classifier(news, origin, olabel)
                        sig_dict = {
                            "shift_type": sig.shift_type,
                            "direction": sig.direction,
                            "magnitude_bucket": sig.magnitude_bucket,
                            "months_from_now": sig.months_from_now,
                            "timing_confidence": sig.timing_confidence,
                        }
                        arms["B"] = crps_from_quantiles(
                            lv_a, apply_overlay(lv_a, vals_a, sig, origin, calibration), actual
                        )
                        # Arm C: oracle classification (mechanism ceiling).
                        arms["C"] = crps_from_quantiles(
                            lv_a,
                            apply_overlay(
                                lv_a,
                                vals_a,
                                oracle_signal(shift, sev, origin),
                                origin,
                                calibration,
                            ),
                            actual,
                        )
                        # Ablation: ramp off.
                        arms["B_noramp"] = crps_from_quantiles(
                            lv_a,
                            apply_overlay(lv_a, vals_a, sig, origin, calibration, use_ramp=False),
                            actual,
                        )
                        # Ablation: fixed delta instead of calibrated buckets.
                        arms["B_fixed"] = crps_from_quantiles(
                            lv_a,
                            apply_overlay(lv_a, vals_a, sig, origin, fixed_cal),
                            actual,
                        )
                    else:
                        # Placebo world: no shock news exists -> all arms = A.
                        arms.update(
                            {"B": crps_a, "C": crps_a, "B_noramp": crps_a, "B_fixed": crps_a}
                        )
                    combo_rows.append(
                        {
                            "shift": shift,
                            "severity": sev,
                            "seed": seed,
                            "origin": origin,
                            "origin_class": origin_class,
                            "series": label,
                            "status": "ok",
                            "error": "",
                            **{f"crps_{k}": v for k, v in arms.items()},
                            "signal": sig_dict,
                        }
                    )
        rows.extend(combo_rows)
        _save(rows)
        print(
            f"  [{i + 1}/{len(combos)}] {shift}/{sev}/seed{seed}, {time.time() - t0:.0f}s",
            flush=True,
        )
    return rows


def analyze(rows: list[dict]) -> dict:
    """Pre-registered primary claim + do-no-harm gates + ablations."""
    import pandas as pd

    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    out: dict = {}

    def _test(frame, arm_x: str, arm_y: str, name: str, harm_only: bool = False) -> None:
        d = (frame[f"crps_{arm_x}"] - frame[f"crps_{arm_y}"]).to_numpy()
        d = d[np.isfinite(d)]
        if len(d) < 3:
            out[name] = {"n": len(d), "verdict": "insufficient"}
            return
        base = float(np.mean(frame["crps_A"]))
        entry: dict[str, object] = {
            "n": len(d),
            "mean_diff": float(np.mean(d)),
            "rel_improvement": float(-np.mean(d) / base) if base else 0.0,
            "win_rate": float(np.mean(d < 0)),
        }
        if np.all(d == 0):
            # No variance: arms identical everywhere. For do-no-harm this is
            # the ideal outcome; for a superiority claim it is a failure.
            entry["paired_p"] = 1.0
            entry["verdict"] = "PASS" if harm_only else "FAIL"
            out[name] = entry
            return
        t, p = stats.ttest_1samp(d, 0.0)
        entry["paired_p"] = float(p)
        if harm_only:
            entry["verdict"] = "PASS" if not (p < 0.05 and np.mean(d) > 0) else "FAIL"
        else:
            entry["verdict"] = "PASS" if (p < 0.05 and np.mean(d) < 0) else "FAIL"
        out[name] = entry

    sh = df[(df["series"] == "shifted")]
    # Primary: B vs A on straddle origins.
    _test(sh[sh["origin_class"] == "straddle"], "B", "A", "primary_B_vs_A_straddle")
    # Mechanism ceiling: C vs A.
    _test(sh[sh["origin_class"] == "straddle"], "C", "A", "ceiling_C_vs_A_straddle")
    # LLM error cost: B vs C.
    _test(sh[sh["origin_class"] == "straddle"], "B", "C", "llm_error_B_vs_C_straddle")
    # Ablations.
    _test(sh[sh["origin_class"] == "straddle"], "B_noramp", "A", "ablation_noramp_vs_A")
    _test(sh[sh["origin_class"] == "straddle"], "B_fixed", "A", "ablation_fixed_vs_A")
    # Do-no-harm: nonews origins (classifier must not hallucinate).
    _test(sh[sh["origin_class"] == "nonews"], "B", "A", "do_no_harm_nonews", harm_only=True)
    # Do-no-harm: post origins (overlay must not fire).
    _test(sh[sh["origin_class"] == "post"], "B", "A", "do_no_harm_post", harm_only=True)
    # Signal audit: what did the LLM output at straddle origins?
    sig = sh[(sh["origin_class"] == "straddle") & (sh["series"] == "shifted")]["signal"]
    out["signal_audit"] = {
        "n_actionable": int(
            sum(1 for s in sig if isinstance(s, dict) and s.get("shift_type") not in (None, "none"))
        ),
        "n_total": int(len(sig)),
        "buckets": pd.Series([s.get("magnitude_bucket") for s in sig if isinstance(s, dict)])
        .value_counts()
        .to_dict(),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--out", default="experiments/exp69_results.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.mode == "real":
        from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: PLC0415
        from td_one.models.tsfm import Chronos2Wrapper  # noqa: PLC0415

        weights_dir = Path(__file__).resolve().parents[1] / "models" / "weights"
        wrapper = Chronos2Wrapper(
            prediction_length=HORIZON, weights_dir=str(weights_dir / "chronos2")
        )
        tsfm_fn = _real_tsfm(wrapper)
        llm_client: object = VectorProxyClient(model="flash-lite", call_budget=200)
    else:
        wrapper = None
        tsfm_fn = _mock_tsfm
        llm_client = MockNewsClient()

    print(f"exp69 [{args.mode}]: anticipatory LLM overlay (pre-registered)")
    t_start = time.time()
    calibration = run_calibration()
    classifier = CachedClassifier(
        AnticipatoryClassifier(llm_client),  # type: ignore[arg-type]
        CACHE_PATH,
    )
    rows = run_eval(tsfm_fn, calibration, classifier, Path(args.out), args.resume)
    summary = analyze(rows)
    print(f"\nLLM calls this run: {classifier.llm_calls} (cached: {len(classifier._cache)})")
    print(json.dumps(summary, indent=1, default=str))
    out = {
        "config": {
            "mode": args.mode,
            "calibration": {f"{k[0]}/{k[1]}": v for k, v in calibration.items()},
        },
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nexp69 [{args.mode}] done in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
