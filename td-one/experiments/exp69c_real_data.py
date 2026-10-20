"""exp69c: real-data anticipatory LLM shift adjustment (pre-registered).

exp69 failed on synthetic data — and the *oracle* failed too, so the
mechanism itself was the limit. exp69b showed broad synthesis blurs
timing. The user's counter-hypothesis: real shocks have richer, uneven,
real textual buildup that a single LLM call on a bag of chunks cannot
exploit. exp69c tests the anticipatory idea ON REAL DATA with better
scaffolding:

  1. Real shock episodes, real series: the 2022 BoC hiking cycle
     (first hike 2022-03; 100bp super-hike 2022-07), the 2021 inflation
     build, the 2020 COVID shock — on Food CPI YoY inflation (2010-2026)
     and Freddie Mac SMM attrition (2021-2026). News: the real
     ``data/news/news_2026.db``, PIT-safe by ``pub_date``.
  2. Timing reminders: the classifier gets explicit timeline anchors
     ("today is X, the forecast window is Y..Z, key dated events: ...").
  3. News as story: a chronological narrative (ordered dated events +
     "what's building") instead of a bag of chunks.
  4. Multi-call pipeline: retrieve (semantic RAG) -> rank (LLM, top-12
     with reasons) -> story (LLM) -> classify (LLM, ShiftSignal).

Arms: A = Chronos-2 alone; B = A + multi-call overlay (with timeline);
B_notl = A + multi-call overlay WITHOUT timeline anchors (ablation);
C = A + single-call overlay (top-30 chunks -> AnticipatoryClassifier,
replicates exp69b on real data; isolates the scaffolding effect).

Calibration: per (series, test origin), from past episodes only
(onset+6 <= origin, strict PIT). Table maps magnitude_bucket ->
|median level change| (absolute: the LLM owns direction, statistics own
size — the signed version would flip an opposite-direction test the
wrong way; see tests/test_story_pipeline.py).

Pre-registered kill (locked in experiments/exp69c_report.md BEFORE the
real run): B beats A at p<0.05 on pooled straddle origins AND B beats C
at p<0.05. Per the standing researcher-mindset instruction, a triggered
kill is a tripwire for investigation, not an automatic execution: the
script records per-origin diagnostics (firing rate, direction accuracy,
timing error, size ratio) and tests at least one variant addressing the
dominant diagnosed failure, all from stored artifacts (no new LLM calls).

Usage:
    .venv/bin/python experiments/exp69c_real_data.py --mode mock   # smoke
    .venv/bin/python experiments/exp69c_real_data.py --mode real   # real
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.anticipatory_adjuster import (  # noqa: E402
    NULL_SIGNAL,
    SHIFT_TYPES,
    AnticipatoryClassifier,
    ShiftSignal,
    apply_overlay,
    calibrate_level_shifts,
)
from td_one.agent.story_pipeline import StoryPipeline  # noqa: E402
from td_one.evaluation.real_benchmark import (  # noqa: E402
    COMMON_GRID,
    crps_from_quantiles,
    interp_quantiles_to_grid,
)
from td_one.models.tsfm import Chronos2Wrapper  # noqa: E402
from td_one.rag.semantic import SemanticNewsIndex  # noqa: E402

HORIZON = 6
CACHE_JSON = ROOT / "experiments" / "exp69c_llm_cache.json"


def results_path(mode: str) -> Path:
    name = "exp69c_results.json" if mode == "real" else "exp69c_results_mock.json"
    return ROOT / "experiments" / name

SERIES_CFG = {
    "food_cpi": {
        "target_desc": (
            "Canada Food CPI year-over-year inflation rate "
            "(fraction, e.g. 0.05 = 5%)"
        ),
        "question": (
            "Will food-price inflation in Canada shift materially "
            "in the next 6 months?"
        ),
        "queries": [
            "Bank of Canada interest rate hike inflation outlook",
            "food prices grocery inflation Canada rising",
            "central bank monetary policy tightening",
        ],
    },
    "smm": {
        "target_desc": (
            "Freddie Mac single-month mortgage prepayment "
            "(attrition) rate (fraction)"
        ),
        "question": (
            "Will mortgage prepayment/attrition shift materially "
            "in the next 6 months?"
        ),
        "queries": [
            "Bank of Canada interest rate decision mortgage rates",
            "mortgage refinancing prepayment rates outlook",
            "central bank rate cut housing market borrowing",
        ],
    },
}

#: (id, series, onset month, role)
CAL_EPISODES = [
    {"id": "C1a", "series": "food_cpi", "onset": "2020-03"},  # COVID (sparse news)
    {"id": "C1b", "series": "food_cpi", "onset": "2021-10"},  # inflation build
    {"id": "C2", "series": "smm", "onset": "2022-03"},  # first hike
]
TEST_EPISODES = [
    {
        "id": "T1",
        "series": "food_cpi",
        "onset": "2022-03",
        "straddle": ["2021-11", "2021-12", "2022-01", "2022-02"],
    },
    {
        "id": "T2",
        "series": "food_cpi",
        "onset": "2022-07",
        "straddle": ["2022-03", "2022-04", "2022-05", "2022-06"],
    },
    {
        "id": "T3",
        "series": "smm",
        "onset": "2024-06",
        "straddle": ["2024-02", "2024-03", "2024-04", "2024-05"],
    },
]
PLACEBO_ORIGINS = {
    "food_cpi": ["2018-06", "2018-07"],
    "smm": ["2023-06", "2023-07"],
}


# ----------------------------------------------------------------------------
# Month helpers
# ----------------------------------------------------------------------------
def add_months(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    m += k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def month_end_ymd(ym: str) -> str:
    last = calendar.monthrange(int(ym[:4]), int(ym[5:7]))[1]
    return f"{ym}-{last:02d}"


def before_ymd(ym: str) -> str:
    """PIT cutoff (exclusive): first day of the month after the origin."""
    return add_months(ym, 1) + "-01"


# ----------------------------------------------------------------------------
# Series loading
# ----------------------------------------------------------------------------
def load_food_cpi() -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(
        ROOT / "benchmarks" / "food_cpi" / "data.csv", parse_dates=["date"]
    ).sort_values("date")
    v = df["value"].to_numpy(dtype=float)
    yoy = v[12:] / v[:-12] - 1.0
    months = df["date"].dt.strftime("%Y-%m").tolist()[12:]
    # YoY can dip marginally negative (2016-10: -0.38%); the overlay still
    # clips forecasts to [0, 1] and Chronos runs with validate_input=False.
    assert bool((yoy > -0.05).all()) and bool((yoy <= 1).all())
    return months, yoy


def load_smm() -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv")
    df = df.sort_values("period").reset_index(drop=True)
    months = [
        f"{int(p) // 100:04d}-{int(p) % 100:02d}" for p in df["period"].tolist()
    ]
    vals = df["smm"].to_numpy(dtype=float)
    assert bool((vals >= 0).all()) and bool((vals <= 1).all())
    return months, vals


# ----------------------------------------------------------------------------
# LLM clients
# ----------------------------------------------------------------------------
class CachingClient:
    """Disk-cached LLM client (resume-safe; one call per unique prompt)."""

    def __init__(self, inner: object, path: Path):
        self._inner = inner
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self.llm_calls = 0

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._cache))

    def complete(self, prompt: str, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._inner.complete(prompt, **kwargs)  # type: ignore[union-attr]
            self.llm_calls += 1
            if self.llm_calls % 5 == 0:
                self._save()
        return self._cache[key]


_NONE_JSON = (
    '{"shift_type": "none", "direction": "none", '
    '"magnitude_bucket": "none", "months_from_now": null, '
    '"timing_confidence": "low", "rationale": "mock quiet", '
    '"driving_trends": []}'
)
_ACTION_JSON = (
    '{"shift_type": "rate", "direction": "up", '
    '"magnitude_bucket": "large", "months_from_now": 2, '
    '"timing_confidence": "high", "rationale": "mock shock", '
    '"driving_trends": ["mock hike expected"]}'
)


class MockLLMClient:
    """Deterministic mock: actionable on straddle months, none elsewhere."""

    def __init__(self, straddle_months: set[str]):
        self._straddle = straddle_months
        self.llm_calls = 0

    def _fire(self, prompt: str) -> bool:
        return any(m in prompt for m in self._straddle)

    def complete(self, prompt: str, **kwargs: object) -> str:  # type: ignore[no-untyped-def]
        self.llm_calls += 1
        if "ranking news articles" in prompt:
            return (
                '{"ranked": [{"idx": 1, "score": 0.9, "reason": "mock"},'
                '{"idx": 2, "score": 0.1, "reason": "mock"}]}'
            )
        if "chronological story" in prompt:
            return (
                '{"timeline": [{"date": "2022-01-26", "event": "mock event"}],'
                '"narrative": "mock narrative.", "building": "mock building."}'
            )
        return _ACTION_JSON if self._fire(prompt) else _NONE_JSON


# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------
def tsfm_forecast(
    wrapper: Chronos2Wrapper, ctx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    levels, values = wrapper.forecast_quantiles(
        ctx, horizon=HORIZON, seed=seed, validate_input=False
    )
    return COMMON_GRID.copy(), interp_quantiles_to_grid(levels, values, COMMON_GRID)


def realized_change(ctx_last: float, actuals: np.ndarray) -> float:
    """Signed level change over the window (for direction diagnosis)."""
    return float(np.mean(actuals) - ctx_last)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    series: dict[str, tuple[list[str], np.ndarray]] = {
        "food_cpi": load_food_cpi(),
        "smm": load_smm(),
    }
    pos = {
        name: {m: i for i, m in enumerate(months)}
        for name, (months, _) in series.items()
    }

    straddle_months = {m for e in TEST_EPISODES for m in e["straddle"]}
    if args.mode == "real":
        from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: PLC0415

        weights = ROOT / "models" / "weights" / "chronos2"
        wrapper = Chronos2Wrapper(prediction_length=HORIZON, weights_dir=str(weights))
        llm: object = CachingClient(
            VectorProxyClient(model="flash-lite", call_budget=400), CACHE_JSON
        )
        index: object = SemanticNewsIndex()
    else:
        wrapper = Chronos2Wrapper(prediction_length=HORIZON, use_mock=True)
        llm = MockLLMClient(straddle_months)
        index = None

    print(f"exp69c [{args.mode}]: real-data anticipatory overlay (pre-registered)")
    t_start = time.time()

    out_json = results_path(args.mode)
    rows: list[dict] = []
    if args.resume and out_json.exists():
        rows = json.loads(out_json.read_text())["rows"]
        print(f"resumed with {len(rows)} rows")
    done = {
        (r["episode"], r["origin"], r.get("origin_class", "straddle")) for r in rows
    }

    def get_pipeline(name: str) -> StoryPipeline:
        cfg = SERIES_CFG[name]
        return StoryPipeline(
            client=llm,
            index=index,
            queries=cfg["queries"],
            question=cfg["question"],
            target_desc=cfg["target_desc"],
            horizon=HORIZON,
        )

    # ---- Calibration: classify each cal episode once (origin = onset-1) ----
    cal_signals: dict[str, ShiftSignal] = {}
    cal_values: dict[str, dict[str, float]] = {}  # ep_id -> {bucket: |delta|}
    for ep in CAL_EPISODES:
        name, onset = ep["series"], ep["onset"]
        months, vals = series[name]
        if args.mode == "real":
            pipe = get_pipeline(name)
            sig, dbg = pipe.run(
                before_ymd(add_months(onset, -1)),
                month_end_ymd(add_months(onset, -1)),
                onset,
                add_months(onset, 5),
                use_timeline=True,
            )
            print(
                f"  cal {ep['id']} ({name} {onset}): {sig.shift_type}/"
                f"{sig.direction}/{sig.magnitude_bucket} "
                f"mfn={sig.months_from_now} conf={sig.timing_confidence} "
                f"trends={list(sig.driving_trends)}"
            )
        else:
            sig = ShiftSignal("rate", "up", "large", 2, "high", "mock")
        cal_signals[ep["id"]] = sig
        if sig.is_actionable:
            raw = calibrate_level_shifts(
                [(sig.shift_type, sig.magnitude_bucket, vals, pos[name][onset])],
                absolute=True,
            )
            # Bucket-keyed (cross-type transfer): expand over all types so
            # apply_overlay's (type, bucket) lookup always hits.
            bval = float(np.median(list(raw.values())))
            cal_values[ep["id"]] = {sig.magnitude_bucket: bval}
            print(f"    |delta| = {bval:.5f} (bucket {sig.magnitude_bucket})")
        else:
            cal_values[ep["id"]] = {}
            print("    no signal -> contributes nothing (honest)")

    def cal_table(name: str, origin_ym: str) -> dict[tuple[str, str], float]:
        """Bucket table from cal episodes with onset+6 <= origin (PIT)."""
        o_pos = pos[name][origin_ym]
        bucket_vals: dict[str, list[float]] = {}
        for ep in CAL_EPISODES:
            if ep["series"] != name:
                continue
            if pos[name][ep["onset"]] + 6 > o_pos:
                continue  # post window not yet observed
            for b, v in cal_values[ep["id"]].items():
                bucket_vals.setdefault(b, []).append(v)
        table: dict[tuple[str, str], float] = {}
        for b, vs in bucket_vals.items():
            v = float(np.median(vs))
            for st in SHIFT_TYPES:
                table[(st, b)] = v
        return table

    def eval_origin(
        name: str,
        origin_ym: str,
        episode_id: str,
        origin_class: str,
        run_notl: bool,
    ) -> None:
        months, vals = series[name]
        o_pos = pos[name][origin_ym]
        ctx = vals[: o_pos + 1]
        actuals = vals[o_pos + 1 : o_pos + 1 + HORIZON]
        assert len(actuals) == HORIZON, f"short window at {name} {origin_ym}"
        cfg = SERIES_CFG[name]
        table = cal_table(name, origin_ym)

        seed = 1000 + o_pos
        _, q_a = tsfm_forecast(wrapper, ctx, seed)
        crps_a = crps_from_quantiles(COMMON_GRID, q_a, actuals)

        row: dict = {
            "episode": episode_id,
            "series": name,
            "origin": origin_ym,
            "origin_class": origin_class,
            "crps_A": crps_a,
            "q_A": q_a.tolist(),  # stored so variants re-score w/o Chronos
            "realized_change": realized_change(float(ctx[-1]), actuals),
            "cal_table": {f"{k[0]}/{k[1]}": v for k, v in table.items()},
        }

        if args.mode == "real":
            pipe = get_pipeline(name)
            b_ymd, o_ymd = before_ymd(origin_ym), month_end_ymd(origin_ym)
            ws, we = add_months(origin_ym, 1), add_months(origin_ym, HORIZON)
            sig_b, dbg_b = pipe.run(b_ymd, o_ymd, ws, we, use_timeline=True)
            signals: dict[str, ShiftSignal] = {"B": sig_b}
            if run_notl:
                sig_n, _ = pipe.run(b_ymd, o_ymd, ws, we, use_timeline=False)
                signals["B_notl"] = sig_n
            # Arm C: single call on top-30 retrieved chunks.
            from td_one.agent.story_pipeline import (  # noqa: PLC0415
                retrieve_candidates,
            )

            cands = retrieve_candidates(cfg["queries"], b_ymd, index)[:30]
            arts = [
                {
                    "article_date": c["pub_date"],
                    "title": c["title"],
                    "body": c.get("body", ""),
                }
                for c in cands
            ]
            clf = AnticipatoryClassifier(llm, target_desc=cfg["target_desc"])  # type: ignore[arg-type]
            signals["C"] = clf(arts, 0, o_ymd)
            row["debug_B"] = {
                "n_retrieved": dbg_b.n_retrieved,
                "ranked": dbg_b.ranked_reasons[:5],
                "story_timeline": [
                    {"date": e.date, "event": e.event}
                    for e in (dbg_b.story.timeline if dbg_b.story else [])
                ],
                "story_building": dbg_b.story.building if dbg_b.story else "",
            }
        else:
            fire = origin_ym in straddle_months
            mk = lambda: (  # noqa: E731
                ShiftSignal("rate", "up", "large", 2, "high", "mock")
                if fire
                else NULL_SIGNAL
            )
            signals = {"B": mk()}
            if run_notl:
                signals["B_notl"] = mk()
            signals["C"] = mk()

        for arm, sig in signals.items():
            q = apply_overlay(COMMON_GRID, q_a, sig, o_pos, table)
            row[f"crps_{arm}"] = crps_from_quantiles(COMMON_GRID, q, actuals)
            row[f"signal_{arm}"] = {
                "shift_type": sig.shift_type,
                "direction": sig.direction,
                "magnitude_bucket": sig.magnitude_bucket,
                "months_from_now": sig.months_from_now,
                "timing_confidence": sig.timing_confidence,
                "rationale": sig.rationale,
                "driving_trends": list(sig.driving_trends),
                "fired": bool(
                    sig.is_actionable and not np.array_equal(q, q_a)
                ),
            }
        rows.append(row)
        if isinstance(llm, CachingClient):
            llm._save()

    # ---- Test episodes ----
    for ep in TEST_EPISODES:
        for origin_ym in ep["straddle"]:
            key = (ep["id"], origin_ym, "straddle")
            if key not in done:
                eval_origin(ep["series"], origin_ym, ep["id"], "straddle", True)
                done.add(key)
        # post origin: onset+1 (do-no-harm)
        po = add_months(ep["onset"], 1)
        key = (ep["id"], po, "post")
        if key not in done:
            eval_origin(ep["series"], po, ep["id"], "post", False)
            done.add(key)
    # ---- Placebo origins (do-no-harm) ----
    for name, origins in PLACEBO_ORIGINS.items():
        for origin_ym in origins:
            key = ("placebo", origin_ym, "placebo")
            if key not in done:
                eval_origin(name, origin_ym, "placebo", "placebo", False)
                done.add(key)

    summary = analyze(rows)
    print(json.dumps(summary, indent=1, default=str))
    diag = diagnose(rows)
    print("\nDIAGNOSIS:")
    print(json.dumps(diag, indent=1, default=str))
    cal_rows = [
        {
            "id": ep["id"],
            "series": ep["series"],
            "onset": ep["onset"],
            "signal": {
                "is_actionable": cal_signals[ep["id"]].is_actionable,
                "months_from_now": cal_signals[ep["id"]].months_from_now,
            },
        }
        for ep in CAL_EPISODES
    ]
    variants = run_variants(rows, series, pos, cal_rows)
    print("\nVARIANTS:")
    print(json.dumps(variants, indent=1, default=str))

    out = {
        "config": {"mode": args.mode, "horizon": HORIZON},
        "cal_signals": {
            k: {
                "shift_type": s.shift_type,
                "direction": s.direction,
                "magnitude_bucket": s.magnitude_bucket,
                "months_from_now": s.months_from_now,
                "timing_confidence": s.timing_confidence,
                "driving_trends": list(s.driving_trends),
            }
            for k, s in cal_signals.items()
        },
        "summary": summary,
        "diagnosis": diag,
        "variants": variants,
        "rows": rows,
    }
    out_json.write_text(json.dumps(out, indent=1, default=str))
    if isinstance(llm, CachingClient):
        llm._save()
        print(f"\nLLM calls this run: {llm.llm_calls}")
    print(f"\nexp69c [{args.mode}] done in {time.time() - t_start:.0f}s")


# ----------------------------------------------------------------------------
# Analysis + diagnosis + variants
# ----------------------------------------------------------------------------
def _paired(d: pd.DataFrame, a: str, b: str, name: str, harm_only: bool = False) -> dict:
    d = d.dropna(subset=[f"crps_{a}", f"crps_{b}"])
    entry: dict = {"n": int(len(d)), "name": name}
    if len(d) == 0:
        entry.update(verdict="NA")
        return entry
    diff = d[f"crps_{a}"].to_numpy() - d[f"crps_{b}"].to_numpy()
    entry["mean_diff"] = float(np.mean(diff))
    entry["rel_pct"] = float(
        100 * np.mean(diff) / np.mean(d[f"crps_{b}"].to_numpy())
    )
    entry["win_rate"] = float(np.mean(diff < 0))
    if np.all(diff == 0):
        entry.update(paired_p=1.0, verdict="PASS" if harm_only else "FAIL")
        return entry
    _, p = stats.ttest_1samp(diff, 0.0)
    entry["paired_p"] = float(p)
    if harm_only:
        entry["verdict"] = (
            "PASS" if not (p < 0.05 and np.mean(diff) > 0) else "FAIL"
        )
    else:
        entry["verdict"] = "PASS" if (p < 0.05 and np.mean(diff) < 0) else "FAIL"
    return entry


def analyze(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out: dict = {}
    sh = df[df["origin_class"] == "straddle"]
    out["primary_B_vs_A"] = _paired(sh, "B", "A", "primary_B_vs_A")
    out["scaffolding_B_vs_C"] = _paired(sh, "B", "C", "scaffolding_B_vs_C")
    out["timeline_B_vs_Bnotl"] = _paired(sh, "B", "B_notl", "timeline_B_vs_Bnotl")
    for ep in sorted(sh["episode"].unique()):
        sub = sh[sh["episode"] == ep]
        out[f"episode_{ep}_B_vs_A"] = _paired(sub, "B", "A", f"episode_{ep}")
    for cls in ("placebo", "post"):
        sub = df[df["origin_class"] == cls]
        if len(sub):
            out[f"do_no_harm_{cls}_B"] = _paired(sub, "B", "A", f"dnh_{cls}_B", True)
            out[f"do_no_harm_{cls}_C"] = _paired(sub, "C", "A", f"dnh_{cls}_C", True)
    return out


def diagnose(rows: list[dict]) -> dict:
    """Per-origin failure anatomy for arm B (researcher-mindset input)."""
    sh = [r for r in rows if r["origin_class"] == "straddle"]
    fired = [r for r in sh if r["signal_B"]["fired"]]
    diag: dict = {
        "n_straddle": len(sh),
        "firing_rate": len(fired) / max(len(sh), 1),
    }
    if not fired:
        diag["note"] = "B never fired: nothing to diagnose beyond abstention"
        return diag
    # Direction accuracy: sign(realized change) vs signal direction.
    dir_ok = [
        r
        for r in fired
        if (r["realized_change"] > 0 and r["signal_B"]["direction"] == "up")
        or (r["realized_change"] < 0 and r["signal_B"]["direction"] == "down")
    ]
    diag["direction_accuracy"] = len(dir_ok) / len(fired)
    # Timing error: predicted onset vs month of max |MoM| change in window.
    # (Approximation: needs the series; recomputed in run_variants.)
    diag["n_fired"] = len(fired)
    diag["fired_origins"] = [
        (r["episode"], r["origin"], r["signal_B"]["direction"],
         r["signal_B"]["months_from_now"], round(r["realized_change"], 5))
        for r in fired
    ]
    return diag


def run_variants(
    rows: list[dict],
    series: dict[str, tuple[list[str], np.ndarray]],
    pos: dict[str, dict[str, int]],
    cal_rows: list[dict],
) -> dict:
    """Cheap re-scorings from stored A quantiles (no new LLM calls).

    Pre-committed decision tree (researcher-mindset instruction):
    - firing rate < 50% -> V_abstain: "low"-confidence signals allowed to
      fire with a ramp (tests whether abstention, not information, is
      the failure).
    - direction accuracy < 50% among fired -> V_oracle_dir: direction
      replaced by the realized sign (DIAGNOSTIC ceiling — quantifies how
      much of the failure is direction vs size/timing; not a method).
    - |median timing bias on cal episodes| >= 1 month -> V_timing:
      months_from_now corrected by that bias (learned from calibration
      episodes only — PIT-clean).
    - calibrated |delta| vs test realized |change| off by >2x ->
      V_size: table scaled by the median ratio (DIAGNOSTIC — post-hoc;
      labels the size mis-calibration, not a deployable method).
    """
    out: dict = {}
    sh = [r for r in rows if r["origin_class"] == "straddle"]
    if not sh:
        return out
    firing = float(np.mean([r["signal_B"]["fired"] for r in sh]))
    out["firing_rate"] = firing
    if firing < 0.5:
        out["V_abstain"] = _rescore(
            sh, series, pos, _fn_abstain, "V_abstain_vs_A"
        )
    fired = [r for r in sh if r["signal_B"]["fired"]]
    if fired:
        dir_acc = float(
            np.mean(
                [
                    (r["realized_change"] > 0)
                    == (r["signal_B"]["direction"] == "up")
                    for r in fired
                ]
            )
        )
        out["direction_accuracy_fired"] = dir_acc
        if dir_acc < 0.5:
            out["V_oracle_dir"] = _rescore(
                fired, series, pos, _fn_oracle_dir, "V_oracle_dir_vs_A"
            )
    bias = _timing_bias(cal_rows, series, pos)
    out["cal_timing_bias"] = bias
    if bias is not None and abs(bias) >= 1:
        out["V_timing"] = _rescore(
            sh, series, pos, lambda r: _fn_timing(r, bias), "V_timing_vs_A"
        )
    ratios = [
        abs(r["realized_change"]) / max(abs(v), 1e-9)
        for r in sh
        if (v := _delta_used(r)) is not None
    ]
    if ratios:
        med_ratio = float(np.median(ratios))
        out["size_ratio_realized_over_calibrated"] = med_ratio
        if med_ratio > 2.0 or med_ratio < 0.5:
            out["V_size"] = _rescore(
                sh, series, pos,
                lambda r: _rebuild_signal(r["signal_B"]),
                "V_size_vs_A",
                table_scale=med_ratio,
            )
    return out


def _delta_used(row: dict) -> float | None:
    sig = row.get("signal_B", {})
    if not sig.get("fired"):
        return None
    vals = [
        v for k, v in row["cal_table"].items()
        if k.endswith("/" + str(sig["magnitude_bucket"]))
    ]
    return float(vals[0]) if vals else None


def _timing_bias(
    cal_rows: list[dict],
    series: dict[str, tuple[list[str], np.ndarray]],
    pos: dict[str, dict[str, int]],
) -> float | None:
    """Median (realized onset offset - predicted mfn) on cal episodes.

    ``cal_rows`` entries carry the cal classification origin (onset-1), the
    predicted months_from_now, and the episode onset. Realized onset =
    month of max |MoM change| in [onset-2, onset+6].
    """
    biases: list[float] = []
    for cr in cal_rows:
        sig = cr["signal"]
        if not sig.get("is_actionable") or sig.get("months_from_now") is None:
            continue
        months, vals = series[cr["series"]]
        onset_pos = pos[cr["series"]][cr["onset"]]
        origin_pos = onset_pos - 1
        lo, hi = max(onset_pos - 2, 1), min(onset_pos + 6, len(vals) - 1)
        diffs = np.abs(np.diff(vals[lo - 1 : hi + 1]))
        realized_pos = lo - 1 + int(np.argmax(diffs)) + 1
        biases.append((realized_pos - origin_pos) - int(sig["months_from_now"]))
    if not biases:
        return None
    return float(np.median(biases))


def _rebuild_signal(d: dict, **overrides: object) -> ShiftSignal:
    kw = {
        "shift_type": d["shift_type"],
        "direction": d["direction"],
        "magnitude_bucket": d["magnitude_bucket"],
        "months_from_now": d["months_from_now"],
        "timing_confidence": d["timing_confidence"],
        "rationale": d.get("rationale", ""),
        "driving_trends": tuple(d.get("driving_trends", ())),
    }
    kw.update(overrides)  # type: ignore[typeddict-unknown-key]
    return ShiftSignal(**kw)  # type: ignore[arg-type]


def _fn_abstain(row: dict) -> ShiftSignal:
    """Allow 'low' confidence to fire as 'medium' (ramp)."""
    sig = _rebuild_signal(row["signal_B"])
    if sig.timing_confidence == "low" and sig.shift_type != "none":
        return _rebuild_signal(
            row["signal_B"], timing_confidence="medium"
        )
    return sig


def _fn_oracle_dir(row: dict) -> ShiftSignal:
    """Direction := realized sign (diagnostic ceiling)."""
    want = "up" if row["realized_change"] > 0 else "down"
    return _rebuild_signal(row["signal_B"], direction=want)


def _fn_timing(row: dict, bias: float) -> ShiftSignal:
    sig = _rebuild_signal(row["signal_B"])
    if sig.months_from_now is None:
        return sig
    return _rebuild_signal(
        row["signal_B"], months_from_now=max(0, sig.months_from_now + int(bias))
    )


def _rescore(
    rows_subset: list[dict],
    series: dict[str, tuple[list[str], np.ndarray]],
    pos: dict[str, dict[str, int]],
    signal_fn: object,
    name: str,
    table_scale: float = 1.0,
) -> dict:
    """Re-apply overlays with modified signals/tables vs stored A rows."""
    diffs, base = [], []
    for r in rows_subset:
        months, vals = series[r["series"]]
        o_pos = pos[r["series"]][r["origin"]]
        actuals = vals[o_pos + 1 : o_pos + 1 + HORIZON]
        q_a = np.array(r["q_A"], dtype=float)
        table = {
            tuple(k.split("/")): v * table_scale
            for k, v in r["cal_table"].items()
        }
        sig = signal_fn(r)  # type: ignore[operator]
        q_v = apply_overlay(COMMON_GRID, q_a, sig, o_pos, table)
        crps_v = crps_from_quantiles(COMMON_GRID, q_v, actuals)
        diffs.append(crps_v - r["crps_A"])
        base.append(r["crps_A"])
    d = np.array(diffs)
    entry: dict = {"name": name, "n": int(len(d))}
    entry["mean_diff"] = float(np.mean(d)) if len(d) else 0.0
    entry["rel_pct"] = float(
        100 * np.mean(d) / np.mean(base) if len(d) else 0.0
    )
    entry["win_rate"] = float(np.mean(d < 0)) if len(d) else 0.0
    if len(d) and not np.all(d == 0):
        _, p = stats.ttest_1samp(d, 0.0)
        entry["paired_p"] = float(p)
    else:
        entry["paired_p"] = 1.0
    return entry


if __name__ == "__main__":
    main()
