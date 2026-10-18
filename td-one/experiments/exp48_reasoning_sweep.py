"""exp48: Reasoning-effort sweep for Groq gpt-oss-120b.

Tests whether `reasoning_effort` ("low"/"medium"/"high") affects agent-task
quality, and at what latency/cost tradeoff.

Three representative tasks (1-2 LLM calls each, rubric-scored):
  A. regime_diagnosis  — classify a series as seasonal/trending/noisy from
     diagnostic features (objective accuracy vs hand-rule ground truth).
  B. boc_discovery     — propose the interaction feature that fixes BoC rule
     failures (binary hit: mentions inflation_gap x rate_momentum).
  C. changepoint_diagnosis — diagnose a series with a level shift and
     recommend a model (keyword rubric: changepoint + model + uncertainty).

Design: 3 tasks x 3 levels x 3 seeds = 27 LLM calls. Same prompt repeated
per seed measures output stability (does "high" give more consistent answers?).

Deterministic data fixtures (fixed numpy seed). LLM calls are the only
stochastic element.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from td_one.agent.groq_client import GroqChatClient

OUT = Path(__file__).with_name("exp48_results.json")
REPORT = Path(__file__).with_name("exp48_report.md")

LEVELS = ("low", "medium", "high")
SEEDS = (0, 1, 2)
MAX_TOKENS = 1500  # concise prompts; keep TPM headroom


# ---------------------------------------------------------------- fixtures


def make_series(kind: str, rng: np.random.Generator, n: int = 72) -> np.ndarray:
    """Deterministic synthetic series with known regime."""
    t = np.arange(n)
    if kind == "seasonal":
        return 10 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
    if kind == "trending":
        return 5 + 0.3 * t + rng.normal(0, 0.8, n)
    # noisy: white noise around a flat level
    return 8 + rng.normal(0, 2.0, n)


def describe_series(y: np.ndarray) -> str:
    """Compact diagnostic summary (what an agent would compute)."""
    n = len(y)
    acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if n > 1 else 0.0
    # seasonal strength proxy: variance explained by monthly means
    monthly = np.array([np.mean(y[i::12]) for i in range(12)])
    seasonal_strength = float(np.var(np.tile(monthly, n // 12 + 1)[:n]) / (np.var(y) + 1e-9))
    trend_slope = float(np.polyfit(np.arange(n), y, 1)[0])
    vol = float(np.std(np.diff(y)))
    return (
        f"n={n}, mean={np.mean(y):.2f}, std={np.std(y):.2f}, "
        f"lag1_acf={acf1:.2f}, seasonal_strength={seasonal_strength:.2f}, "
        f"trend_slope={trend_slope:.3f}, diff_vol={vol:.2f}"
    )


# ---------------------------------------------------------------- tasks


def task_a_regime(level: str, seed: int, rng: np.random.Generator) -> dict:
    """Classify series regime. Objective accuracy."""
    kinds = ["seasonal", "trending", "noisy"]
    results = []
    client = GroqChatClient(call_budget=6, max_tokens=MAX_TOKENS, reasoning_effort=level)
    for kind in kinds:
        y = make_series(kind, rng)
        prompt = (
            "You are a time-series diagnostician. Given summary statistics of a "
            "monthly series, classify its regime as exactly one of: seasonal, "
            "trending, noisy.\n\n"
            f"Statistics: {describe_series(y)}\n\n"
            "Reply with ONLY the single regime word, nothing else."
        )
        t0 = time.time()
        try:
            out = client.complete(prompt).strip().lower()
            dt = time.time() - t0
            pred = (
                "seasonal"
                if "seasonal" in out
                else (
                    "trending" if "trending" in out else ("noisy" if "noisy" in out else "invalid")
                )
            )
            results.append(
                {
                    "kind": kind,
                    "pred": pred,
                    "correct": pred == kind,
                    "latency_s": round(dt, 2),
                    "raw": out[:40],
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                {
                    "kind": kind,
                    "pred": "error",
                    "correct": False,
                    "latency_s": -1,
                    "raw": f"{type(e).__name__}",
                }
            )
    acc = sum(r["correct"] for r in results) / len(results)
    lat = [r["latency_s"] for r in results if r["latency_s"] >= 0]
    return {
        "task": "regime_diagnosis",
        "level": level,
        "seed": seed,
        "accuracy": acc,
        "mean_latency_s": round(float(np.mean(lat)), 2) if lat else -1,
        "detail": results,
    }


def task_b_boc(level: str, seed: int) -> dict:
    """Propose the interaction feature fixing BoC rule failures. Binary hit."""
    client = GroqChatClient(call_budget=3, max_tokens=MAX_TOKENS, reasoning_effort=level)
    prompt = (
        "You are a macroeconomist diagnosing a Bank of Canada rate-decision model.\n"
        "The simple rule (based on rate momentum) predicts HOLD well but fails on hikes/cuts.\n"
        "Failure analysis: hikes are missed when inflation is high but rates haven't moved yet; "
        "cuts are missed when unemployment is rising but rates are flat.\n"
        "Available base features: rate_momentum, inflation_gap (inflation minus 2% target), "
        "unemployment_momentum, yield_spread.\n\n"
        "Propose the SINGLE most promising interaction feature (feature_a x feature_b) "
        "that would fix these failures, and one sentence of reasoning.\n"
        "Format: FEATURE: <name> | REASON: <one sentence>"
    )
    t0 = time.time()
    try:
        out = client.complete(prompt)
        dt = time.time() - t0
        low = out.lower()
        hit = ("inflation" in low and ("rate_momentum" in low or "momentum" in low)) or (
            "inflation_gap" in low
        )
        return {
            "task": "boc_discovery",
            "level": level,
            "seed": seed,
            "hit": hit,
            "latency_s": round(dt, 2),
            "raw": out[:300],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "task": "boc_discovery",
            "level": level,
            "seed": seed,
            "hit": False,
            "latency_s": -1,
            "raw": f"{type(e).__name__}",
        }


def task_c_changepoint(level: str, seed: int, rng: np.random.Generator) -> dict:
    """Diagnose a level-shift series; rubric: changepoint + model + uncertainty."""
    y = np.concatenate(
        [np.full(30, 5.0) + rng.normal(0, 0.5, 30), np.full(30, 8.0) + rng.normal(0, 0.5, 30)]
    )
    client = GroqChatClient(call_budget=3, max_tokens=MAX_TOKENS, reasoning_effort=level)
    prompt = (
        "You are a forecasting diagnostician. A monthly series has these stats:\n"
        f"{describe_series(y)}\n"
        f"First-half mean: {np.mean(y[:30]):.2f}, second-half mean: {np.mean(y[30:]):.2f}.\n\n"
        "In 3 short bullet points: (1) what structural feature do you see, "
        "(2) which forecasting model handles it best (ETS/ARIMA/naive) and why, "
        "(3) how confident are you (high/medium/low) and why."
    )
    t0 = time.time()
    try:
        out = client.complete(prompt)
        dt = time.time() - t0
        low = out.lower()
        rubric = {
            "mentions_shift": any(
                w in low
                for w in (
                    "changepoint",
                    "change point",
                    "break",
                    "shift",
                    "level shift",
                    "structural",
                )
            ),
            "recommends_model": any(w in low for w in ("ets", "arima", "naive")),
            "states_uncertainty": any(
                w in low for w in ("confident", "confidence", "uncertain", "high", "medium", "low")
            ),
        }
        return {
            "task": "changepoint_diagnosis",
            "level": level,
            "seed": seed,
            "rubric": rubric,
            "score": sum(rubric.values()) / 3,
            "latency_s": round(dt, 2),
            "raw": out[:400],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "task": "changepoint_diagnosis",
            "level": level,
            "seed": seed,
            "rubric": {},
            "score": 0.0,
            "latency_s": -1,
            "raw": f"{type(e).__name__}",
        }


# ---------------------------------------------------------------- runner


def main() -> None:
    rng = np.random.default_rng(42)
    records: list[dict] = []
    ckpt = Path(__file__).with_name("exp48_checkpoint.json")
    if ckpt.exists():
        records = json.loads(ckpt.read_text())
        print(f"resumed {len(records)} records from checkpoint")
    done_keys = {(r["task"], r["level"], r["seed"]) for r in records}
    total = len(LEVELS) * len(SEEDS) * 3
    done = len(records)

    def save() -> None:
        ckpt.write_text(json.dumps(records))

    for level in LEVELS:
        for seed in SEEDS:
            if ("regime_diagnosis", level, seed) not in done_keys:
                records.append(task_a_regime(level, seed, rng))
                save()
            done += 1
            print(f"[{done}/{total}] A {level} s{seed}", flush=True)
            if ("boc_discovery", level, seed) not in done_keys:
                records.append(task_b_boc(level, seed))
                save()
            done += 1
            print(f"[{done}/{total}] B {level} s{seed}", flush=True)
            if ("changepoint_diagnosis", level, seed) not in done_keys:
                records.append(task_c_changepoint(level, seed, rng))
                save()
            done += 1
            print(f"[{done}/{total}] C {level} s{seed}", flush=True)
    ckpt.unlink(missing_ok=True)

    # aggregate
    summary: dict = {}
    for task in ("regime_diagnosis", "boc_discovery", "changepoint_diagnosis"):
        summary[task] = {}
        for level in LEVELS:
            cells = [r for r in records if r["task"] == task and r["level"] == level]
            metric = (
                "accuracy"
                if task == "regime_diagnosis"
                else ("hit" if task == "boc_discovery" else "score")
            )
            vals = [c[metric] for c in cells]
            lat_key = "latency_s" if task != "regime_diagnosis" else "mean_latency_s"
            lats = [c[lat_key] for c in cells if c.get(lat_key, -1) >= 0]
            summary[task][level] = {
                f"mean_{metric}": round(float(np.mean(vals)), 3),
                "vals": vals,
                "mean_latency_s": round(float(np.mean(lats)), 2) if lats else -1,
                "errors": sum(1 for c in cells if c.get(lat_key, -1) < 0),
            }

    payload = {
        "records": records,
        "summary": summary,
        "note": "27 LLM calls (3 tasks x 3 levels x 3 seeds); max_tokens=1500; gpt-oss-120b",
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")

    # report
    lines = [
        "# exp48: Reasoning-Effort Sweep",
        "",
        "27 LLM calls: 3 tasks x 3 levels (low/medium/high) x 3 seeds.",
        "Model: openai/gpt-oss-120b, max_tokens=1500.",
        "",
    ]
    for task, s in summary.items():
        lines.append(f"## {task}")
        for level in LEVELS:
            d = s[level]
            mkey = next(k for k in d if k.startswith("mean_") and "latency" not in k)
            lat = d["mean_latency_s"]
            lines.append(
                f"- **{level}**: {mkey}={d[mkey]}, latency={lat}s, "
                f"errors={d['errors']}, vals={d['vals']}"
            )
        lines.append("")
    # recommendation
    lines.append("## Recommendation")
    a = summary["regime_diagnosis"]
    b = summary["boc_discovery"]
    c = summary["changepoint_diagnosis"]
    for name, s, mkey in (
        ("regime_diagnosis", a, "mean_accuracy"),
        ("boc_discovery", b, "mean_hit"),
        ("changepoint_diagnosis", c, "mean_score"),
    ):
        vals = " / ".join(f"{lv}={s[lv][mkey]}" for lv in LEVELS)
        lines.append(f"- {name}: {vals}")
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
