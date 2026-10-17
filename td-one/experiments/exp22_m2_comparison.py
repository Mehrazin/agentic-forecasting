"""exp22: M2 v2 (code-executing) vs M2 v1 (3-tool) on event-augmented data.

Original M2 (ReAct, ``td_one.agent.react``) had 3 tools and no code
execution — it failed because it could only call ``forecasting`` blindly.
M2 v2 (``td_one.agent.react_v2``) adds sandboxed code execution plus the 10
statistical tools, turning the LLM into a "data scientist".

This experiment isolates the effect of *capabilities* (not LLM quality):
both agents are driven by deterministic scripted policies (no real LLM).
v1's policy: data_segment -> evidence -> forecasting -> finalize (the
forecasting tool's m0 output, verbatim — mirrors exp3's scripted trace).
v2's policy: data_segment -> code_execute (changepoint + stationarity
analysis) -> evidence (news triangulation, always) -> code_execute
(post-break adjustment, only when a recent changepoint is found) or
forecasting (honest fallback) -> finalize.

Data: event-augmented simulator (Phase 1). Origins span the last 12 months
with no event-based pre-selection; results are stratified by whether v2's
analysis actually fired ("adjusted" vs "fallback").

Hypothesis: on origins where v2 detects a recent break, its post-break
adjustment beats v1's stale trailing mean; where it finds nothing, it
ties (correctly refusing to hallucinate).

Usage:
    .venv/bin/python experiments/exp22_m2_comparison.py [--seeds 42 7 123]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from td_one.agent.react import ReActAgent, ReActResult  # noqa: E402
from td_one.agent.react_v2 import ReActAgentV2  # noqa: E402
from td_one.data_gen.events import generate_event_timeline  # noqa: E402
from td_one.data_gen.news_synthesis import (  # noqa: E402
    articles_visible_asof,
    generate_news_feed,
    strip_eval_fields,
)
from td_one.data_gen.simulator import MortgageSimulator  # noqa: E402
from td_one.evaluation.real_benchmark import crps_from_quantiles  # noqa: E402
from td_one.tools.forecasting import forecast as forecast_tool  # noqa: E402

#: Quantile levels every forecast must emit.
LEVELS = [0.1, 0.5, 0.9]
#: z for the 80% normal interval (matches exp3's normal-approx convention).
_Z80 = 1.2815515655446004
#: A changepoint this many months (or fewer) before the origin counts as "recent".
#: Set to 5 so breaks inside m0's trailing-6 staleness window are caught.
RECENT_CP_MONTHS = 5
#: Simulator config (matches exp20).
N_MONTHS = 36
N_CUSTOMERS = 2000
START_DATE = date(2022, 1, 31)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return _month_end(y, m)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(seed: int) -> dict:
    """Generate event-augmented data and the monthly panel."""
    events = generate_event_timeline(
        start_date=START_DATE, n_months=N_MONTHS, seed=seed, event_rate_per_year=4.0
    )
    articles = strip_eval_fields(generate_news_feed(events, seed=seed + 1000))
    sim = MortgageSimulator(
        n_customers=N_CUSTOMERS, n_months=N_MONTHS, start_date=START_DATE, seed=seed
    )
    history = sim.generate_history()
    sim.rng = np.random.default_rng(10_000 + seed)
    outcomes = sim.generate_outcomes(history=history, event_timeline=events)

    month_dates = sim._month_dates()
    n = np.zeros(len(month_dates))
    a = np.zeros(len(month_dates))
    for o in outcomes:
        m = o.scheduled_maturity_date
        eom = _month_end(m.year, m.month)
        try:
            idx = month_dates.index(eom)
        except ValueError:
            continue
        n[idx] += 1
        if o.switched_lender_flag:
            a[idx] += 1
    rates = np.divide(a, n, out=np.full_like(a, np.nan), where=n > 0)
    panel = pd.DataFrame(
        {
            "month": pd.to_datetime(month_dates),
            "n_renewals": n.astype(int),
            "n_switches": a.astype(int),
        }
    )
    return {
        "events": events,
        "articles": articles,
        "panel": panel,
        "month_dates": month_dates,
        "rates": rates,
        "n": n,
        "a": a,
    }


# ---------------------------------------------------------------------------
# Tool wiring (PIT-safe)
# ---------------------------------------------------------------------------

ToolFn = Callable[[dict[str, object]], str]


def make_tools(
    panel: pd.DataFrame,
    articles: list[dict[str, object]],
    holder: dict[str, object],
) -> dict[str, ToolFn]:
    """Wire v1 tools to the event-augmented dataset (PIT-safe)."""

    def _run_date(args: dict[str, object]) -> date:
        return date.fromisoformat(str(args["run_date"]))

    def data_segment(args: dict[str, object]) -> str:
        rd = _run_date(args)
        sub = panel[panel["month"] <= pd.Timestamp(rd)]
        rows = [
            {
                "month": d.date().isoformat(),
                "n_renewals": int(n_),
                "n_switches": int(a_),
            }
            for d, n_, a_ in zip(sub["month"], sub["n_renewals"], sub["n_switches"], strict=True)
        ]
        obs = json.dumps(rows)
        holder["segment_obs"] = obs
        holder["segment_rows"] = rows
        return obs

    def evidence(args: dict[str, object]) -> str:
        rd = _run_date(args)
        query = str(args.get("query", ""))
        toks = {t.lower() for t in re.findall(r"[a-z]+", query)}
        scored: list[tuple[int, dict[str, object]]] = []
        for art in articles_visible_asof(articles, rd):
            text = f"{art.get('headline', '')} {art.get('body', '')}".lower()
            score = sum(1 for t in toks if t in text)
            if score > 0:
                scored.append((score, art))
        scored.sort(key=lambda p: (-p[0], str(p[1].get("article_date", ""))))
        top = []
        for _, art in scored[:5]:
            top.append(
                {
                    "date": str(art.get("article_date", "")),
                    "headline": art.get("headline", ""),
                    "body": str(art.get("body", ""))[:600],
                    "event_type": art.get("event_type", ""),
                }
            )
        obs = json.dumps(top)
        holder["evidence_obs"] = obs
        holder["n_articles"] = len(top)
        return obs

    def forecasting(args: dict[str, object]) -> str:
        rd = _run_date(args)
        quantiles = forecast_tool("m0", panel, rd)
        obs = json.dumps({str(k): float(v) for k, v in quantiles.items()})
        holder["forecast_obs"] = obs
        return obs

    return {
        "data_segment": data_segment,
        "evidence": evidence,
        "forecasting": forecasting,
    }


def _pick_levels(raw: dict[str, float]) -> dict[float, float]:
    """Extract {0.1, 0.5, 0.9} from a many-level quantile mapping."""
    items = sorted((float(k), float(v)) for k, v in raw.items())
    out: dict[float, float] = {}
    for want in LEVELS:
        k, v = min(items, key=lambda kv: abs(kv[0] - want))
        out[want] = v
    return out


# ---------------------------------------------------------------------------
# v1 policy: fixed 4-step script (mirrors exp3)
# ---------------------------------------------------------------------------


class V1Policy:
    """Deterministic v1 policy: data_segment -> evidence -> forecasting."""

    def __init__(self, origin: date, holder: dict[str, object]) -> None:
        self.origin = origin
        self.holder = holder
        self.state = "start"

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        iso = self.origin.isoformat()
        if self.state == "start":
            self.state = "evidence"
            return json.dumps(
                {
                    "action": "data_segment",
                    "args": {"segment_id": "agg", "run_date": iso},
                }
            )
        if self.state == "evidence":
            self.state = "forecasting"
            return json.dumps(
                {
                    "action": "evidence",
                    "args": {
                        "query": "rate shock policy bank unemployment",
                        "run_date": iso,
                    },
                }
            )
        if self.state == "forecasting":
            self.state = "finalize"
            return json.dumps(
                {
                    "action": "forecasting",
                    "args": {"segment_id": "agg", "run_date": iso},
                }
            )
        # finalize: use the forecasting tool's quantiles verbatim
        raw = json.loads(str(self.holder.get("forecast_obs", "{}")))
        q = _pick_levels({k: float(v) for k, v in raw.items()})
        return json.dumps(
            {
                "action": "finalize",
                "forecast": {str(k): v for k, v in q.items()},
            }
        )


# ---------------------------------------------------------------------------
# v2 policy: adaptive data-scientist workflow
# ---------------------------------------------------------------------------

_ANALYSIS_CODE = """\
series = {series_json}
cp = T.detect_changepoints(series, min_distance=6).model_dump()
_show({{"n": len(series), "changepoints": cp["changepoints"],
        "segment_means": cp["segment_means"], "n_segments": cp["n_segments"]}})
st = T.test_stationarity(series).model_dump()
_show({{"verdict": st["verdict"], "adf_p": st["adf_pvalue"],
        "kpss_p": st["kpss_pvalue"]}})
# Recent-shift test: does the last 3 months differ from the prior 6?
# (changepoint detection needs >=6-point segments, so it is blind to very
# recent breaks; this t-test catches them.)
import numpy as _np
from scipy import stats as _st
_recent = _np.array(series[-3:])
_trail = _np.array(series[-9:-3])
_rm = float(_recent.mean()); _tm = float(_trail.mean())
_rsd = float(_recent.std(ddof=1)) if len(_recent) > 1 else 0.005
_t, _p = _st.ttest_ind(_recent, _trail, equal_var=False)
_shift = _rm - _tm
_show({{"recent_mean": _rm, "recent_sd": _rsd, "trail_mean": _tm,
        "shift": _shift, "shift_p": float(_p)}})
print("SHIFT_TEST %.6f %.6f %.6f %.6f" % (_shift, float(_p), _rm, _rsd))
"""

_ADJUST_CODE = """\
# Adjustment from an explicit level estimate (post-break or recent mean).
level_mean = {level_mean}
level_sd = {level_sd}
level_n = {level_n}
import math as _m
sd = max(level_sd, 0.002)
q10 = max(0.001, level_mean - {z} * sd)
q50 = min(0.99, max(0.001, level_mean))
q90 = min(0.99, level_mean + {z} * sd)
_show({{"level_n": level_n, "level_mean": level_mean, "level_sd": sd,
        "q10": q10, "q50": q50, "q90": q90}})
print("FINAL_Q %.6f %.6f %.6f" % (q10, q50, q90))
"""

_CP_STATS_CODE = """\
series = {series_json}
cp_idx = {cp_idx}
post = series[cp_idx + 1:]
import statistics as _s
m = _s.mean(post)
sd = _s.pstdev(post) if len(post) > 1 else 0.005
print("CP_STATS %.6f %.6f %d" % (m, sd, len(post)))
"""


class V2Policy:
    """Deterministic v2 policy: analyze -> triangulate -> adjust-or-fallback."""

    def __init__(self, origin: date, holder: dict[str, object]) -> None:
        self.origin = origin
        self.holder = holder
        self.state = "start"
        self.cp_idx: int | None = None
        self._pending_cp: int | None = None
        self._adj_mode: str | None = None
        self._adj_mean = 0.0
        self._adj_sd = 0.005
        self._adj_n = 0
        self.used_evidence = False
        self.adjusted = False
        self.fallback = False

    def _series(self) -> list[float]:
        rows = self.holder.get("segment_rows", [])
        rates = []
        for r in rows:
            n_ = r["n_renewals"]
            rates.append(r["n_switches"] / n_ if n_ > 0 else float("nan"))
        s = pd.Series(rates).ffill().bfill()
        return [float(x) for x in s]

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        iso = self.origin.isoformat()
        if self.state == "start":
            self.state = "analyze"
            return json.dumps(
                {
                    "action": "data_segment",
                    "args": {"segment_id": "agg", "run_date": iso},
                }
            )
        if self.state == "analyze":
            self.state = "decide"
            code = _ANALYSIS_CODE.format(series_json=json.dumps(self._series()))
            return json.dumps({"action": "code_execute", "args": {"code": code, "timeout_s": 60}})
        if self.state == "decide":
            obs = str(self.holder.get("last_code_obs", ""))
            m = re.search(r'"changepoints":\s*\[([0-9,\s]*)\]', obs)
            cps = (
                [int(x) for x in m.group(1).split(",") if x.strip()]
                if m and m.group(1).strip()
                else []
            )
            n = len(self._series())
            recent = [c for c in cps if 1 <= (n - 1 - c) <= RECENT_CP_MONTHS]
            # Recent-shift t-test (catches breaks too fresh for changepoints).
            sm = re.search(
                r"SHIFT_TEST\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+"
                r"([0-9.eE+\-]+)\s+([0-9.eE+\-]+)",
                obs,
            )
            self._adj_mode: str | None = None
            self._adj_mean = 0.0
            self._adj_sd = 0.005
            self._adj_n = 0
            if sm:
                shift, pval, rmean, rsd = (float(sm.group(i)) for i in range(1, 5))
                if pval < 0.10 and abs(shift) > 0.02:
                    self._adj_mode = "shift"
                    self._adj_mean, self._adj_sd, self._adj_n = rmean, rsd, 3
            if self._adj_mode is None and recent:
                self._adj_mode = "changepoint"
                self._pending_cp = recent[-1]
            # Evidence is always queried: triangulation for the writeup, and the
            # news feed is the information-asymmetry channel v1 cannot exploit.
            self.used_evidence = True
            self.state = "post_evidence"
            return json.dumps(
                {
                    "action": "evidence",
                    "args": {
                        "query": "rate shock policy bank unemployment mortgage",
                        "run_date": iso,
                    },
                }
            )
        if self.state == "post_evidence":
            if self._adj_mode == "shift":
                self.state = "finalize_adj"
                code = _ADJUST_CODE.format(
                    level_mean=self._adj_mean,
                    level_sd=self._adj_sd,
                    level_n=self._adj_n,
                    z=_Z80,
                )
                return json.dumps(
                    {"action": "code_execute", "args": {"code": code, "timeout_s": 60}}
                )
            if self._adj_mode == "changepoint":
                # First get post-changepoint stats, then build quantiles.
                self.state = "cp_stats"
                assert self._pending_cp is not None
                code = _CP_STATS_CODE.format(
                    series_json=json.dumps(self._series()),
                    cp_idx=self._pending_cp,
                )
                return json.dumps(
                    {"action": "code_execute", "args": {"code": code, "timeout_s": 60}}
                )
            self.fallback = True
            self.state = "finalize_m0"
            return json.dumps(
                {
                    "action": "forecasting",
                    "args": {"segment_id": "agg", "run_date": iso},
                }
            )
        if self.state == "cp_stats":
            obs = str(self.holder.get("last_code_obs", ""))
            m = re.search(r"CP_STATS\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+(\d+)", obs)
            self.state = "finalize_adj"
            if m:
                mean, sd, cnt = float(m.group(1)), float(m.group(2)), int(m.group(3))
            else:
                series = self._series()
                mean, sd, cnt = float(np.mean(series[-3:])), 0.005, 3
            code = _ADJUST_CODE.format(level_mean=mean, level_sd=sd, level_n=cnt, z=_Z80)
            return json.dumps({"action": "code_execute", "args": {"code": code, "timeout_s": 60}})
        if self.state == "finalize_adj":
            obs = str(self.holder.get("last_code_obs", ""))
            m = re.search(r"FINAL_Q\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)", obs)
            if m:
                self.adjusted = True
                q10, q50, q90 = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            else:
                # Sandbox adjustment failed: fall back to m0 median +/- default.
                self.fallback = True
                raw = json.loads(str(self.holder.get("forecast_obs", "{}") or "{}"))
                if raw:
                    q = _pick_levels({k: float(v) for k, v in raw.items()})
                    q10, q50, q90 = q[0.1], q[0.5], q[0.9]
                else:
                    q10, q50, q90 = 0.02, 0.05, 0.10
            return json.dumps(
                {
                    "action": "finalize",
                    "forecast": {"0.1": q10, "0.5": q50, "0.9": q90},
                }
            )
        # finalize_m0
        raw = json.loads(str(self.holder.get("forecast_obs", "{}")))
        q = _pick_levels({k: float(v) for k, v in raw.items()})
        return json.dumps({"action": "finalize", "forecast": {str(k): v for k, v in q.items()}})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _score(forecast: dict[float, float] | None, actual: float) -> dict[str, float]:
    if not forecast:
        return {"crps": float("nan"), "mae": float("nan"), "finalized": 0.0}
    lv = np.array(LEVELS)
    qv = np.array([forecast[lvl] for lvl in LEVELS])
    return {
        "crps": crps_from_quantiles(lv, qv, actual),
        "mae": float(abs(forecast[0.5] - actual)),
        "finalized": 1.0,
    }


def run_origin(dataset: dict, origin: date, actual: float) -> dict:
    """Run v1 and v2 at one origin; return scored comparison."""
    panel, articles = dataset["panel"], dataset["articles"]

    # ---- v1 ----
    h1: dict[str, object] = {}
    tools1 = make_tools(panel, articles, h1)
    agent1 = ReActAgent(llm_client=V1Policy(origin, h1), tools=tools1, max_steps=6)
    r1: ReActResult = agent1.run(
        task="Forecast next month's aggregate switching rate.", run_date=origin
    )
    s1 = _score(r1["forecast"], actual)
    v1_actions = [t["action"] for t in r1["trace"]]

    # ---- v2 ----
    h2: dict[str, object] = {}
    tools2 = make_tools(panel, articles, h2)
    agent2 = ReActAgentV2(llm_client=V2Policy(origin, h2), tools=tools2, max_steps=8)
    # Record code_execute observations for the adaptive policy.
    orig_ce = agent2.tools["code_execute"]

    def _recording_ce(args: dict[str, object]) -> str:
        obs = orig_ce(args)
        h2["last_code_obs"] = obs
        return obs

    agent2.tools["code_execute"] = _recording_ce
    r2: ReActResult = agent2.run(
        task="Forecast next month's aggregate switching rate.", run_date=origin
    )
    s2 = _score(r2["forecast"], actual)
    v2_actions = [t["action"] for t in r2["trace"]]
    policy2: V2Policy = agent2.llm_client  # type: ignore[assignment]

    return {
        "origin": origin.isoformat(),
        "actual": actual,
        "v1": {
            "forecast": r1["forecast"],
            "actions": v1_actions,
            **s1,
        },
        "v2": {
            "forecast": r2["forecast"],
            "actions": v2_actions,
            "used_code_execute": "code_execute" in v2_actions,
            "used_evidence": policy2.used_evidence,
            "adjusted": policy2.adjusted,
            "adj_mode": policy2._adj_mode,
            "fallback_to_m0": policy2.fallback,
            "cp_idx": policy2.cp_idx,
            **s2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="exp22: M2 v2 vs v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for seed in args.seeds:
        print(f"exp22: seed={seed}", flush=True)
        ds = build_dataset(seed)
        month_dates: list[date] = ds["month_dates"]
        rates: np.ndarray = ds["rates"]
        # All candidate origins: no event-based pre-selection. v2's policy
        # decides per-origin whether its analysis fires; we stratify by that.
        n_eval = 0
        for origin in month_dates[-13:-1]:
            nxt = _add_months(origin, 1)
            try:
                actual = float(rates[month_dates.index(nxt)])
            except (ValueError, IndexError):
                continue
            if math.isnan(actual):
                continue
            row = run_origin(ds, origin, actual)
            row["seed"] = seed
            row["kind"] = "adjusted" if row["v2"]["adjusted"] else "fallback"
            all_rows.append(row)
            n_eval += 1
            v1c, v2c = row["v1"]["crps"], row["v2"]["crps"]
            flag = (
                "ADJ"
                if row["v2"]["adjusted"]
                else ("FB " if row["v2"]["fallback_to_m0"] else "???")
            )
            print(
                f"  [{row['kind']:8s}] origin={origin} actual={actual:.4f} "
                f"v1_crps={v1c:.5f} v2_crps={v2c:.5f} v2[{flag}]",
                flush=True,
            )
        print(f"  evaluated {n_eval} origins", flush=True)

    # Paired comparison
    d = np.array([r["v1"]["crps"] - r["v2"]["crps"] for r in all_rows])
    d = d[~np.isnan(d)]
    wins = int(np.sum(d > 0))
    mean_d = float(np.mean(d))
    try:
        from scipy import stats

        t_stat, p_val = stats.ttest_rel(
            [r["v1"]["crps"] for r in all_rows],
            [r["v2"]["crps"] for r in all_rows],
            nan_policy="omit",
        )
        t_stat, p_val = float(t_stat), float(p_val)
    except ImportError:
        t_stat, p_val = float("nan"), float("nan")
    v1_mean = float(np.nanmean([r["v1"]["crps"] for r in all_rows]))
    v2_mean = float(np.nanmean([r["v2"]["crps"] for r in all_rows]))
    adj_rate = float(np.mean([1.0 if r["v2"]["adjusted"] else 0.0 for r in all_rows]))
    ce_rate = float(np.mean([1.0 if r["v2"]["used_code_execute"] else 0.0 for r in all_rows]))

    def _split(kind: str) -> list[dict]:
        return [r for r in all_rows if r.get("kind") == kind]

    def _stats(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        v1m = float(np.nanmean([r["v1"]["crps"] for r in rows]))
        v2m = float(np.nanmean([r["v2"]["crps"] for r in rows]))
        dd = np.array([r["v1"]["crps"] - r["v2"]["crps"] for r in rows])
        dd = dd[~np.isnan(dd)]
        return {
            "n": len(rows),
            "v1_mean_crps": v1m,
            "v2_mean_crps": v2m,
            "rel_improvement": (v1m - v2m) / v1m if v1m > 0 else 0.0,
            "v2_wins": int(np.sum(dd > 0)),
        }

    summary = {
        "experiment": "exp22_m2_v2_vs_v1",
        "seeds": args.seeds,
        "n_origins": len(all_rows),
        "v1_mean_crps": v1_mean,
        "v2_mean_crps": v2_mean,
        "mean_diff_v1_minus_v2": mean_d,
        "rel_improvement": (v1_mean - v2_mean) / v1_mean if v1_mean > 0 else 0.0,
        "v2_wins": wins,
        "paired_t": t_stat,
        "paired_p": p_val,
        "v2_code_execute_rate": ce_rate,
        "v2_adjustment_rate": adj_rate,
        "by_kind": {
            "adjusted": _stats(_split("adjusted")),
            "fallback": _stats(_split("fallback")),
        },
        "rows": all_rows,
    }
    print("\n=== SUMMARY ===")
    print(f"origins: {len(all_rows)}")
    print(f"v1 mean CRPS: {v1_mean:.5f}")
    print(f"v2 mean CRPS: {v2_mean:.5f}")
    print(f"relative improvement: {summary['rel_improvement']:+.1%}")
    print(f"v2 wins: {wins}/{len(d)}  paired t={t_stat:.2f} p={p_val:.4f}")
    print(f"v2 code_execute used: {ce_rate:.0%}, adjusted: {adj_rate:.0%}")
    for kind in ("adjusted", "fallback"):
        s = summary["by_kind"][kind]
        if s["n"]:
            print(
                f"  [{kind}] n={s['n']} v1={s['v1_mean_crps']:.5f} "
                f"v2={s['v2_mean_crps']:.5f} improv={s['rel_improvement']:+.1%} "
                f"wins={s['v2_wins']}/{s['n']}"
            )

    out_path = args.out or str(Path(__file__).parent / "exp22_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
