"""Experiment 19: News-regime-augmented SMM forecasting.

Tests whether the news/RAG regime signal (exp17 machinery) improves on the
exp18 baselines for the 63-month Freddie Mac SMM series.

For each of the 50 exp18 origins (2022-01..2026-02):
  1. Base forecasts: m0_w3 point and m0_conformal quantiles (same as exp18).
  2. Regime: BM25-only RAG over the frozen news KB (queries "mortgage
     rates" + "refinance", PIT cutoff = first day of the forecast month),
     then read_regime_with_news (hardened v2 prompt).
  3. Adjustment: if regime in {UP, DOWN} and the conservatively-shrunk
     confidence >= 0.6, shift the forecast by
         shift = sign * scale * shrunk_confidence
     where scale = std of trailing-12m month-over-month SMM changes
     (adapts to volatility regimes). Applied to the m0_w3 point and to
     the full conformal quantile lattice (location shift).
     NONE / ABSTAIN / low confidence -> no adjustment.
  4. Score: CRPS=MAE for points; pinball CRPS for quantiles; 80% coverage.

PIT discipline: train uses months <= origin; every news article satisfies
pub_date < first-day-of-(origin+1) (RAG enforces in SQL + client-side).
Groq budget: 50 regime calls (1/origin); retrieval is BM25-only (free).
A failed Groq call for an origin falls back to the base forecast.

Usage:
    cd td-one && .venv/bin/python experiments/exp19_news_smm.py
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.groq_client import GroqChatClient  # noqa: E402
from td_one.agent.regime_reader import (  # noqa: E402
    RegimeReading,
    conservative_shrinkage,
    read_regime_with_news,
)
from td_one.evaluation.real_benchmark import crps_from_quantiles  # noqa: E402
from td_one.news.rag import RAGTool  # noqa: E402

SMM_CSV = ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv"

#: Same origins as exp18.
N_ORIGINS = 50
MIN_HISTORY = 12

#: Quantile grid for the conformal model (must include 0.1/0.9/0.5).
CONF_LEVELS = np.linspace(0.05, 0.95, 19)

#: RAG queries (FTS5 keyword style — natural language returns zero hits).
QUERIES = ["mortgage rates", "refinance"]
ARTICLES_PER_QUERY = 5

#: Regime gate: apply the shift only when the shrunk confidence clears this.
CONF_GATE = 0.6

#: Groq calls: exactly 1 regime reading per origin.
GROQ_BUDGET = 60


def load_smm() -> pd.DataFrame:
    """Load the monthly SMM series, sorted ascending, with a datetime index."""
    df = pd.read_csv(SMM_CSV)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    df = df.sort_values("date").reset_index(drop=True)
    assert len(df) == 63, f"expected 63 months, got {len(df)}"
    return df


def m0_point(train: np.ndarray, window: int) -> float:
    """Trailing-window mean point forecast."""
    return float(np.mean(train[-window:]))


def conformal_quantiles(
    train: np.ndarray, levels: np.ndarray, window: int = 12, alpha: float = 0.2
) -> np.ndarray:
    """Trailing-window mean +/- empirical residual quantiles (exp16/exp18 method)."""
    hist = train[-window:]
    mu = float(np.mean(hist))
    resid = np.abs(hist - mu)
    width = float(np.quantile(resid, 1 - alpha)) if len(resid) > 0 else 0.0
    lo = max(mu - width, 0.0)
    hi = mu + width
    q = np.where(
        levels < 0.5,
        lo + (mu - lo) * (levels / 0.5),
        mu + (hi - mu) * ((levels - 0.5) / 0.5),
    )
    return np.clip(q, 0.0, None)


def interval_from_quantiles(
    levels: np.ndarray, qvals: np.ndarray, lo: float = 0.1, hi: float = 0.9
) -> tuple[float, float]:
    """Interpolate (lo, hi) interval bounds from a quantile lattice."""
    order = np.argsort(levels, kind="stable")
    lv = np.asarray(levels, dtype=float)[order]
    qv = np.asarray(qvals, dtype=float)[order]
    qv = np.maximum.accumulate(qv)
    return float(np.interp(lo, lv, qv)), float(np.interp(hi, lv, qv))


def median_from_quantiles(levels: np.ndarray, qvals: np.ndarray) -> float:
    """Interpolated median from a quantile lattice."""
    order = np.argsort(levels, kind="stable")
    lv = np.asarray(levels, dtype=float)[order]
    qv = np.asarray(qvals, dtype=float)[order]
    qv = np.maximum.accumulate(qv)
    return float(np.interp(0.5, lv, qv))


def score_quantiles(levels: np.ndarray, qvals: np.ndarray, actual: float) -> dict[str, float]:
    """CRPS / MAE / 80% coverage from a quantile forecast."""
    crps = float(crps_from_quantiles(np.asarray(levels), np.asarray(qvals), actual))
    med = median_from_quantiles(levels, qvals)
    q10, q90 = interval_from_quantiles(levels, qvals)
    return {
        "crps": crps,
        "mae": abs(med - actual),
        "coverage_80": float(q10 <= actual <= q90),
        "pred_median": med,
        "q10": q10,
        "q90": q90,
    }


def mom_scale(train: np.ndarray, window: int = 12) -> float:
    """Std of trailing month-over-month SMM changes (regime-shift scale)."""
    diffs = np.diff(train[-(window + 1):])
    if len(diffs) < 2:
        return 0.0
    return float(np.std(diffs, ddof=1))


def scenario_description(origin_label: str, forecast_label: str, smm_now: float) -> str:
    """Build the SMM scenario for the regime reader."""
    return (
        f"US mortgage market, end of {origin_label}. The Freddie Mac single-month "
        f"mortality (SMM) prepayment rate stands at {smm_now:.4f} "
        f"({smm_now * 100:.2f}% of outstanding balance prepaid this month). "
        f"Borrowers prepay mainly by refinancing into lower rates or selling "
        f"homes; falling mortgage rates encourage refinancing (SMM up), rising "
        f"rates cause lock-in (SMM down). Assess the expected DIRECTION of the "
        f"SMM shift for the coming month ({forecast_label})."
    )


def get_regime(
    rag: RAGTool,
    groq: GroqChatClient,
    origin_label: str,
    forecast_label: str,
    cutoff: str,
    smm_now: float,
) -> dict[str, object]:
    """Retrieve news (PIT-safe) and read the regime. Never raises."""
    triples: list[tuple[str, str, str]] = []
    n_hits = 0
    try:
        for q in QUERIES:
            res = rag.search(q, before=cutoff, limit=ARTICLES_PER_QUERY)
            n_hits += len(res.articles)
            for a in res.articles:
                # Belt-and-braces PIT check: RAG already enforces pub_date < cutoff.
                assert a.pub_date < cutoff, f"PIT violation: {a.pub_date} >= {cutoff}"
                t = (a.pub_date, a.title, a.url)
                if t not in triples:
                    triples.append(t)
        triples.sort(key=lambda t: t[0], reverse=True)
    except Exception as exc:  # noqa: BLE001 - retrieval failure -> no signal
        return {"regime": "ERROR", "confidence": 0.0, "applied": False,
                "error": f"rag: {exc}", "n_articles": 0}

    reading: RegimeReading | None = None
    try:
        reading = read_regime_with_news(
            client=groq,
            description=scenario_description(origin_label, forecast_label, smm_now),
            base_rate=smm_now,
            articles=triples,
            use_v2_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001 - LLM failure -> no signal
        return {"regime": "ERROR", "confidence": 0.0, "applied": False,
                "error": f"reader: {exc}", "n_articles": len(triples)}

    shrunk = conservative_shrinkage(reading.confidence)
    applied = reading.regime in ("UP", "DOWN") and shrunk >= CONF_GATE
    sign = {"UP": 1.0, "DOWN": -1.0}.get(reading.regime, 0.0)
    return {
        "regime": reading.regime,
        "confidence": round(reading.confidence, 3),
        "shrunk_confidence": round(shrunk, 3),
        "reason": reading.reason,
        "sign": sign,
        "applied": applied,
        "n_articles": len(triples),
        "error": None,
    }


def main() -> int:
    t0 = time.time()
    df = load_smm()
    smm = df["smm"].to_numpy(dtype=float)
    dates = df["date"]
    print(f"[exp19] {len(smm)} months: {dates.iloc[0].date()}..{dates.iloc[-1].date()}",
          flush=True)

    origin_idx = list(range(MIN_HISTORY, len(smm) - 1))
    assert len(origin_idx) == N_ORIGINS
    print(f"[exp19] {N_ORIGINS} origins: "
          f"{dates.iloc[origin_idx[0]].strftime('%Y-%m')}.."
          f"{dates.iloc[origin_idx[-1]].strftime('%Y-%m')}", flush=True)

    rag = RAGTool(groq=None)  # BM25-only retrieval: zero Groq spend
    groq = GroqChatClient(call_budget=GROQ_BUDGET, max_tokens=4000)

    per_origin: list[dict[str, object]] = []
    partial_path = ROOT / "experiments" / "exp19_news_smm_partial.json"
    done_origins: set[str] = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            per_origin = prev.get("per_origin", [])
            done_origins = {r["origin"] for r in per_origin}
            print(f"[exp19] resuming: {len(done_origins)} origins already done",
                  flush=True)
        except Exception:  # noqa: BLE001 - corrupt partial file: start fresh
            pass

    for oi, idx in enumerate(origin_idx):
        origin_label = dates.iloc[idx].strftime("%Y-%m")
        if origin_label in done_origins:
            continue
        forecast_label = dates.iloc[idx + 1].strftime("%Y-%m")
        # PIT cutoff: first day of the forecast month (news strictly before).
        cutoff = dates.iloc[idx + 1].strftime("%Y-%m-%d")
        train = smm[: idx + 1]
        actual = float(smm[idx + 1])
        smm_now = float(train[-1])

        # --- base forecasts (same as exp18) ---
        p_w3 = m0_point(train, 3)
        base_point = {"crps": abs(p_w3 - actual), "mae": abs(p_w3 - actual),
                      "pred": p_w3}
        cq = conformal_quantiles(train, CONF_LEVELS)
        base_conf = score_quantiles(CONF_LEVELS, cq, actual)

        # --- regime signal ---
        reg = get_regime(rag, groq, origin_label, forecast_label, cutoff, smm_now)

        # --- regime-adjusted forecasts ---
        scale = mom_scale(train)
        shift = 0.0
        if reg["applied"]:
            shift = float(reg["sign"]) * scale * float(reg["shrunk_confidence"])  # type: ignore[arg-type]

        p_adj = max(p_w3 + shift, 0.0)
        adj_point = {"crps": abs(p_adj - actual), "mae": abs(p_adj - actual),
                     "pred": p_adj}
        cq_adj = np.clip(cq + shift, 0.0, None)
        adj_conf = score_quantiles(CONF_LEVELS, cq_adj, actual)

        per_origin.append({
            "origin": origin_label,
            "forecast_month": forecast_label,
            "cutoff": cutoff,
            "actual": actual,
            "mom_scale": scale,
            "shift": shift,
            "regime": dict(reg),
            "models": {
                "m0_w3": base_point,
                "m0_w3_regime": adj_point,
                "m0_conformal": base_conf,
                "m0_conformal_regime": adj_conf,
            },
        })
        if len(per_origin) % 5 == 0:
            partial_path.write_text(json.dumps({"per_origin": per_origin}, default=float))
        if (oi + 1) % 5 == 0 or oi == 0:
            r = reg["regime"]
            ap = "APPLIED" if reg["applied"] else "no-action"
            print(f"[exp19] {oi + 1}/{N_ORIGINS} {origin_label}: regime={r} "
                  f"conf={reg.get('confidence')} {ap} shift={shift:+.6f} "
                  f"({time.time() - t0:.0f}s, groq={groq.calls})", flush=True)

    per_origin.sort(key=lambda r: r["origin"])  # type: ignore[arg-type]
    model_names = ["m0_w3", "m0_w3_regime", "m0_conformal", "m0_conformal_regime"]
    summary: dict[str, dict[str, float]] = {}
    for m in model_names:
        crps_v = [r["models"][m]["crps"] for r in per_origin
                  if np.isfinite(r["models"][m]["crps"])]
        mae_v = [r["models"][m]["mae"] for r in per_origin
                 if np.isfinite(r["models"][m]["mae"])]
        cov_v = [r["models"][m].get("coverage_80", float("nan")) for r in per_origin
                 if np.isfinite(r["models"][m].get("coverage_80", float("nan")))]
        summary[m] = {
            "mean_crps": float(np.mean(crps_v)),
            "mean_mae": float(np.mean(mae_v)),
            "coverage_80": float(np.mean(cov_v)) if cov_v else float("nan"),
            "n_origins": len(crps_v),
        }

    # Paired comparison: adjusted vs base (per-origin CRPS deltas).
    comparison: dict[str, dict[str, float]] = {}
    for base, adj in (("m0_w3", "m0_w3_regime"),
                      ("m0_conformal", "m0_conformal_regime")):
        deltas = [r["models"][adj]["crps"] - r["models"][base]["crps"]
                  for r in per_origin
                  if np.isfinite(r["models"][adj]["crps"])
                  and np.isfinite(r["models"][base]["crps"])]
        d = np.asarray(deltas)
        wins = int(np.sum(d < 0))
        losses = int(np.sum(d > 0))
        ties = int(np.sum(d == 0))
        # Paired t-test on the deltas.
        if len(d) > 1 and np.std(d, ddof=1) > 0:
            t_stat = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))
        else:
            t_stat = 0.0
        from math import erf, sqrt  # noqa: E402
        # Two-sided p via normal approximation (|t| large-n); report t and let readers judge.
        p_approx = float(erf(abs(t_stat) / sqrt(2)))  # P(|Z| <= |t|); p = 1 - this
        comparison[f"{adj}_vs_{base}"] = {
            "mean_delta_crps": float(np.mean(d)),
            "std_delta_crps": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "t_stat": t_stat,
            "p_two_sided_normal_approx": round(1.0 - p_approx, 4),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "n": len(d),
        }

    # Regime-action audit: how often did we act, and did acting help?
    acted = [r for r in per_origin if r["regime"]["applied"]]
    abstained = [r for r in per_origin if r["regime"]["regime"] == "ABSTAIN"]
    regime_counts: dict[str, int] = {}
    for r in per_origin:
        k = str(r["regime"]["regime"])
        regime_counts[k] = regime_counts.get(k, 0) + 1
    acted_delta = [r["models"]["m0_w3_regime"]["crps"] - r["models"]["m0_w3"]["crps"]
                   for r in acted]

    out = {
        "experiment": "exp19_news_smm",
        "config": {
            "series": "Freddie Mac monthly SMM, 2021-01..2026-03 (63 months)",
            "origins": N_ORIGINS,
            "origin_range": [per_origin[0]["origin"], per_origin[-1]["origin"]],
            "min_history_months": MIN_HISTORY,
            "models": model_names,
            "rag": "BM25-only (groq=None), queries=['mortgage rates','refinance'], "
                   "5 articles/query, PIT cutoff = first day of forecast month",
            "regime_reader": "read_regime_with_news v2 prompt, 1 Groq call/origin",
            "adjustment": "shift = sign * trailing-12m MoM std * shrunk_conf; "
                          f"applied iff regime in {{UP,DOWN}} and shrunk_conf >= {CONF_GATE}; "
                          "NONE/ABSTAIN/ERROR -> no adjustment",
            "pit_note": "train months <= origin; news pub_date < forecast-month start "
                        "(asserted per article); RAG enforces in SQL + client-side",
            "crps_method": "point: CRPS=MAE; quantiles: pinball estimator (same as exp18)",
        },
        "summary": summary,
        "comparison": comparison,
        "regime_audit": {
            "regime_counts": regime_counts,
            "n_acted": len(acted),
            "n_abstained": len(abstained),
            "mean_delta_when_acted": float(np.mean(acted_delta)) if acted_delta else 0.0,
            "groq_calls_used": groq.calls,
        },
        "per_origin": per_origin,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp19_news_smm_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    # Clean up the partial file on success.
    with contextlib.suppress(FileNotFoundError):
        partial_path.unlink()
    print(f"[exp19] wrote {out_path}", flush=True)
    print("[exp19] summary:", json.dumps(summary, indent=2), flush=True)
    print("[exp19] comparison:", json.dumps(comparison, indent=2), flush=True)
    print("[exp19] regime_audit:",
          json.dumps(out["regime_audit"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
