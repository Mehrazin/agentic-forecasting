"""exp69b: broad-synthesis variant of exp69 (user-directed follow-up).

exp69's LLM saw rate/policy announcements only. The user's hypothesis:
anticipation should *synthesize across a broad macro landscape* — e.g.
immigration tightening + tax change + trade friction jointly implying an
attrition shock, not just "rates going up."

Design (controlled comparison, same grid/calibration as exp69):
- Corpus conditions per (shift, severity):
  - ``direct``: exp69's corpus (3 shock announcements + 5 distractors).
  - ``broad``: ~25 articles across 12+ macro topics (immigration, food
    inflation, tax rules, elections, trade, AI/startups + S&P 500,
    unemployment, FX, recession risk, small-business insolvencies,
    oil/geopolitics, housing) = 3 direct announcements + corroborating
    macro signals + topical noise.
- Arm B': LLM classifies the ``direct`` corpus (all visible articles).
- Arm D: semantic RAG (local MiniLM embeddings, generic query) retrieves
  top-8 chunks from the ``broad`` corpus (PIT-filtered), LLM classifies
  the retrieved set and must cite ``driving_trends``.
- Arms A (TSFM alone) and C (oracle) as in exp69. Same calibrated mapping.
- Primary comparison: D vs B' on straddle origins (paired t-test). If
  synthesis adds nothing beyond direct news, D ≈ B'. The retrieval audit
  (which topics surfaced) and driving_trends audit say what the LLM used.

Note: the classifier prompt gained the ``driving_trends`` field after
exp69 ran, so B' is a replication of exp69's B under the new prompt —
the comparison B' vs D holds the classifier constant and isolates the
corpus/retrieval effect.

Usage:
    .venv/bin/python experiments/exp69b_synthesis_variant.py --mode mock
    .venv/bin/python experiments/exp69b_synthesis_variant.py --mode real
"""

from __future__ import annotations

import argparse
import hashlib
import json

# Reuse exp69's harness verbatim (same grid, calibration, oracle).
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from td_one.agent.anticipatory_adjuster import (
    AnticipatoryClassifier,
    ShiftSignal,
    apply_overlay,
    month_label,
)
from td_one.evaluation.real_benchmark import crps_from_quantiles
from td_one.rag.semantic import EmbeddingClient, cosine_sim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp69_anticipatory_llm import (  # noqa: E402
    HORIZON,
    NONEWS_ORIGINS,
    ONSET,
    POST_ORIGINS,
    SEVS,
    SHIFTS,
    STRADDLE_ORIGINS,
    TEST_SEEDS,
    MockNewsClient,
    _mock_tsfm,
    _real_tsfm,
    make_news,
    make_placebo,
    make_series,
    oracle_signal,
    run_calibration,
)

RETRIEVAL_K = 8
RETRIEVAL_QUERY = (
    "upcoming changes affecting mortgage borrowers: interest rates, lending "
    "rules, taxes, jobs, trade, economic risks in the next few months"
)
CACHE_PATH = Path(__file__).resolve().parent / "exp69b_llm_cache.json"

#: Corroborating macro articles per shift: (month, topic, title, body).
#: These jointly paint the macro regime behind the announced shock; none of
#: them names the shock directly.
_CORRO_S1 = [
    (
        26,
        "recession_risk",
        "Recession risk rises as yield curve inverts",
        "Bond markets are pricing a rising probability of recession over the next year.",
    ),
    (
        27,
        "unemployment",
        "Jobless claims tick up for third straight month",
        "Initial unemployment claims rose again, and hiring in rate-sensitive sectors slowed.",
    ),
    (
        27,
        "housing",
        "Housing market cools as listings sit longer",
        "Home sales fell and average days on market rose as borrowing costs climbed.",
    ),
    (
        28,
        "small_business",
        "Small-business insolvencies edge higher",
        "Insolvency filings among small firms rose, with owners citing debt-service costs.",
    ),
    (
        28,
        "fx",
        "Canadian dollar slides to two-year low",
        "The loonie weakened as rate differentials and risk sentiment moved against it.",
    ),
    (
        29,
        "food_inflation",
        "Food inflation stays stubbornly high",
        "Grocery prices rose again, squeezing household budgets already hit by payments.",
    ),
]
_CORRO_S2 = [
    (
        26,
        "elections",
        "New government vows housing affordability crackdown",
        "The incoming administration campaigned on cooling housing speculation and "
        "tightening lending standards.",
    ),
    (
        27,
        "tax_rules",
        "Ottawa floats tax changes for property investors",
        "Finance officials are studying higher taxes on secondary properties and short-term flips.",
    ),
    (
        27,
        "immigration",
        "Immigration targets under review amid housing strain",
        "Ottawa is reviewing intake levels, citing pressure on housing supply.",
    ),
    (
        28,
        "trade",
        "Trade friction clouds export outlook",
        "New tariff threats are raising input costs for builders and manufacturers.",
    ),
    (
        28,
        "housing",
        "Affordability crisis dominates Question Period",
        "Opposition MPs hammered the government over record unaffordability in major cities.",
    ),
    (
        29,
        "unemployment",
        "Construction employment softens",
        "Building-sector jobs fell for a second month as projects were shelved.",
    ),
]
#: Topical noise (no bearing on either shock), both shifts.
_NOISE = [
    (
        20,
        "ai_startups",
        "AI startups raise record funding round",
        "Canadian AI firms closed a record quarter of venture funding.",
    ),
    (
        21,
        "sp500",
        "S&P 500 notches another record close",
        "US equities rose on strong tech earnings.",
    ),
    (
        22,
        "oil_geopolitics",
        "Oil steady as OPEC holds output",
        "Crude prices were little changed amid stable supply.",
    ),
    (
        23,
        "food_inflation",
        "Harvest outlook improves for Prairie wheat",
        "Crop forecasts were revised up on favourable weather.",
    ),
    (
        24,
        "elections",
        "Provincial election called for the fall",
        "The premier announced a snap provincial election.",
    ),
    (
        25,
        "fx",
        "Euro firms against the US dollar",
        "The euro gained on hawkish central-bank commentary.",
    ),
    (
        26,
        "ai_startups",
        "Toronto startup hub expands",
        "A new downtown tech campus will house 40 startups.",
    ),
    (
        28,
        "oil_geopolitics",
        "Pipeline decision delayed again",
        "Regulators pushed back a decision on a major pipeline expansion.",
    ),
    (
        29,
        "sp500",
        "Retail sales beat expectations in the US",
        "American consumers kept spending despite higher prices.",
    ),
]


def make_broad_news(shift: str, severity: float) -> list[dict[str, object]]:
    """Broad macro corpus: direct announcements + corroborating + noise."""
    articles = list(make_news(shift, severity))  # direct + exp69 distractors
    for a in articles:
        a.setdefault("topic", "direct")
    corro = _CORRO_S1 if shift == "S1" else _CORRO_S2
    for m, topic, title, body in corro:
        articles.append(
            {
                "title": title,
                "body": body,
                "article_date": month_label(m),
                "source": "synth",
                "topic": topic,
            }
        )
    for m, topic, title, body in _NOISE:
        articles.append(
            {
                "title": title,
                "body": body,
                "article_date": month_label(m),
                "source": "synth",
                "topic": topic,
            }
        )
    assert all(int(str(a["article_date"])[1:]) < ONSET for a in articles)
    return articles


class BroadRetriever:
    """Semantic retrieval over a per-combo corpus (local MiniLM embeddings).

    Articles are embedded once per corpus; retrieval is PIT-filtered to
    article_date <= origin before ranking. One article = one chunk (the
    synthetic articles are short; fixed-512 chunking from exp66d applies
    to long-form bodies, not these).
    """

    def __init__(self, encoder: EmbeddingClient):
        self._enc = encoder
        self._cache: dict[str, tuple[list[dict[str, object]], np.ndarray]] = {}

    def retrieve(
        self,
        corpus_key: str,
        corpus: list[dict[str, object]],
        origin_label: str,
        k: int = RETRIEVAL_K,
    ) -> list[dict[str, object]]:
        if corpus_key not in self._cache:
            texts = [f"{a['title']}: {a.get('body', '')}" for a in corpus]
            self._cache[corpus_key] = (corpus, self._enc.encode(texts))
        articles, embs = self._cache[corpus_key]
        visible_idx = [i for i, a in enumerate(articles) if str(a["article_date"]) <= origin_label]
        if not visible_idx:
            return []
        q = self._enc.encode([RETRIEVAL_QUERY])[0]
        scored = sorted(visible_idx, key=lambda i: -cosine_sim(q, embs[i]))[:k]
        return [articles[i] for i in scored]


class CachedClassifierB:
    """Classifier with disk cache; stores driving_trends (exp69b schema)."""

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

    @staticmethod
    def _key(articles: list[dict[str, object]], origin_label: str) -> str:
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
                tuple(d.get("driving_trends", [])),
            )
        sig = self._clf(articles, origin, origin_label)
        self._cache[key] = {
            "shift_type": sig.shift_type,
            "direction": sig.direction,
            "magnitude_bucket": sig.magnitude_bucket,
            "months_from_now": sig.months_from_now,
            "timing_confidence": sig.timing_confidence,
            "rationale": sig.rationale,
            "driving_trends": list(sig.driving_trends),
        }
        self.llm_calls += 1
        import contextlib

        with contextlib.suppress(OSError):
            self._path.write_text(json.dumps(self._cache, indent=1))
        return sig


def run_eval(
    tsfm_fn,
    calibration: dict[tuple[str, str], float],
    classifier: CachedClassifierB,
    retriever: BroadRetriever,
    out_path: Path,
    resume: bool = False,
) -> list[dict]:
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
    for i, (shift, sev, seed) in enumerate(combos):
        if (shift, sev, seed) in done:
            continue
        combo_rows: list[dict] = []
        series = make_series(shift, sev, seed)
        placebo = make_placebo(seed)
        direct_news = make_news(shift, sev)
        broad_news = make_broad_news(shift, sev)
        corpus_key = f"{shift}/{sev}"
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
                    sig_b: dict[str, object] = {}
                    sig_d: dict[str, object] = {}
                    retrieved_topics: list[str] = []
                    if label == "shifted":
                        # Arm B': direct corpus, all visible articles.
                        sb = classifier(direct_news, origin, olabel)
                        sig_b = {
                            "shift_type": sb.shift_type,
                            "magnitude_bucket": sb.magnitude_bucket,
                            "months_from_now": sb.months_from_now,
                            "driving_trends": list(sb.driving_trends),
                        }
                        arms["B"] = crps_from_quantiles(
                            lv_a,
                            apply_overlay(lv_a, vals_a, sb, origin, calibration),
                            actual,
                        )
                        # Arm D: broad corpus via semantic retrieval.
                        retrieved = retriever.retrieve(corpus_key, broad_news, olabel)
                        retrieved_topics = [str(a.get("topic", "?")) for a in retrieved]
                        sd_sig = classifier(retrieved, origin, olabel)
                        sig_d = {
                            "shift_type": sd_sig.shift_type,
                            "magnitude_bucket": sd_sig.magnitude_bucket,
                            "months_from_now": sd_sig.months_from_now,
                            "driving_trends": list(sd_sig.driving_trends),
                        }
                        arms["D"] = crps_from_quantiles(
                            lv_a,
                            apply_overlay(lv_a, vals_a, sd_sig, origin, calibration),
                            actual,
                        )
                        # Arm C: oracle (ceiling), as in exp69.
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
                    else:
                        arms.update({"B": crps_a, "D": crps_a, "C": crps_a})
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
                            "signal_B": sig_b,
                            "signal_D": sig_d,
                            "retrieved_topics": retrieved_topics,
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
            entry["paired_p"] = 1.0
            entry["verdict"] = "PASS" if harm_only else "FAIL"
            out[name] = entry
            return
        _t, p = stats.ttest_1samp(d, 0.0)
        entry["paired_p"] = float(p)
        if harm_only:
            entry["verdict"] = "PASS" if not (p < 0.05 and np.mean(d) > 0) else "FAIL"
        else:
            entry["verdict"] = "PASS" if (p < 0.05 and np.mean(d) < 0) else "FAIL"
        out[name] = entry

    sh = df[df["series"] == "shifted"]
    st = sh[sh["origin_class"] == "straddle"]
    _test(st, "D", "B", "primary_D_vs_B_straddle")
    _test(st, "D", "A", "D_vs_A_straddle")
    _test(st, "B", "A", "B_vs_A_straddle_replication")
    _test(st, "D", "C", "D_vs_C_straddle")
    _test(sh[sh["origin_class"] == "nonews"], "D", "A", "do_no_harm_nonews_D", harm_only=True)
    _test(sh[sh["origin_class"] == "post"], "D", "A", "do_no_harm_post_D", harm_only=True)

    # Audits: what did retrieval surface, and what did the LLM cite?
    from collections import Counter

    topics: Counter = Counter()
    trends_d: Counter = Counter()
    n_direct_retrieved = 0
    n_retr = 0
    for r in rows:
        if r["status"] == "ok" and r["series"] == "shifted" and r["origin_class"] == "straddle":
            topics.update(r["retrieved_topics"])
            n_retr += len(r["retrieved_topics"])
            n_direct_retrieved += sum(1 for t in r["retrieved_topics"] if t == "direct")
            s = r["signal_D"]
            if isinstance(s, dict):
                trends_d.update(s.get("driving_trends", []))
    out["retrieval_audit"] = {
        "topics": dict(topics),
        "direct_share": (n_direct_retrieved / n_retr) if n_retr else 0.0,
    }
    out["driving_trends_D"] = dict(trends_d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--out", default="experiments/exp69b_results.json")
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
        llm_client: object = VectorProxyClient(model="flash-lite", call_budget=300)
    else:
        wrapper = None
        tsfm_fn = _mock_tsfm
        llm_client = MockNewsClient()

    print(f"exp69b [{args.mode}]: broad-synthesis variant (pre-registered)")
    t_start = time.time()
    calibration = run_calibration()
    classifier = CachedClassifierB(
        AnticipatoryClassifier(llm_client),  # type: ignore[arg-type]
        CACHE_PATH,
    )
    retriever = BroadRetriever(EmbeddingClient())
    rows = run_eval(tsfm_fn, calibration, classifier, retriever, Path(args.out), args.resume)
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
    print(f"\nexp69b [{args.mode}] done in {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
