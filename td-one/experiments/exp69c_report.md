# exp69c: Real-Data Anticipatory LLM Shift Adjustment — Report

**Date:** 2026-10-20
**Status:** PRE-REGISTERED (design locked; results pending).
**Data:** `experiments/exp69c_results.json`. Code:
`experiments/exp69c_real_data.py`, `src/td_one/agent/story_pipeline.py`.

## Why this exists

exp69 failed on synthetic data — and the *oracle* failed too (+11.4%,
p=0.214), so the mechanism itself was the limit, not the LLM. exp69b
showed broad synthesis *blurs timing* (−5.0%, win 5.6%). The user's
counter-hypothesis: real shocks have richer, uneven, real textual buildup
that a single LLM call on a bag of chunks cannot exploit. exp69c tests
the anticipatory idea **on real data, with better scaffolding**:

1. **Real shock episodes, real series:** the 2022 BoC hiking cycle
   (first hike 2022-03; 100bp super-hike 2022-07), the 2020 COVID shock,
   and the 2021 inflation build — on Food CPI year-over-year inflation
   (2010–2026) and Freddie Mac SMM attrition (2021–2026). News: the real
   `data/news/news_2026.db`, PIT-safe by `pub_date`.
2. **Timing reminders:** the classifier gets explicit timeline anchors —
   "today is {date}, the forecast window is {months}, key dated events:
   ..." (exp69/69b's fragile component, now scaffolded).
3. **News as story:** a chronological narrative (ordered dated events +
   "what's building") instead of a bag of chunks.
4. **Multi-call pipeline:** retrieve (semantic RAG) → rank (LLM, top-12
   with reasons) → story (LLM) → classify (LLM, ShiftSignal). One focused
   job per call.

A sibling agent is building a Macro Signal Service in parallel; the
retrieval function exposes an `extra_articles` seam for its outputs, but
exp69c does not block on it.

## Pre-registered design (locked before running)

**Series & targets.** Food CPI → year-over-year inflation *fraction*
(`value/100`, in [0, 1] so the existing overlay machinery applies
unchanged). SMM → the rate itself. Horizon h=6. Baseline forecaster A =
Chronos-2 zero-shot quantiles (the same A as exp69/exp67).

**Episodes.**

| ID | Series | Onset | Event | Role |
|---|---|---|---|---|
| C1a | Food CPI | 2020-03 | COVID shock | calibration (sparse news: 3 pre-onset articles — expected "none") |
| C1b | Food CPI | 2021-10 | inflation build (YoY 4.6→6.3%) | calibration |
| C2 | SMM | 2022-03 | first BoC hike | calibration |
| T1 | Food CPI | 2022-03 | first hike | test (origins 2021-11…2022-02) |
| T2 | Food CPI | 2022-07 | 100bp super-hike | test (origins 2022-03…2022-06) |
| T3 | SMM | 2024-06 | first cut | test (origins 2024-02…2024-05) |

**Calibration.** Per (series, test origin): classify each calibration
episode with the story pipeline at origin = onset−1 (news only, PIT-safe);
measure `|median(post[onset:onset+6]) − median(pre[onset−6:onset])|`;
table maps `magnitude_bucket → |level change|`, median over episodes.
An episode contributes only if `onset+6 ≤ origin` (realized post window
fully observed — strict PIT). **Absolute value** (new `absolute=True`
flag): the LLM owns *direction*; statistics own *size*. The signed
version would flip an opposite-direction test (e.g. hiking calibration →
cutting test) the wrong way — regression-tested.

**Arms** (per straddle origin):
- **A** = Chronos-2 alone.
- **B** = A + overlay from the multi-call pipeline *with* timeline anchors.
- **B_notl** = A + overlay from the multi-call pipeline *without* timeline
  anchors (ablation: does the timeline earn its keep?).
- **C** = A + overlay from a single-call baseline: top-30 retrieved chunks
  → `AnticipatoryClassifier` (replicates exp69b's approach on real data;
  isolates the scaffolding effect). Same calibration table as B (the arms
  differ only in test-time signal generation — controlled comparison).

**Do-no-harm.** Placebo origins (Food CPI 2018-06/07, SMM 2023-06/07 —
calm windows, no shock in the following 6 months) and post origins
(onset+1 per test episode): expect ΔCRPS = 0 (classifier abstains / the
overlay's own origin ≥ onset guard fires).

**Primary claims & kill criteria.**
1. B beats A on pooled straddle origins (paired t-test on CRPS, p<0.05).
2. B beats C on pooled straddle origins (p<0.05) — the scaffolding must
   earn its keep.
3. Secondary: B vs B_notl (timeline ablation); per-episode B vs A
   (2022 hikes vs 2024 cut may differ — reported honestly).

If (1) or (2) fails: **dead, no iteration.**

**Known limitations (stated upfront).**
- n ≈ 12 straddle rows (3 episodes × 4 origins) — weak power; a null
  result cannot distinguish "no effect" from "underpowered".
- Food CPI tests depend on C1b's classification (C1a's news is too sparse
  to classify); if both yield "none", Food CPI rows no-op by construction.
- SMM's calibrated |Δ| is small (~0.0003) relative to monthly noise
  (~±0.001) — the overlay may be correctly sized yet undetectable.
- Chunks are article *titles* (no bodies for 2022/2024 articles) — the
  story is built from dated headlines.

## Results

*(pending — filled after the run)*

## Verdict

*(pending)*
