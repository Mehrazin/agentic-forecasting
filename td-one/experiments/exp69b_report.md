# exp69b: Broad-Synthesis Variant of exp69 — Report

**Date:** 2026-10-20
**Status:** COMPLETE — synthesis does not help (controlled comparison).
**Data:** `experiments/exp69b_results.json` (192 rows, 0 errors). Code:
`experiments/exp69b_synthesis_variant.py`.

## The hypothesis under test (user-directed)

exp69's LLM saw rate/policy announcements only. The user's hypothesis:
anticipation should *synthesize across a broad macro landscape* —
immigration tightening + tax change + trade friction jointly implying an
attrition shock — not just "rates going up."

## Design (controlled comparison; same grid/calibration as exp69)

- **Corpora per (shift, severity):**
  - `direct`: exp69's corpus (3 shock announcements + 5 distractors).
  - `broad`: ~25 articles across 12+ macro topics — the user's landscape:
    immigration policy, food/basic-needs inflation, income/tax rule
    changes, elections/political shifts, global commerce/trade, AI/startups
    + S&P 500, unemployment/labour, FX, recession risks, small-business
    insolvencies, oil/geopolitics, housing — = 3 direct announcements +
    6 corroborating macro signals (regime color behind the shock) +
    topical noise.
- **Arm B′:** LLM classifies the `direct` corpus (all visible articles).
  (Replication of exp69's B under the new prompt, which adds the
  `driving_trends` field — the comparison holds the classifier constant.)
- **Arm D:** semantic RAG (local MiniLM embeddings, generic query,
  top-8 chunks, PIT-filtered) retrieves from the `broad` corpus; the LLM
  classifies the retrieved set and must cite `driving_trends`.
- Arms A (Chronos-2 alone) and C (oracle) as in exp69; same calibration
  (seeds 200–202), same test seeds (300–302), same origins.

## Verdict: synthesis hurts (D < B′)

| Test | Result |
|---|---|
| **Primary:** D vs B′, straddle origins | mean ΔCRPS +0.0025 (**−5.0%** rel.), p=0.148, win rate **5.6%** → synthesis **worse** |
| D vs A | +6.9% rel., p=0.431, win 39% (n.s.) |
| B′ vs A (replication of exp69) | +11.9% rel., p=0.191, win 42% — consistent with exp69's +10.2% (p=0.231) |
| D vs C (oracle) | −4.4% rel., p=0.194, win 2.8% |
| Do-no-harm (NONEWS / POST, arm D) | Δ=0.0000 exactly → **PASS** both |

## Why synthesis lost: timing dilution

The LLM *did* synthesize — `driving_trends` for arm D cite corroborating
macro trends alongside the direct signal ("rate hike expected" +
"borrowing costs rising", "mortgage payment shock", "stricter lending
standards"). Classification of type and bucket stayed correct. What
degraded was **timing**, already the fragile component in exp69:

- S1/2σ at origin 27: B′ reports `months_from_now=3` (correct; onset=30).
  D reports **2** — the corroborating articles' vaguer timing language
  pulls the predicted onset a month early, wrongly shifting one pre-shock
  step.
- S1/4σ at origin 27: B′ reports 3 (correct). D reports **0** → the
  anticipatory guard abstains → D=A on exactly the rows where exp69's
  oracle gained most (0.127→0.073).

Retrieval audit (straddle origins): the direct announcements are only
**33%** of retrieved chunks; the rest is regime color (food inflation 30,
FX 24, oil/geopolitics 24, housing/recession/elections/immigration/tax 18
each). The color doesn't change the (correct) classification — it blurs
the (fragile) timing.

## Honest limitation

This test **cannot** detect synthesis value for *unannounced* shocks:
in this DGP the shock is always directly announced, so synthesis has
nothing to add and only noise to inject. The user's stronger scenario —
converging macro signals implying a shock nobody announced — remains
**untested, not refuted**. Testing it needs a DGP variant where the shift
has no direct announcement and the news carries only the converging
signals (the LLM would have to infer type, size *and* timing from color
alone — a harder task than anything in exp69/69b).

## Engineering notes

- `ShiftSignal` gained `driving_trends` (validated, capped at 5);
  classifier prompt requires it. 21/21 adjuster tests green (+3 new).
- Retrieval: `BroadRetriever` over `rag/semantic.py`'s `EmbeddingClient`
  (local all-MiniLM-L6-v2); one synthetic article = one chunk.
  56 real LLM calls (flash-lite), cached locally, not committed.
- exp69's headline result is unaffected: B′ replicates B (+11.9% vs
  +10.2%, both n.s.).
