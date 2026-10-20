# exp66d: Cohort Recovery + LLM Risk Classifier — Report

**Date:** 2026-10-19
**Status:** Complete. Follow-up to exp66c (deterministic grouping + LLM explanations).
**Setup:** seed 42, 4,000 customers, 36 months, 10 deterministic groups
(kmeans + tree-surrogate rules), same config as exp66c.

## Task 1 — Cohort recovery: ❌ honest negative

The simulator plants ground-truth cohorts that drive heterogeneous
switching propensity:

| Cohort | Definition (DGP) | n |
|---|---|---|
| `low_rate` | fixed, originated 2021, rate ≤ 3% | 1,158 |
| `osfi_exempt` | uninsured + 720+ credit | 385 |
| `family_squeeze` | income < 75k + ≥3 kids | 286 |
| `control` | everything else | 2,171 |

**Does the deterministic grouper recover them? No.**

- **ARI = 0.010** — chance-level agreement between discovered groups
  and planted cohorts.
- Per-cohort best-match:

| Cohort | Best group | Precision | Recall | F1 |
|---|---|---|---|---|
| `low_rate` | g0 | 0.42 | 0.47 | 0.45 |
| `control` | g8 | 0.72 | 0.21 | 0.32 |
| `family_squeeze` | g0 | 0.12 | 0.54 | 0.20 |
| `osfi_exempt` | g8 | 0.14 | 0.23 | 0.18 |

`low_rate` is partially recovered (the tree's rate splits pick up the
low-origination-rate borrowers), but the sharp interaction rules that
define the cohorts (fixed × 2021 × ≤3%; income × ≥3 kids) are not the
variance-dominant structure kmeans clusters on. `family_squeeze` and
`osfi_exempt` are essentially missed.

**But the groups are not noise.** Diagnostic ARIs for the same groups:
rate_type 0.24, income quintile 0.10, province −0.00, credit band −0.00.
The grouper finds real, reproducible structure — just not the DGP's
causal joints.

**Implication for explainability:** the risk reports (exp66c) are true
descriptions of real borrower segments, but the segments are not the
ones that drive attrition heterogeneity in the DGP. The explanations
explain the wrong (if real) thing for forecasting purposes. They remain
useful as segment narratives; they are not evidence that we have
identified the true risk drivers.

## Task 2 — LLM as at-risk classifier

Deterministic training-tertile bins were inverted on test (exp66c):
HIGH 0.087, MEDIUM 0.068, LOW 0.128 realized test attrition.

The LLM classifier (per-group aggregate stats + training attrition +
PIT-safe news chunks, HIGH/MEDIUM/LOW output) on the same 10 groups:

| Method | HIGH | MEDIUM | LOW | Monotonic? |
|---|---|---|---|---|
| Deterministic bins | 0.087 | 0.068 | 0.128 | ❌ inverted |
| LLM classifier | 0.114 | 0.071 | 0.091 | ✅ HIGH > MED > LOW |

The LLM was more conservative (HIGH for 2/10 groups vs bins' 4/10) and
its ordering came out monotonic on test. It explicitly used news the
bins ignore — e.g. g0 justification: "low historical attrition rate of
10.71% ... stability is supported by stable interest rate environments
[url:0], [url:1], [url:3]" → LOW (test realized 0.070, correct).

**Honest caveats:**
1. n=10 groups — this is directional evidence, not a significance claim.
2. The LLM had strictly more information (news chunks) than the bins.
   The comparison is "bins vs bins+LLM+news", not a controlled ablation
   of the LLM alone.
3. Both methods fail together on g3 (train realized 0.0000 → LOW from
   both; test realized 0.1667, the highest). When the signal does not
   transfer, the LLM cannot fix it — it shares the bins' blindness.

## Task 3 — Worked examples

Four full worked examples live in
[`docs/GROUPING_EXAMPLES.md`](../docs/GROUPING_EXAMPLES.md): g0 (cleanest
cohort mapping + LLM/bin disagreement, LLM right), g8 (mixed group,
both agree HIGH), g5 (affluent segment, LLM corrects down), g3 (honest
failure, both wrong). Explanations passed citations 4/4.

## Verdict

1. ❌ **Cohort recovery: honest negative.** ARI 0.01. Deterministic
   grouping describes real segments but does not find the DGP's planted
   risk structure. Any future "explainability" claim must carry this
   caveat.
2. ⚠️ **LLM risk classifier: directionally better than bins**
   (monotonic vs inverted), but n=10 and info-asymmetric. Worth a
   controlled follow-up (LLM-with-stats-only vs LLM-with-news) before
   any stronger claim.
3. ✅ **Examples shipped** — `docs/GROUPING_EXAMPLES.md` is the showcase
   for the grouping track, including the failure case.
