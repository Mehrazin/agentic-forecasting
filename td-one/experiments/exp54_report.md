# exp54 — Rung 1: dataset-registry reproduction check

**Date:** 2026-10-18T09:37:27.878119+00:00 | **Mode:** full | **Elapsed:** 175s

## Verdict

**PASS: registry reproduces exp49 exactly (rung 1 expected null confirmed)**

Rung 1's hypothesis was that formalizing the dataset -> regime-spec mapping
changes nothing for classical experts (they don't read text). The registry
wiring is confirmed behavior-preserving: the value of this rung is the
infrastructure (``td_one/llm_rungs/datasets.py``), which rungs 2-5 build on.

## What was checked

- **Structural:** ``ProposedForecaster(dataset="m4")`` selects the m4 regime
  spec + the four classical models; ``dataset="food_cpi"`` selects the
  generic spec + the same models — the entries exp47/exp49 passed by hand.
- **Internal consistency:** registry-wired vs explicit ``regime_spec`` /
  ``models`` construction give bit-identical per-origin CRPS on the same
  records (max diff 0.0 on both datasets).

- **External:** every reported number (mean CRPS for proposed / equal /
  best-single; paired t, p, win rate, mean diff for proposed-vs-equal and
  proposed-vs-best) reproduces ``exp49_results.json`` to < 1e-12, the
  majority bit-identical (diff exactly 0.0).

## Per-dataset detail

| dataset | n records | registry vs explicit (max diff) | proposed mean CRPS |
|---|---|---|---|
| m4 | 600 | 0 | 0.145057 |
| food_cpi | 24 | 0 | 0.014127 |

## vs exp49 reference

| dataset | max abs diff | exactly-zero fields | worst field |
|---|---|---|---|
| m4 | 1.78e-15 | 7/15 | proposed_vs_best.t |
| food_cpi | 0 | 15/15 | mean_crps.proposed |

## PIT note

No new data in this rung — the registry descriptions are timeless metadata
(no numbers, no dates), so PIT safety holds trivially. The record pipeline
is exp49's own (PIT-safe by its construction).
