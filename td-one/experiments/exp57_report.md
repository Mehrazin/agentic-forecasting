# exp57 — Rung 4: per-dataset prompts (PIT-safe)

**Date:** 2026-10-19T15:04:55.038202+00:00 | **Elapsed:** 0s
Evaluated on **59 common origins** (generic / promptV1 / promptV2 caches).

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
| promptV1 vs generic | 0.4262 vs 0.5158 | -0.0896 | -1.94 | 0.05713 | 0.61 |
| promptV2 vs generic | 0.4599 vs 0.5158 | -0.0559 | -1.11 | 0.273 | 0.59 |
| promptV1 vs promptV2 | 0.4262 vs 0.4599 | -0.0337 | -1.29 | 0.2037 | 0.51 |
Standalone mean CRPS (recomputed from 9-level grids): generic=0.5158, promptV1=0.4262, promptV2=0.4599.
(Worker-stored means: generic=0.5158, promptV1=0.4262, promptV2=0.4599.)

## Mini-router over prompt variants

Blocked 5-fold CV: each fold picks the variant with the best *train-fold*
mean CRPS and scores the test fold with it.

- Fold picks: fold 0->promptV1, fold 1->promptV1, fold 2->promptV1, fold 3->promptV1, fold 4->promptV1
- Mini-router mean CRPS: 0.4262
- vs best single variant (promptV1): identical — the router picked it in every fold, selection adds nothing
- vs generic: diff=-0.0896, p=0.05713

## Setup notes

- Per-origin CRPS is recomputed from the 9-level (0.1..0.9) subset of each
  cached 11-level grid with `grid_to_crps`, so all three variants are scored
  identically.
- Prompt templates are audited for PIT safety in
  `docs/PROMPT_LEAKAGE_AUDIT.md` (every template field traced to a pre-origin
  source; macro table versioned with as-of dates). No new data was used here.
