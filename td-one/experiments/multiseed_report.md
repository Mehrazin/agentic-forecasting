# Multi-Seed Scale-Up Report

**Question:** Do the scale-up findings survive across DGP seeds, or was
seed 0 lucky?

**Design:** seeds 0–4, 10k×66, 14 segments × 5 origins each. Models:
M0 window=6 (published baseline), M0 window=3 (ablation challenger),
Chronos-2 (real weights), TimesFM-2.5 (real weights). New flag:
`scripts/scale_up.py --m0-window` (default 6; window=6 uses the
unchanged tools path).

**Completed:** 2026-10-16. 19/20 runs finished; seed1-m0w6 lost to OOM
during sibling-agent memory contention (3.2GB exp5 process). TimesFM
seeds 2–4 needed one retry after transient PyPI timeouts on
`uv run --with` env setup.

## Per-seed mean CRPS

| Seed | M0 w6 | M0 w3 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|---|
| 0 | 5.360 | 3.196 | 4.463 | 4.591 |
| 1 | — (OOM) | 2.976 | 4.263 | 4.508 |
| 2 | 2.916 | 3.695 | 3.886 | 4.197 |
| 3 | 4.372 | 2.070 | 3.464 | 3.621 |
| 4 | 3.873 | 2.794 | 3.618 | 3.886 |

## Q1: Does M0 window=3 beat window=6 on all seeds?

**No — 3 of 4, and the exception is significant.** Paired t-tests
(14 shared segments, challenger = w3):

| Seed | w3 wins | mean Δ (w3−w6) | p |
|---|---|---|---|
| 0 | 11/14 | −2.164 | 0.036 ✅ |
| 2 | 2/14 | +0.779 | 0.022 ❌ (w6 better) |
| 3 | 12/14 | −2.302 | 0.031 ✅ |
| 4 | 12/14 | −1.080 | 0.035 ✅ |

The window is a genuine tuning parameter: the effect is significant
on every seed, but its **direction flips on seed 2**. Seed 0's w3
advantage was not a fluke in direction (w3 wins the majority), but the
magnitude is seed-dependent. **Consequence:** future M1-vs-M0
comparisons must tune the window or report the envelope — the
"published M0" (w6) is not the best M0.

## Q2: Do the real TSFMs beat M0_w6 on any seed? Majority?

**Neither TSFM beats M0_w6 significantly on any seed.** Paired t-tests
vs M0_w6 (14 segments):

| Seed | Chronos wins | p | TimesFM wins | p |
|---|---|---|---|---|
| 0 | 7/14 (−0.897) | 0.212 n.s. | 5/14 (−0.769) | 0.341 n.s. |
| 2 | 1/14 (+0.970) | 0.007 ❌ | 2/14 (+1.281) | 0.004 ❌ |
| 3 | 7/14 (−0.908) | 0.284 n.s. | 6/14 (−0.751) | 0.296 n.s. |
| 4 | 5/14 (−0.255) | 0.516 n.s. | 3/14 (+0.013) | 0.962 n.s. |

(Mean Δ in parentheses; negative favors the challenger.)

- **Chronos-2** beats m0w6 on the mean in 3/4 seeds (0, 3, 4) but
  never significantly. On seed 2, m0w6 wins significantly.
- **TimesFM-2.5** beats m0w6 on the mean in 2/4 seeds (0, 3), never
  significantly; essentially tied on seed 4 (+0.013). On seed 2, m0w6
  wins significantly.

**Verdict:** the seed-0 "TSFMs beat M0 on mean CRPS" does not survive
as a general claim. The honest summary across 5 seeds: *no model
significantly beats the baseline; the baseline's tuned window (w3)
beats both TSFMs on 3/5 seeds.* This strengthens exp2 v2's conclusion.

## Q3: Coverage patterns across seeds

Mean 80% interval coverage per seed:

| Seed | M0 w6 | M0 w3 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|---|
| 0 | 0.671 | 0.814 | 0.914 | 0.800 |
| 1 | — | 0.800 | 0.914 | 0.843 |
| 2 | 0.886 | 0.757 | 0.900 | 0.871 |
| 3 | 0.786 | 0.986 | 0.971 | 0.943 |
| 4 | 0.757 | 0.829 | 0.900 | 0.829 |

- **Chronos-2 overcovers on every seed** (0.90–0.97) — the most
  consistent pattern in the table, reproducing exp2.
- **TimesFM-2.5 is closest to nominal** on 4/5 seeds (0.80–0.87;
  0.94 on seed 3) — the best-calibrated model overall.
- **M0_w6 undercovers** on 3/4 seeds (0.67–0.79; 0.89 on seed 2).
- **M0_w3** is better centered (0.76–0.99) but overcovers on seed 3.

Calibration ranking (by |coverage − 0.80| averaged over seeds):
TimesFM-2.5 (0.06) < M0_w3 (0.10) < Chronos-2 (0.12) < M0_w6 (0.13).

## Limitations

- seed1-m0w6 lost to OOM (sibling agent); the w3-vs-w6 comparison on
  seed 1 is one-sided.
- 5 seeds of one synthetic DGP family — not a general TSFM claim.
- No multiple-comparison correction across the 3 challengers × 4 seeds
  = 12 tests (doesn't change the "no significant wins" conclusion).

## Artifacts

- `experiments/multiseed_results.json` — per-seed table + paired tests
- `experiments/scale_up_results_seed{s}-{m0w6,m0w3,chronos2,timesfm25}.json`
  (19 files; seed1-m0w6 missing)
- `experiments/collect_multiseed.py` — aggregation script
- `scripts/scale_up.py --m0-window` — new CLI flag
- Per-run logs: `/tmp/multiseed/*.log` (ephemeral)
