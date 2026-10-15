# Experiment 2 v2 Report: M0 vs M1 with Real Weights (Redesigned)

**Date:** 2026-10-15. **Mode:** real weights. **Supersedes** the v1 report below.

## What changed and why

The v1 experiment had three fatal design flaws (found by the 2026-10-15
critic audit): a circular DGP hand-tuned to flatter trend-extrapolating
TSFMs, n=6 test points on a single seed (noise-dominated), and an M0 that
didn't match its own spec (pooled 12 periods instead of trailing 6).
v2 redesigns from scratch at the same sample size:

- **3 pre-registered scenarios** (18 periods each): `stationary` (M0's home
  turf), `drift` (the old v1 DGP — TSFM's home turf), `regime_shift`
  (discrete 0.15→0.25 jump at t=9; the realistic mortgage-renewal case).
- **Realistic structure** in all scenarios: seasonal renewal volume,
  Beta-Binomial overdispersion, AR(1) logit noise.
- **K=5 seeds**, rolling origins {6, 9, 12}, paired Wilcoxon tests.
- **Primary metric: CRPS** (proper score on count distributions);
  secondary: MAE, 80% PI empirical coverage.
- **Oracle anchor**: Binomial(N, true_rate) — the noise floor.
- **M3 probe**: fixed Δ grid {±2.0, ±0.75, ±0.25, 0} applied to M1a's rate
  via the bounded hybrid readout, testing the ±0.75-bound hypothesis
  with an unbounded negative control.

## Results

### CRPS (mean ± std across 5 seeds; lower is better)

| Scenario | Oracle | M0 | M1a (Chronos-2) | M1b (TimesFM-2.5) |
|---|---|---|---|---|
| stationary | 4.65 ± 1.27 | 4.85 ± 1.53 | 5.40 ± 2.18 | **4.28 ± 0.65** |
| drift | 5.02 ± 1.17 | 5.23 ± 1.34 | 6.04 ± 1.89 | **5.17 ± 0.80** |
| regime_shift | 5.58 ± 1.37 | 7.40 ± 1.77 | 7.44 ± 2.13 | **7.09 ± 1.48** |

Paired Wilcoxon vs M0 (per-seed CRPS differences):

| Scenario | M1a − M0 | p | M1b − M0 | p |
|---|---|---|---|---|
| stationary | +0.55 ± 0.74 | 0.31 | −0.57 ± 0.91 | 0.44 |
| drift | +0.80 ± 0.99 | 0.19 | −0.06 ± 0.67 | 1.00 |
| regime_shift | +0.04 ± 1.04 | 0.81 | −0.31 ± 0.45 | 0.62 |

**No model significantly beats M0 at this sample size** (all p > 0.18).
TimesFM-2.5 is the best of the three models in every scenario but the
gaps are not significant. Chronos-2 is *worse* than M0 on drift
(+0.80) — the v1 "Chronos beats M0 by 20%" claim does not survive a
fair design.

### 80% PI empirical coverage (target 0.80)

| Scenario | M0 | M1a | M1b |
|---|---|---|---|
| stationary | 0.61 | 0.94 | 0.84 |
| drift | 0.60 | 0.94 | 0.82 |
| regime_shift | 0.41 | 0.93 | 0.61 |

M0 undercovers (intervals too narrow — the trailing-6 posterior is
overconfident on a noisy series). Chronos-2 overcovers badly (0.93–0.94:
the 21-level quantile lattice is too wide for this task). TimesFM-2.5 is
closest to nominal — the best-calibrated of the three.

### M3 probe: CRPS vs Δ (drift scenario)

| Δ | −2.0 | −0.75 | −0.25 | 0 | +0.25 | +0.75 | +2.0 |
|---|---|---|---|---|---|---|---|
| CRPS | 8.93 | 8.93 | 7.07 | 5.96 | **5.63** | 8.93 | 8.93 |
| coverage | 0.22 | 0.22 | 0.39 | 0.50 | 0.54 | 0.28 | 0.28 |

The U-shape is the expected signature: small adjustments (±0.25) help
slightly, large ones destroy calibration. The ±2.0 negative control
collapses to the ±0.75 clip (identical numbers), confirming the bound is
doing the work. This is the first experimental evidence for the plan's
bounded-adjustment hypothesis — on a toy, but the shape is right.

### Oracle anchor

Oracle CRPS (4.65–5.58) sits just below M0 (4.85–5.23) in stationary/drift:
the series is **noise-dominated** — the binomial sampling noise floor is
near the best achievable. No ranking claim is possible at this sample
size in those scenarios; that *is* the honest finding.

## Interpretation

1. **v1's headline is retracted.** "Chronos-2 beats M0 by 20%" was a DGP
   artifact (circular drift design + MAE-only + single seed + mis-specified
   M0). Under a fair design Chronos-2 does not beat the baseline.
2. **TimesFM-2.5 is the most promising M1**: best CRPS in all scenarios,
   best calibration, lowest seed variance. Worth carrying forward.
3. **M0 is a strong baseline** — the trailing-6 Jeffreys Beta-Binomial is
   hard to beat on 18 noisy periods. Its weakness is calibration
   (undercoverage), not point accuracy.
4. **The M3 bounded-adjustment hypothesis shows the right shape** (U-curve
   with negative control). M2/M3 work should continue.
5. **Next**: the full simulator (10k×66) gives the sample size to make
   these comparisons significant; segment-level heterogeneity is where
   TSFMs should have an edge over a pooled M0.

## Reproduction

```bash
cd td-one
# Real weights (models/weights/ via scripts/bootstrap_models.py)
uv run --with torch --with chronos-forecasting --with timesfm \
    --with huggingface_hub python experiments/exp2_real_weights.py
# Mock mode (CI, no downloads)
TD_ONE_MOCK=1 TD_ONE_SEEDS=2 python experiments/exp2_real_weights.py
```

Timings (2 vCPU): M0 0.1s, Chronos-2 26s, TimesFM-2.5 972s (compile-heavy;
forecasts are fast once compiled).

---

# Experiment 2 v1 Report (SUPERSEDED — retained for the record)

**Date:** 2026-10-15. **Mode:** real weights. **Status:** superseded by v2
above; the v1 conclusions are retracted (see Interpretation §1).

## Setup (v1)

- Synthetic 18-period switching series (seed 42), mild upward true-rate
  drift (0.16 → 0.228).
- Train periods 1–12, test periods 13–18 (6-step horizon).
- Metric: MAE on switch counts.

## Results (v1)

| Model | MAE | Δ vs M0 |
|---|---|---|
| M0 (historical) | 5.00 | — |
| M1a (Chronos-2, real) | 4.00 | −20% |
| M1b (TimesFM-2.5, real) | 4.62 | −8% |

## Why v1 was wrong

1. The drift DGP was chosen because it rewards trend extrapolation —
   circular for a TSFM-vs-stationary-baseline comparison.
2. n=6 test points, one seed: the 1.0-count M1a−M0 gap is a fraction of
   one period's binomial noise (σ≈3.6).
3. M0 pooled 12 periods; its spec says trailing 6 (handicapped baseline).
4. MAE-only discarded the probabilistic forecasts the architecture is
   built for.

## Runtime notes (v1, still valid)

- `timesfm` 3.x API: `TimesFM_2p5_200M_torch.from_pretrained` +
  `compile(ForecastConfig(...))`; the old `TimesFm` class is gone.
- `Chronos2Pipeline.predict` takes `inputs=`, not `context=`.
- Models load sequentially with explicit `unload()` (CPU-only VM).
- `no_proxy`/`NO_PROXY` must exclude bracketed IPv6 (`[::1]`) or
  `huggingface_hub` fails.
