# Experiment 2 Report: M0 vs M1 with Real TSFM Weights

**Date:** 2026-10-15
**Mode:** real weights (not mocks)
**Status:** ✅ Complete

## Models (pinned, see `models/MANIFEST.json`)

| Key | HF repo | Revision | Size |
|-----|---------|----------|------|
| chronos2 | `amazon/chronos-2` | `29ec3766d36d` | 455.8 MB |
| timesfm25 | `google/timesfm-2.5-200m-pytorch` | `1d952420fba8` | 882.3 MB |

Weights bootstrapped via `scripts/bootstrap_models.py` (idempotent,
resumable, SHA-pinned). Raw weights are gitignored; the manifest is committed.

## Setup

- **Data:** synthetic 18-period switching series (seed=42), mild upward
  drift in the true rate (0.16 → 0.228) — a gentle non-stationarity that
  rewards models that extrapolate trend vs. pure averaging.
- **Train:** periods 1–12. **Test:** periods 13–18 (6-step horizon).
- **Metric:** MAE on predicted switch counts.

## Results

| Model | MAE | vs M0 |
|-------|-----|-------|
| **M1a Chronos-2** | **4.00** | **−20%** ✅ |
| M1b TimesFM-2.5 | 4.62 | −8% |
| M0 (Beta-Binomial) | 5.00 | — |

## Interpretation

1. **Real Chronos-2 beats the historical baseline by 20%.** The zero-shot
   TSFM picks up the upward drift that the stationary M0 baseline averages
   away. This is the first real-weights evidence that the M1 rung adds
   value over M0 on this task family.

2. **TimesFM-2.5 improves 8% over M0** — directionally consistent, smaller
   gain. The two architectures disagree slightly, which is exactly what the
   M1a/M1b ablation is designed to surface.

3. **Caveats:** tiny synthetic pilot (18 periods, one seed). Effect sizes
   will move on the full 10k×66mo data and across seeds. The next step is
   a multi-seed run on the full simulator output.

## Runtime notes

- CPU-only VM (2 vCPU, ~7GB RAM). Models loaded **one at a time** and
  unloaded between runs — peak RSS stayed within limits.
- Chronos-2 inference: ~45s for 6-step forecast (100 samples).
- TimesFM-2.5: comparable; `timesfm>=3.x` API
  (`TimesFM_2p5_200M_torch.from_pretrained` + `compile(ForecastConfig)`).
- Package API gotchas hit and fixed during this run (documented in
  `REPRODUCIBILITY.md` §5):
  - `timesfm` 3.x removed the old `TimesFm` class.
  - `Chronos2Pipeline.predict` takes `inputs=`, not `context=`.

## Reproducibility

```bash
cd td-one
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
uv run --with huggingface_hub python scripts/bootstrap_models.py
uv run --with torch --with chronos-forecasting --with timesfm \
    --with huggingface_hub python experiments/exp2_real_weights.py
```

Full steps: `experiments/REPRODUCIBILITY.md`.
Raw results: `experiments/exp2_results.json`.
