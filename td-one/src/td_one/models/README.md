# `models/` — the model ladder

| Module | What it is |
|---|---|
| `baseline.py` | M0: trailing-6 Jeffreys Beta-Binomial |
| `conformal.py` | Conformalized M0 (coverage 0.67→0.91) — the default baseline |
| `tsfm.py` | Chronos-2 / TimesFM-2.5 wrappers (real weights via manifest) |
| `tsfm_calibration.py` | TSFM calibration (sorted quantiles, volatility floor) |
| `hybrid.py` | M3: M0 + bounded log-odds adjustment |
| `macro_m0.py` | MacroM0: FRED OLS overlay (track closed — hurts) |
| `ensemble.py` | Inverse-CRPS ensemble |
| `regime_blend.py` | Per-regime expert blending (exp43) |
| `shrinkage.py` | James-Stein shrinkage toward zero |
| `hierarchical.py` | Hierarchical model for thin segments |
| `ewm.py` / `newm.py` | EWMA variants |

Weights: [`models/MANIFEST.json`](../../models/MANIFEST.json) (fail-fast if missing).
Up: [`td_one/`](../README.md).
