# `models/` — the model ladder

Every forecaster the research compares, from the dumb baseline up to
foundation models. The ladder exists so each new idea has something honest to
beat — no method ships without beating M0 first.

| Module | What it is |
|---|---|
| `baseline.py` | M0: trailing-6 Jeffreys Beta-Binomial (the floor) |
| `conformal.py` | Conformalized M0 (coverage 0.67→0.91) — the default baseline |
| `tsfm.py` | Chronos-2 / TimesFM-2.5 wrappers (real weights via manifest) |
| `tsfm_calibration.py` | TSFM calibration: sorted quantiles, volatility floor, log-scale |
| `tsfm_macro.py` | Macro-conditioned TSFM via the residual approach (track closed — hurts) |
| `hybrid.py` | M3: M0 + bounded log-odds adjustment |
| `macro_m0.py` | MacroM0: FRED OLS overlay (track closed — hurts) |
| `ensemble.py` | Inverse-CRPS ensemble |
| `regime_blend.py` | Per-regime expert blending (exp43 — the router backbone) |
| `shrinkage.py` | James-Stein shrinkage toward zero |
| `hierarchical.py` | Hierarchical model for thin segments |
| `ewm.py` / `newm.py` | EWMA variants |

Weights: [`models/MANIFEST.json`](../../models/MANIFEST.json) (fail-fast if missing).
Up: [`td_one/`](../README.md).
