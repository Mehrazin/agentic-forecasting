# Survey: Time-Series Foundation Models (TSFMs) for Zero-Shot Forecasting

**Date:** 2026-10-15. **Purpose:** pick TSFM backends for mortgage attrition
forecasting (66 monthly history points, 6-month horizon, laptop CPU, ~8GB RAM).
All HuggingFace IDs below were verified via search results, not guessed.

## 1. Chronos-2 (recommended primary)

- **HF ID:** `amazon/chronos-2` (verified via huggingface.co page)
- **Size:** 120M params, encoder-only (T5-inspired)
- **Capabilities:** univariate, multivariate, past-only + known-future covariates
  in one architecture. Context 8192, prediction length 1024.
- **Probabilistic:** native multi-step quantile forecasts, well-calibrated
  (~95% coverage in independent benchmark).
- **Install:** `pip install chronos-forecasting` (repo: amazon-science/chronos-forecasting)
- **CPU:** explicitly supported. ~300 forecasts/sec on A10G; CPU is slower but
  fine for our scale (dozens of segments, not thousands).
- **License:** Apache-2.0. ~25.5M monthly downloads (Aug 2026).
- **2026 note:** native HuggingFace Transformers integration was still pending
  (tracked issue); use the `chronos-forecasting` package, not `AutoModel`.
- **Inference sketch:**
  ```python
  import torch
  from chronos import Chronos2Pipeline  # from chronos-forecasting
  pipeline = Chronos2Pipeline.from_pretrained(
      "amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
  forecast = pipeline.predict(
      context=torch.tensor(series, dtype=torch.float32),
      prediction_length=6, num_samples=1000)
  # forecast: (1000, 6) sampled trajectories -> quantiles, Beta-Binomial readout
  ```

## 2. TimesFM-2.5 (recommended secondary, for M1a/M1b comparison)

- **HF IDs:** `google/timesfm-2.5-200m-pytorch` (native),
  `google/timesfm-2.5-200m-transformers` (HF port, PEFT-friendly)
- **Size:** 200M params, decoder-only, 16k context. ~800MB weights, ~1.5GB RAM on CPU.
- **Probabilistic:** optional 30M continuous-quantile head -> mean + 10 quantiles
  (q10..q90). 80% PI = quantiles index 1 (q10) and 9 (q90).
- **Install:** `pip install timesfm[torch]` — **caveat:** as of Apr 2026, PyPI's
  `timesfm 1.3.0` was the v1/v2 archive and 2.5 required GitHub install
  (`pip install "git+https://github.com/google-research/timesfm.git[torch]"`).
  Verify PyPI caught up before pinning.
- **Caveat:** recursive (autoregressive) generation accumulates error at long
  horizons; fine at H=6.
- **Inference sketch:**
  ```python
  import timesfm, numpy as np
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      "google/timesfm-2.5-200m-pytorch")
  model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=64,
      normalize_inputs=True, use_continuous_quantile_head=True,
      fix_quantile_crossing=True))
  point, quantiles = model.forecast(horizon=6, inputs=[series.astype(np.float32)])
  ```

## 3. Moirai-2 (fast alternative)

- **HF ID:** `Salesforce/moirai-2.0-R-small` (verified via usage snippet)
- **Size:** ~11M params (small), decoder-only. **Fastest of the big three** (0.025s).
- **Accuracy:** best MASE (0.307) in an independent energy benchmark, beating
  Chronos-2 (0.334). **Caveat:** overconfident uncertainty (~70% coverage) —
  worse for our calibrated PI requirement.
- **Install:** `pip install uni2ts` (PyPI), needs GluonTS for inference helpers.
- **Verdict:** good speed/accuracy fallback, but weaker calibration makes it a
  poor fit for the 80% PI coverage metric. Optional M1c.

## 4. TinyTimeMixers / CITRAS-FM (ultra-light)

- **Size:** <1M (TTM) to <10M (CITRAS-FM). Sub-0.1s CPU inference.
- **Verdict:** interesting for edge/on-device, but accuracy lags larger models.
  Not needed for our laptop-scale workload. Skip unless latency becomes critical.

## 5. Practical shortcut: Darts unified API

- **Darts** (unit8co/darts, `pip install darts[torch]`, v0.47+) ships a
  `FoundationModel` collection: `chronos2`, `chronos2_small`, `timesfm2p5`,
  `patchtst_fm`, `tirex` — one API, zero-shot or fine-tuned, with built-in
  probabilistic forecasting and backtesting.
- **Why it matters:** switching backends becomes a name change; joint
  evaluation/ensembling is trivial. Strongly consider building
  `tsfm_chronos.py` / `tsfm_timesfm.py` on Darts instead of raw packages.

## Recommendation for this project

| Slot | Model | Rationale |
|---|---|---|
| M1a (primary TSFM) | **Chronos-2** via Darts (or raw package) | Best calibration (matters for 80% PI), native probabilistic output, CPU OK |
| M1b (comparison) | **TimesFM-2.5** via Darts | Independent architecture family; validates the ablation isn't model-specific |
| Fallback | Moirai-2 small | If Chronos-2 proves too slow on CPU |

**Sizing sanity:** 66 monthly points context, H=6 — trivially small for all
models above (context limits are 8k-16k). RAM is not a constraint; expect
~1-2GB peak per model on CPU. Download weights once, cache in `~/.cache`.

**Deprecations to avoid:** original Chronos-T5 (superseded by Bolt/2);
TimesFM 1.x/2.0 (superseded by 2.5); Nixtla TimeGPT (closed API, not local).

## Sources

- HF: amazon/chronos-2; google/timesfm-2.5-200m-pytorch; google/timesfm-2.5-200m-transformers; Salesforce/moirai-2.0-R-small
- GitHub: amazon-science/chronos-forecasting; google-research/timesfm; unit8co/darts; Salesforce/uni2ts
- arXiv: 2510.15821 (Chronos-2), 2606.27438 (Darts FoundationModel)
