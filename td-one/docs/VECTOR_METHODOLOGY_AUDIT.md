# Vector Methodology Audit

How Vector's `aieng-forecasting` evaluation stack differs from TD-One's, and what
it means for the exp52/exp53 comparisons. Written 2026-10-18 after running
Vector's actual predictor classes head-to-head against our proposed method.

## 1. CRPS implementation — DIFFERENT, but ranking-robust

| | Vector (`backtest._crps_for_prediction`) | TD-One (`crps_quantile`) |
|---|---|---|
| Formula | `properscoring.crps_ensemble(actual, sorted_quantiles)` — treats the 11 quantile values as ensemble members | Mean pinball loss × 2 over quantile levels: `mean(2·((y−q)·τ − (y−q)·1{y<q}))` |
| Their own caveat | Docstring admits: "quantile values are not independent samples … reasonable CRPS approximation when the quantile grid is sufficiently fine" | Standard quantile-score definition (Gneiting & Raftery 2007) |
| Quantile grid | 11 levels: 0.05–0.95 (`STANDARD_QUANTILES`) | 9 levels: 0.10–0.90 |

**Measured relationship** (2000 synthetic normal forecasts, this audit):
- Vector-ensemble vs our-11-level: Pearson r = 0.9998, ratio ≈ 1.06
- Our-9-level vs our-11-level: Pearson r = 0.9988, ratio ≈ 1.13
- Our-9-level vs closed-form normal CRPS: ratio = 1.0895 ± 0.012 (the 9% is the
  standard coarse-grid discretization bias; it is constant across methods)

**Implication:** rankings are essentially identical under either scorer, so
exp52/exp53's *relative* conclusions are valid. But **absolute CRPS numbers are
not comparable** across the two scorers — never quote our 1.0662 against a
number from Vector's backtest harness without re-scoring.

## 2. Evaluation protocol

| | Vector `backtest()` | TD-One (exp47–53) |
|---|---|---|
| Origin selection | `BacktestSpec`: explicit `origin_dates` or start/end/stride grid; `warmup` filter; unresolvable origins skipped with warning | Blocked 5-fold CV over a fixed origin list (e.g. last 59–60 months) |
| PIT safety | `ForecastContext(as_of)` — cutoff enforced at data access; sound | `PITStore` + manual windowing; sound (audited, see `td-one/audits/`) |
| Failure handling | Per-origin retry (default 2), failures counted as skipped | `try/except` per origin; failures excluded from means (reported as n_ok) |
| Metric choice | Implied by `payload_type` (CRPS/Brier/RPS) | Fixed per experiment, documented |

Both are defensible. Blocked CV gives us honest standard errors; their spec
system gives shareable, YAML-serializable evaluation windows. Neither leaks.

## 3. Baselines — asymmetric strength

- Vector's documented "performance floor" is `LastValuePredictor` (predict last
  value, degenerate quantiles). It is *intentionally* weak.
- Our internal baselines (ETS, ARIMA, seasonal naive, equal-weights combination)
  are far stronger.
- **Consequence:** exp52 ("we beat Vector's baseline 11.8%") is weak evidence —
  beating a floor is expected. exp53 (vs their LLM predictor) is the real test.

## 4. LLM predictor configuration (as deployed by Vector vs as tested here)

`QuantileGridLLMPredictor` defaults and what we changed for exp53:

| Setting | Vector default | exp53 | Why changed |
|---|---|---|---|
| Model | `gemini-3.1-flash-lite-preview` via Vector proxy | `openai/gpt-oss-120b` via Groq | No access to Vector's proxy |
| `temperature` | **1.0** | 1.0 (kept) | Faithful, but note: high variance; single-run comparisons are noisy |
| `max_tokens` | 16384 | 2048 | 16384 risks Groq HTTP 413; grid response needs <200 tokens |
| `reasoning_effort` | None (provider default) | `"low"` | exp48: "low" ≥ default for diagnostics, 3.5–6× faster |
| `history_window` | None (full history) | None (kept) | Faithful; costs ~2300 tokens/prompt → TPM pressure (see §5) |
| Response format | `json_schema` strict, `additionalProperties` stripped (Gemini-proxy compat) | Same, re-added `additionalProperties:false` in proxy | Groq strict mode requires it; their strip is proxy-specific |

**Portability finding:** Vector's LLM stack is tuned for their Gemini proxy and
does *not* run unmodified against a generic OpenAI-compatible endpoint. Three
adaptations were required (model-id prefix, `additionalProperties`, token
budget). None change the elicitation logic, but exp53 tests their predictor
*architecture*, not their exact deployment.

## 5. Rate-limit behavior (observed)

- Vector's `_client` has transient retry (429/5xx) with backoff — it handled
  Groq's 8000 TPM limit gracefully during exp53.
- Their default full-history prompts (~2300 tokens) burn the TPM budget fast:
  expect frequent 429s on free-tier keys. `history_window` is the lever.

## 6. Stochasticity caveat

At `temperature=1.0`, the LLM predictor is stochastic. exp53 runs each origin
once (59 LLM calls ≈ 25–60 min under TPM limits; multi-seed was infeasible).
The paired t-test treats per-origin CRPS as fixed, but an unknown fraction of
the variance is LLM sampling noise. A 3-seed repeat is the correct follow-up
when budget allows.
