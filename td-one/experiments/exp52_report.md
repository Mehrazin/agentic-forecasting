# exp52: Proposed Method vs Vector's Official Baselines

## Question

How does TD-One's proposed method fare against Vector Institute's official
baselines from the agentic-forecasting bootcamp?

## Vector's Baselines

From `aieng-forecasting/aieng/forecasting/methods/baselines/`:

| Predictor | Description |
|---|---|
| `LastValuePredictor` | Naive: predict last observed value at all quantiles (degenerate distribution). The "performance floor" — every model should beat it. |
| `HistoricalFrequencyPredictor` | For categorical targets: predict historical class frequencies. Not applicable to continuous forecasting. |
| `CategoricalFrequencyPredictor` | For categorical targets. Not applicable here. |

For continuous forecasting, Vector's baseline is `LastValuePredictor`.

## Method

Direct apples-to-apples comparison on identical data and origins:

- **Data:** TD-One Food CPI (200 obs, 2010-01 to 2026-09)
- **Task:** h=1 monthly forecasting
- **Evaluation:** Blocked 5-fold CV over last 59 origins, paired t-test
- **Metric:** Quantile CRPS (9 levels: 0.1–0.9)
- **Vector side:** `LastValuePredictor` logic (predict last value, flat quantiles)
- **Our side:** `ProposedForecaster` with regime router + 4 experts (naive, snaive, ets, arima)

Both see identical PIT-safe inputs at each origin.

## Results

| Method | Mean CRPS | vs LastValue |
|---|---|---|
| **TD-One Proposed** | **1.0662** | **−0.1429** |
| Vector LastValue | 1.2091 | — |

- **Difference:** −0.1429 (11.8% improvement)
- **Paired t-test:** t=−3.58, **p=7.1e-04** (significant)
- **Win rate:** 50/59 origins (84.7%)

## Interpretation

Our proposed method **significantly beats Vector's official baseline** on Food CPI.

This is expected — `LastValuePredictor` is deliberately a "performance floor."
But it's the external validation we were missing: our method doesn't just beat
our own internal baselines (equal weights, best single), it beats Vector's
published baseline too.

### Caveats

1. **Single dataset:** This test ran on Food CPI only. Vector's canonical
   benchmarks use StatCan data via their DataService (not run here due to
   network requirements).
2. **Baseline strength:** `LastValuePredictor` is intentionally weak. The
   stronger test — against Vector's LLM predictor — is exp53. **Spoiler: the
   LLM wins there** (0.5688 vs 1.0339, p=5.4e-05). See `exp53_report.md`.
3. **Our experts:** The proposed method used 4 experts (naive, snaive, ets, arima)
   with regime routing. On Food CPI, the seasonal-naive expert likely dominates
   given the strong seasonality.

## Follow-up: exp53 (vs their LLM predictor)

exp53 ran Vector's actual `QuantileGridLLMPredictor` class (gpt-oss-120b via
Groq) on the same 59 origins. Result: **LLM 0.5688 vs proposed 1.0339,
p=5.4e-05, LLM wins 43/58**. The year-by-year pattern (worst in volatile
2022, no training-cutoff step) indicates genuine forecasting skill, not
memorization — likely via better calibration on this smooth series. Full
writeup in `exp53_report.md`.

## Methodological differences (summary)

Full audit in `docs/VECTOR_METHODOLOGY_AUDIT.md`. Headlines:
- **CRPS implementations differ** (theirs: `crps_ensemble` on quantiles;
  ours: mean quantile score) but are ranking-equivalent (r=0.9998); absolute
  values are NOT cross-comparable.
- Their quantile grid is 11 levels (0.05–0.95); ours is 9 (0.10–0.90).
- Their LLM stack is tuned for their Gemini proxy and needed 3 adaptations
  to run on Groq (documented, none touch elicitation logic).
- Their `temperature=1.0` default makes single-run LLM comparisons noisy.

## Integrity

See `docs/SELF_AUDIT.md` for the adversarial self-review (multiple
comparisons, M4 development bias, BoC discrete downgrade, PIT re-verification,
CRPS scorer verification). Headline results survive; marginal claims were
downgraded where warranted.

## Files

- `experiments/exp52_vector_baselines.py` — comparison script
- `experiments/exp52_results.json` — numbers

## Bottom Line

**We beat Vector's baseline significantly (p=7.1e-04, 50/59 wins).**
The proposed method clears Vector's performance floor with room to spare.
