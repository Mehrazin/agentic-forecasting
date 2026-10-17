# BoC Rate Decisions — Survey

## What the example does

Predicts Bank of Canada's CUT/HOLD/HIKE decision 28 days before each fixed announcement date (8/year). Scored with Ranked Probability Score (RPS) on ordered categories.

## Data

| Ingredient | Source | Our equivalent |
|---|---|---|
| BoC target rate | StatCan 10-10-0139-01 | FRED IRSTCB01CAM156N (monthly, 1955-2023) |
| Meeting dates | meeting_schedule.yaml (144 meetings, 2009-2026) | Same file |
| 2yr GoC yield | StatCan | **Missing** — we have 5yr (BD.CDN.5YR.DQ.YLD) |
| CPI | StatCan 18-10-0004-11 | FRED CPALTT01CAM659N (1915-2025) |
| Unemployment | FRED LRUNTTTTCAM156S | FRED LRHUTTTTCAM156S (1955-2026) |

## Their baselines

1. **Climatology**: Constant distribution (76% hold). RPS floor.
2. **Logistic regression**: 4 features (yield_spread, rate_momentum, inflation_gap, unemployment_momentum), refit at each origin.
3. **LLMP**: Direct distribution elicitation via LLM.
4. **Agent**: Same 4 features as logistic, LLM reasoning.

## Evaluation

- 120 origins (2010-2024), T-28 lead
- 12 origins (2025-2026), honest eval
- Metric: RPS (ordered), skill score vs climatology

## Key design choices

- **28-day lead**: Tests anticipation, not market reading
- **Cutoff discipline**: Drop newest month of monthly data (publication lag)
- **Ordered categories**: RPS penalizes cut-vs-hike errors more than cut-vs-hold

## What we're testing

Can our agent methodology (react_v2 + 10 tools + decision policy) beat their logistic baseline on discrete data?

Our tools that apply:
- `detect_changepoints` → regime changes in macro series
- `analyze_residuals` → where does logistic fail?
- `test_stationarity` → is inflation stationary?
- `backtest_event_impact` → do BoC decisions follow macro surprises?

The discrete nature is new for us. Our tools were built for continuous series. This tests their generality.
