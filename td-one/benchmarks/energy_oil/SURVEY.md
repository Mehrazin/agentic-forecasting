# Energy/Oil Benchmark Survey

**Date:** 2026-10-17
**Example:** `implementations/energy_oil_forecasting/` (Vector bootcamp reference #3 of 4)

## What it is

Daily WTI crude oil forecasting — the bootcamp's "high-frequency, context-driven"
reference. Unlike our monthly SMM work, daily resolution makes genuinely prospective
evaluation practical.

## Data

- **Source:** Yahoo Finance `CL=F` (front-month futures close), cached to `data/yfinance/`
- **Our substitute:** FRED `DCOILWTICO` (WTI spot price), `td-one/data/fred/DCOILWTICO.parquet`
  - 10,640 daily obs, 1986-01-02 → 2026-10-14
  - Spot vs futures: nearly identical for forecasting purposes (basis is small/stable)
  - **No cache needed** — we have the full series locally
- **Covariates available** (their `data.py`): Brent, nat gas, RBOB, gold, USD index,
  futures curve (USL/USO contango proxy), VIX — all from Yahoo, all lagged 1 day

## Tasks (their spec: `specs/energy_oil_backtest.yaml`)

| Task | Horizons | Output | Metric |
|------|----------|--------|--------|
| A: Trajectory | 5, 10, 21 business days | Point + quantiles | CRPS, MAE |
| B: Up-shock prob | P(>$5 in 5 bdays) | Binary prob | Brier |
| C: Scenarios | 60-day narrative | 3 scenario cards | Qualitative |

**Backtest:** 51 weekly origins in 2025, 250-day warmup.

## Their 4 capability levels (notebook 02)

1. **L1 Basic:** LLM reasons from price history alone (no tools)
2. **L2 News:** + bounded Google Search with temporal cutoffs + leakage verifier
3. **L3 Code-exec:** + Gemini native code execution + 3 forecasting skills
4. **L4 Tool:** + AutoARIMA ForecastTool (rigid alternative to open-ended code)

Plus notebook 05-06: **adaptive agent** that learns a strategy from 2025 data,
frozen before/after comparison on 2026.

## Their baselines

- **Prophet:** daily refit, yearly seasonality, 80-95% CI → Gaussian quantiles
- **AutoARIMA:** via ForecastTool (notebook 05 demo)

## How our methodology maps

| Our component | Applies here? | Notes |
|---------------|---------------|-------|
| Event simulator | **Partially.** Real WTI has real events (OPEC, geopolitics). We can't *simulate* them, but we can *detect* them via `detect_changepoints` + `backtest_event_impact`. |
| 10 tools | **Yes.** `detect_changepoints`, `test_stationarity`, `decompose_series`, `analyze_residuals`, `plot_diagnostics` all directly applicable to daily oil. `backtest_event_impact` needs a real event list (we can use OPEC meeting dates). |
| react_v2 | **Yes.** Code-executing agent can run the tools on WTI data. |
| Decision policy | **Yes.** Shrinkage + corroboration gating applies to any detection-based adjustment. |
| LogScaleCalibrator | **No.** WTI is ~$60-100, not near-zero. Not needed. |
| News/RAG | **Partially.** We have no live news API in this benchmark, but the *question* is whether statistical detection alone helps. |

## Key differences from our SMM work

1. **Frequency:** Daily (WTI) vs monthly (SMM). More data, faster adaptation.
2. **Volatility:** WTI is far more volatile (~30% annualized vs SMM's smooth trends).
3. **Events are real:** OPEC meetings, wars, pandemics — not synthetic.
4. **No near-zero problem:** Standard scaling works; no log-calibration needed.
5. **Their agents use live search:** We can't replicate L2/L3 exactly (no Gemini API in benchmark), but we can test whether *statistical* agentic methods (our tools + decision policy) add value over pure ETS/ARIMA.

## Pilot plan

1. **Baselines on 2025 weekly origins:** Naive (last), ETS, ARIMA — horizons 5/10/21
2. **Our agent:** react_v2 scripted to run changepoint + stationarity + shift detection, apply decision policy
3. **Compare:** Does the agent beat ETS? Where? Why/why not?
4. **Diagnose:** If it fails here too, the method needs rethinking. If it works here but failed on synthetic, our synthetic DGP needs calibration.
