# Red-Teaming TSFMs on Loan-Attrition Shifts — Protocol (exp67)

**Status:** complete (2026-10-20). Real-TSFM grid finished; results in
`experiments/exp67_report.md`.

## Key result (summary)

All four models (Chronos-2, TimesFM-2.5, naive, M0) break together at 4σ on
S1/S2/S3 — TSFMs show no relative robustness advantage, but Chronos-2 remains
best in absolute CRPS on 6/8 shifts. S4/S5 never break (honest negatives).
Coverage collapses under level shifts (M0 80% coverage 0.61→0.08 at S1/4σ).
Full breaking-point table: `experiments/exp67_report.md`.

## Why

Chronos-2 and TimesFM-2.5 are strictly univariate: forecast = f(history).
Shifts whose signature is absent from the history window but present in side
information (policy announcement dates, rate levels, borrower mix, regional
splits) are *structurally invisible* to them. This experiment maps exactly
where they break, at what severity, and whether classical baselines break
less.

## Attack surface (verified from `models/tsfm.py`)

- **Chronos-2** (120M): single univariate float tensor in; fixed 21-level
  quantile lattice out. Tokenizes the series, extrapolates the distribution.
- **TimesFM-2.5** (200M): same, 9 quantiles, `normalize_inputs=True`
  (z-scores the window — partially erases level/variance regime info).
- "Context" = the numeric history only. No covariates, dates, text, metadata.

Failure modes under test: (a) mixture windows post-break, (b) covariate
jumps predictable from metadata, (c) normalization blindness, (d) mechanism
blindness (can't represent a changing DGP).

## Shift taxonomy

| # | Shift | Series mechanism | DGP mechanism |
|---|---|---|---|
| S1 | Rate shock | Level shift at T | `macro_shifts` dict |
| S2 | Policy intervention | Partial jump, 2-month lag | Event timeline |
| S3 | Regional divergence | Opposite shifts, aggregate cancels | Per-region series |
| S4 | Seasonal break | Amplitude/phase change at T | Seasonal-break hook |
| S5 | Correlation breakdown | AR persistence flips at T | `beta_schedule` (sign flip) |
| S6 | Volatility regime | Noise scale inflates at T | Overdispersion drop |
| S7 | Composition drift | Linear drift post-T | Cohort aging |

Severity ladder: 0.5σ / 1σ / 2σ / 4σ in logit space (1σ = 0.25 logit).
Rolling origins: 3 pre-shift, 3 straddling T, 3 post-shift (h=6).

## Protocol

- **Unit:** one (series, origin). **Metrics:** CRPS on the common 19-level
  grid (`evaluation/real_benchmark.py`), 80% interval coverage.
- **Failure =** significant CRPS degradation vs the model's own pre-shift
  baseline **and** vs classical baselines (naive, M0) on identical shifted
  data. Paired t-test + win rate; never a bare mean.
- **Placebo:** identical protocol, no shift — must replicate the known
  calm-DGP result (TSFM matches/beats classical).
- **Output:** breaking-point curves per model (degradation vs severity),
  not binary pass/fail.

## What comes next (exp68, AFTER these numbers)

The agentic guard — CUSUM shift detection from covariates, context surgery,
shift-aware quantile widening — is designed against the measured breaking
points, on dev shifts S1/S4/S6, evaluated on held-out S2/S3/S5/S7.
See `docs/LOAN_ATTRITION_PLAN.md`.
