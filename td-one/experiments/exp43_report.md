# exp43: Regime-Probability Router — Report

**Date:** 2026-10-17 · **Status:** complete · **Verdict: PARTIAL POSITIVE** —
the first combination method in this program to significantly beat equal
weights (M4, p=0.018), but the win is modest and regime value-add collapses
on 3 of 5 datasets.

## 1. What was built

`experiments/exp43_regime_router.py` (ruff clean, ~72s full run):

1. **Regimes** (all PIT-safe, from train window / pre-meeting data only):
   - M4: `seasonal` (seasonal_strength>0.5) / `trending` (trend_strength>0.7) / `noisy` (else)
   - Synthetic: true DGP kind (`trending`/`seasonal`/`rw`) — the clean latent-regime test
   - SMM: `calm` / `stressed` (recent changepoint or vol_ratio>1.0)
   - Food CPI: `calm` / `volatile` (per-fold median split on train vol_ratio — no leakage)
   - BoC: `EASY` / `HARD` (macro-signal agreement, from exp32)
2. **Router:** `LogisticRegression` → P(regime | 10 features). Simple, calibrated.
3. **Experts:** per-regime best model on *training* records of that regime
   (fallback: global best single if <10 records).
4. **Forecast:** Σ_r P(r|x) · expert_r quantiles (Vincentized, same as exp42).
5. **Baselines:** equal weights, best single *model* (not hindsight), hard router
   (argmax), oracle. Paired t-tests, CV by series (M4/synthetic) or time blocks
   (SMM/Food CPI/BoC).

## 2. Results

| Dataset | n | Router | Equal | Best single model | Oracle | Router vs equal (p) |
|---|---|---|---|---|---|---|
| M4 | 529 | **0.1282** | 0.1295 | arima 0.1295 | 0.1184 | **0.018** ✅ |
| Synthetic | 120 | 0.0855 | 0.0898 | naive 0.0856 | 0.0675 | 0.194 |
| SMM | 50 | 0.1843 | 0.1975 | naive 0.1843 | 0.1597 | 0.019* |
| Food CPI | 24 | 0.0114 | 0.0140 | arima 0.0114 | 0.0114 | ~1e-17* |
| BoC (acc) | 143 | 0.846 | — | rule 0.846 | — | n/a (identical) |

\* SMM/Food CPI: experts collapsed to a single model in every fold
(router = naive / arima = best single). The "win" over equal weights is the
best-single-model win, **not** regime value-add. Food CPI p-value is overstated
(24 correlated origins from one series).

**M4 detail** (the real test — 5-fold CV by series, 529 records):
- Experts discovered, stable across folds: `seasonal→arima`, `trending→ets`,
  `noisy→ets`. Genuine specialization, textbook-sensible.
- Router regime-classification accuracy: 0.88–0.95.
- Soft ≈ hard router (classifier is confident; softening adds little).
- Router vs best single model (arima): 0.1282 vs 0.1295, fold-level paired
  p=0.20 (n.s.; wins 4/5 folds numerically). Gap closed vs best-single-model:
  **+11.8%** of the (best model → oracle) gap.
- Pre-registered success criterion (beat equal weights, p<0.05): **MET**.

**BoC detail:** experts `EASY→rule`, `HARD→climatology` (3/5 folds) confirm
exp32's finding, but soft/hard/router all equal the plain rule (0.846) —
the router is confident and the weighted vote reproduces the rule. No value-add.

## 3. Interpretation

**Why M4 works:** the regime structure is real — different DGP families genuinely
favor different models (seasonal→arima, trending→ets). The router's structural
prior (learn a 3-way classification, use fixed experts) needs far less data than
exp42's flexible per-model weight regression. Constrained > flexible at n=529.

**Why the others collapse:** on SMM, Food CPI, and BoC, one model/policy dominates
every regime in training, so all experts are identical and the router degenerates
to best-single. Regime machinery adds nothing when there is no regime-specific
specialization to exploit. This is an honest negative for those datasets — and
a useful diagnostic: *expert diversity is the precondition for routing to help.*

**vs exp42 (the key comparison):**

| Method | M4 vs equal weights |
|---|---|
| exp42 learned combination (flexible) | 0.1300 vs 0.1295, p=0.156 (n.s.) |
| exp43 regime router (constrained) | 0.1282 vs 0.1295, **p=0.018** |

The structural prior helped where flexible learning drowned in estimation error.
This is the forecast-combination-puzzle lesson applied: constrain the hypothesis
space and you can beat equal weights with 500 records; leave it flexible and you
need 100k (FFORMA).

**vs exp41 (learned selection):** selection closed +0.9% of the gap (tied with
best single); the router closes +11.8% on M4. Soft regime weighting with
specialized experts > hard selection. The literature was right: average, don't
select — but average *within regimes that mean something*.

## 4. Limitations

1. M4 regimes are rule-defined from the same features the router uses — the
   classifier partly re-learns the rule (regacc ~0.9). The synthetic DGP-kind
   test (true latent regimes) is cleaner but underpowered there (p=0.19).
2. Food CPI: n=24 correlated origins; treat its p-value as directional only.
3. Router vs best single model is not significant (p=0.20, 5 folds) — the win
   over equal weights is solid, the win over arima is suggestive.
4. No temporal regime persistence (Markov smoothing) — records are treated as
   independent; v2 could smooth P(regime_t) with P(regime_{t-1}) on single-series
   datasets.

## 5. What's next

- The router is worth keeping as a combination baseline: it beats equal weights
  where exp42 failed, costs little, and degrades gracefully to best-single.
- Biggest open lever: *learned* regimes (unsupervised clustering on features)
  instead of hand rules — the current rules may not carve nature at its joints.
- Forward-looking regime signals (news density, event calendars) as router
  inputs — exp44 (news as selector) is the natural complement: use news to
  predict the regime, not the forecast.

## 6. Reproducibility

`cd td-one && .venv/bin/python experiments/exp43_regime_router.py [--quick] [--skip-boc]`
Deterministic (fixed seeds), ~72s full run. Results: `exp43_results.json`.
