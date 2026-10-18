# exp45: Learned Regime Router — Report

**Date:** 2026-10-17 · **Status:** complete · **Verdict: HONEST NEGATIVE** —
hand-written rules significantly beat unsupervised regime discovery on M4
(p=0.00046). The hand rules weren't arbitrary: they encode the structural
prior that made exp43 work, and clustering can't recover it.

## 1. What was built

`experiments/exp45_learned_regimes.py` (ruff clean + formatted, ~82s full run):

Per fold, fit **only on TRAIN records** (PIT-safe):

1. **GMM branch:** `GaussianMixture(K)` on standardized features → posterior
   P(regime|x) used directly as the router (soft, no separate classifier).
2. **KMeans branch:** `KMeans(K)` → hard clusters → `LogisticRegression`
   router (same structure as exp43).
3. **Hand branch:** exp43's hand-rule labels → LR router (re-run on the SAME
   folds for apples-to-apples paired comparison).

Per-regime experts = best model on train records of that regime; soft
combination Σ_r P(r|x) · expert_r quantiles (Vincentized), same as exp43.
K: M4=3, synthetic=3, SMM=2, Food CPI=2 (matches exp43's regime counts).

Interpretability diagnostics: silhouette scores, ARI(learned clusters, hand
labels), and on synthetic ARI vs the true DGP kind (ground truth).

## 2. Results

| Dataset | Hand | GMM | KMeans | Equal | gmm vs hand (p) |
|---|---|---|---|---|---|
| M4 | **0.1282** | 0.1302 | 0.1302 | 0.1295 | **0.00046** (hand better) |
| Synthetic | 0.0855 | 0.0856 | 0.0856 | 0.0898 | 0.97 (tie) |
| SMM | 0.1843 | 0.1843 | 0.1843 | 0.1975 | tie (experts collapse) |
| Food CPI | 0.0114 | 0.0114 | 0.0114 | 0.0140 | tie (experts collapse) |

Key: GMM does **not** beat equal weights on M4 (0.1302 vs 0.1295, p=0.23).
Only the hand-rule router did (exp43, p=0.018).

### Interpretability

| Dataset | Silhouette (GMM) | ARI(GMM, hand) | ARI(GMM, truth) |
|---|---|---|---|
| M4 | 0.27 | **0.162** | — |
| Synthetic | 0.35 | 0.581 | 0.581 |
| SMM | 0.52 | 0.259 | — |
| Food CPI | 0.32 | 0.342 | — |

- M4: learned clusters barely resemble hand regimes (ARI 0.162). The
  per-fold GMM experts are degenerate — all one model (arima×3 or ets×3) —
  while hand-rule experts show genuine specialization (seasonal→ets,
  trending→arima, consistent across folds).
- Synthetic: GMM partially recovers the true DGP kinds (ARI 0.581) but
  experts collapse to naive everywhere, so routing adds nothing over hand.
- SMM / Food CPI: learned clusters map onto the same binary split as the
  hand rules (identical CRPS); silhouette is higher (0.52/0.32) but experts
  collapse either way.

## 3. Interpretation

**Hand rules win because they encode domain knowledge.** "Seasonal" and
"trending" are the right axes for separating M4 series because they are the
axes along which ARIMA vs ETS performance differs. GMM, clustering the same
10 features without a target, finds groups that are statistically coherent
(silhouette 0.27) but not *model-relevant* — the clusters don't separate
series by which forecaster wins. This is why the per-fold GMM experts are
degenerate: the clusters carry no signal about the loss surface.

This is the positive mirror of the exp43 lesson. exp43 showed **constrained
structural priors beat flexible learning at small-n**; exp45 shows *why*:
the prior is doing real work. "If vol > X → stressed" looks like a hack,
but it points at the right latent structure. Unsupervised discovery, with no
access to the loss, can't find it.

A corollary: the hand rules are not the ceiling. The ARI of 0.581 on
synthetic shows clustering *can* find structure when it's clean. The gap is
that our 10 features don't cluster M4 into model-relevant groups. Better
features (the tsfeatures-lite gap from Option 3: ACF/PACF, entropy, Hurst,
lumpiness) or semi-supervised discovery (cluster to *predict model loss*,
not feature density) are the remaining directions — not more flexible
unsupervised methods on the same features.

## 4. What this closes

- **Learned regimes (unsupervised):** dead as a drop-in replacement for
  hand rules. GMM/KMeans on our 10 features find no model-relevant
  structure on M4.
- **exp43's win stands:** it was the hand rules, not luck. Keep the regime
  router with hand-defined regimes as the combination baseline.

## 5. Groq reasoning-effort check

The parent asked whether the Groq API supports a reasoning-level parameter.

**Yes.** Groq's Chat Completions API supports `reasoning_effort` for
GPT-OSS models (`openai/gpt-oss-120b`): values `"low"`, `"medium"`, `"high"`
(per Groq's reasoning docs, console.groq.com/docs/reasoning). Maximum = `"high"`.

**Limitation:** our client (`src/td_one/agent/groq_client.py`,
`GroqChatClient`) does **not** expose it. The request body in
`_do_complete` sends only `model`, `messages`, `max_tokens`, and optional
`response_format`. To set it to max, add:

```python
body["reasoning_effort"] = "high"
```

next to `"max_tokens": self.max_tokens,` in `_do_complete`
(~line 160 of `groq_client.py`), or better: add a
`reasoning_effort: str = "medium"` constructor arg (defaulting to the
API's current default so behavior is unchanged unless opted in).

Caveat: higher effort = more reasoning tokens = more of the ~8000 TPM
budget consumed per call. The client already documents a "reasoning tax"
(`GroqEmptyContentError`, line 47). Recommend: keep `"medium"` default,
expose the knob, use `"high"` only for the real-LLM validation runs
(exp36 policy re-run) where diagnosis quality matters more than throughput.

## 6. Reproducibility

`cd td-one && .venv/bin/python experiments/exp45_learned_regimes.py`
(deterministic, seed=42, ~82s). `--quick` for the 38s smoke test.
`exp45_results.json` has per-fold experts, silhouette, and ARI diagnostics.
