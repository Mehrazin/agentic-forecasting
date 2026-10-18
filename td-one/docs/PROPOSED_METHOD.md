# TD-One Proposed Method: Architecture

**Date:** 2026-10-18
**Status:** Implemented (`src/td_one/agent/proposed_method.py`), tested (exp47)
**Verdict:** Positive — beats equal weights, best single, and exp43 router on M4 (all p < 1e-8)

---

## 1. Overview

The proposed method is a **regime-probability router with gates** — the
distillation of 47 experiments into a single system. Every component earned
its place through a positive result; every exclusion was killed by a
definitive negative (see `docs/POSITIVE_LIFT_AUDIT.md`).

```mermaid
flowchart TB
    IN["Input: time series<br/>(+ optional news, LLM reading)"]
    FEAT["10 PIT-safe features<br/>exp42"]
    ROUTER["Regime router (exp43)<br/>hand rules → LogisticReg<br/>→ P(regime | features)"]
    NEWS["News modulation<br/>(exp44, GATED)<br/>only if dense"]
    LLM["LLM sensor<br/>(exp11, GATED)<br/>only if conf ≥ 0.75"]
    GATE["'Do nothing' gate<br/>uncertain + weak → global best<br/>exp6 + decision_policy"]
    EXPERTS["Per-regime experts<br/>Σ P(r) × expert_r<br/>exp43"]
    CONF["Conformal calibration<br/>volatility floor<br/>exp5 / exp10"]
    OUT["Output: quantiles<br/>+ regime probs<br/>+ trust (EASY/HARD) [exp32]<br/>+ explanation"]

    IN --> FEAT --> ROUTER
    ROUTER --> NEWS
    ROUTER --> LLM
    NEWS --> GATE
    LLM --> GATE
    GATE --> EXPERTS --> CONF --> OUT
```

## 2. Components and their evidence

### 2.1 Regime-probability router (exp43) — THE BACKBONE

**Evidence:** M4 0.1282 vs equal 0.1295, **p=0.018** — first method in the
program to significantly beat equal weights.

**Design:** Hand-defined regimes (seasonal/trending/noisy for M4;
calm/stressed for SMM-style) → LogisticRegression → P(regime|features).
Per-regime expert = best model on TRAIN records of that regime.
Forecast = Σ P(r) × expert_r quantiles (Vincentized).

**Why it works:** Constrained structural priors beat flexible learning at
small-n. Learning a 3-way classification needs far less data than regressing
4 continuous weights (exp42 failed, p=0.156). exp45 proved the hand rules
encode real domain knowledge — unsupervised clustering can't recover it
(p=0.00046, hand wins).

### 2.2 Trust flag (exp32)

**Evidence:** BoC EASY/HARD split gives +7.2pp (0.887 vs 0.815).

**Design:** `trust = "EASY"` if top regime P ≥ 0.7, else `"HARD"`.
Tells the user when the model is in a regime where it's trustworthy.

### 2.3 "Do nothing" gate (exp6 + decision_policy)

**Evidence:** M3+gate 7.60 vs M0 7.86 held-out; kills catastrophic spurious
adjustments. `corroboration_gate`: |t| > 3.5 or news corroboration.

**Design:** If router is uncertain (top P < 0.6) AND evidence is weak
(|t| < 3.5, |shift| < 0.03), fall back to global best single instead of
the soft combination. Fired on 5% of M4 origins in exp47 — and those are
exactly where the soft combination is most fragile.

### 2.4 Conformal calibration (exp5/exp10)

**Evidence:** Coverage 0.671 → 0.914.

**Design:** Quantile half-width floored at 0.5 × historical volatility.
Prevents the degenerate intervals exp22 diagnosed (3-point sd → width 0.03).

### 2.5 News modulation (exp44) — GATED, off by default

**Evidence:** Mechanism proven on synthetic (p=0.0068); dead on real
monthly data (exp46 definitive negative).

**Design:** `use_news=True` only when news coverage ≥ 50%. Shifts P(regime)
toward stressed/noisy by `news_weight × stress_signal`. Wired in, off by
default — the gate honors exp46's verdict.

### 2.6 LLM sensor (exp11) — GATED, off by default

**Evidence:** M4a 3.59 vs M0 4.82 (real Groq); LLM as noisy sensor with
abstention. Fragile — single confident error catastrophic.

**Design:** `use_llm=True` only with explicit reading. Blends 50/50 with
router iff confidence ≥ 0.75 (abstention threshold). LLM never touches
numbers — outputs (regime, confidence) only.

## 3. What was deliberately excluded

| Excluded | Experiments | Reason |
|---|---|---|
| Learned selection | exp23, exp37, exp41 | = random or tied with best single |
| Learned combination | exp42 | Forecast combination puzzle (p=0.156) |
| Unsupervised regimes | exp45 | Hand rules win (p=0.00046) |
| News-as-feature | exp19, exp21, exp25 | Clean/definitive negatives |
| News monthly | exp46 | Definitive negative; track closed |
| MNLogit (small-n) | exp30 | Overfits; simple rule wins |
| M5 LLM router | exp12 | Loses to inverse-CRPS |
| Macro overlay | exp8, exp15 | Retracted / hurts |

## 4. Performance (exp47)

| Dataset | Proposed | Equal | Best single | Router-only | vs equal | vs router |
|---|---|---|---|---|---|---|
| M4 (n=600) | **0.1451** | 0.1507 | 0.1505 | 0.1490 | **p=6.5e-09** | **p=6.7e-10** |
| Synthetic (n=480) | 0.1036 | 0.1068 | 0.1028 | 0.1028 | p=0.072 | p=0.28 |
| SMM (n=27) | 0.1389 | 0.1424 | 0.1377 | 0.1377 | p=0.66 | p=0.61 |
| Food CPI (n=24) | 0.0141 | 0.0176 | 0.0143 | 0.0143 | p=3e-18* | p=0.005* |

\* Correlated origins — p overstated.

**Key:** proposed beats the exp43 router alone (p=6.7e-10 on M4). The
system is more than the sum of its parts — gates + conformal add value
beyond the router.

## 5. Usage

```python
from td_one.agent.proposed_method import ProposedForecaster, m4_regime_spec

# fit on train records
fc = ProposedForecaster(h=6, regime_spec=m4_regime_spec())
fc.fit(train_records, feature_names)  # records: features, quantiles, crps, actual

# predict one origin
out = fc.predict(
    train_window,          # np.ndarray (for volatility floor)
    model_quantiles,       # {model: (9, H)}
    feats,                 # {feature_name: value}
    feature_names,
    news_features=None,    # optional {"coverage": frac, "stress_signal": float}
    llm_reading=None,      # optional {"regime": str, "confidence": float}
    t_stat=0.0,            # for the corroboration gate
    shift=0.0,
)
print(out.explanation)
# "Regime router (m4): noisy=0.72, seasonal=0.18, trending=0.10 |
#  experts: seasonal->arima, trending->ets, noisy->ets |
#  active: ets | trust: EASY"
```

## 6. Open extensions (not yet implemented)

1. **TSFM expert:** add calibrated TimesFM/Chronos-2 to the expert zoo
   (exp14: −24%, exp18b: −12.2%).
2. **BoC discrete variant:** EASY/HARD → rule/climatology for CUT/HOLD/HIKE.
3. **Semi-supervised regimes:** cluster to predict model loss, not feature
   density (exp45's suggested direction).
4. **Learned gate threshold:** tune the 0.6/3.5/0.03 gate parameters.

---

*This is TD-One's method. Everything else was scaffolding.*
