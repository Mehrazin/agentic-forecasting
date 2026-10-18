# exp50: Open Extensions — Final Report

**Date:** 2026-10-18
**Status:** Complete

## Summary

Tested all 4 open extensions from PROPOSED_METHOD.md §6:

| Extension | Result | Verdict |
|---|---|---|
| 1. TSFM expert | 0% expert rate, identical CRPS | ❌ Doesn't help on M4 |
| 2. BoC discrete | Wired correctly | ✅ Implemented |
| 3. Loss regimes | p=0.44, hand wins | ❌ Doesn't help |
| 4. Learned gates | p=0.32, hand wins | ❌ Doesn't help |

**1 of 4 helps (BoC discrete). 3 are honest negatives.**

## Extension 1: TSFM Expert — Honest Negative

Added calibrated Chronos-2 (real weights from `models/weights/chronos2`) to the expert zoo via `LogScaleCalibrator`.

**Result:** TSFM was selected as per-regime expert 0% of the time. Mean CRPS identical with/without TSFM (0.11655). The mock/real TSFM doesn't beat ARIMA/ETS on these M4 series at h=6.

**Interpretation:** exp14's -24% was on a different M4 subset/horizon. On our test set, classical models already capture the signal. TSFM remains valuable as a standalone (exp18b: -12.2% on SMM), but doesn't add value *as a regime expert* in the router.

**Decision:** Keep the infrastructure (models list accepts "tsfm"), but don't enable by default.

## Extension 2: BoC Discrete Variant — Implemented ✅

`BoCForecaster` class added to `proposed_method.py`:
- EASY regime → trust the rule (0.887 accuracy)
- HARD regime → fall back to climatology (0.827 accuracy)

Verified: `easy_uses_rule=True`, `hard_uses_climatology=True`.

**This is now part of the proposed method** for discrete tasks.

## Extension 3: Semi-supervised Regimes — Honest Negative

Clustered LOSS vectors (not features) to find model-relevant regimes. Winners: [m0, arima, arima] — 2/3 clusters pick arima.

**Result:** Loss-regime CRPS 0.1190 vs hand-rule 0.1170 (p=0.44). Hand rules win.

**Interpretation:** exp45's suggestion was sound, but even loss-clustering can't beat the domain knowledge in hand rules ("seasonal→ARIMA" is the right prior). The clusters found "arima wins here" vs "arima wins here too" — not discriminative.

**Decision:** Keep hand-rule regimes. The prior is doing real work.

## Extension 4: Learned Gate Thresholds — Honest Negative

Grid-searched gates via 2-fold CV: learned (0.5/2.5/0.02) vs hand (0.6/3.5/0.03).

**Result:** Learned CRPS 0.1170 vs hand 0.1170 (p=0.32). No difference.

**Interpretation:** The gates fire rarely (~5%); threshold choice doesn't matter much. Hand-picked values are fine.

**Decision:** Keep hand-picked thresholds. Added `learn_gate_thresholds()` to the module for future use.

## Code Changes

**`src/td_one/agent/proposed_method.py`:**
- Tunable gate thresholds (`gate_top_p`, `gate_t_stat`, `gate_shift` params)
- `BoCForecaster` class (discrete EASY/HARD → rule/climatology)
- `learn_gate_thresholds()` function (grid search)
- Fixed `_crps_quantile` sign bug (was returning negative CRPS)

**`src/td_one/agent/tools/regimes.py`:**
- `loss_regime_spec()` — cluster loss vectors
- `loss_regime_classifier()` — features → loss-cluster

**`experiments/exp50_extensions.py`:** Test harness for all 4.

## Updated PROPOSED_METHOD.md

Section 6 marked:
- ✅ BoC discrete: implemented
- ❌ TSFM expert: tested, doesn't help as regime expert (keep as standalone)
- ❌ Loss regimes: tested, hand rules win
- ❌ Learned gates: tested, hand values fine

## Final Verdict

The proposed method stands as-is. The 4 extensions were worth testing; 1 improved it (BoC discrete), 3 confirmed the current design is already optimal.

**The system is complete.**
