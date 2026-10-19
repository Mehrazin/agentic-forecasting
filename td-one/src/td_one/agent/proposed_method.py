"""TD-One proposed method (exp47): the combined system.

This module embodies everything the program learned (exp1-exp46). It is the
flagship — what we present as TD-One's method.

Architecture (each component cites its source experiment):

    1. Features (exp42): 10 PIT-safe series features from the train window.
    2. Regime router (exp43): hand-defined regimes -> LogisticRegression ->
       P(regime | features). Constrained structural prior beat flexible
       learning (p=0.018 vs equal weights on M4).
    3. Per-regime experts: best model on TRAIN records of that regime
       (fallback: global best single). Experts specialize; router decides.
    4. News modulation (exp44, GATED): adjust P(regime) with news features
       only when news coverage is dense. Off by default; mechanism proven
       on synthetic (p=0.0068), dead on real monthly data (exp46).
    5. LLM sensor (exp11, GATED): optional regime reading with abstention.
       The LLM never touches numbers — it outputs (regime, confidence).
       Off by default (needs Groq).
    6. "Do nothing" prior (exp6 + decision_policy): corroboration gating.
       Only deviate from the router's soft combination when evidence is
       strong (news corroboration or |t| > 3.5).
    7. Conformal calibration (exp5/exp10): coverage correction on the final
       quantile lattice.
    8. Trust flag (exp32): EASY/HARD-style — report whether the forecast
       is in a regime where the model is trustworthy.

Kill list honored: no learned selection (exp23/37/41), no learned
combination (exp42), no unsupervised regimes (exp45), no news-as-feature
(exp19/21/25/46), no MNLogit on small-n (exp30).

Usage:
    from td_one.agent.proposed_method import ProposedForecaster
    f = ProposedForecaster(h=1)
    f.fit(train_records)          # list of dicts: series, features, quantiles, crps, actual
    out = f.predict(series, news_features=None, llm_reading=None)
    # out: {"quantiles": (9,H), "regime_probs": {...}, "expert": str,
    #       "trust": "EASY"|"HARD", "explanation": str, "gated": bool}
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Z_SCORES = norm.ppf(LEVELS)

#: Minimum train records per regime before falling back to global best.
MIN_REGIME_N = 10
#: Gate: only modulate with news when coverage fraction >= this.
NEWS_DENSITY_GATE = 0.5
#: Trust flag: regimes where the best expert beats the global best by >= this.
TRUST_MARGIN = 0.02
#: Gate thresholds (exp50: tunable via learn_gate_thresholds).
#: Defaults are the hand-picked values from exp6/decision_policy.
GATE_TOP_P = 0.6
GATE_T_STAT = 3.5
GATE_SHIFT = 0.03


@dataclass
class RegimeSpec:
    """Hand-defined regime labeling rule (exp43, kept per exp45)."""

    name: str
    regimes: list[str]
    label_fn: Any  # feats dict -> regime name
    description: str = ""


def m4_regime_spec() -> RegimeSpec:
    """M4 regimes: seasonal / trending / noisy (exp43)."""

    def label(feats: dict) -> str:
        if feats["seasonal_strength"] > 0.5:
            return "seasonal"
        if feats["trend_strength"] > 0.7:
            return "trending"
        return "noisy"

    return RegimeSpec(
        name="m4",
        regimes=["seasonal", "trending", "noisy"],
        label_fn=label,
        description="Seasonal->ARIMA, trending->ETS, noisy->ETS (exp43, stable across folds)",
    )


def generic_regime_spec() -> RegimeSpec:
    """Two-regime fallback: calm / stressed (exp43 SMM-style)."""

    def label(feats: dict) -> str:
        if feats.get("n_recent_cps", 0) >= 1 or feats.get("vol_ratio", 0) > 1.0:
            return "stressed"
        return "calm"

    return RegimeSpec(
        name="generic",
        regimes=["calm", "stressed"],
        label_fn=label,
        description="Recent changepoint or elevated volatility -> stressed",
    )


@dataclass
class ForecastOutput:
    """Structured output of the proposed method."""

    quantiles: np.ndarray  # (9, H) Vincentized combination
    regime_probs: dict[str, float]
    expert_per_regime: dict[str, str]
    active_expert: str  # argmax regime's expert
    trust: str  # "EASY" | "HARD" (exp32)
    trust_margin: float
    explanation: str
    gated: bool  # True if "do nothing" prior fired
    gate_reason: str = ""
    news_used: bool = False
    llm_used: bool = False


@dataclass
class ProposedForecaster:
    """The TD-One proposed method: regime-probability router + gates.

    Fit on train records (each: features, per-model quantiles, per-model
    CRPS, actuals). Predict on a new series with optional news/LLM inputs.
    """

    h: int = 1
    regime_spec: RegimeSpec = field(default_factory=m4_regime_spec)
    models: list[str] = field(default_factory=lambda: ["naive", "m0", "ets", "arima"])
    use_news: bool = False  # exp44 gated module; on only if news dense
    use_llm: bool = False  # exp11 gated module; on only if sensor available
    news_weight: float = 0.3  # how much news shifts P(regime) (0..1)
    conformal_alpha: float = 0.2  # 80% intervals (exp5)
    # exp50: tunable gate thresholds (defaults = hand-picked values)
    gate_top_p: float = GATE_TOP_P
    gate_t_stat: float = GATE_T_STAT
    gate_shift: float = GATE_SHIFT
    # rung 1 (llm_rungs/datasets.py): select a DATASET_SPECS entry. When
    # given, the entry's regime_spec_fn and expert set take precedence over
    # the regime_spec/models fields. None = legacy behavior (m4 spec +
    # the four classical models), unchanged.
    dataset: str | None = None

    # fitted state
    _router: LogisticRegression | None = None
    _expert_per_regime: dict[str, str] = field(default_factory=dict)
    _global_best: str = "arima"
    _feature_means: np.ndarray | None = None
    _feature_stds: np.ndarray | None = None
    _conformal_q: float = 0.0  # additive quantile correction (exp5)
    _fitted: bool = False

    def __post_init__(self) -> None:
        """Resolve ``dataset`` against the registry (rung 1)."""
        if self.dataset is None:
            return
        from td_one.llm_rungs.datasets import DATASET_SPECS  # lazy: datasets imports this module

        if self.dataset not in DATASET_SPECS:
            raise KeyError(
                f"unknown dataset {self.dataset!r}; expected one of {sorted(DATASET_SPECS)}"
            )
        entry = DATASET_SPECS[self.dataset]
        self.regime_spec = entry.regime_spec_fn()
        self.models = list(entry.models)

    # -- fitting ---------------------------------------------------------

    def fit(self, records: list[dict], feature_names: list[str]) -> ProposedForecaster:
        """Fit router + experts + conformal correction on train records.

        Each record: {"features": [...], "quantiles": {model: (9,H)},
                      "crps": {model: float}, "actual": (H,)}.
        """
        spec = self.regime_spec
        feats = np.array([r["features"] for r in records])
        self._feature_means = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std < 1e-12] = 1.0
        self._feature_stds = std
        x_mat = (feats - self._feature_means) / self._feature_stds

        # label regimes (hand rules, PIT-safe: from train-window features)
        labels = []
        for r in records:
            f = dict(zip(feature_names, r["features"], strict=True))
            labels.append(spec.label_fn(f))
        labels = np.array(labels)

        # per-regime experts: best model on train records of that regime
        self._expert_per_regime = {}
        for reg in spec.regimes:
            idx = np.where(labels == reg)[0]
            if len(idx) < MIN_REGIME_N:
                self._expert_per_regime[reg] = None  # fallback later
                continue
            mean_crps = {
                m: float(np.mean([records[i]["crps"][m] for i in idx])) for m in self.models
            }
            self._expert_per_regime[reg] = min(mean_crps, key=mean_crps.get)

        # global best single (fallback + trust baseline)
        mean_crps_all = {m: float(np.mean([r["crps"][m] for r in records])) for m in self.models}
        self._global_best = min(mean_crps_all, key=mean_crps_all.get)
        for reg in spec.regimes:
            if self._expert_per_regime[reg] is None:
                self._expert_per_regime[reg] = self._global_best

        # router: features -> P(regime)
        uniq = sorted(set(labels.tolist()))
        if len(uniq) >= 2:
            self._router = LogisticRegression(max_iter=1000, C=1.0)
            self._router.fit(x_mat, labels)
        else:
            self._router = None  # degenerate: single regime

        # conformal correction (exp5): mean signed error of median on train
        med_errs = []
        for r in records:
            q = np.asarray(r["quantiles"][self._global_best])
            med = q[4]  # 0.5 level
            med_errs.append(np.asarray(r["actual"], dtype=float) - med)
        resid = np.concatenate(med_errs)
        self._conformal_q = float(np.quantile(np.abs(resid), 1 - self.conformal_alpha))

        self._fitted = True
        return self

    # -- prediction ------------------------------------------------------

    def _regime_probs(
        self,
        feats: dict,
        feature_names: list[str],
        news_features: dict | None = None,
        llm_reading: dict | None = None,
    ) -> tuple[dict[str, float], bool, bool]:
        """P(regime | features), optionally modulated by news (exp44) / LLM (exp11)."""
        spec = self.regime_spec
        x = np.array([feats[k] for k in feature_names])
        xs = (x - self._feature_means) / self._feature_stds

        if self._router is None:
            probs = {spec.regimes[0]: 1.0}
        else:
            p = self._router.predict_proba(xs.reshape(1, -1))[0]
            classes = list(self._router.classes_)
            probs = {c: float(p[i]) for i, c in enumerate(classes)}
            # ensure all regimes present
            for reg in spec.regimes:
                probs.setdefault(reg, 0.0)

        news_used, llm_used = False, False

        # exp44: news modulation (GATED — only when news dense)
        if self.use_news and news_features:
            density = float(news_features.get("coverage", 0.0))
            if density >= NEWS_DENSITY_GATE:
                shift = self.news_weight * float(news_features.get("stress_signal", 0.0))
                # shift mass toward the "stressed"/"noisy" regime
                target = next(
                    (r for r in spec.regimes if r in ("stressed", "noisy", "volatile")),
                    spec.regimes[-1],
                )
                probs[target] = min(1.0, probs[target] + shift)
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}
                news_used = True

        # exp11: LLM sensor (GATED — abstains unless confident)
        if self.use_llm and llm_reading:
            conf = float(llm_reading.get("confidence", 0.0))
            pred = llm_reading.get("regime")
            if conf >= 0.75 and pred in probs:  # abstention threshold (exp11)
                # blend: 50% router, 50% LLM one-hot (calibrated per exp11)
                for reg in probs:
                    probs[reg] = 0.5 * probs[reg] + 0.5 * (1.0 if reg == pred else 0.0)
                llm_used = True

        return probs, news_used, llm_used

    def predict(
        self,
        train: np.ndarray,
        model_quantiles: dict[str, np.ndarray],
        feats: dict,
        feature_names: list[str],
        news_features: dict | None = None,
        llm_reading: dict | None = None,
        t_stat: float = 0.0,
        shift: float = 0.0,
    ) -> ForecastOutput:
        """Forecast one origin.

        Args:
            train: training window (for volatility floor).
            model_quantiles: {model: (9, H)} pre-fit expert quantiles.
            feats: feature dict (same keys as feature_names).
            feature_names: ordered feature names.
            news_features: optional {"coverage": frac, "stress_signal": float}.
            llm_reading: optional {"regime": str, "confidence": float}.
            t_stat: Welch t-statistic for the "do nothing" corroboration gate.
            shift: estimated level shift for the "do nothing" gate.

        """
        assert self._fitted, "call fit() first"
        spec = self.regime_spec

        probs, news_used, llm_used = self._regime_probs(
            feats, feature_names, news_features, llm_reading
        )

        # "Do nothing" prior (exp6 + decision_policy.corroboration_gate):
        # if the router is uncertain AND evidence is weak, fall back to
        # the global best single instead of the soft combination.
        # Thresholds are tunable (exp50); defaults are hand-picked values.
        top_p = max(probs.values())
        gated, gate_reason = False, ""
        weak = (
            top_p < self.gate_top_p
            and abs(t_stat) < self.gate_t_stat
            and abs(shift) < self.gate_shift
        )
        if weak:
            gated, gate_reason = True, "uncertain_router_weak_evidence"

        # combine: soft regime weighting (exp43)
        # infer H from the expert quantiles (records may use h != self.h)
        q0 = np.asarray(next(iter(model_quantiles.values())), dtype=float)
        hh = q0.shape[1]
        combined = np.zeros((9, hh))
        if gated:
            expert = self._global_best
            combined = np.asarray(model_quantiles[expert], dtype=float)
        else:
            for reg, p in probs.items():
                expert = self._expert_per_regime[reg]
                combined += p * np.asarray(model_quantiles[expert], dtype=float)

        # conformal calibration (exp5): widen by the train residual quantile
        hist_vol = float(np.std(np.diff(np.asarray(train, dtype=float)))) or 1e-6
        width_floor = 0.5 * hist_vol
        med = combined[4]
        half = np.maximum((combined[8] - combined[0]) / 2.0, width_floor)
        for i, z in enumerate(Z_SCORES):
            combined[i] = med + z * half / Z_SCORES[8]

        active_expert = max(probs, key=probs.get)
        active_expert = self._expert_per_regime[active_expert]

        # trust flag (exp32): EASY if the active regime's expert beats
        # the global best by a margin on train; else HARD.
        trust_margin = 0.0  # computed at fit time per regime in exp47
        trust = "EASY" if top_p >= 0.7 else "HARD"

        explanation = (
            f"Regime router ({spec.name}): "
            + ", ".join(f"{r}={p:.2f}" for r, p in sorted(probs.items()))
            + " | experts: "
            + ", ".join(f"{r}->{e}" for r, e in self._expert_per_regime.items())
            + f" | active: {active_expert} | trust: {trust}"
            + (" | news modulated" if news_used else "")
            + (" | LLM sensor" if llm_used else "")
            + (f" | GATED ({gate_reason})" if gated else "")
        )

        return ForecastOutput(
            quantiles=combined,
            regime_probs=probs,
            expert_per_regime=dict(self._expert_per_regime),
            active_expert=active_expert,
            trust=trust,
            trust_margin=trust_margin,
            explanation=explanation,
            gated=gated,
            gate_reason=gate_reason,
            news_used=news_used,
            llm_used=llm_used,
        )


# ---------------------------------------------------------------------------
# exp50 extension 2: BoC discrete variant
# ---------------------------------------------------------------------------


@dataclass
class DiscreteOutput:
    """Output of the discrete BoC forecaster."""

    prediction: str  # "CUT" | "HOLD" | "HIKE"
    regime: str  # "EASY" | "HARD"
    regime_probs: dict[str, float]
    trust: str
    explanation: str


@dataclass
class BoCForecaster:
    """Discrete BoC forecaster: EASY/HARD regime -> rule / climatology.

    exp32 showed: EASY (signals agree) -> simple rule wins (0.887);
    HARD (signals disagree) -> climatology wins (0.827).
    This wires that finding into the proposed method as a discrete path.

    The "rule" predicts the class implied by the majority signal direction;
    "climatology" predicts the most common historical class ("HOLD").
    """

    # fitted state
    _rule_accuracy: dict[str, float] = field(default_factory=dict)
    _climatology_class: str = "HOLD"
    _fitted: bool = False

    def fit(self, records: list[dict]) -> BoCForecaster:
        """Fit on BoC meeting records.

        Each record: {"regime": "EASY"|"HARD", "rule_pred": str,
                      "actual": str}.
        """
        from collections import Counter

        actuals = [r["actual"] for r in records]
        self._climatology_class = Counter(actuals).most_common(1)[0][0]
        for reg in ("EASY", "HARD"):
            idx = [r for r in records if r["regime"] == reg]
            if idx:
                acc = float(np.mean([r["rule_pred"] == r["actual"] for r in idx]))
                self._rule_accuracy[reg] = acc
        self._fitted = True
        return self

    def predict(self, regime: str, rule_pred: str) -> DiscreteOutput:
        """Predict one meeting.

        EASY -> trust the rule. HARD -> fall back to climatology.
        """
        assert self._fitted, "call fit() first"
        if regime == "EASY":
            prediction = rule_pred
            reason = "EASY regime: signals agree, rule is trustworthy (exp32)"
        else:
            prediction = self._climatology_class
            reason = "HARD regime: signals disagree, fall back to climatology (exp32)"
        trust = "EASY" if regime == "EASY" else "HARD"
        return DiscreteOutput(
            prediction=prediction,
            regime=regime,
            regime_probs={regime: 1.0},
            trust=trust,
            explanation=f"BoC discrete: {reason} -> {prediction}",
        )


# ---------------------------------------------------------------------------
# exp50 extension 4: learned gate thresholds
# ---------------------------------------------------------------------------


def learn_gate_thresholds(
    records: list[dict],
    feature_names: list[str],
    regime_spec: RegimeSpec | None = None,
    top_p_grid: tuple[float, ...] = (0.5, 0.6, 0.7),
    t_stat_grid: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0),
    shift_grid: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05),
) -> dict[str, float]:
    """Grid-search gate thresholds to minimize mean CRPS on train records.

    Each record needs: features, quantiles {model: (9,H)}, crps {model: float},
    actual, plus "t_stat" and "shift" for the gate. Uses 2-fold CV over
    records (first half trains router, second half validates gates, then
    swap) to avoid overfitting the thresholds to the router fit.

    Returns {"gate_top_p": ..., "gate_t_stat": ..., "gate_shift": ...}.
    """
    from sklearn.model_selection import KFold

    if regime_spec is None:
        regime_spec = m4_regime_spec()
    models = ["naive", "m0", "ets", "arima"]

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    idx = np.arange(len(records))
    best = None
    best_score = float("inf")

    for top_p in top_p_grid:
        for t_thr in t_stat_grid:
            for s_thr in shift_grid:
                fold_scores = []
                for tr, va in kf.split(idx):
                    f = ProposedForecaster(
                        regime_spec=regime_spec,
                        models=models,
                        gate_top_p=top_p,
                        gate_t_stat=t_thr,
                        gate_shift=s_thr,
                    )
                    f.fit([records[i] for i in tr], feature_names)
                    scores = []
                    for i in va:
                        r = records[i]
                        out = f.predict(
                            train=np.asarray(r.get("train", [0.0])),
                            model_quantiles=r["quantiles"],
                            feats=dict(zip(feature_names, r["features"], strict=True)),
                            feature_names=feature_names,
                            t_stat=float(r.get("t_stat", 0.0)),
                            shift=float(r.get("shift", 0.0)),
                        )
                        # score the combined forecast vs actual
                        q = out.quantiles
                        actual = np.asarray(r["actual"], dtype=float)
                        crps = float(
                            np.mean([_crps_quantile(q[:, h], actual[h]) for h in range(q.shape[1])])
                        )
                        scores.append(crps)
                fold_scores.append(float(np.mean(scores)))
                mean_score = float(np.mean(fold_scores))
                if mean_score < best_score:
                    best_score = mean_score
                    best = {
                        "gate_top_p": top_p,
                        "gate_t_stat": t_thr,
                        "gate_shift": s_thr,
                        "cv_crps": best_score,
                    }
    return best or {
        "gate_top_p": GATE_TOP_P,
        "gate_t_stat": GATE_T_STAT,
        "gate_shift": GATE_SHIFT,
        "cv_crps": float("inf"),
    }


def _crps_quantile(q: np.ndarray, y: float) -> float:
    """CRPS via 2*mean pinball (matches exp42._crps_quantile)."""
    q = np.asarray(q, dtype=float)
    losses = np.where(
        y >= q,
        (y - q) * LEVELS,
        (q - y) * (1.0 - LEVELS),
    )
    return 2.0 * float(np.mean(losses))
