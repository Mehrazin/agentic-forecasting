"""TD-One proposed method v2 (exp64): unified forecaster with gated LLM expert.

v2 extends the v1 ``ProposedForecaster`` (regime router + gates + conformal)
with an LLM expert that fires only on series types where it helps.

Design insight (LLM ladder, exp55-59):
- LLM wins decisively on smooth macro series with semantic context
  (Food CPI: 0.4262 vs 0.5158 generic, beats Vector's 0.5688).
- LLM loses badly on seasonal multi-series data
  (Tourism: 64% worse; M3: 36% worse than classical).
- Per-dataset prompts (V1) are the biggest lever; the router adds nothing
  when one expert dominates.

v2 architecture:
    1. Series-type gate: LLM expert included only when seasonal_strength < 0.5
       AND the series has semantic context (macro-like). Otherwise v2 = v1.
    2. Regime router over {classical experts + LLM expert (if gated on)}.
    3. Same gates (do-nothing prior), conformal calibration, trust flag as v1.

The LLM expert's quantiles come from cached grids (offline) or live calls
via ``VectorProxyClient``. The LLM never outputs numbers directly into the
combination — its quantile grids are treated as one expert among many,
and the router learns per-regime whether to trust it.

Usage:
    from td_one.agent.proposed_method_v2 import ProposedForecasterV2
    f = ProposedForecasterV2(h=1, dataset="food_cpi", use_llm_expert=True)
    f.fit(train_records)  # records include "llm" in quantiles/crps where available
    out = f.predict(...)  # same interface as v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from td_one.agent.proposed_method import (
    ForecastOutput,
    ProposedForecaster,
    m4_regime_spec,
)

#: LLM expert name in the models list and quantiles dict.
LLM_EXPERT = "llm"

#: Gate threshold: LLM excluded when seasonal strength exceeds this.
#: From exp59: Tourism/M3 (seasonal_strength > 0.5) saw 36-64% degradation.
LLM_SEASONAL_GATE = 0.5


def llm_gate(
    feats: dict[str, float],
    dataset: str | None = None,
) -> tuple[bool, str]:
    """Decide whether the LLM expert should fire for this series/origin.

    Args:
        feats: feature dict (must include "seasonal_strength" when available).
        dataset: dataset name; macro-like datasets ("food_cpi", "smm")
            allow the LLM, seasonal/multi-series datasets do not.

    Returns:
        (use_llm, reason): whether to include the LLM expert and why.

    """
    # Dataset-level prior: seasonal benchmarks never benefit (exp59).
    seasonal_datasets = {"tourism", "m3", "m4"}
    if dataset in seasonal_datasets:
        return False, f"dataset {dataset!r} is seasonal/multi-series (exp59)"

    # Series-level gate: strong seasonality -> LLM hurts (exp59).
    seasonal_strength = float(feats.get("seasonal_strength", 0.0))
    if seasonal_strength >= LLM_SEASONAL_GATE:
        return False, f"seasonal_strength {seasonal_strength:.2f} >= {LLM_SEASONAL_GATE}"

    # Macro-like datasets with semantic context: LLM helps (exp55-58).
    macro_datasets = {"food_cpi", "smm"}
    if dataset in macro_datasets:
        return True, f"dataset {dataset!r} is macro-like with semantic context"

    # Default: allow LLM when seasonality is weak; the router learns the rest.
    return True, "weak seasonality, router decides per-regime"


@dataclass
class ProposedForecasterV2(ProposedForecaster):
    """Proposed method v2: v1 + gated LLM expert.

    Inherits the full v1 machinery (regime router, gates, conformal,
    trust flag). Adds:

    - ``use_llm_expert``: master switch for the LLM expert.
    - ``llm_gate_fn``: callable (feats, dataset) -> (bool, str);
      defaults to :func:`llm_gate`.
    - When the gate fires, "llm" must be in ``models`` and present in
      the per-origin ``model_quantiles`` dicts. When it doesn't, v2
      behaves exactly as v1.

    """

    use_llm_expert: bool = False
    llm_gate_fn: Any = None  # (feats: dict, dataset: str|None) -> (bool, str)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.llm_gate_fn is None:
            self.llm_gate_fn = llm_gate
        if self.use_llm_expert and LLM_EXPERT not in self.models:
            self.models = [*self.models, LLM_EXPERT]

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
        """Forecast one origin, with LLM gating.

        If ``use_llm_expert`` is on but the gate says no for this origin,
        the LLM expert is excluded from the combination (v1 behavior).
        """
        if self.use_llm_expert:
            dataset = getattr(self, "_logical_dataset", self.dataset)
            use_llm, reason = self.llm_gate_fn(feats, dataset)
            if not use_llm:
                # Exclude LLM for this origin: v1 behavior.
                # Map any regime whose expert is "llm" to the best classical
                # model (not the global best, which might itself be the LLM).
                mq = {m: q for m, q in model_quantiles.items() if m != LLM_EXPERT}
                classical_models = [m for m in self.models if m != LLM_EXPERT]
                # Best classical from the fitted per-regime experts, else first.
                fallback = next(
                    (
                        e
                        for e in self._expert_per_regime.values()
                        if e in classical_models
                    ),
                    classical_models[0],
                )
                orig_experts = self._expert_per_regime
                self._expert_per_regime = {
                    r: (e if e != LLM_EXPERT else fallback)
                    for r, e in orig_experts.items()
                }
                try:
                    out = super().predict(
                        train, mq, feats, feature_names,
                        news_features, llm_reading, t_stat, shift,
                    )
                finally:
                    self._expert_per_regime = orig_experts
                out.explanation += f" | LLM gated off ({reason})"
                return out
        return super().predict(
            train, model_quantiles, feats, feature_names,
            news_features, llm_reading, t_stat, shift,
        )


def v2_for_dataset(dataset: str, **kwargs: Any) -> ProposedForecasterV2:
    """Configure v2 for a dataset, enabling the LLM expert where it helps.

    - "food_cpi", "smm": LLM expert enabled (macro-like).
    - "m4", "tourism", "m3": LLM disabled (seasonal, exp59).
    - "boc_rates": use BoCForecaster instead (discrete path).

    Datasets not in the registry ("tourism", "m3") use the m4 regime
    spec with classical models.
    """
    llm_datasets = {"food_cpi", "smm"}
    use_llm = dataset in llm_datasets
    # Bypass the registry for datasets it doesn't know; set spec manually.
    if dataset in ("tourism", "m3"):
        fc = ProposedForecasterV2(
            dataset=None,
            regime_spec=m4_regime_spec(),
            models=["naive", "snaive", "ets", "arima"],
            use_llm_expert=use_llm,
            **kwargs,
        )
        # stash the logical dataset name for the gate
        fc._logical_dataset = dataset
        return fc
    return ProposedForecasterV2(
        dataset=dataset,
        use_llm_expert=use_llm,
        **kwargs,
    )
