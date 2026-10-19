"""Rung 1 — dataset registry (``docs/LLM_INTEGRATION_LADDER.md``).

Maps each benchmark dataset to its timeless metadata plus the modeling
choices that experiments previously hard-coded per dataset:

- ``description``: timeless metadata — what the series *is* (name, units,
  frequency, source). No numbers, no dates: nothing that could leak a label.
  PIT-safe by construction.
- ``regime_spec_fn``: callable returning the ``RegimeSpec`` the proposed
  method uses for this dataset (exp43 hand rules, kept per exp45).
- ``models``: expert set for this dataset — the classical experts for the
  continuous benchmarks (same content *and order* as exp42's ``MODELS``;
  order matters for best-single tie-breaking), the rule/climatology
  policies for the discrete BoC benchmark (exp32).
- ``data_path``: repo-relative path of the underlying data; ``None`` when
  the data is generated in code (synthetic).

``ProposedForecaster(dataset=...)`` selects the registry entry. With no
``dataset`` the constructor behaves exactly as before (m4 spec + the four
classical models), so wiring the registry in changes nothing — rung 1 is
expected null; the value is the seam rungs 2-5 build on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from td_one.agent.proposed_method import (
    RegimeSpec,
    generic_regime_spec,
    m4_regime_spec,
)

#: Expert set for the continuous benchmarks — same content and order as
#: exp42's ``MODELS`` (order matters for best-single tie-breaking).
CLASSICAL_MODELS: tuple[str, ...] = ("naive", "m0", "ets", "arima")


def boc_regime_spec() -> RegimeSpec:
    """BoC regimes: EASY (all macro signals agree) / HARD (exp32).

    Rule replicated from ``td_one/agent/tools/regimes.py::_label_boc`` so
    the registry does not depend on that module's private helper.
    """

    def label(feats: dict[str, Any]) -> str:
        def implied(value: float, lo: float, hi: float) -> int:
            return -1 if value < lo else (1 if value > hi else 0)

        signals = {
            implied(float(feats.get("rate_momentum", 0.0)), -0.25, 0.25),
            implied(float(feats.get("inflation_gap", 0.0)), -1.0, 1.0),
            implied(float(feats.get("yield_spread", 0.0)), -0.5, 0.5),
        }
        return "EASY" if len(signals) == 1 else "HARD"

    return RegimeSpec(
        name="boc",
        regimes=["EASY", "HARD"],
        label_fn=label,
        description="EASY (signals agree) -> rule; HARD (signals disagree) -> climatology (exp32)",
    )


@dataclass(frozen=True)
class DatasetSpec:
    """One registry entry: timeless metadata + per-dataset modeling choices."""

    description: str
    regime_spec_fn: Callable[[], RegimeSpec]
    models: tuple[str, ...]
    data_path: str | None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "food_cpi": DatasetSpec(
        description="Canada Food CPI (headline), monthly index, source StatCan",
        regime_spec_fn=generic_regime_spec,
        models=CLASSICAL_MODELS,
        data_path="benchmarks/food_cpi/data.csv",
    ),
    "m4": DatasetSpec(
        description="Monthly competition benchmark series, univariate real-world time series",
        regime_spec_fn=m4_regime_spec,
        models=CLASSICAL_MODELS,
        data_path="data/m4",
    ),
    "smm": DatasetSpec(
        description="Freddie Mac single-month mortgage prepayment speeds, monthly series",
        regime_spec_fn=generic_regime_spec,
        models=CLASSICAL_MODELS,
        data_path="data/freddie/smm_monthly_2021_2026.csv",
    ),
    "boc_rates": DatasetSpec(
        description=(
            "Bank of Canada policy-rate decisions: scheduled meeting outcomes "
            "with PIT-safe macro features; discrete benchmark"
        ),
        regime_spec_fn=boc_regime_spec,
        models=("rule", "climatology"),
        data_path="benchmarks/boc_rates/pilot_data.json",
    ),
    "synthetic": DatasetSpec(
        description=(
            "Synthetic monthly series: mixture of trending, seasonal and "
            "random-walk regimes with calibrated event shifts"
        ),
        regime_spec_fn=m4_regime_spec,
        models=CLASSICAL_MODELS,
        data_path=None,  # generated in code (exp42_fforma.load_synthetic)
    ),
}
"""Dataset name -> :class:`DatasetSpec`."""
