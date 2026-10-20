"""Agentic shift guard (exp68).

Detect distributional shifts from covariates *before* the target series
moves, then respond.

Pipeline:
1. **Detect** — two-sided CUSUM on a covariate history (e.g. policy rates)
   plus PIT-filtered announcements via ``events_visible_asof``. The detector
   only ever sees data with index <= asof.
2. **Classify** — shift-kind → response-template mapping. An LLM hook
   (structure only, never numbers) may override the heuristic classifier.
3. **Respond** — context surgery (truncate the TSFM input window to
   post-detection data, or switch to a classical expert on the new regime)
   and shift-aware quantile widening. The widening factor is conformal-style:
   calibrated on *past* (dev) shift episodes' residuals, never on eval data.

PIT safety: every public method takes ``asof`` (an integer time index) and
indexes all inputs with ``<= asof``. Events pass through
``events_visible_asof``. Tests assert this contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np


class LLMClient(Protocol):
    """Structural type for the LLM hook client (structure-only output)."""

    def complete(self, prompt: str) -> str: ...


class ShiftKind(StrEnum):
    NONE = "none"
    LEVEL = "level"  # abrupt mean shift (S1/S2-like)
    VOLATILITY = "volatility"  # spread change (S6-like)
    SEASONAL = "seasonal"  # seasonal pattern break (S4-like)
    COMPOSITION = "composition"  # slow drift (S7-like)
    UNKNOWN = "unknown"


class ResponseTemplate(StrEnum):
    TRUNCATE = "truncate"  # TSFM on post-detection context only
    SWITCH = "switch"  # classical expert on post-detection context
    WIDEN = "widen"  # same model, widened intervals
    NONE = "none"  # no action


@dataclass(frozen=True)
class GuardConfig:
    """Tunable guard parameters (no LLM involved)."""

    cusum_threshold_h: float = 5.0  # CUSUM decision threshold (σ units)
    cusum_drift_k: float = 0.75  # reference drift (σ units)
    baseline_window: int = 24  # months used to estimate covariate baseline
    min_post_periods: int = 3  # min post-detection points to allow truncation
    widen_alpha: float = 0.1  # miscoverage for the conformal widening factor


@dataclass(frozen=True)
class Detection:
    detected: bool
    onset_idx: int | None  # estimated first post-shift index (None if none)
    detection_idx: int | None  # index at which CUSUM fired (None if none)
    statistic: float  # max CUSUM statistic (σ units)
    kind: ShiftKind = ShiftKind.NONE


@dataclass(frozen=True)
class Classification:
    kind: ShiftKind
    template: ResponseTemplate
    confidence: str  # "high" | "medium" | "low" — structure only
    rationale: str


@dataclass
class GuardedForecast:
    levels: np.ndarray
    values: np.ndarray  # post-widening quantiles, shape (K, H)
    template: ResponseTemplate
    detection: Detection
    classification: Classification
    context_start: int  # first index of the context actually used
    widen_factor: float  # 1.0 when no widening applied


def cusum_detect(
    x: np.ndarray,
    asof: int,
    config: GuardConfig | None = None,
) -> Detection:
    """Two-sided CUSUM for a mean shift in ``x[:asof+1]``.

    Baseline (mean/sd) is estimated on the first ``baseline_window`` points.
    Returns the first index where the CUSUM exceeds ``h`` and the estimated
    onset (last index where the CUSUM was at zero before firing).

    PIT-safe by construction: never indexes beyond ``asof``.
    """
    cfg = config or GuardConfig()
    xs = np.asarray(x, dtype=float)[: asof + 1]
    n = len(xs)
    w = min(cfg.baseline_window, n)
    if w < 3 or n <= w:
        return Detection(False, None, None, 0.0)
    mu = float(np.mean(xs[:w]))
    sd = float(np.std(xs[:w]))
    if sd < 1e-9:
        return Detection(False, None, None, 0.0)
    z = (xs[w:] - mu) / sd
    # Two-sided CUSUM.
    gp = np.zeros(len(z))
    gm = np.zeros(len(z))
    for i in range(1, len(z)):
        gp[i] = max(0.0, gp[i - 1] + z[i] - cfg.cusum_drift_k)
        gm[i] = max(0.0, gm[i - 1] - z[i] - cfg.cusum_drift_k)
    stat = float(max(np.max(gp), np.max(gm)))
    fire = np.where((gp > cfg.cusum_threshold_h) | (gm > cfg.cusum_threshold_h))[0]
    if len(fire) == 0:
        return Detection(False, None, None, stat)
    i_fire = int(fire[0])
    side = gp if gp[i_fire] > gm[i_fire] else gm
    # Onset: last zero of the firing side before the fire index.
    zeros = np.where(side[: i_fire + 1] == 0.0)[0]
    onset_rel = int(zeros[-1]) + 1 if len(zeros) else 0
    onset_idx = w + onset_rel
    return Detection(True, onset_idx, w + i_fire, stat, ShiftKind.LEVEL)


def visible_events(events: list[dict[str, Any]], asof: int) -> list[dict[str, Any]]:
    """PIT filter for guard event dicts (integer month index ``t``).

    Returns ``[e for e in events if e["t"] <= asof]`` — the integer-index
    analogue of ``events_visible_asof``. The guard NEVER sees ``t > asof``.
    """
    return [e for e in events if int(e["t"]) <= asof]


def heuristic_classifier(
    detection: Detection,
    visible_events: list[dict[str, Any]],
    covariate_shift_mag: float,
) -> Classification:
    """Deterministic shift-kind → template mapping (no LLM).

    - Covariate CUSUM fired + rate/policy announcement visible → LEVEL →
      TRUNCATE (fresh regime, drop pre-shift history).
    - CUSUM fired, no announcement → UNKNOWN → WIDEN (don't trust the
      context, but don't discard it either).
    - No detection → NONE → NONE.
    """
    if not detection.detected:
        return Classification(
            ShiftKind.NONE, ResponseTemplate.NONE, "high", "no covariate shift detected"
        )
    announced = any(
        str(e.get("type", "")).upper() in ("RATE_SHIFT", "POLICY_CHANGE", "POLICY")
        or "rate" in str(e.get("description", "")).lower()
        for e in visible_events
    )
    if announced or abs(covariate_shift_mag) > 0.0:
        # A named, covariate-confirmed level shift: the old history is poison.
        return Classification(
            ShiftKind.LEVEL,
            ResponseTemplate.TRUNCATE,
            "high" if announced else "medium",
            "covariate CUSUM fired with supporting announcement"
            if announced
            else "covariate CUSUM fired without announcement",
        )
    return Classification(
        ShiftKind.UNKNOWN,
        ResponseTemplate.WIDEN,
        "low",
        "CUSUM fired but shift magnitude unclear",
    )


class LLMShiftClassifier:
    """LLM hook: classify shift kind + response template from metadata.

    The LLM outputs **structure only** (kind, template, confidence,
    rationale) — never numbers. ``client`` is any object with
    ``complete(prompt: str) -> str`` returning a JSON object string.
    """

    KINDS = {k.value for k in ShiftKind}
    TEMPLATES = {t.value for t in ResponseTemplate}

    def __init__(self, client: LLMClient):
        self._client = client

    def __call__(self, metadata: dict[str, object]) -> Classification:
        prompt = (
            "You are a shift-type classifier for a loan-attrition forecasting "
            "guard. Given the metadata below, output ONLY a JSON object with "
            "keys: shift_kind (one of: none, level, volatility, seasonal, "
            "composition, unknown), template (one of: truncate, switch, "
            "widen, none), confidence (high/medium/low), rationale (one "
            "sentence). No numbers, no other keys.\n\n"
            f"Metadata: {json.dumps(metadata, default=str)}"
        )
        raw = self._client.complete(prompt)
        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return Classification(
                ShiftKind.UNKNOWN,
                ResponseTemplate.WIDEN,
                "low",
                "LLM output unparseable; falling back to widen",
            )
        kind = str(d.get("shift_kind", "unknown")).lower()
        template = str(d.get("template", "widen")).lower()
        conf = str(d.get("confidence", "low")).lower()
        if kind not in self.KINDS:
            kind = "unknown"
        if template not in self.TEMPLATES:
            template = "widen"
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return Classification(
            ShiftKind(kind),
            ResponseTemplate(template),
            conf,
            str(d.get("rationale", ""))[:200],
        )


def calibrate_widening(
    residuals: np.ndarray,
    halfwidths: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Conformal-style widening factor from *past* shift episodes.

    ``residuals``: |actual − median| per (episode, horizon step) on dev shifts.
    ``halfwidths``: the model's (q90−q10)/2 at the same points.
    Returns q_hat = quantile_{1−alpha}(residual / halfwidth), floored at 1.0.
    Applying ``median ± q_hat * halfwidth`` restores ~(1−alpha) coverage on
    data drawn from the same shift distribution.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    h = np.asarray(halfwidths, dtype=float).ravel()
    mask = np.isfinite(r) & np.isfinite(h) & (h > 1e-12)
    if not np.any(mask):
        return 1.0
    scores = r[mask] / h[mask]
    q_hat = float(np.quantile(scores, 1.0 - alpha))
    return max(1.0, q_hat)


def widen_quantiles(
    levels: np.ndarray,
    values: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Expand quantiles around the median: median + factor * (q − median)."""
    lv = np.asarray(levels, dtype=float)
    v = np.asarray(values, dtype=float)
    i50 = int(np.argmin(np.abs(lv - 0.5)))
    med: np.ndarray = v[i50]
    out: np.ndarray = med + factor * (v - med)
    return out


class ShiftGuard:
    """The agentic guard: detect → classify → respond.

    Parameters
    ----------
        config: GuardConfig.
        classifier: callable(detection, visible_events, covariate_shift_mag)
            -> Classification. Defaults to :func:`heuristic_classifier`;
            pass ``LLMShiftClassifier(client)`` adapted to that signature via
            :meth:`with_llm_classifier`.
        widen_factor: conformal factor from dev shift episodes (1.0 = off).

    """

    def __init__(
        self,
        config: GuardConfig | None = None,
        classifier: (
            Callable[[Detection, list[dict[str, Any]], float], Classification] | None
        ) = None,
        widen_factor: float = 1.0,
    ):
        self.config = config or GuardConfig()
        self.classifier = classifier or heuristic_classifier
        self.widen_factor = widen_factor

    def detect(
        self,
        covariate: np.ndarray,
        events: list[dict[str, Any]],
        asof: int,
    ) -> tuple[Detection, list[dict[str, Any]]]:
        """Run CUSUM on the covariate and PIT-filter events. Pure/PIT-safe."""
        det = cusum_detect(covariate, asof, self.config)
        vis = visible_events(events, asof)
        return det, vis

    def guard(
        self,
        context: np.ndarray,
        covariate: np.ndarray,
        events: list[dict[str, Any]],
        asof: int,
        tsfm_forecast: Callable[[np.ndarray, int, int], tuple[np.ndarray, np.ndarray]],
        classical_forecast: Callable[[np.ndarray, int, int], tuple[np.ndarray, np.ndarray]],
        horizon: int,
        seed: int,
    ) -> GuardedForecast:
        """Full guard pipeline at one origin.

        Args:
            context: target history with ``context[asof]`` the last visible
                point (indexing is by position; ``asof`` bounds everything).
            covariate: covariate history, same indexing.
            events: full event timeline (PIT-filtered internally to
                ``t <= asof``).
            asof: integer time index of the forecast origin.
            tsfm_forecast: callable(ctx, h, seed) -> (levels, values).
            classical_forecast: callable(ctx, h, seed) -> (levels, values).
            horizon: forecast horizon.
            seed: RNG seed for the forecasters.

        Returns:
            GuardedForecast with the (possibly surgically altered) forecast.

        """
        ctx = np.asarray(context, dtype=float)[: asof + 1]
        det, vis = self.detect(covariate, events, asof)

        cov = np.asarray(covariate, dtype=float)[: asof + 1]
        w = min(self.config.baseline_window, len(cov))
        shift_mag = float(np.mean(cov[-3:]) - np.mean(cov[:w])) if len(cov) > w + 3 else 0.0
        cls = self.classifier(det, vis, shift_mag)

        template = cls.template
        context_start = 0
        forecast_fn = tsfm_forecast
        if template == ResponseTemplate.TRUNCATE and det.onset_idx is not None:
            if asof + 1 - det.onset_idx >= self.config.min_post_periods:
                context_start = det.onset_idx
            else:
                template = ResponseTemplate.WIDEN  # too little post data
        if template == ResponseTemplate.SWITCH:
            forecast_fn = classical_forecast
            if det.onset_idx is not None and (
                asof + 1 - det.onset_idx >= self.config.min_post_periods
            ):
                context_start = det.onset_idx

        eff_ctx = ctx[context_start:]
        levels, values = forecast_fn(eff_ctx, horizon, seed)
        levels = np.asarray(levels, dtype=float)
        values = np.asarray(values, dtype=float)

        widen = 1.0
        if template in (ResponseTemplate.TRUNCATE, ResponseTemplate.WIDEN):
            widen = self.widen_factor
            values = widen_quantiles(levels, values, widen)
        # Re-sort to keep quantiles monotone after widening.
        values = np.sort(values, axis=0)
        return GuardedForecast(
            levels=levels,
            values=values,
            template=template,
            detection=det,
            classification=cls,
            context_start=context_start,
            widen_factor=widen,
        )


# Backwards-compatible alias for the LLM hook with the classifier signature.
def llm_classifier_adapter(
    llm: LLMShiftClassifier,
) -> Callable[[Detection, list[dict[str, Any]], float], Classification]:
    """Adapt LLMShiftClassifier to (detection, events, shift_mag) -> Classification."""

    def _adapt(
        detection: Detection, visible_events: list[dict[str, Any]], shift_mag: float
    ) -> Classification:
        metadata: dict[str, object] = {
            "detection": {
                "detected": detection.detected,
                "onset_idx": detection.onset_idx,
                "detection_idx": detection.detection_idx,
                "statistic": round(detection.statistic, 3),
            },
            "n_visible_events": len(visible_events),
            "event_types": sorted({str(e.get("type", "?")) for e in visible_events}),
            "covariate_shift_direction": (
                "up" if shift_mag > 0 else "down" if shift_mag < 0 else "flat"
            ),
        }
        return llm(metadata)

    return _adapt


#: Default metadata fields the LLM hook receives (documented for audit).
LLM_METADATA_FIELDS = [
    "detection.detected",
    "detection.onset_idx",
    "detection.detection_idx",
    "detection.statistic",
    "n_visible_events",
    "event_types",
    "covariate_shift_direction",
]
