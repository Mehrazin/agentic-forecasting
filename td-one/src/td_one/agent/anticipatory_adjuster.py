"""Anticipatory LLM shift adjustment (exp69).

Where exp68 reacted to shifts *after* onset (and failed: context truncation
hurts, +0.5% p=0.824), this module acts *before* onset: an LLM reads PIT-safe
news, anticipates an announced shock, and a calibrated judgmental overlay
shifts the TSFM's quantile forecast for the horizon steps the shock will
hit.

Division of labor (the key design choice):
- The LLM outputs **structure only** — shift type, direction, magnitude
  bucket, timing. Never numbers.
- **Statistics quantify**: a calibration table maps (shift_type,
  magnitude_bucket) -> level shift in rate space, learned from *past* shock
  episodes (disjoint seeds from any test episode). The LLM classifies;
  statistics set the size.

PIT safety: the classifier only ever receives articles with
``article_date <= origin`` (via ``articles_visible_asof``). The calibration
table uses only past episodes. Tests assert both contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from td_one.data_gen.news_synthesis import articles_visible_asof

#: Valid magnitude buckets the LLM may emit.
MAGNITUDE_BUCKETS = ("small", "medium", "large")

#: Valid shift types the LLM may emit.
SHIFT_TYPES = ("rate", "policy", "volatility")

#: Valid directions.
DIRECTIONS = ("up", "down")

#: Valid timing confidences.
TIMING_CONFIDENCES = ("high", "medium", "low")


class LLMClient(Protocol):
    """Structural type for the classifier client (JSON string out)."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class ShiftSignal:
    """Structured anticipatory shift signal.

    No numbers from the LLM except timing; the *size* of the shift comes
    from the calibration table.
    """

    shift_type: str  # "rate" | "policy" | "volatility" | "none"
    direction: str  # "up" | "down" | "none"
    magnitude_bucket: str  # "small" | "medium" | "large" | "none"
    months_from_now: int | None  # predicted onset relative to origin
    timing_confidence: str  # "high" | "medium" | "low"
    rationale: str = ""
    driving_trends: tuple[str, ...] = ()  # trends the LLM cites (exp69b)

    @property
    def is_actionable(self) -> bool:
        """Whether the signal warrants an overlay."""
        return (
            self.shift_type in SHIFT_TYPES
            and self.direction in DIRECTIONS
            and self.magnitude_bucket in MAGNITUDE_BUCKETS
            and self.months_from_now is not None
            and self.months_from_now >= 0
        )


#: The null signal: no overlay.
NULL_SIGNAL = ShiftSignal("none", "none", "none", None, "low", "no signal")


def _validate_signal(d: dict[str, object]) -> ShiftSignal:
    """Validate a parsed LLM JSON object into a ShiftSignal.

    Anything malformed or out of schema -> NULL_SIGNAL (do-no-harm).
    """
    try:
        st = str(d.get("shift_type", "none")).lower()
        dr = str(d.get("direction", "none")).lower()
        mb = str(d.get("magnitude_bucket", "none")).lower()
        tc = str(d.get("timing_confidence", "low")).lower()
        mfn_raw = d.get("months_from_now")
        mfn: int | None = None
        if isinstance(mfn_raw, bool):
            mfn = None
        elif isinstance(mfn_raw, (int, float)):
            mfn = int(mfn_raw)
        elif isinstance(mfn_raw, str) and mfn_raw.strip().lstrip("-").isdigit():
            mfn = int(mfn_raw.strip())
    except (TypeError, ValueError):
        return NULL_SIGNAL
    if st not in (*SHIFT_TYPES, "none"):
        return NULL_SIGNAL
    if dr not in (*DIRECTIONS, "none"):
        return NULL_SIGNAL
    if mb not in (*MAGNITUDE_BUCKETS, "none"):
        return NULL_SIGNAL
    if tc not in TIMING_CONFIDENCES:
        tc = "low"
    if mfn is not None and mfn < 0:
        return NULL_SIGNAL
    # Partial signals are not actionable: if any of type/direction/bucket is
    # "none" while others are set, the LLM was hedging -> treat as none.
    parts = (st, dr, mb)
    if any(p == "none" for p in parts) and not all(p == "none" for p in parts):
        return NULL_SIGNAL
    rationale = str(d.get("rationale", ""))[:300]
    trends_raw = d.get("driving_trends", [])
    trends: tuple[str, ...] = ()
    if isinstance(trends_raw, list):
        trends = tuple(str(t)[:120] for t in trends_raw if str(t).strip())[:5]
    return ShiftSignal(st, dr, mb, mfn, tc, rationale, trends)


def _extract_json(raw: str) -> dict[str, object]:
    """Parse an LLM JSON response, tolerating markdown fences.

    Returns the parsed dict, or raises ValueError if unparseable.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    return obj


class AnticipatoryClassifier:
    """LLM hook: read PIT-safe news, output a structured ShiftSignal.

    The prompt carries only articles with ``article_date <= origin`` and the
    current month label. The LLM must not see future articles (PIT gate) and
    must not emit numbers beyond ``months_from_now``.
    """

    PROMPT_TEMPLATE = (
        "You are an anticipatory shift classifier for a forecasting system. "
        "Target: {target_desc}. Today is month {origin_label}. Below are "
        "news articles published on or before today (PIT-safe). An economic "
        "development in the news — a central bank rate decision, a policy "
        "announcement, a supply disruption — may shift the target in the "
        "coming months.\n\n"
        "Output ONLY a JSON object with keys:\n"
        '  "shift_type": one of "rate", "policy", "volatility", "none"\n'
        '  "direction": one of "up", "down", "none" (direction of the TARGET)\n'
        '  "magnitude_bucket": one of "small", "medium", "large", "none"\n'
        '  "months_from_now": integer >= 0, or null if no shift is coming\n'
        '  "timing_confidence": one of "high", "medium", "low"\n'
        '  "rationale": one sentence citing the article that informed you\n'
        '  "driving_trends": list of short trend labels that drove your\n'
        "    classification (e.g. 'rate hike expected', 'recession risk\n"
        "    rising', 'housing cooling'). Empty list if no shift.\n\n"
        "Rules: if no article announces a coming change, output "
        '"none" for shift_type/direction/magnitude_bucket and null timing. '
        "Do NOT invent shifts from unrelated news. Base magnitude on the "
        "article's language (basis points, adjectives like "
        "modest/aggressive/sweeping). Each article shows its publication "
        "month in brackets. Phrases like 'next month' or 'in 2 months' are "
        "relative to the ARTICLE's publication month, not today: compute "
        "months_from_now = (article month + stated lag) - today. If the "
        "announced change already took effect (months_from_now < 0), output "
        '"none" (the shock is in the data already).\n\n'
        "Articles:\n{articles}"
    )

    def __init__(self, client: LLMClient, target_desc: str = "loan attrition rate"):
        self._client = client
        self._target_desc = target_desc

    def __call__(
        self, articles: list[dict[str, object]], origin: int, origin_label: str
    ) -> ShiftSignal:
        """Classify visible news into a ShiftSignal. PIT-safe by construction."""
        visible = articles_visible_asof(articles, origin_label)
        lines = []
        for a in visible:
            lines.append(f"[{a.get('article_date')}] {a.get('title')}: {a.get('body', '')}")
        prompt = self.PROMPT_TEMPLATE.format(
            target_desc=self._target_desc,
            origin_label=origin_label,
            articles="\n".join(lines) if lines else "(no articles)",
        )
        try:
            raw = self._client.complete(prompt)
            d = _extract_json(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return NULL_SIGNAL
        return _validate_signal(d)


def month_label(m: int) -> str:
    """Zero-padded month label so lexicographic == chronological.

    ``articles_visible_asof`` compares labels as strings; zero-padding keeps
    the PIT filter correct across single/double digit months.
    """
    return f"M{m:03d}"


def calibrate_level_shifts(
    episodes: list[tuple[str, str, np.ndarray, int]],
    absolute: bool = False,
) -> dict[tuple[str, str], float]:
    """Learn magnitude-bucket -> level-shift from *past* shock episodes.

    Args:
        episodes: ``(shift_type, magnitude_bucket, attrition_rate, onset)``
            for calibration episodes only (disjoint seeds from test).
        absolute: If True, store ``|median change|`` (magnitude only).
            The sign then comes entirely from the LLM's ``direction`` at
            test time. This is the correct generalization when calibration
            and test episodes can move in opposite directions (e.g. a
            hiking-cycle calibration applied to a cutting-cycle test on
            real data): the signed version would flip the test shift the
            wrong way. Default False preserves exp69's behavior.

    Returns:
        ``{(shift_type, bucket): median level change in rate space}``,
        where level change = median(post[onset:onset+6]) -
        median(pre[onset-6:onset]) (or its absolute value).

    """
    from collections import defaultdict

    changes: dict[tuple[str, str], list[float]] = defaultdict(list)
    for stype, bucket, rate, onset in episodes:
        r = np.asarray(rate, dtype=float)
        if onset < 6 or onset + 6 > len(r):
            raise ValueError(f"episode too short around onset {onset}")
        pre = float(np.median(r[onset - 6 : onset]))
        post = float(np.median(r[onset : onset + 6]))
        delta = post - pre
        changes[(stype, bucket)].append(abs(delta) if absolute else delta)
    return {k: float(np.median(v)) for k, v in changes.items()}


def apply_overlay(
    levels: np.ndarray,
    values: np.ndarray,
    signal: ShiftSignal,
    origin: int,
    calibration: dict[tuple[str, str], float],
    use_ramp: bool = True,
) -> np.ndarray:
    """Apply the anticipatory level shift to a quantile forecast.

    The shift applies to horizon steps ``k`` whose month ``origin + 1 + k``
    falls at or after the predicted onset ``origin + months_from_now``.
    Steps before the predicted onset are untouched.

    Timing confidence controls the shape:
    - "high": step function at the predicted onset.
    - "medium": linear ramp over [onset-1, onset+1] (partial weight).
    - "low": no overlay (abstain — do-no-harm).

    Args:
        levels: Quantile levels, shape (K,).
        values: Quantile values, shape (K, H).
        signal: The classified shift signal.
        origin: Integer forecast origin (context = rate[:origin+1]).
        calibration: (shift_type, bucket) -> level shift in rate space.
        use_ramp: If False, "medium" confidence also uses a step function
            (ablation: ramp on/off).

    Returns:
        Adjusted quantile values, shape (K, H), clipped to [0, 1] and
        re-sorted along the level axis.

    """
    v = np.asarray(values, dtype=float).copy()
    if not signal.is_actionable:
        return v
    delta = calibration.get((signal.shift_type, signal.magnitude_bucket))
    if delta is None:
        return v
    signed = delta if signal.direction == "up" else -delta
    onset_pred = origin + int(signal.months_from_now)  # type: ignore[arg-type]
    h = v.shape[1]

    if signal.timing_confidence == "low":
        return v
    if origin >= onset_pred:
        # The predicted onset is already in the context: the TSFM has seen
        # the shock and adapts on its own. Shifting again would double-count
        # (this is the reactive case exp68 showed does not help).
        return v
    for k in range(h):
        m = origin + 1 + k
        if use_ramp and signal.timing_confidence == "medium":
            # Linear ramp: 0 at onset-2, 0.5 at onset-1, 1 at onset and after.
            w = min(1.0, max(0.0, (m - onset_pred + 2) / 2.0))
        else:
            w = 1.0 if m >= onset_pred else 0.0
        if w > 0:
            v[:, k] = v[:, k] + w * signed
    v = np.clip(v, 0.0, 1.0)
    v = np.sort(v, axis=0)
    return v
