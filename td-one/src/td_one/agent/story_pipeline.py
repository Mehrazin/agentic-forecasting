"""Multi-call anticipatory story pipeline (exp69c).

exp69 (synthetic): a single LLM call classified a bag of retrieved chunks.
Honest negative — timing was the fragile component, and the oracle ceiling
showed the mechanism itself was the limit on that DGP.

exp69c (real data): the user's hypothesis is that real shocks have richer,
uneven textual buildup that a single call cannot exploit. This module
implements the user-directed scaffolding as four focused LLM calls:

  Call 1 (retrieve): semantic RAG over the real news DB, PIT-gated,
      2-4 query phrasings per episode, dedup, cap N candidates.
  Call 2 (rank):     LLM ranks candidates by relevance to the forecast
      question, keeps top-k with per-article reasons.
  Call 3 (story):    LLM builds a chronological narrative from the ranked
      chunks: ordered dated events + a "what's building" synthesis.
  Call 4 (classify): LLM emits a structured :class:`ShiftSignal` from the
      story, with explicit timeline anchors ("today is X, the forecast
      window is Y..Z, months_from_now is measured from today").

The timeline-ablation variant drops the anchors from Call 4.

Division of labor (unchanged from exp69): the LLM outputs structure only
(shift type, direction, magnitude bucket, timing — never numbers);
a calibration table maps magnitude_bucket -> |level shift|, learned from
*past* episodes only. The LLM classifies; statistics quantify.

Corpus seam (Macro Signal Service): :func:`retrieve_candidates` accepts
``extra_articles`` — a list of ``{pub_date, title, body?, source}`` dicts
merged with the RAG hits (dedup by title, PIT-gated like everything
else). The Macro Signal Service's per-month category signals can be
injected here later without changing the pipeline; the story builder
treats them as dated evidence like any other article. (Not wired yet —
the service is being built in parallel; this module does not block on it.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from td_one.agent.anticipatory_adjuster import (
    NULL_SIGNAL,
    ShiftSignal,
    _extract_json,
    _validate_signal,
)

logger = logging.getLogger(__name__)

#: Queries are episode-specific; this is the default cap per query.
RETRIEVE_PER_QUERY_K = 50
#: Cap on merged candidates handed to the ranker.
RETRIEVE_CAP = 60
#: Candidates the ranker keeps for the story.
RANK_KEEP_K = 12


@dataclass(frozen=True)
class TimelineEvent:
    """One dated event from the story (all dates < origin)."""

    date: str  # YYYY-MM-DD
    event: str


@dataclass(frozen=True)
class NewsStory:
    """Output of Call 3: chronological narrative."""

    timeline: tuple[TimelineEvent, ...]
    narrative: str
    building: str  # "what's building" synthesis, 2-4 sentences


@dataclass
class PipelineDebug:
    """Per-origin debug trace (cached alongside the signal)."""

    n_retrieved: int = 0
    n_ranked: int = 0
    ranked_reasons: list[dict] = field(default_factory=list)
    story: NewsStory | None = None
    raw_signal: dict = field(default_factory=dict)


def render_timeline_anchors(
    origin_ymd: str,
    horizon: int,
    window_start_ym: str,
    window_end_ym: str,
    events: list[TimelineEvent],
) -> str:
    """Render the explicit timing-reminder block for the classifier.

    The user-directed fix for exp69/exp69b's timing fragility: the LLM gets
    a pointer of references — today, the forecast window, and the key
    dated events — instead of relying on its own date sense.
    """
    lines = [
        "TIMELINE ANCHORS — use these, not your own sense of dates:",
        f"- TODAY: {origin_ymd}. The forecast is made at the end of this "
        "month; every event above was published on or before today.",
        f"- FORECAST WINDOW: the next {horizon} months, i.e. "
        f"{window_start_ym} .. {window_end_ym}. Only shifts that take "
        "effect INSIDE this window matter.",
        "- months_from_now is measured from TODAY: a shift taking effect "
        "in the 3rd forecast month is months_from_now = 3.",
        "- A shift that already took effect (before the window) gets "
        'shift_type "none" — it is in the data already.',
    ]
    if events:
        lines.append("- KEY DATED EVENTS (from the story, all before today):")
        lines.extend(f"  * {e.date}: {e.event}" for e in events[:10])
    else:
        lines.append("- KEY DATED EVENTS: none — the story found nothing dated.")
    return "\n".join(lines)


def retrieve_candidates(
    queries: list[str],
    before_ymd: str,
    index: object,
    extra_articles: list[dict] | None = None,
    per_query_k: int = RETRIEVE_PER_QUERY_K,
    cap: int = RETRIEVE_CAP,
) -> list[dict]:
    """Call 1: semantic retrieval, PIT-gated, with the corpus seam.

    Args:
        queries: Episode-specific query phrasings.
        before_ymd: PIT cutoff (exclusive) as ``YYYY-MM-DD``.
        index: Object with ``search(query, before, top_k)`` returning
            items with ``.pub_date`` / ``.title`` / ``.url`` / ``.source``.
        extra_articles: Optional ``{pub_date, title, body?, source}``
            dicts (Macro Signal Service seam) merged with RAG hits.
        per_query_k: Candidates per query phrasing.
        cap: Max merged candidates (best cosine score wins).

    Returns:
        ``[{pub_date, title, url, source, score}]`` with
        ``pub_date < before_ymd`` for every row (asserted — PIT gate).

    """
    seen: dict[str, dict] = {}
    for q in queries:
        for c in index.search(q, before_ymd, top_k=per_query_k):  # type: ignore[union-attr]
            key = (c.title or "").strip().lower()
            if not key or key in seen:
                continue
            seen[key] = {
                "pub_date": c.pub_date,
                "title": c.title,
                "url": getattr(c, "url", ""),
                "source": getattr(c, "source", ""),
                "score": float(getattr(c, "score", 0.0)),
            }
    for a in extra_articles or []:
        key = str(a.get("title", "")).strip().lower()
        if not key or key in seen:
            continue
        seen[key] = {
            "pub_date": str(a.get("pub_date", "")),
            "title": str(a.get("title", "")),
            "url": str(a.get("url", "")),
            "source": str(a.get("source", "")),
            "score": 1.0,  # injected evidence outranks retrieved
            "body": str(a.get("body", "")),
        }
    cands = sorted(seen.values(), key=lambda d: d["score"], reverse=True)[:cap]
    # PIT gate: hard fail, never silently leak.
    leaked = [c for c in cands if not (c["pub_date"] < before_ymd)]
    if leaked:
        raise AssertionError(f"PIT violation: {len(leaked)} articles with pub_date >= {before_ymd}")
    return cands


_RANK_PROMPT = """\
You are ranking news articles by relevance to a forecasting question.
Forecast question: {question}

Rate EACH article's relevance to anticipating a shift in the target over
the next {horizon} months. Consider: does it describe a coming rate
decision, policy change, supply disruption, or demand shock that would
move the target? Pure background noise scores near 0.

Articles (index: [date] title):
{articles}

Reply with JSON only:
{{"ranked": [{{"idx": <1-based index>, "score": <0-1>, "reason": "<one line>"}}]}}
Score every article; the caller keeps the top {keep_k}.
"""


def rank_articles(
    client: object,
    candidates: list[dict],
    question: str,
    horizon: int,
    keep_k: int = RANK_KEEP_K,
) -> list[dict]:
    """Call 2: LLM ranker — relevance scores + reasons, keep top-k."""
    if not candidates:
        return []
    lines = "\n".join(f"{i + 1}. [{c['pub_date']}] {c['title']}" for i, c in enumerate(candidates))
    prompt = _RANK_PROMPT.format(
        question=question,
        horizon=horizon,
        articles=lines,
        keep_k=keep_k,
    )
    try:
        raw = client.complete(prompt)  # type: ignore[union-attr]
        obj = _extract_json(raw)
        ranked = obj.get("ranked", [])
        scored: list[dict] = []
        for r in ranked:
            if not isinstance(r, dict):
                continue
            i = r.get("idx")
            if isinstance(i, int) and 1 <= i <= len(candidates):
                c = dict(candidates[i - 1])
                c["rank_score"] = float(r.get("score", 0.0) or 0.0)
                c["rank_reason"] = str(r.get("reason", ""))[:200]
                scored.append(c)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning("ranker failed (%s); keeping retrieval order", e)
        scored = [
            {**c, "rank_score": c.get("score", 0.0), "rank_reason": "fallback"} for c in candidates
        ]
    scored.sort(key=lambda d: d["rank_score"], reverse=True)
    return scored[:keep_k]


_STORY_PROMPT = """\
You are a news analyst building a chronological story for a forecaster.
Today is {today}. Below are {n} news articles published on or before
today, ranked by relevance to this question: {question}

Build the story of what happened, in what order, and what is building:
1. A timeline of the dated events that matter (date + one-line event).
   Drop noise; keep only events relevant to the question.
2. A 3-5 sentence narrative connecting them chronologically.
3. "What's building": 2-4 sentences on what these events jointly imply
   for the coming months — the setup, not a forecast number.

Articles (most relevant first):
{articles}

Reply with JSON only:
{{"timeline": [{{"date": "YYYY-MM-DD", "event": "..."}}],
  "narrative": "...",
  "building": "..."}}
"""


def build_story(
    client: object,
    ranked: list[dict],
    today_ymd: str,
    question: str,
) -> NewsStory:
    """Call 3: chronological narrative from the ranked chunks."""
    if not ranked:
        return NewsStory((), "", "")
    lines = "\n".join(
        f"- [{c['pub_date']}] {c['title']} (why relevant: {c.get('rank_reason', '')})"
        for c in ranked
    )
    prompt = _STORY_PROMPT.format(
        today=today_ymd,
        n=len(ranked),
        question=question,
        articles=lines,
    )
    try:
        raw = client.complete(prompt)  # type: ignore[union-attr]
        obj = _extract_json(raw)
        tl = obj.get("timeline", [])
        events = []
        for t in tl:
            if isinstance(t, dict) and t.get("date") and t.get("event"):
                d = str(t["date"])[:10]
                # Story events must not postdate today (PIT).
                if d <= today_ymd:
                    events.append(TimelineEvent(date=d, event=str(t["event"])[:200]))
        events.sort(key=lambda e: e.date)
        return NewsStory(
            timeline=tuple(events),
            narrative=str(obj.get("narrative", ""))[:1500],
            building=str(obj.get("building", ""))[:1500],
        )
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning("story builder failed (%s); empty story", e)
        return NewsStory((), "", "")


_CLASSIFY_PROMPT = """\
You are an anticipatory shift classifier for a forecasting system.
Target: {target_desc}

You are given a chronological NEWS STORY assembled from articles
published on or before today. A real economic development — rate
decision, policy change, supply disruption, demand shock — may shift
the target in the coming months.

{anchors}

Output ONLY a JSON object with keys:
  "shift_type": one of "rate", "policy", "volatility", "none"
  "direction": one of "up", "down", "none" (direction of the TARGET)
  "magnitude_bucket": one of "small", "medium", "large", "none"
  "months_from_now": integer >= 0, or null if no shift is coming
  "timing_confidence": one of "high", "medium", "low"
  "rationale": one sentence citing the story's dated events
  "driving_trends": list of short trend labels that drove your
    classification (e.g. 'rate hike expected', 'supply crunch building').
    Empty list if no shift.

Rules: if the story contains no coming change, output "none" for
shift_type/direction/magnitude_bucket and null timing. Do NOT invent
shifts from unrelated news. Base magnitude on the story's language
(basis points, adjectives like modest/aggressive/sweeping). Dates in
the story are publication dates; "next month" in an event is relative
to that event's date.

NEWS STORY:
Timeline:
{timeline}
Narrative: {narrative}
What's building: {building}
"""


def classify_from_story(
    client: object,
    story: NewsStory,
    target_desc: str,
    origin_ymd: str,
    horizon: int,
    window_start_ym: str,
    window_end_ym: str,
    use_timeline: bool = True,
) -> ShiftSignal:
    """Call 4: structured ShiftSignal from the story (+/- timeline anchors)."""
    tl_lines = "\n".join(f"- {e.date}: {e.event}" for e in story.timeline)
    if use_timeline:
        anchors = render_timeline_anchors(
            origin_ymd,
            horizon,
            window_start_ym,
            window_end_ym,
            list(story.timeline),
        )
    else:
        anchors = (
            "Classify from the story alone. Today is "
            f"{origin_ymd}; the forecast covers the next {horizon} months."
        )
    prompt = _CLASSIFY_PROMPT.format(
        target_desc=target_desc,
        anchors=anchors,
        timeline=tl_lines or "(no dated events)",
        narrative=story.narrative or "(none)",
        building=story.building or "(none)",
    )
    try:
        raw = client.complete(prompt)  # type: ignore[union-attr]
        return _validate_signal(_extract_json(raw))
    except (ValueError, TypeError, AttributeError):
        return NULL_SIGNAL


class StoryPipeline:
    """Four-call anticipatory pipeline with a disk cache.

    ``run(...)`` returns ``(ShiftSignal, PipelineDebug)``. The cache key
    covers (queries, before_ymd, arm, prompt inputs) so reruns are free.
    """

    def __init__(
        self,
        client: object,
        index: object,
        queries: list[str],
        question: str,
        target_desc: str,
        horizon: int = 6,
    ) -> None:
        self._client = client
        self._index = index
        self._queries = queries
        self._question = question
        self._target_desc = target_desc
        self._horizon = horizon
        self._mem: dict[str, tuple[ShiftSignal, PipelineDebug]] = {}
        self.llm_calls = 0

    def _complete(self, prompt: str) -> str:
        self.llm_calls += 1
        return self._client.complete(prompt)  # type: ignore[union-attr,no-any-return]

    def run(
        self,
        before_ymd: str,
        origin_ymd: str,
        window_start_ym: str,
        window_end_ym: str,
        use_timeline: bool = True,
        extra_articles: list[dict] | None = None,
    ) -> tuple[ShiftSignal, PipelineDebug]:
        """Run retrieve -> rank -> story -> classify (cached)."""
        key = "|".join(
            [
                before_ymd,
                origin_ymd,
                str(use_timeline),
                *self._queries,
                str(len(extra_articles or [])),
            ]
        )
        if key in self._mem:
            return self._mem[key]
        dbg = PipelineDebug()

        cands = retrieve_candidates(self._queries, before_ymd, self._index, extra_articles)
        dbg.n_retrieved = len(cands)

        # Inline the client call counting: rank/story/classify each count.
        ranked = self._ranked(cands)
        dbg.n_ranked = len(ranked)
        dbg.ranked_reasons = [
            {
                "date": c["pub_date"],
                "title": c["title"],
                "score": c["rank_score"],
                "reason": c["rank_reason"],
            }
            for c in ranked
        ]
        story = self._storied(ranked, origin_ymd)
        dbg.story = story
        signal = classify_from_story(
            _CountingClient(self),
            story,
            self._target_desc,
            origin_ymd,
            self._horizon,
            window_start_ym,
            window_end_ym,
            use_timeline=use_timeline,
        )
        dbg.raw_signal = {
            "shift_type": signal.shift_type,
            "direction": signal.direction,
            "magnitude_bucket": signal.magnitude_bucket,
            "months_from_now": signal.months_from_now,
            "timing_confidence": signal.timing_confidence,
            "driving_trends": list(signal.driving_trends),
        }
        self._mem[key] = (signal, dbg)
        return signal, dbg

    def _ranked(self, cands: list[dict]) -> list[dict]:
        client = _CountingClient(self)
        return rank_articles(client, cands, self._question, self._horizon, RANK_KEEP_K)

    def _storied(self, ranked: list[dict], origin_ymd: str) -> NewsStory:
        client = _CountingClient(self)
        return build_story(client, ranked, origin_ymd, self._question)


class _CountingClient:
    """Adapter: route module-level calls through the pipeline's counter."""

    def __init__(self, pipe: StoryPipeline) -> None:
        self._pipe = pipe

    def complete(self, prompt: str) -> str:
        return self._pipe._complete(prompt)
