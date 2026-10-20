"""Tests for exp69c's multi-call story pipeline + absolute calibration."""

from __future__ import annotations

import numpy as np
import pytest

from td_one.agent.anticipatory_adjuster import (
    NULL_SIGNAL,
    calibrate_level_shifts,
)
from td_one.agent.story_pipeline import (
    NewsStory,
    StoryPipeline,
    TimelineEvent,
    build_story,
    classify_from_story,
    rank_articles,
    render_timeline_anchors,
    retrieve_candidates,
)


class FakeArticle:
    def __init__(self, pub_date: str, title: str, score: float = 0.5):
        self.pub_date = pub_date
        self.title = title
        self.url = f"https://example.com/{title[:10]}"
        self.source = "fake"
        self.score = score


class FakeIndex:
    """In-memory SemanticNewsIndex stand-in (PIT filter included)."""

    def __init__(self, articles: list[FakeArticle]):
        self._articles = articles

    def search(self, query: str, before: str, top_k: int = 50):
        hits = [a for a in self._articles if a.pub_date < before]
        return sorted(hits, key=lambda a: a.score, reverse=True)[:top_k]


class ScriptedClient:
    """LLM stand-in: dispatches on prompt content."""

    def __init__(self, signal_json: str):
        self._signal_json = signal_json
        self.n_calls = 0

    def complete(self, prompt: str) -> str:
        self.n_calls += 1
        if "ranking news articles" in prompt:
            return (
                '{"ranked": [{"idx": 1, "score": 0.9, "reason": "hike signal"},'
                '{"idx": 2, "score": 0.2, "reason": "noise"}]}'
            )
        if "chronological story" in prompt:
            return (
                '{"timeline": [{"date": "2022-01-26", "event": "BoC signals hikes"}],'
                '"narrative": "Inflation hit a 30-year high. The BoC signaled hikes.",'
                '"building": "A first hike in March is widely expected."}'
            )
        if "anticipatory shift classifier" in prompt:
            return self._signal_json
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def _articles() -> list[FakeArticle]:
    return [
        FakeArticle("2022-01-19", "Inflation hits highest in 30 years", 0.9),
        FakeArticle("2022-01-26", "BoC signals hikes to come", 0.85),
        FakeArticle("2022-02-10", "Random lifestyle piece", 0.3),
        FakeArticle("2022-03-15", "FUTURE ARTICLE after origin", 0.99),
    ]


def test_retrieve_candidates_pit_gate() -> None:
    idx = FakeIndex(_articles())
    cands = retrieve_candidates(["q"], "2022-03-01", idx)
    assert cands
    assert all(c["pub_date"] < "2022-03-01" for c in cands)
    assert not any("FUTURE" in c["title"] for c in cands)


def test_retrieve_candidates_corpus_seam() -> None:
    """Macro Signal Service seam: extra articles merge, PIT-gated too."""
    idx = FakeIndex(_articles())
    extra = [
        {"pub_date": "2022-02-01", "title": "Macro signal: rates up", "source": "mss"},
        {"pub_date": "2022-04-01", "title": "Macro signal: FUTURE leak", "source": "mss"},
    ]
    with pytest.raises(AssertionError, match="PIT violation"):
        retrieve_candidates(["q"], "2022-03-01", idx, extra_articles=extra)
    ok = retrieve_candidates(["q"], "2022-03-01", idx, extra_articles=extra[:1])
    assert any(c["title"] == "Macro signal: rates up" for c in ok)


def test_rank_articles_parses_and_keeps_topk() -> None:
    client = ScriptedClient("{}")
    cands = [
        {"pub_date": "2022-01-26", "title": "A", "score": 0.9},
        {"pub_date": "2022-01-19", "title": "B", "score": 0.8},
    ]
    ranked = rank_articles(client, cands, "will rates shift?", 6, keep_k=1)
    assert len(ranked) == 1
    assert ranked[0]["rank_score"] == 0.9
    assert ranked[0]["rank_reason"] == "hike signal"


def test_build_story_filters_future_events() -> None:
    client = ScriptedClient("{}")
    ranked = [
        {"pub_date": "2022-01-26", "title": "A", "rank_reason": "r"},
        {"pub_date": "2022-01-19", "title": "B", "rank_reason": "r"},
    ]
    story = build_story(client, ranked, "2022-02-28", "q?")
    assert story.timeline
    assert all(e.date <= "2022-02-28" for e in story.timeline)
    assert "March" in story.building


def test_timeline_anchors_content() -> None:
    events = [TimelineEvent("2022-01-26", "BoC signals hikes")]
    out = render_timeline_anchors("2022-02-28", 6, "2022-03", "2022-08", events)
    assert "TODAY: 2022-02-28" in out
    assert "2022-03 .. 2022-08" in out
    assert "months_from_now is measured from TODAY" in out
    assert "2022-01-26: BoC signals hikes" in out


def test_classify_from_story_with_and_without_timeline() -> None:
    sig_json = (
        '{"shift_type": "rate", "direction": "up", '
        '"magnitude_bucket": "large", "months_from_now": 1, '
        '"timing_confidence": "high", "rationale": "Jan 26 signal", '
        '"driving_trends": ["rate hike expected"]}'
    )
    client = ScriptedClient(sig_json)
    story = NewsStory(
        (TimelineEvent("2022-01-26", "BoC signals hikes"),),
        "Inflation high.",
        "Hike coming.",
    )
    sig = classify_from_story(
        client,
        story,
        "Food CPI YoY",
        "2022-02-28",
        6,
        "2022-03",
        "2022-08",
        use_timeline=True,
    )
    assert sig.is_actionable
    assert sig.direction == "up"
    assert sig.driving_trends == ("rate hike expected",)
    # Ablation path also parses.
    sig2 = classify_from_story(
        client,
        story,
        "Food CPI YoY",
        "2022-02-28",
        6,
        "2022-03",
        "2022-08",
        use_timeline=False,
    )
    assert sig2.is_actionable


def test_classify_malformed_json_is_null() -> None:
    class BadClient:
        def complete(self, prompt: str) -> str:
            return "not json at all"

    story = NewsStory((), "", "")
    sig = classify_from_story(BadClient(), story, "X", "2022-02-28", 6, "2022-03", "2022-08")
    assert sig == NULL_SIGNAL


def test_story_pipeline_end_to_end() -> None:
    sig_json = (
        '{"shift_type": "rate", "direction": "up", '
        '"magnitude_bucket": "large", "months_from_now": 1, '
        '"timing_confidence": "high", "rationale": "r", '
        '"driving_trends": ["rate hike expected"]}'
    )
    client = ScriptedClient(sig_json)
    pipe = StoryPipeline(
        client=client,
        index=FakeIndex(_articles()),
        queries=["Bank of Canada rate hike"],
        question="Will food inflation shift in the next 6 months?",
        target_desc="Canada Food CPI year-over-year inflation rate",
        horizon=6,
    )
    sig, dbg = pipe.run("2022-03-01", "2022-02-28", "2022-03", "2022-08")
    assert sig.is_actionable
    assert dbg.n_retrieved == 3  # future article excluded by PIT
    assert dbg.story is not None and dbg.story.timeline
    assert client.n_calls == 3  # rank + story + classify
    # Cache: second identical run makes no new calls.
    sig2, _ = pipe.run("2022-03-01", "2022-02-28", "2022-03", "2022-08")
    assert sig2 == sig
    assert client.n_calls == 3


def test_absolute_calibration_flips_sign_correctly() -> None:
    """Regression: signed calibration flips opposite-direction tests.

    A signed calibration would flip an opposite-direction test shift the
    wrong way (hiking cal -> cutting test).
    """
    rate = np.array([0.005] * 24)
    rate[12:18] = 0.003  # episode: level FALLS at onset 12
    episodes = [("rate", "large", rate, 12)]
    signed = calibrate_level_shifts(episodes)
    assert signed[("rate", "large")] < 0
    absolute = calibrate_level_shifts(episodes, absolute=True)
    assert absolute[("rate", "large")] > 0
    assert abs(absolute[("rate", "large")] - 0.002) < 1e-9
