"""Synthetic PIT-safe news generator for the event-augmented simulator.

Generates template-based news articles for exogenous events. The news feed is
the agent's window into the event timeline — but it's noisy and delayed, just
like real news:

- Coverage is uneven: some events get 20 articles, some get 0.
- Magnitudes are misreported (±30% noise).
- Publication is delayed: housing ~1 month, rate transmission 2-4 weeks,
  policy announcements within days, lender campaigns may never be covered.
- Articles never reference future events (PIT-safe by construction).

The ``event_id`` field is included for evaluation only and MUST be stripped
before articles are shown to an agent. The agent sees headline/body/date.

Design notes:
- Templates are deliberately varied in tone (wire-service neutral, opinionated
  columnist, data-heavy analyst) so agents can't just pattern-match.
- Each article has a ``reported_magnitude`` that may differ from the true
  magnitude — agents must triangulate across articles.
"""

from __future__ import annotations

import numpy as np

from td_one.data_gen.events import Event, EventType

#: Publication delay (days) by event type: (min_days, max_days, coverage_prob).
#: coverage_prob is P(at least one article is written about the event).
_NEWS_PARAMS: dict[EventType, tuple[int, int, float]] = {
    EventType.POLICY_SHOCK: (1, 7, 0.95),  # announcements covered fast
    EventType.RATE_SHIFT: (1, 5, 0.95),  # surprise decisions are news
    EventType.HOUSING_DISRUPTION: (20, 45, 0.70),  # ~1 month delay, survivorship bias
    EventType.LENDER_SHIFT: (5, 30, 0.40),  # asymmetric: often not covered
    EventType.MACRO_SURPRISE: (10, 30, 0.80),  # labor data lags
}

#: Number of articles per covered event: Poisson(λ) + 1.
_ARTICLES_LAMBDA: dict[EventType, float] = {
    EventType.POLICY_SHOCK: 6.0,
    EventType.RATE_SHIFT: 8.0,
    EventType.HOUSING_DISRUPTION: 4.0,
    EventType.LENDER_SHIFT: 2.0,
    EventType.MACRO_SURPRISE: 3.0,
}


# --- Templates ---
# Each template: (headline_template, body_template). Placeholders:
#   {magnitude} — reported magnitude (with noise)
#   {cohort} — human-readable cohort description
#   {direction} — "rise"/"fall"/"tighten"/"ease" as appropriate

_TEMPLATES: dict[EventType, list[tuple[str, str]]] = {
    EventType.POLICY_SHOCK: [
        (
            "OSFI {direction} mortgage stress test by {magnitude}pp",
            "The Office of the Superintendent of Financial Institutions announced "
            "{direction_long} to the mortgage stress test, moving the qualifying rate "
            "by {magnitude} percentage points for {cohort}. Analysts say uninsured "
            "borrowers renewing in the coming months will feel the impact first.",
        ),
        (
            "Stress test shift: what {cohort} need to know",
            "Regulators have adjusted the benchmark used to qualify borrowers. "
            "The {magnitude}pp move {direction_long} the bar for {cohort}, "
            "potentially pushing more renewals toward alternative lenders.",
        ),
    ],
    EventType.RATE_SHIFT: [
        (
            "Bank of Canada surprises with {magnitude}pp rate {direction}",
            "In an unscheduled move, the Bank of Canada {direction_long} its policy "
            "rate by {magnitude} percentage points. Mortgage borrowers with renewals "
            "in the next quarter — {cohort} — face an immediate repricing.",
        ),
        (
            "Markets reel as BoC moves {magnitude}pp",
            "The central bank's surprise {direction} of {magnitude}pp caught traders "
            "off guard. Fixed mortgage rates are expected to follow within weeks, "
            "hitting {cohort} hardest.",
        ),
    ],
    EventType.HOUSING_DISRUPTION: [
        (
            "{cohort} prices drop {magnitude}% in correction",
            "Home prices for {cohort} have fallen {magnitude}% from recent peaks, "
            "according to new data. Economists warn that underwater borrowers may "
            "struggle to refinance, while distressed sales could rise.",
        ),
        (
            "Housing correction deepens: {cohort} down {magnitude}%",
            "The downturn in {cohort} accelerated this month, with prices off "
            "{magnitude}%. Lenders are watching loan-to-value ratios closely.",
        ),
    ],
    EventType.LENDER_SHIFT: [
        (
            "Major bank launches aggressive mortgage retention push",
            "One of Canada's big banks has begun an unusually aggressive campaign "
            "to retain renewing mortgage clients, offering {cohort} preferential "
            "rates. Rivals may be forced to respond.",
        ),
        (
            "Retention wars: bank targets {cohort} with sharp pricing",
            "A leading lender is offering below-market renewal rates to {cohort} "
            "in a bid to stem attrition. Industry watchers call it the most "
            "aggressive retention push in years.",
        ),
    ],
    EventType.MACRO_SURPRISE: [
        (
            "Unemployment jumps {magnitude}pp for {cohort}",
            "New labour data shows unemployment up {magnitude} percentage points "
            "among {cohort}. Economists say household cash flows will tighten, "
            "raising the risk of payment stress at renewal.",
        ),
        (
            "Job losses mount: {cohort} hit hardest",
            "The latest employment figures paint a grim picture for {cohort}, "
            "with joblessness up {magnitude}pp. Mortgage arrears typically follow "
            "with a lag of several months.",
        ),
    ],
}

_COHORT_LABELS: dict[str, str] = {
    "uninsured_720plus": "uninsured borrowers with strong credit",
    "uninsured_all": "uninsured borrowers",
    "all_borrowers": "mortgage borrowers",
    "variable_rate": "variable-rate borrowers",
    "resetting_3mo": "borrowers renewing this quarter",
    "toronto_condo": "Toronto condo owners",
    "vancouver_detached": "Vancouver detached homeowners",
    "alberta_all": "Alberta homeowners",
    "ontario_all": "Ontario homeowners",
    "bank_td": "TD mortgage clients",
    "bank_rbc": "RBC mortgage clients",
    "bank_bmo": "BMO mortgage clients",
    "bank_scotia": "Scotia mortgage clients",
    "ontario_manufacturing": "Ontario manufacturing workers",
    "alberta_energy": "Alberta energy-sector workers",
    "bc_tech": "BC tech workers",
    "quebec_all": "Quebec households",
}


def _format_magnitude(ev_type: EventType, magnitude: float, rng: np.random.Generator) -> str:
    """Format the reported magnitude with ±30% noise."""
    noisy = magnitude * float(rng.uniform(0.7, 1.3))
    if ev_type == EventType.HOUSING_DISRUPTION:
        return f"{abs(noisy) * 100:.1f}"
    if ev_type == EventType.LENDER_SHIFT:
        return f"{abs(noisy):.1f}"
    return f"{abs(noisy):.2f}"


def _direction_word(ev_type: EventType, magnitude: float) -> tuple[str, str]:
    """Return (short, long) direction words based on sign of magnitude."""
    pos = magnitude >= 0
    if ev_type == EventType.POLICY_SHOCK:
        return ("tightens", "tightening") if pos else ("eases", "easing")
    if ev_type == EventType.RATE_SHIFT:
        return ("hike", "hiked") if pos else ("cut", "cut")
    if ev_type == EventType.HOUSING_DISRUPTION:
        return ("fall", "falling")  # magnitude is always positive (a drop)
    if ev_type == EventType.LENDER_SHIFT:
        return ("push", "pushing")
    return ("rise", "rising") if pos else ("fall", "falling")


def generate_news_feed(
    events: list[Event],
    seed: int = 42,
) -> list[dict[str, object]]:
    """Generate a synthetic PIT-safe news feed for the event timeline.

    Each event may produce zero or more articles. Article dates are always
    >= event_date (PIT-safe). Magnitudes are noisy. The ``event_id`` field is
    for evaluation only and must be stripped before showing to agents.

    Args:
        events: The full event timeline (already generated).
        seed: Random seed for reproducibility (independent of event seed).

    Returns:
        List of article dicts with keys: article_date (date), headline (str),
        body (str), event_id (str), event_type (str), reported_magnitude (str).
        Sorted by article_date.

    """
    rng = np.random.default_rng(seed)
    articles: list[dict[str, object]] = []

    for ev in events:
        min_delay, max_delay, coverage_p = _NEWS_PARAMS[ev.event_type]
        if rng.random() > coverage_p:
            continue  # event gets zero coverage

        n_articles = int(rng.poisson(_ARTICLES_LAMBDA[ev.event_type])) + 1
        # Cap to avoid runaway feeds
        n_articles = min(n_articles, 20)

        templates = _TEMPLATES[ev.event_type]
        cohort_label = _COHORT_LABELS.get(ev.affected_cohort, ev.affected_cohort)

        for _ in range(n_articles):
            delay_days = int(rng.integers(min_delay, max_delay + 1))
            # Article date: event_date + delay, snapped to a valid date.
            # We approximate by adding days directly.
            from datetime import timedelta

            art_date = ev.event_date + timedelta(days=delay_days)

            tmpl = templates[int(rng.integers(0, len(templates)))]
            mag_str = _format_magnitude(ev.event_type, ev.magnitude, rng)
            dir_short, dir_long = _direction_word(ev.event_type, ev.magnitude)

            headline = tmpl[0].format(
                magnitude=mag_str,
                cohort=cohort_label,
                direction=dir_short,
                direction_long=dir_long,
            )
            body = tmpl[1].format(
                magnitude=mag_str,
                cohort=cohort_label,
                direction=dir_short,
                direction_long=dir_long,
            )
            articles.append(
                {
                    "article_date": art_date,
                    "headline": headline,
                    "body": body,
                    "event_id": ev.event_id,  # EVAL ONLY — strip before agent sees
                    "event_type": ev.event_type.value,
                    "reported_magnitude": mag_str,
                }
            )

    articles.sort(key=lambda a: (str(a["article_date"]), str(a["event_id"])))
    return articles


def strip_eval_fields(articles: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove evaluation-only fields before showing articles to an agent.

    Strips ``event_id`` (which would reveal the ground-truth event linkage).
    Keeps ``event_type`` since a competent reader could infer it from text.

    Args:
        articles: Raw articles from generate_news_feed.

    Returns:
        Articles with ``event_id`` removed.

    """
    return [
        {k: v for k, v in a.items() if k != "event_id"} for a in articles
    ]


def articles_visible_asof(
    articles: list[dict[str, object]], asof: object
) -> list[dict[str, object]]:
    """Return articles with ``article_date <= asof`` (PIT-safe filter).

    Args:
        articles: Full news feed (eval fields already stripped or not).
        asof: The as-of date.

    Returns:
        Articles published on or before asof.

    """
    return [a for a in articles if str(a["article_date"]) <= str(asof)]


def coverage_stats(
    articles: list[dict[str, object]], events: list[Event]
) -> dict[str, object]:
    """Compute news coverage statistics (for diagnostics, not for agents).

    Args:
        articles: Full news feed (with event_id).
        events: Full event timeline.

    Returns:
        Dict with: n_events, n_articles, coverage_rate (fraction of events
        with >=1 article), mean_articles_per_covered_event.

    """
    covered = {str(a["event_id"]) for a in articles}
    n_covered = len(covered)
    return {
        "n_events": len(events),
        "n_articles": len(articles),
        "coverage_rate": n_covered / len(events) if events else 0.0,
        "mean_articles_per_covered_event": len(articles) / n_covered if n_covered else 0.0,
    }
