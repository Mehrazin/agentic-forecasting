"""Perplexity sonar client for date-scoped news search.

Same interface as the GDELT-backed NewsStore: every query takes an
explicit ``before`` cutoff and the client NEVER requests articles on or
after it. When the user provides a Perplexity API key, this becomes the
quality upgrade path — richer snippets and summaries than GDELT.

Setup:
    1. Get a key at https://www.perplexity.ai/settings/api
    2. Share it via the secure vault (like the Groq key)
    3. Set PERPLEXITY_API_KEY in the environment or pass it explicitly

Then build a DB fast:
    uv run python scripts/fetch_news_perplexity.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import date

API_URL = "https://api.perplexity.ai/chat/completions"


@dataclass(frozen=True)
class PerplexityArticle:
    pub_date: str  # YYYY-MM-DD (best-effort from search results)
    title: str
    url: str
    snippet: str = ""


class PerplexityClient:
    """Date-scoped news search via Perplexity sonar.

    Parameters
    ----------
    api_key:
        Perplexity API key. Defaults to PERPLEXITY_API_KEY env var.
    model:
        Sonar model. "sonar" is cheapest; "sonar-pro" is better.
    """

    def __init__(self, api_key: str | None = None, model: str = "sonar") -> None:
        key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not key:
            raise ValueError(
                "Perplexity API key required: set PERPLEXITY_API_KEY "
                "or pass api_key=. Get one at perplexity.ai/settings/api"
            )
        self.api_key = key
        self.model = model

    def search(
        self,
        query: str,
        before: str | date,
        limit: int = 10,
    ) -> list[PerplexityArticle]:
        """Search news published strictly before ``before``.

        Uses Perplexity's search_before_date_filter so the cutoff is
        enforced server-side, not just in our parsing.
        """
        cutoff = before.isoformat() if isinstance(before, date) else before
        date.fromisoformat(cutoff)  # fail loudly on malformed dates

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a news retrieval assistant. Return factual "
                        "news article results as JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Find {limit} recent news articles about: {query}. "
                        "Return a JSON array of objects with keys: "
                        "title, url, date (YYYY-MM-DD), snippet (1-2 sentences). "
                        "Only articles published before " + cutoff + "."
                    ),
                },
            ],
            "search_before_date_filter": cutoff,
            "return_citations": True,
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        content = data["choices"][0]["message"]["content"]
        # The model returns a JSON array; be tolerant of markdown fences.
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            return []

        articles = []
        for it in items:
            if not isinstance(it, dict):
                continue
            d = str(it.get("date", ""))[:10]
            # Client-side PIT guard: drop anything on/after the cutoff,
            # even if the server filter slipped.
            if not d or d >= cutoff:
                continue
            articles.append(
                PerplexityArticle(
                    pub_date=d,
                    title=str(it.get("title", ""))[:500],
                    url=str(it.get("url", "")),
                    snippet=str(it.get("snippet", ""))[:1000],
                )
            )
        return articles[:limit]
