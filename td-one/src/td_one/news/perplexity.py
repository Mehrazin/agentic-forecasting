"""Perplexity Agent API client for date-scoped news search.

Uses the Agent API (POST /v1/agent, GA Feb 2026) — Sonar chat completions
retired Sep 27, 2026. Preset "fast" replaces the old "sonar" model.

Same interface as the GDELT-backed NewsStore: every query takes an
explicit ``before`` cutoff and the client NEVER returns articles on or
after it (server-side date instruction + client-side guard).

Setup:
    1. Get a key at https://www.perplexity.ai/settings/api
    2. Share it via the secure vault (credentials.request_api_access)
    3. The key lives in the custom.perplexity connector; this module
       reads it via the vault surrogate (never in env/files).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import (  # noqa: E402
    add_surrogate_to_request,
    read_json_response,
)

API_URL = "https://api.perplexity.ai/v1/agent"


@dataclass(frozen=True)
class PerplexityArticle:
    pub_date: str  # YYYY-MM-DD
    title: str
    url: str
    snippet: str = ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):  # type: ignore[override]
        raise RuntimeError("redirect refused, verify the endpoint")


_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class PerplexityClient:
    """Date-scoped news search via the Perplexity Agent API.

    Parameters
    ----------
    preset:
        Agent API preset. "fast" (ex-sonar) is cheapest; "low" is better.
    """

    def __init__(self, preset: str = "fast") -> None:
        self.preset = preset

    def _call(self, prompt: str) -> dict:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({"preset": self.preset, "input": prompt}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": _UA},
        )
        add_surrogate_to_request(
            req, "custom.perplexity", allowed_hosts=["api.perplexity.ai"]
        )
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=120) as resp:
            return read_json_response(resp)  # type: ignore[no-any-return]

    @staticmethod
    def _extract_text(data: dict) -> str:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return str(c.get("text", ""))
        return ""

    def search(
        self,
        query: str,
        before: str | date,
        limit: int = 10,
    ) -> list[PerplexityArticle]:
        """Search news published strictly before ``before``.

        The cutoff is enforced twice: in the prompt (server-side date
        restriction) and by dropping any result with date >= cutoff.
        """
        cutoff = before.isoformat() if isinstance(before, date) else before
        date.fromisoformat(cutoff)  # fail loudly on malformed dates

        prompt = (
            f"Find {limit} news articles about: {query}. "
            f"Only articles published BEFORE {cutoff} — exclude anything on "
            f"or after that date. Return a JSON array of objects with keys: "
            f"title, url, date (YYYY-MM-DD), snippet (1-2 sentences). "
            f"Return ONLY the JSON array, no other text."
        )
        data = self._call(prompt)
        text = self._extract_text(data).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(items, list):
            return []

        articles = []
        for it in items:
            if not isinstance(it, dict):
                continue
            d = str(it.get("date", ""))[:10]
            # Client-side PIT guard: drop anything on/after the cutoff.
            if not d or d >= cutoff:
                continue
            url = str(it.get("url", ""))
            if not url:
                continue
            articles.append(
                PerplexityArticle(
                    pub_date=d,
                    title=str(it.get("title", ""))[:500],
                    url=url,
                    snippet=str(it.get("snippet", ""))[:1000],
                )
            )
        return articles[:limit]
