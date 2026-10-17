"""PIT-safe news store for TD-One.

Articles are indexed with their publication dates. All queries take an
explicit ``before`` cutoff — the store NEVER returns articles published on
or after the cutoff. This is the temporal guard that keeps the LLM's
"news context" honest: at a forecast origin of 2026-06-01, the agent can
only see news from before 2026-06-01.

Backends:
  - SQLite + FTS5 (local, free, built from GDELT via scripts/fetch_news_gdelt.py)
  - Perplexity sonar (API, date-scoped web search — needs PERPLEXITY_API_KEY)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Article:
    """A single indexed news article."""

    pub_date: str  # YYYY-MM-DD
    title: str
    url: str
    source: str
    snippet: str = ""


def _default_db() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "news" / "news_2026.db"


class NewsStore:
    """Date-scoped full-text search over the local 2026 news DB.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB. Defaults to ``td-one/data/news/news_2026.db``.

    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _default_db()
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"News DB not found at {self.db_path}. "
                "Build it with: uv run python scripts/fetch_news_gdelt.py"
            )

    def search(
        self,
        query: str,
        before: str | date,
        limit: int = 10,
    ) -> list[Article]:
        """Full-text search for articles published strictly before ``before``.

        Parameters
        ----------
        query:
            FTS5 query (keywords, quoted phrases, AND/OR).
        before:
            Cutoff as ``YYYY-MM-DD`` (or date). Articles with
            ``pub_date >= before`` are excluded — the PIT guard.
        limit:
            Max articles returned, newest first.

        """
        cutoff = before.isoformat() if isinstance(before, date) else before
        # Validate the cutoff format early — a malformed date must fail
        # loudly, never silently widen the window.
        date.fromisoformat(cutoff)

        con = sqlite3.connect(self.db_path)
        try:
            rows = con.execute(
                """
                SELECT a.pub_date, a.title, a.url, a.source,
                       snippet(articles_fts, 0, '[', ']', '...', 20) AS snip
                FROM articles_fts
                JOIN articles a ON a.rowid = articles_fts.rowid
                WHERE articles_fts MATCH ? AND a.pub_date < ?
                ORDER BY a.pub_date DESC
                LIMIT ?
                """,
                (query, cutoff, limit),
            ).fetchall()
        finally:
            con.close()
        return [
            Article(
                pub_date=r[0], title=r[1], url=r[2], source=r[3], snippet=r[4] or ""
            )
            for r in rows
        ]

    def count_before(self, before: str | date) -> int:
        """How many articles are visible before the cutoff (coverage check)."""
        cutoff = before.isoformat() if isinstance(before, date) else before
        con = sqlite3.connect(self.db_path)
        try:
            (n,) = con.execute(
                "SELECT COUNT(*) FROM articles WHERE pub_date < ?", (cutoff,)
            ).fetchone()
        finally:
            con.close()
        return int(n)


def init_db(db_path: Path) -> None:
    """Create the articles schema (idempotent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                pub_date TEXT NOT NULL,
                title    TEXT NOT NULL,
                url      TEXT NOT NULL UNIQUE,
                source   TEXT NOT NULL DEFAULT ''
            )
            """
        )
        con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
            USING fts5(title, content='articles', content_rowid='rowid')
            """
        )
        con.execute(
            """
            CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles
            BEGIN
                INSERT INTO articles_fts(rowid, title)
                VALUES (new.rowid, new.title);
            END
            """
        )
        con.commit()
    finally:
        con.close()
