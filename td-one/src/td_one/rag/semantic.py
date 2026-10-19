"""Semantic RAG for news retrieval: embeddings + cosine + LLM rerank.

Upgrades the BM25-only path in ``td_one.news.rag`` with a semantic
retrieval stage. Embeddings come from a local ``all-MiniLM-L6-v2``
sentence-transformer (384-dim) — the Vector proxy does not expose
embedding models to this project (403 model_access_denied, verified
2026-10-19), so there is no API dependency for retrieval.

Pipeline
--------
1. **Semantic retrieval**: cosine similarity between the query embedding
   and cached article embeddings (title text), PIT-gated to
   ``articles_visible_asof(origin)``.
2. **LLM rerank**: top candidates rescored by Gemini Flash via the Vector
   proxy (same JSON protocol as ``td_one.news.rag``).
3. **Abstention**: empty result when top relevance < threshold.

Top-k tuning (2026-10-19, ``scripts/tune_rag_k.py``, 20 hand-labeled
query->article pairs from the news DB):

======  ========  ========  ========
k       P@k       R@k       F1
======  ========  ========  ========
5       0.507     0.408     0.452
10      0.427     0.518     0.468
20      0.360     0.613     0.453
50      0.251     0.818     0.384
100     0.182     0.910     0.303
======  ========  ========  ========

Chosen: semantic_k=50, rerank_final_k=5. F1 peaks at k=10, but the
retrieval stage deliberately biases toward recall (0.818 at k=50) —
the LLM reranker filters noise cheaply, and a missed relevant article
can never be recovered downstream. The final k=5 biases toward
precision (the LLM prompt budget is fixed).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Tuned top-k values (see module docstring).
SEMANTIC_TOP_K = 50
BM25_TOP_K = 50
RERANK_FINAL_K = 5
ABSTAIN_THRESHOLD = 0.4

_MODEL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
    / "manual"
)

_EMBED_CACHE_NAME = "news_embeddings_v1.npz"


@dataclass(frozen=True)
class ScoredArticle:
    pub_date: str
    title: str
    url: str
    source: str
    score: float


def _default_db() -> Path:
    return (
        Path(__file__).resolve().parents[3] / "data" / "news" / "news_2026.db"
    )


def _embed_cache_path(db_path: Path) -> Path:
    return db_path.parent / _EMBED_CACHE_NAME


class EmbeddingClient:
    """Local sentence-transformer embeddings (no API calls)."""

    def __init__(self, model_dir: Path | None = None) -> None:
        import os

        # Threading guard: torch can deadlock on import without these
        # (observed 2026-10-19 after SIGTERM'd embedding processes).
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer

        d = model_dir or _MODEL_DIR
        if not d.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {d}. "
                "Download all-MiniLM-L6-v2 first."
            )
        self._model = SentenceTransformer(str(d))

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts -> (n, 384) float32 array, L2-normalized."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class SemanticNewsIndex:
    """Cosine-similarity index over cached news article embeddings.

    Embeddings are built once from article titles and cached to disk
    (gitignored ``.npz`` next to the news DB). PIT-gating happens at
    query time: only articles with ``pub_date < cutoff`` participate.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        client: EmbeddingClient | None = None,
    ) -> None:
        self.db_path = db_path or _default_db()
        if not self.db_path.exists():
            raise FileNotFoundError(f"News DB not found at {self.db_path}")
        self._client = client or EmbeddingClient()
        self._cache = _embed_cache_path(self.db_path)
        self._ids: list[int] = []
        self._dates: list[str] = []
        self._titles: list[str] = []
        self._urls: list[str] = []
        self._sources: list[str] = []
        self._emb: np.ndarray | None = None
        self._load_or_build()

    def _load_or_build(self) -> None:
        if self._cache.exists():
            logger.info("semantic RAG: loading embedding cache %s", self._cache)
            z = np.load(self._cache, allow_pickle=True)
            self._ids = list(z["ids"])
            self._dates = list(z["dates"])
            self._titles = list(z["titles"])
            self._urls = list(z["urls"])
            self._sources = list(z["sources"])
            self._emb = z["emb"]
            return
        logger.info("semantic RAG: building embeddings (one-time, ~17k articles)")
        con = sqlite3.connect(self.db_path)
        rows = con.execute(
            "SELECT rowid, pub_date, title, url, source FROM articles ORDER BY rowid"
        ).fetchall()
        con.close()
        self._ids = [r[0] for r in rows]
        self._dates = [r[1] for r in rows]
        self._titles = [r[2] for r in rows]
        self._urls = [r[3] for r in rows]
        self._sources = [r[4] or "" for r in rows]
        self._emb = self._client.encode(self._titles)
        np.savez_compressed(
            self._cache,
            ids=np.array(self._ids),
            dates=np.array(self._dates, dtype=object),
            titles=np.array(self._titles, dtype=object),
            urls=np.array(self._urls, dtype=object),
            sources=np.array(self._sources, dtype=object),
            emb=self._emb,
        )
        logger.info("semantic RAG: cache written to %s", self._cache)

    def search(
        self, query: str, before: str | date, top_k: int = SEMANTIC_TOP_K
    ) -> list[ScoredArticle]:
        """Cosine retrieval, PIT-gated to ``pub_date < before``."""
        if isinstance(before, date):
            before = before.isoformat()
        if not query.strip():
            raise ValueError("query must be non-empty")
        assert self._emb is not None
        q = self._client.encode([query.strip()])[0]
        sims = self._emb @ q  # embeddings are L2-normalized
        # PIT mask
        eligible = [i for i, d in enumerate(self._dates) if d < before]
        if not eligible:
            return []
        ranked = sorted(eligible, key=lambda i: sims[i], reverse=True)[:top_k]
        return [
            ScoredArticle(
                pub_date=self._dates[i],
                title=self._titles[i],
                url=self._urls[i],
                source=self._sources[i],
                score=float(sims[i]),
            )
            for i in ranked
        ]


class SemanticRAGTool:
    """Semantic retrieval + LLM rerank, PIT-gated.

    Parameters
    ----------
    llm:
        Callable ``(prompt) -> str`` for reranking (e.g.
        ``VectorProxyClient(...).complete``). If None, returns semantic
        order without reranking.

    """

    def __init__(
        self,
        db_path: Path | None = None,
        llm: object | None = None,
        abstain_threshold: float = ABSTAIN_THRESHOLD,
    ) -> None:
        self._index = SemanticNewsIndex(db_path=db_path)
        self._llm = llm
        self.abstain_threshold = abstain_threshold

    def search(
        self,
        query: str,
        before: str | date,
        limit: int = RERANK_FINAL_K,
    ) -> list[ScoredArticle]:
        """Retrieve -> rerank -> top-``limit`` (PIT-gated throughout)."""
        cands = self._index.search(query, before, top_k=SEMANTIC_TOP_K)
        if not cands:
            return []
        if self._llm is None:
            return cands[:limit]
        ranked = self._rerank(query, cands)
        if ranked and ranked[0].score < self.abstain_threshold:
            return []
        return ranked[:limit]

    def _rerank(
        self, query: str, cands: list[ScoredArticle]
    ) -> list[ScoredArticle]:
        """One LLM call scoring each candidate 0..1 for relevance."""
        assert self._llm is not None
        lines = "\n".join(
            f"{i + 1}. [{c.pub_date}] {c.title}" for i, c in enumerate(cands)
        )
        prompt = (
            "Rate each article's relevance to the query on a 0-1 scale.\n"
            f"Query: {query}\n\nArticles:\n{lines}\n\n"
            'Reply with JSON only: {"scores": [s1, s2, ...]} in article order.'
        )
        try:
            text = self._llm.complete(prompt, json_mode=True)  # type: ignore[union-attr]
            scores = json.loads(text).get("scores", [])
            if len(scores) != len(cands):
                raise ValueError("score count mismatch")
            out = [
                ScoredArticle(
                    pub_date=c.pub_date,
                    title=c.title,
                    url=c.url,
                    source=c.source,
                    score=float(s),
                )
                for c, s in zip(cands, scores, strict=True)
            ]
            return sorted(out, key=lambda c: c.score, reverse=True)
        except Exception as e:  # noqa: BLE001 — degraded but functional
            logger.warning("semantic RAG rerank failed (%s); keeping order", e)
            return cands


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0
