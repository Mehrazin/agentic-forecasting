"""Chunking strategies for news article bodies.

Four methods, one interface: ``chunk(text) -> list[Chunk]`` where each
Chunk carries text + char offsets. Compared in
``scripts/tune_chunking.py`` on the 20-pair tuning set; the winner feeds
``rag/semantic.py`` chunk embeddings.

Methods
-------
- fixed: 512 tokens (~2000 chars), 50-token (~200 char) overlap.
- sentence: 3-5 sentences per chunk (NLTK-free sentence splitter).
- paragraph: natural paragraph boundaries (blank-line splits), merged if
  tiny (<100 chars), split if huge (>3000 chars).
- semantic: embedding-similarity breakpoints — split sentences, embed,
  cut where consecutive cosine similarity drops below (mean - std).

"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

# Rough char<->token conversion for the 384-dim MiniLM model.
CHARS_PER_TOKEN = 4
FIXED_TOKENS = 512
FIXED_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    text: str
    start: int  # char offset in source
    end: int
    method: str

    @property
    def key(self) -> str:
        """Stable cache key for this chunk's embedding."""
        h = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        return f"{self.method}:{h}"


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=[.!?])\n+")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no NLTK dependency)."""
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_fixed(text: str) -> list[Chunk]:
    size = FIXED_TOKENS * CHARS_PER_TOKEN
    overlap = FIXED_OVERLAP_TOKENS * CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(Chunk(text[i:end], i, end, "fixed"))
        if end == n:
            break
        i = end - overlap
    return chunks


def chunk_sentence(text: str, per_chunk: int = 4) -> list[Chunk]:
    sents = split_sentences(text)
    chunks: list[Chunk] = []
    i = 0
    pos = 0
    while i < len(sents):
        group = sents[i : i + per_chunk]
        joined = " ".join(group)
        start = text.find(group[0], pos)
        end = start + len(joined)
        # account for single-space joins vs original whitespace
        end = min(end + len(group), len(text))
        chunks.append(Chunk(joined, start, end, "sentence"))
        pos = end
        i += per_chunk
    return chunks


def chunk_paragraph(text: str) -> list[Chunk]:
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    for p in raw:
        if merged and len(merged[-1]) < 100:
            merged[-1] = merged[-1] + "\n" + p
        elif len(p) > 3000:
            # fall back to fixed within oversized paragraphs
            for c in chunk_fixed(p):
                merged.append(c.text)
        else:
            merged.append(p)
    chunks: list[Chunk] = []
    pos = 0
    for m in merged:
        start = text.find(m[:40], pos)
        if start < 0:
            start = pos
        end = min(start + len(m), len(text))
        chunks.append(Chunk(m, start, end, "paragraph"))
        pos = end
    return chunks


def chunk_semantic(
    text: str,
    encode=None,
    threshold_std: float = 1.0,
) -> list[Chunk]:
    """Split where consecutive sentence-embedding similarity dips.

    ``encode``: callable(list[str]) -> (n, d) L2-normalized array.
    Falls back to sentence chunking when encode is None.
    """
    sents = split_sentences(text)
    if len(sents) < 3 or encode is None:
        return chunk_sentence(text)
    emb = encode(sents)
    sims = np.sum(emb[:-1] * emb[1:], axis=1)
    cutoff = sims.mean() - threshold_std * sims.std()
    breaks = [0]
    for j, s in enumerate(sims, start=1):
        if s < cutoff:
            breaks.append(j)
    breaks.append(len(sents))
    chunks: list[Chunk] = []
    pos = 0
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        group = sents[a:b]
        joined = " ".join(group)
        start = text.find(group[0][:40], pos)
        if start < 0:
            start = pos
        end = min(start + len(joined) + len(group), len(text))
        chunks.append(Chunk(joined, start, end, "semantic"))
        pos = end
    return chunks


CHUNKERS = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "paragraph": chunk_paragraph,
    "semantic": chunk_semantic,
}
