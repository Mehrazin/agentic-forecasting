"""Tests for rag/chunking.py — chunker contracts, not exact splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.rag.chunking import (  # noqa: E402
    CHUNKERS,
    chunk_fixed,
    chunk_paragraph,
    chunk_semantic,
    chunk_sentence,
    split_sentences,
)

TEXT = (
    "The Bank of Canada held its policy rate steady on Wednesday. "
    "Economists had expected no change amid cooling inflation. "
    "Governor Macklem cited persistent shelter cost pressures.\n\n"
    "In a separate statement, the bank noted that wage growth remains "
    "elevated. Markets now price a cut in the spring. "
    "Analysts remain divided on the timing."
)


def test_split_sentences_basic():
    sents = split_sentences(TEXT)
    assert len(sents) >= 5
    assert all(s for s in sents)


def test_fixed_covers_full_text():
    chunks = chunk_fixed(TEXT)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)
    assert all(c.method == "fixed" for c in chunks)


def test_fixed_overlap():
    long = "word " * 2000
    chunks = chunk_fixed(long)
    assert len(chunks) > 1
    # overlap: second chunk starts before first ends
    assert chunks[1].start < chunks[0].end


def test_sentence_chunk_count():
    chunks = chunk_sentence(TEXT, per_chunk=2)
    assert len(chunks) >= 3
    assert all(c.method == "sentence" for c in chunks)


def test_paragraph_respects_boundaries():
    chunks = chunk_paragraph(TEXT)
    assert len(chunks) == 2  # two paragraphs in TEXT
    assert all(c.method == "paragraph" for c in chunks)


def test_semantic_falls_back_without_encoder():
    chunks = chunk_semantic(TEXT, encode=None)
    assert len(chunks) > 0
    assert all(c.method == "sentence" for c in chunks)


def test_semantic_with_encoder():
    import numpy as np

    def fake_encode(texts):
        # decreasing similarity -> forces a break
        n = len(texts)
        base = np.eye(n, 4, dtype=np.float32) + 0.1
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        return base / np.maximum(norms, 1e-9)

    chunks = chunk_semantic(TEXT, encode=fake_encode, threshold_std=-10)
    assert len(chunks) >= 1
    assert all(c.method == "semantic" for c in chunks)


def test_chunk_keys_stable():
    a = chunk_fixed(TEXT)
    b = chunk_fixed(TEXT)
    assert [c.key for c in a] == [c.key for c in b]


def test_all_chunkers_registered():
    assert set(CHUNKERS) == {"fixed", "sentence", "paragraph", "semantic"}


def test_empty_text():
    assert chunk_fixed("") == []
    assert chunk_sentence("") == []
    assert chunk_paragraph("") == []
