"""Tests for td_one.rag.semantic (mock embeddings; no model download)."""


import numpy as np
import pytest

from td_one.rag.semantic import (
    SemanticNewsIndex,
    SemanticRAGTool,
    cosine_sim,
)


class FakeClient:
    """Deterministic toy embeddings: query words map to fixed vectors."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            tl = t.lower()
            if "ontario" in tl or "toronto" in tl:
                vecs.append([1.0, 0.0, 0.0])
            elif "alberta" in tl or "calgary" in tl:
                vecs.append([0.0, 1.0, 0.0])
            else:
                vecs.append([0.0, 0.0, 1.0])
        return np.array(vecs, dtype=np.float32)


@pytest.fixture()
def tiny_db(tmp_path):
    import sqlite3

    db = tmp_path / "news.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE articles (pub_date TEXT, title TEXT, url TEXT, source TEXT)"
    )
    rows = [
        ("2024-01-05", "Ontario housing market cools in Toronto", "u1", "s"),
        ("2024-02-10", "Alberta oil jobs boost Calgary mortgages", "u2", "s"),
        ("2025-06-01", "Ontario first-time buyer incentive expands", "u3", "s"),
        ("2023-12-01", "Bank of Canada holds policy rate", "u4", "s"),
    ]
    con.executemany("INSERT INTO articles VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return db


def _index(tiny_db, monkeypatch):
    monkeypatch.setattr(
        "td_one.rag.semantic.SemanticNewsIndex._load_or_build",
        lambda self: _fake_load(self, tiny_db),
    )
    return SemanticNewsIndex(db_path=tiny_db, client=FakeClient())


def _fake_load(self, db_path):
    import sqlite3

    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT rowid, pub_date, title, url, source FROM articles ORDER BY rowid"
    ).fetchall()
    con.close()
    self._ids = [r[0] for r in rows]
    self._dates = [r[1] for r in rows]
    self._titles = [r[2] for r in rows]
    self._urls = [r[3] for r in rows]
    self._sources = [r[4] for r in rows]
    self._emb = FakeClient().encode(self._titles)


def test_semantic_search_ranks_by_topic(tiny_db, monkeypatch) -> None:
    idx = _index(tiny_db, monkeypatch)
    hits = idx.search("Ontario mortgages", before="2026-01-01", top_k=4)
    assert hits[0].title.startswith("Ontario housing")
    assert hits[1].title.startswith("Ontario first-time")


def test_pit_gate_excludes_future(tiny_db, monkeypatch) -> None:
    idx = _index(tiny_db, monkeypatch)
    hits = idx.search("Ontario", before="2025-01-01", top_k=10)
    assert all(h.pub_date < "2025-01-01" for h in hits)
    assert not any("first-time buyer" in h.title for h in hits)


def test_pit_gate_empty_before_all(tiny_db, monkeypatch) -> None:
    idx = _index(tiny_db, monkeypatch)
    assert idx.search("Ontario", before="2020-01-01") == []


def test_empty_query_raises(tiny_db, monkeypatch) -> None:
    idx = _index(tiny_db, monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        idx.search("  ", before="2026-01-01")


def test_cosine_sim() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_sim(a, b) == pytest.approx(1.0)
    assert cosine_sim(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_sim(a, np.zeros(2)) == 0.0


class FakeLLM:
    def complete(self, prompt: str, json_mode: bool = False) -> str:
        import json
        import re

        n = len(re.findall(r"^\d+\.", prompt, flags=re.M))
        # Score Ontario articles high, others low.
        scores = []
        for line in prompt.splitlines():
            m = re.match(r"\d+\.\s", line)
            if m:
                scores.append(0.9 if "ontario" in line.lower() else 0.1)
        return json.dumps({"scores": scores[:n]})


def test_rag_tool_rerank(tiny_db, monkeypatch) -> None:
    monkeypatch.setattr(
        "td_one.rag.semantic.SemanticNewsIndex._load_or_build",
        lambda self: _fake_load(self, tiny_db),
    )
    tool = SemanticRAGTool(db_path=tiny_db, llm=FakeLLM())
    # Bypass index construction: patch the instance after init fails? Instead
    # construct via the same monkeypatched path.
    tool._index = SemanticNewsIndex.__new__(SemanticNewsIndex)
    tool._index.db_path = tiny_db
    tool._index._client = FakeClient()
    tool._index._cache = tiny_db.parent / "x.npz"
    _fake_load(tool._index, tiny_db)
    hits = tool.search("mortgages", before="2026-01-01", limit=2)
    assert len(hits) == 2
    assert all("ontario" in h.title.lower() for h in hits)


def test_rag_tool_abstains_on_low_relevance(tiny_db, monkeypatch) -> None:
    class LowLLM:
        def complete(self, prompt: str, json_mode: bool = False) -> str:
            import json
            import re

            n = len(re.findall(r"^\d+\.", prompt, flags=re.M))
            return json.dumps({"scores": [0.05] * n})

    monkeypatch.setattr(
        "td_one.rag.semantic.SemanticNewsIndex._load_or_build",
        lambda self: _fake_load(self, tiny_db),
    )
    tool = SemanticRAGTool(
        db_path=tiny_db, llm=LowLLM(), abstain_threshold=0.4
    )
    tool._index = SemanticNewsIndex.__new__(SemanticNewsIndex)
    tool._index.db_path = tiny_db
    tool._index._client = FakeClient()
    tool._index._cache = tiny_db.parent / "x.npz"
    _fake_load(tool._index, tiny_db)
    assert tool.search("mortgages", before="2026-01-01") == []


def test_rag_tool_no_llm_returns_semantic_order(tiny_db, monkeypatch) -> None:
    monkeypatch.setattr(
        "td_one.rag.semantic.SemanticNewsIndex._load_or_build",
        lambda self: _fake_load(self, tiny_db),
    )
    tool = SemanticRAGTool(db_path=tiny_db, llm=None)
    tool._index = SemanticNewsIndex.__new__(SemanticNewsIndex)
    tool._index.db_path = tiny_db
    tool._index._client = FakeClient()
    tool._index._cache = tiny_db.parent / "x.npz"
    _fake_load(tool._index, tiny_db)
    hits = tool.search("Alberta", before="2026-01-01", limit=1)
    assert hits[0].title.startswith("Alberta oil")
