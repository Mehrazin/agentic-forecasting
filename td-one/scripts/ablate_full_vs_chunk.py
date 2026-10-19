"""Ablate full-article vs chunk-only context for RAG-fed LLM outputs.

For each tuning query:
  1. Retrieve top-k chunks (winning chunker from tune_chunking.py).
  2. Condition A (full): pass the entire bodies of the chunk-source articles.
  3. Condition B (chunk): pass only the retrieved chunk texts.
  4. Metric 1 (cheap, all queries): keyword coverage — fraction of the
     query's relevance keywords present in the context.
  5. Metric 2 (LLM, 5 queries): citation-checker pass rate on generated
     explanations via risk_reporter (mock LLM unless --live).

Run:
    .venv/bin/python scripts/ablate_full_vs_chunk.py
    .venv/bin/python scripts/ablate_full_vs_chunk.py --live  # real LLM calls
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

# Threading guard (see tune_chunking.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tune_rag_k import QUERIES  # noqa: E402

from td_one.rag.chunking import CHUNKERS
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
CHUNK_EMB_CACHE = DB.parent / "news_chunk_embeddings.npz"
METHOD = "sentence"  # override after tune_chunking.py picks a winner
TOP_K = 5


def load() -> tuple[dict[str, str], dict[str, str]]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT b.url, a.title, b.body FROM article_bodies b "
        "JOIN articles a ON a.url = b.url WHERE b.fetch_status='ok'"
    ).fetchall()
    con.close()
    bodies = {u: b for u, _, b in rows}
    titles = {u: t for u, t, _ in rows}
    return bodies, titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    bodies, titles = load()
    print(f"bodies: {len(bodies)}")
    if len(bodies) < 20:
        print("need more bodies first")
        return
    client = EmbeddingClient()
    chunk_fn = CHUNKERS[METHOD]

    # build chunk index (reuse cache from tune_chunking.py when present)
    cache: dict[str, np.ndarray] = {}
    if CHUNK_EMB_CACHE.exists():
        z = np.load(CHUNK_EMB_CACHE, allow_pickle=True)
        cache = {k: v for k, v in zip(z["keys"], z["vecs"])}
    chunk_list: list[tuple[str, str]] = []
    for url, body in bodies.items():
        for c in chunk_fn(body):
            chunk_list.append((url, c.text))
    keys = [hashlib.sha256(t.encode()).hexdigest()[:16] for _, t in chunk_list]
    missing = [t for t, k in zip([t for _, t in chunk_list], keys) if k not in cache]
    if missing:
        for t, v in zip(missing, client.encode(missing)):
            cache[hashlib.sha256(t.encode()).hexdigest()[:16]] = v
    mat = np.stack([cache[k] for k in keys])
    urls = [u for u, _ in chunk_list]

    # Metric 1: keyword coverage
    cov_full, cov_chunk = [], []
    len_full, len_chunk = [], []
    for query, keywords in QUERIES:
        q = client.encode([query])[0]
        top = np.argsort(mat @ q)[::-1][:TOP_K]
        src_urls = [urls[i] for i in top]
        chunk_texts = [chunk_list[i][1] for i in top]
        # dedupe urls preserving order for full condition
        seen, full_urls = set(), []
        for u in src_urls:
            if u not in seen:
                seen.add(u)
                full_urls.append(u)
        full_ctx = "\n\n".join(bodies[u] for u in full_urls)
        chunk_ctx = "\n\n".join(chunk_texts)
        len_full.append(len(full_ctx))
        len_chunk.append(len(chunk_ctx))
        for kws, store in ((cov_full, full_ctx), (cov_chunk, chunk_ctx)):
            ctx = store.lower()
            hits = sum(1 for kw in keywords if kw.lower() in ctx)
            kws.append(hits / max(1, len(keywords)))

    print(f"\n--- keyword coverage ({len(QUERIES)} queries) ---")
    print(f"full-article ctx: mean coverage={np.mean(cov_full):.3f} "
          f"mean chars={np.mean(len_full):.0f}")
    print(f"chunk-only ctx:   mean coverage={np.mean(cov_chunk):.3f} "
          f"mean chars={np.mean(len_chunk):.0f}")
    print(f"compression: {np.mean(len_full)/max(1,np.mean(len_chunk)):.1f}x fewer chars")

    if args.live:
        print("\n--live not yet wired to reporter; skipping LLM metric")
        print("wire: risk_reporter.generate(urls->context) x 5 queries, "
              "check_citations() pass rate per condition")


if __name__ == "__main__":
    main()
