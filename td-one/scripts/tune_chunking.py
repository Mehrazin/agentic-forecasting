"""Compare chunking methods for news RAG on the 20-pair tuning set.

For each method (fixed / sentence / paragraph / semantic):
  1. Chunk all cached article bodies.
  2. Embed chunks (local MiniLM, cached by chunk key — never recompute).
  3. For each tuning query: retrieve top-k chunks by cosine, map chunks
     back to articles, score P/R/F1 against keyword relevance labels.

Also reports avg chunks/article and avg chunk length per method.

Run:
    .venv/bin/python scripts/tune_chunking.py
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

# Threading guard: sentence-transformers/torch can deadlock on import
# without these (observed 2026-10-19). Must be set before torch loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tune_rag_k import QUERIES, _relevant  # noqa: E402

from td_one.rag.chunking import CHUNKERS
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
CHUNK_EMB_CACHE = DB.parent / "news_chunk_embeddings.npz"
TOP_K = 10


def load_bodies(limit: int = 100) -> list[tuple[str, str, str]]:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT b.url, a.title, b.body FROM article_bodies b "
        "JOIN articles a ON a.url = b.url "
        "WHERE b.fetch_status = 'ok' LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return rows


def main() -> None:
    bodies = load_bodies()
    print(f"bodies available: {len(bodies)}", flush=True)
    if len(bodies) < 20:
        print("not enough bodies yet — run fetch_article_bodies.py first")
        return
    print("creating embedding client...", flush=True)
    client = EmbeddingClient()
    print("client ready", flush=True)

    # chunk-embedding cache: key -> vector
    cache: dict[str, np.ndarray] = {}
    if CHUNK_EMB_CACHE.exists():
        z = np.load(CHUNK_EMB_CACHE, allow_pickle=True)
        cache = {k: v for k, v in zip(z["keys"], z["vecs"])}
        print(f"loaded {len(cache)} cached chunk embeddings")

    # article titles for relevance labels
    titles = {url: title for url, title, _ in bodies}

    results: dict[str, dict] = {}
    for method, fn in CHUNKERS.items():
        if method == "semantic" and "--include-semantic" not in sys.argv:
            print(f"{method:10s} skipped (pass --include-semantic to run; slow)")
            continue
        # chunk all bodies
        all_chunks: list[tuple[str, str]] = []  # (url, chunk_text)
        for url, _, body in bodies:
            kw = {"encode": client.encode} if method == "semantic" else {}
            for c in fn(body, **kw):
                all_chunks.append((url, c.text))
        # embed (cached)
        texts = [t for _, t in all_chunks]
        print(f"  {method}: {len(texts)} chunks to embed", flush=True)
        keys = [
            hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts
        ]
        missing = [t for t, k in zip(texts, keys) if k not in cache]
        print(f"  {method}: {len(missing)} missing from cache", flush=True)
        if missing:
            vecs = client.encode(missing)
            print(f"  {method}: encoded", flush=True)
            for t, v in zip(missing, vecs):
                k = hashlib.sha256(t.encode()).hexdigest()[:16]
                cache[k] = v
        mat = np.stack([cache[k] for k in keys])  # (n_chunks, 384)
        urls = [u for u, _ in all_chunks]

        # evaluate on tuning queries
        precs, recs = [], []
        for query, keywords in QUERIES:
            q = client.encode([query])[0]
            sims = mat @ q
            top = np.argsort(sims)[::-1][:TOP_K]
            retrieved_urls = {urls[i] for i in top}
            # relevant = articles whose title matches keywords
            rel = {
                u for u, t in titles.items() if _relevant(t, keywords)
            }
            if not rel:
                continue
            hit = len(retrieved_urls & rel)
            precs.append(hit / max(1, len(retrieved_urls)))
            recs.append(hit / len(rel))
        p = float(np.mean(precs)) if precs else 0.0
        r = float(np.mean(recs)) if recs else 0.0
        f1 = 2 * p * r / max(1e-9, p + r)
        avg_chunks = len(all_chunks) / max(1, len(bodies))
        avg_len = float(np.mean([len(t) for _, t in all_chunks]))
        results[method] = {
            "P": p, "R": r, "F1": f1,
            "avg_chunks": avg_chunks, "avg_len": avg_len,
        }
        print(
            f"{method:10s} P={p:.3f} R={r:.3f} F1={f1:.3f} "
            f"chunks/article={avg_chunks:.1f} avg_len={avg_len:.0f}"
        )

    # persist chunk embedding cache
    keys = np.array(list(cache.keys()))
    vecs = np.stack(list(cache.values()))
    np.savez_compressed(CHUNK_EMB_CACHE, keys=keys, vecs=vecs)
    print(f"chunk embedding cache: {len(cache)} vectors -> {CHUNK_EMB_CACHE}")

    best = max(results, key=lambda m: results[m]["F1"])
    print(f"\nwinner (F1): {best}")


if __name__ == "__main__":
    main()
