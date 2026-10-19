"""Chunking comparison via title-as-query retrieval (20 bodies).

For each body, query with its own title. Measure: is the source article's
chunk in the top-k? Reports MRR and hit@5 per chunking method.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from td_one.rag.chunking import chunk_fixed, chunk_paragraph, chunk_sentence
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT b.url, a.title, b.body FROM article_bodies b "
    "JOIN articles a ON a.url = b.url WHERE b.fetch_status='ok' LIMIT 30"
).fetchall()
con.close()
print(f"bodies: {len(rows)}", flush=True)

client = EmbeddingClient()
print("client ready", flush=True)

for name, fn in [("fixed", chunk_fixed), ("paragraph", chunk_paragraph),
                 ("sentence", lambda b: chunk_sentence(b, per_chunk=4))]:
    chunks = []
    for url, _, body in rows:
        chunks.extend((url, c.text) for c in fn(body))
    mat = client.encode([t for _, t in chunks])
    urls = [u for u, _ in chunks]
    rr, hits = [], 0
    for url, title, _ in rows:
        q = client.encode([title])[0]
        sims = mat @ q
        # rank of first chunk from the source article
        ranks = np.argsort(sims)[::-1]
        rank = next(
            (i + 1 for i, j in enumerate(ranks) if urls[j] == url), 999
        )
        rr.append(1.0 / rank)
        if rank <= 5:
            hits += 1
    print(f"{name:10s} MRR={np.mean(rr):.3f} hit@5={hits}/{len(rows)} "
          f"chunks/article={len(chunks)/len(rows):.1f}", flush=True)

print("DONE", flush=True)
