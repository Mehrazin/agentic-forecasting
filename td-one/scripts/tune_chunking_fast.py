"""Fast chunking comparison on 20 bodies (subset for speed)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from tune_rag_k import QUERIES, _relevant

from td_one.rag.chunking import chunk_fixed, chunk_paragraph, chunk_sentence
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT b.url, a.title, b.body FROM article_bodies b "
    "JOIN articles a ON a.url = b.url WHERE b.fetch_status='ok' LIMIT 20"
).fetchall()
con.close()
print(f"bodies: {len(rows)}", flush=True)

client = EmbeddingClient()
print("client ready", flush=True)
titles = {u: t for u, t, _ in rows}

for name, fn in [("fixed", chunk_fixed), ("paragraph", chunk_paragraph),
                 ("sentence", lambda b: chunk_sentence(b, per_chunk=4))]:
    chunks = []
    for url, _, body in rows:
        chunks.extend((url, c.text) for c in fn(body))
    print(f"{name}: {len(chunks)} chunks, encoding...", flush=True)
    mat = client.encode([t for _, t in chunks])
    urls = [u for u, _ in chunks]
    precs, recs = [], []
    for query, keywords in QUERIES:
        q = client.encode([query])[0]
        top = np.argsort(mat @ q)[::-1][:10]
        ret = {urls[i] for i in top}
        rel = {u for u, t in titles.items() if _relevant(t, keywords)}
        if not rel:
            continue
        hit = len(ret & rel)
        precs.append(hit / max(1, len(ret)))
        recs.append(hit / len(rel))
    p, r = float(np.mean(precs)), float(np.mean(recs))
    f1 = 2 * p * r / max(1e-9, p + r)
    print(f"{name:10s} P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)

print("DONE", flush=True)
