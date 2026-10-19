"""Quick paragraph vs fixed comparison (subset of tune_chunking)."""

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

from td_one.rag.chunking import chunk_paragraph
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT b.url, a.title, b.body FROM article_bodies b "
    "JOIN articles a ON a.url = b.url WHERE b.fetch_status='ok' LIMIT 100"
).fetchall()
con.close()
print(f"bodies: {len(rows)}", flush=True)

client = EmbeddingClient()
print("client ready", flush=True)

all_chunks = []
for url, _, body in rows:
    for c in chunk_paragraph(body):
        all_chunks.append((url, c.text))
print(f"paragraph chunks: {len(all_chunks)}", flush=True)

texts = [t for _, t in all_chunks]
vecs = client.encode(texts)
print("encoded", flush=True)
mat = vecs
urls = [u for u, _ in all_chunks]
titles = {u: t for u, t, _ in rows}

precs, recs = [], []
for query, keywords in QUERIES:
    q = client.encode([query])[0]
    top = np.argsort(mat @ q)[::-1][:10]
    retrieved = {urls[i] for i in top}
    rel = {u for u, t in titles.items() if _relevant(t, keywords)}
    if not rel:
        continue
    hit = len(retrieved & rel)
    precs.append(hit / max(1, len(retrieved)))
    recs.append(hit / len(rel))
p, r = float(np.mean(precs)), float(np.mean(recs))
f1 = 2 * p * r / max(1e-9, p + r)
print(f"paragraph  P={p:.3f} R={r:.3f} F1={f1:.3f} "
      f"chunks/article={len(all_chunks)/len(rows):.1f}", flush=True)
