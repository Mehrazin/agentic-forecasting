"""Full-article vs chunk-only context: keyword coverage + compression.

Fast version: 30 bodies, title-as-query, top-5 chunks.
Metric: does the context contain the title's key terms?
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from td_one.rag.chunking import chunk_fixed
from td_one.rag.semantic import EmbeddingClient

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
STOP = set(["the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been", "by", "with", "from", "as"])


def keywords(title: str) -> list[str]:
    words = re.findall(r"[a-z]{4,}", title.lower())
    return [w for w in words if w not in STOP][:6]


con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT b.url, a.title, b.body FROM article_bodies b "
    "JOIN articles a ON a.url = b.url WHERE b.fetch_status='ok' LIMIT 30"
).fetchall()
con.close()

client = EmbeddingClient()
chunks = []
for url, _, body in rows:
    chunks.extend((url, c.text) for c in chunk_fixed(body))
mat = client.encode([t for _, t in chunks])
urls = [u for u, _ in chunks]
bodies = {u: b for u, _, b in rows}

cov_f, cov_c, len_f, len_c = [], [], [], []
for url, title, _ in rows:
    kws = keywords(title)
    if not kws:
        continue
    q = client.encode([title])[0]
    top = np.argsort(mat @ q)[::-1][:5]
    src = list(dict.fromkeys(urls[i] for i in top))  # dedupe
    full_ctx = "\n".join(bodies[u] for u in src).lower()
    chunk_ctx = "\n".join(chunks[i][1] for i in top).lower()
    cov_f.append(sum(k in full_ctx for k in kws) / len(kws))
    cov_c.append(sum(k in chunk_ctx for k in kws) / len(kws))
    len_f.append(len(full_ctx))
    len_c.append(len(chunk_ctx))

print(f"full-article: coverage={np.mean(cov_f):.3f} chars={np.mean(len_f):.0f}")
print(f"chunk-only:   coverage={np.mean(cov_c):.3f} chars={np.mean(len_c):.0f}")
print(f"compression: {np.mean(len_f)/max(1,np.mean(len_c)):.1f}x")
print("DONE", flush=True)
