"""Tune RAG top-k parameters on hand-labeled query -> article pairs.

Builds 20 query/relevance pairs from the news DB (title-as-query against
its own article = positive; articles from unrelated topics = negatives),
then grid-searches (semantic_k, rerank_k) reporting precision@k / recall@k
per stage. Run:

    .venv/bin/python scripts/tune_rag_k.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from td_one.rag.semantic import EmbeddingClient, SemanticNewsIndex

# Hand-labeled: (query, keywords that mark a relevant article)
QUERIES: list[tuple[str, list[str]]] = [
    ("Ontario first-time home buyers", ["ontario", "first-time", "buyer"]),
    ("Bank of Canada interest rate decision", ["bank of canada", "rate"]),
    ("Alberta oil jobs mortgage", ["alberta", "oil"]),
    ("Toronto housing market prices", ["toronto", "housing"]),
    ("mortgage renewal rates Canada", ["mortgage", "renewal", "rate"]),
    ("HELOC home equity loan rates", ["heloc", "home equity"]),
    ("Vancouver real estate", ["vancouver", "real estate"]),
    ("CMHC insured mortgage rules", ["cmhc", "insured"]),
    ("variable vs fixed mortgage", ["variable", "fixed", "mortgage"]),
    ("housing affordability crisis", ["affordability", "housing"]),
    ("mortgage stress test OSFI", ["stress test", "osfi"]),
    ("rental market rent increase", ["rental", "rent"]),
    ("condo market downtown Toronto", ["condo", "toronto"]),
    ("mortgage broker commission", ["broker", "mortgage"]),
    ("retirement mortgage payoff", ["retirement", "mortgage"]),
    ("newcomer immigrant mortgage Canada", ["immigrant", "newcomer", "mortgage"]),
    ("property tax increase city", ["property tax"]),
    ("home renovation loan", ["renovation", "loan"]),
    ("foreclosure power of sale Ontario", ["foreclosure", "power of sale"]),
    ("GIC vs mortgage prepayment", ["gic", "prepayment", "mortgage"]),
]


def _relevant(title: str, keywords: list[str]) -> bool:
    tl = title.lower()
    return sum(kw in tl for kw in keywords) >= 2


def main() -> None:
    db = (
        Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
    )
    idx = SemanticNewsIndex(db_path=db)
    assert idx._emb is not None

    print(f"{'k':>4} {'P@k':>6} {'R@k':>6} {'F1':>6}")
    best = None
    for k in [5, 10, 20, 50, 100]:
        ps, rs = [], []
        for query, kws in QUERIES:
            hits = idx.search(query, before="2027-01-01", top_k=k)
            # relevance judged against the full DB (approximated by top-200)
            pool = idx.search(query, before="2027-01-01", top_k=200)
            n_rel = sum(1 for h in pool if _relevant(h.title, kws))
            if n_rel == 0:
                continue
            n_hit_rel = sum(1 for h in hits if _relevant(h.title, kws))
            ps.append(n_hit_rel / len(hits) if hits else 0.0)
            rs.append(n_hit_rel / n_rel)
        p = float(np.mean(ps)) if ps else 0.0
        r = float(np.mean(rs)) if rs else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{k:>4} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
        if best is None or f1 > best[1]:
            best = (k, f1)
    print(f"\nBest F1 at k={best[0]} (F1={best[1]:.3f})")


if __name__ == "__main__":
    main()
