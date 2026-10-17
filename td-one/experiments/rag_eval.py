"""RAG evaluation: recall@k and precision@k on labeled queries.

Uses RAGTool in BM25-only mode (no Groq — deterministic, free).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.news.rag import RAGTool  # noqa: E402

LABELS_PATH = ROOT / "experiments" / "rag_eval_labels.json"
RESULTS_PATH = ROOT / "experiments" / "rag_eval_results.json"
KS = [1, 3, 5, 10]

# Stopwords to remove when converting natural language to FTS keywords.
STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "go",
    "get",
    "rise",
    "fall",
    "drop",
}


def to_fts_query(natural: str) -> str:
    """Convert natural language to FTS5 keyword query.

    Keeps only the 2-3 most significant content words. FTS5 ANDs them,
    so we must be conservative to avoid zero-result queries.
    """
    import re

    # Topic keywords that are highly discriminative in this KB
    TOPIC_TERMS = {
        "bank",
        "canada",
        "mortgage",
        "housing",
        "inflation",
        "rate",
        "rates",
        "policy",
        "market",
        "price",
        "prices",
        "home",
        "cpi",
    }
    words = re.findall(r'"[^"]+"|\S+', natural.lower())
    keywords = []
    for w in words:
        if w.startswith('"'):
            keywords.append(w)
        else:
            w = re.sub(r"[^a-z0-9]", "", w)
            if w in TOPIC_TERMS:
                keywords.append(w)
    # Deduplicate while preserving order, keep max 3
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return " ".join(unique[:3]) if unique else natural


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def main() -> int:
    labels = json.loads(LABELS_PATH.read_text())
    print(f"[rag_eval] {len(labels)} queries", flush=True)

    tool = RAGTool(groq=None)  # BM25-only: deterministic, no API cost
    per_query = []
    agg: dict[int, dict[str, float]] = {k: {"recall": 0.0, "precision": 0.0} for k in KS}

    for i, lab in enumerate(labels):
        query = lab["query"]
        before = lab["before"]
        relevant = set(lab["relevant_urls"])
        # Convert natural language to FTS keywords (what an agent should do)
        fts_query = to_fts_query(query)
        try:
            res = tool.search(fts_query, before, limit=10)
            retrieved = [a.url for a in res.articles]
        except Exception as exc:  # noqa: BLE001
            print(f"[rag_eval] query {i} failed: {exc}", flush=True)
            retrieved = []

        q_metrics: dict[str, float] = {}
        for k in KS:
            r = recall_at_k(retrieved, relevant, k)
            p = precision_at_k(retrieved, relevant, k)
            q_metrics[f"recall@{k}"] = round(r, 4)
            q_metrics[f"precision@{k}"] = round(p, 4)
            agg[k]["recall"] += r
            agg[k]["precision"] += p
        per_query.append(
            {
                "query": query,
                "before": before,
                "n_relevant": len(relevant),
                "n_retrieved": len(retrieved),
                **q_metrics,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[rag_eval] {i + 1}/{len(labels)}", flush=True)

    n = len(labels)
    summary = {
        k: {
            "recall": round(agg[k]["recall"] / n, 4),
            "precision": round(agg[k]["precision"] / n, 4),
        }
        for k in KS
    }
    out = {
        "n_queries": n,
        "mode": "bm25-only",
        "summary": {f"@{k}": v for k, v in summary.items()},
        "per_query": per_query,
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[rag_eval] wrote {RESULTS_PATH}", flush=True)
    for k in KS:
        print(
            f"  recall@{k}={summary[k]['recall']:.3f}  precision@{k}={summary[k]['precision']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
