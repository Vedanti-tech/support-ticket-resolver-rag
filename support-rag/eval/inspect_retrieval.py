"""
Debug helper: shows exactly what context was retrieved for a query,
WITHOUT calling the Gemini API (so it costs nothing / no rate limit).

Use this to investigate why a query was answered incorrectly or
escalated unnecessarily -- you can see exactly what the LLM saw.

Usage:
    python eval/inspect_retrieval.py "your query here"
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieve import HybridRetriever


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/inspect_retrieval.py \"your query here\"")
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    retriever = HybridRetriever()
    results = retriever.search(query)

    for i, chunk in enumerate(results):
        print("=" * 70)
        print(f"Rank {i+1} | rerank_score={chunk['rerank_score']:.4f}")
        print("=" * 70)
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
