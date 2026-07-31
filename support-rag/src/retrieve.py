"""
Hybrid retrieval: dense vector search + BM25 keyword search, combined
via weighted score fusion, then refined with a cross-encoder reranker.

Why hybrid: pure vector search often misses exact terms customers use
verbatim (order numbers, product names, "2FA", "refund" vs "return").
BM25 catches those; dense search catches paraphrases and synonyms.
"""

import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

import config


class HybridRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

        print("Loading reranker model...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

        print("Connecting to Qdrant...")
        self.qdrant = QdrantClient(path=config.VECTOR_DB_PATH)
        self.collection = "support_kb"

        bm25_path = f"{config.DATA_DIR}/bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.chunks = bm25_data["chunks"]

    def _dense_search(self, query: str, top_k: int):
        query_vec = self.embed_model.encode(query, normalize_embeddings=True)
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
        )
        return {r.payload["id"]: r.score for r in results}

    def _bm25_search(self, query: str, top_k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # normalize BM25 scores to 0-1 range for fair fusion with cosine sim
        max_score = max(scores) if max(scores) > 0 else 1.0
        scored = [(i, s / max_score) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return dict(scored[:top_k])

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or config.TOP_K_RETRIEVE

        dense_scores = self._dense_search(query, top_k)
        bm25_scores = self._bm25_search(query, top_k)

        # weighted fusion over the union of both result sets
        all_ids = set(dense_scores) | set(bm25_scores)
        fused = {}
        for cid in all_ids:
            d_score = dense_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)
            fused[cid] = (config.DENSE_WEIGHT * d_score) + (config.BM25_WEIGHT * b_score)

        ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)[:top_k]
        candidates = [self.chunks[cid] for cid in ranked_ids]
        return candidates

    def rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or config.TOP_K_RERANK
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]

    def search(self, query: str) -> list[dict]:
        """Full pipeline: hybrid retrieve -> rerank."""
        candidates = self.retrieve(query)
        return self.rerank(query, candidates)
