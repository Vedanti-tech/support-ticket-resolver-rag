"""
Ingestion pipeline: load the KB CSV, treat each Q&A pair as a chunk
(no fixed-size splitting needed since responses are already short and
self-contained -- a good real-world lesson: chunk strategy depends on
your source structure, don't cargo-cult fixed-size chunking everywhere),
embed each chunk, and index into Qdrant (dense) + BM25 (sparse).

Run this once (or whenever the KB changes) before querying.
"""

import pandas as pd
import pickle
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import config


def load_kb(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["instruction", "response"])
    return df.reset_index(drop=True)


def build_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Each chunk = one canonical Q&A pair. We embed a combination of the
    instruction + response so retrieval matches both on user phrasing
    (instruction) and on content (response).
    """
    chunks = []
    for i, row in df.iterrows():
        chunk_text = f"Category: {row['category']}\nIntent: {row['intent']}\nExample question: {row['instruction']}\nAnswer: {row['response']}"
        chunks.append({
            "id": i,
            "text": chunk_text,
            "instruction": row["instruction"],
            "response": row["response"],
            "category": row["category"],
            "intent": row["intent"],
        })
    return chunks


def build_dense_index(chunks: list[dict], model: SentenceTransformer):
    client = QdrantClient(path=config.VECTOR_DB_PATH)
    collection = "support_kb"

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        client.delete_collection(collection)

    vector_size = model.get_sentence_embedding_dimension()
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    points = [
        PointStruct(id=c["id"], vector=emb.tolist(), payload=c)
        for c, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant collection '{collection}'.")
    return client


def build_bm25_index(chunks: list[dict]):
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_path = os.path.join(config.DATA_DIR, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Saved BM25 index to {bm25_path}.")


def main():
    print("Loading KB...")
    df = load_kb(config.KB_CSV_PATH)
    print(f"Loaded {len(df)} KB rows.")

    chunks = build_chunks(df)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Building dense (vector) index...")
    build_dense_index(chunks, model)

    print("Building sparse (BM25) index...")
    build_bm25_index(chunks)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
