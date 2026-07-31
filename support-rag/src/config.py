"""
Central configuration for the Support Ticket Resolver RAG system.
Tune thresholds here after running eval/calibrate.py.
"""

import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KB_CSV_PATH = os.path.join(DATA_DIR, "kb_sample.csv")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "qdrant_local")
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "eval_set.json")

# --- Embedding model ---
# Small, free, local model. Swap for "BAAI/bge-base-en-v1.5" for better quality
# or an API embedding model if you prefer not to run locally.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Reranker model ---
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Retrieval settings ---
TOP_K_RETRIEVE = 10       # candidates pulled from hybrid search before reranking
TOP_K_RERANK = 4          # candidates kept after reranking, passed to the LLM
BM25_WEIGHT = 0.4         # weight given to keyword search in fusion (0-1)
DENSE_WEIGHT = 0.6        # weight given to vector search in fusion (0-1)

# --- Confidence / abstention thresholds ---
# Calibrated against the full 26,872-row Bitext dataset using a 102-query
# labeled eval set (eval/eval_set.json). See README "Results" section for
# the false-answer-rate / escalation-rate tradeoff behind these numbers.
# Re-run eval/calibrate.py + eval/resweep.py if you swap in a different KB.
MIN_RERANK_SCORE = 0.3           # below this, the top result isn't trustworthy
MIN_SCORE_GAP = 0.0              # disabled: on a large KB with many near-duplicate
                                  # paraphrases per intent, a small gap usually means
                                  # "several equally good matches", not ambiguity
MIN_SELF_REPORTED_CONFIDENCE = 0.6  # LLM's own confidence self-rating (0-1)

# --- LLM settings ---
# Flash-Lite has the most generous free-tier limits (higher RPM and RPD
# than plain Flash), so it's the safer choice while on the free tier.
LLM_MODEL = "gemini-flash-lite-latest"
LLM_MAX_TOKENS = 1000

# --- Escalation ---
ESCALATION_LOG_PATH = os.path.join(DATA_DIR, "escalations.jsonl")
