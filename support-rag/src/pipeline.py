"""
End-to-end pipeline for a single support ticket:

  query -> hybrid retrieve -> rerank -> generate (with self-confidence)
        -> confidence check -> [faithfulness check if passing so far]
        -> final decision: AUTO-ANSWERED or ESCALATED

Escalated tickets are logged with the retrieved context attached, so a
human agent picks up with useful groundwork already done instead of
starting cold.
"""

import json
import time
from dataclasses import dataclass, asdict

import config
from retrieve import HybridRetriever
from generate import generate_answer, check_faithfulness
from confidence import evaluate_confidence


@dataclass
class TicketResult:
    query: str
    status: str  # "auto_answered" | "escalated"
    answer: str
    cited_sources: list[str]
    retrieval_score: float
    score_gap: float
    llm_self_confidence: float
    faithful: bool
    escalation_reasons: list[str]


def log_escalation(result: TicketResult, chunks: list[dict]):
    entry = {
        "timestamp": time.time(),
        "query": result.query,
        "reasons": result.escalation_reasons,
        "retrieved_context": [
            {"instruction": c["instruction"], "response": c["response"], "rerank_score": c.get("rerank_score")}
            for c in chunks
        ],
    }
    with open(config.ESCALATION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def resolve_ticket(query: str, retriever: HybridRetriever) -> TicketResult:
    # 1. Hybrid retrieval + rerank
    reranked = retriever.search(query)

    # 2. Generation (grounded, with self-reported confidence)
    generation = generate_answer(query, reranked)
    llm_confidence = generation.get("confidence", 0.0)

    # 3. Multi-signal confidence check (retrieval-based, BEFORE trusting the answer)
    conf_result = evaluate_confidence(reranked, llm_confidence)

    # 4. If retrieval/generation confidence already fails, escalate immediately
    #    without even bothering with a faithfulness check.
    if conf_result.should_escalate or generation.get("answerable") == "no":
        reasons = conf_result.reasons
        if generation.get("answerable") == "no":
            reasons = reasons + ["Model itself reported the context as insufficient."]
        result = TicketResult(
            query=query,
            status="escalated",
            answer="",
            cited_sources=[],
            retrieval_score=conf_result.retrieval_score,
            score_gap=conf_result.score_gap,
            llm_self_confidence=llm_confidence,
            faithful=False,
            escalation_reasons=reasons,
        )
        log_escalation(result, reranked)
        return result

    # 5. Faithfulness check -- catch hallucination even when confidence looked fine
    faithfulness = check_faithfulness(generation["answer"], reranked)
    if not faithfulness.get("faithful", False):
        reasons = [f"Faithfulness check failed: {faithfulness.get('unsupported_claims')}"]
        result = TicketResult(
            query=query,
            status="escalated",
            answer="",
            cited_sources=[],
            retrieval_score=conf_result.retrieval_score,
            score_gap=conf_result.score_gap,
            llm_self_confidence=llm_confidence,
            faithful=False,
            escalation_reasons=reasons,
        )
        log_escalation(result, reranked)
        return result

    # 6. All checks passed -- auto-answer
    cited_ids = set(generation.get("cited_chunk_ids", []))
    cited_sources = [c["instruction"] for c in reranked if c["id"] in cited_ids]

    return TicketResult(
        query=query,
        status="auto_answered",
        answer=generation["answer"],
        cited_sources=cited_sources,
        retrieval_score=conf_result.retrieval_score,
        score_gap=conf_result.score_gap,
        llm_self_confidence=llm_confidence,
        faithful=True,
        escalation_reasons=[],
    )


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_queries = [
        "How do I delete my account?",
        "Can I pay using cryptocurrency?",
    ]
    for q in test_queries:
        result = resolve_ticket(q, retriever)
        print("=" * 60)
        print(json.dumps(asdict(result), indent=2))
