"""
Multi-signal confidence scoring for abstention decisions.

This is the heart of the project. We combine THREE independent signals
rather than trusting any single one, because each fails differently:

  1. Retrieval confidence   - is there even a good match in the KB?
  2. Score gap              - is the top match clearly better than the rest,
                              or is retrieval just guessing among similar options?
  3. Generation self-rating - does the LLM itself think it can answer,
                              based on ONLY the retrieved context?

A ticket is auto-answered only if ALL signals clear their thresholds.
Otherwise it escalates to a human, with the retrieved context attached
so the human doesn't start from zero.
"""

from dataclasses import dataclass
import config


@dataclass
class ConfidenceResult:
    should_escalate: bool
    reasons: list[str]
    retrieval_score: float
    score_gap: float
    llm_self_confidence: float


def retrieval_confidence(reranked_candidates: list[dict]) -> tuple[float, float]:
    """Returns (top_score, gap_between_top1_and_top2)."""
    if not reranked_candidates:
        return 0.0, 0.0
    top_score = reranked_candidates[0]["rerank_score"]
    if len(reranked_candidates) > 1:
        second_score = reranked_candidates[1]["rerank_score"]
        gap = top_score - second_score
    else:
        gap = top_score  # only one candidate; treat full score as the "gap"
    return top_score, gap


def evaluate_confidence(
    reranked_candidates: list[dict],
    llm_self_confidence: float,
) -> ConfidenceResult:
    reasons = []

    top_score, gap = retrieval_confidence(reranked_candidates)

    if not reranked_candidates:
        reasons.append("No retrieved chunks cleared the retrieval stage at all.")
        return ConfidenceResult(True, reasons, top_score, gap, llm_self_confidence)

    if top_score < config.MIN_RERANK_SCORE:
        reasons.append(
            f"Top rerank score {top_score:.2f} is below threshold {config.MIN_RERANK_SCORE}. "
            "Nothing in the KB closely matches this query."
        )

    if gap < config.MIN_SCORE_GAP:
        reasons.append(
            f"Score gap between top candidates is only {gap:.2f} (threshold {config.MIN_SCORE_GAP}). "
            "Retrieval is ambiguous -- several KB entries look equally (ir)relevant."
        )

    if llm_self_confidence < config.MIN_SELF_REPORTED_CONFIDENCE:
        reasons.append(
            f"Model's self-reported confidence {llm_self_confidence:.2f} is below "
            f"threshold {config.MIN_SELF_REPORTED_CONFIDENCE}."
        )

    should_escalate = len(reasons) > 0
    return ConfidenceResult(should_escalate, reasons, top_score, gap, llm_self_confidence)
