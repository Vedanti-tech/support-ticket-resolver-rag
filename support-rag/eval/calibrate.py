"""
Calibration script: runs the pipeline against the labeled eval set
(answerable vs unanswerable queries) and produces a precision-recall
style report so you pick thresholds backed by data, not guesses.

Usage:
    python eval/calibrate.py

Interpretation:
  - "False answer rate" = fraction of UNANSWERABLE queries the system
    still answered instead of escalating. This is the number that matters
    most in production -- a wrong confident answer is worse than an
    unnecessary escalation.
  - "Unnecessary escalation rate" = fraction of ANSWERABLE queries the
    system escalated when it could have answered correctly. Costs human
    time but doesn't mislead the customer.

Tune config.MIN_RERANK_SCORE / MIN_SCORE_GAP / MIN_SELF_REPORTED_CONFIDENCE
and re-run until false-answer rate is near zero, without pushing
unnecessary-escalation rate too high.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from retrieve import HybridRetriever
from pipeline import resolve_ticket


def load_eval_set():
    with open(config.EVAL_SET_PATH) as f:
        return json.load(f)


def run_calibration():
    eval_set = load_eval_set()
    retriever = HybridRetriever()

    results = []
    for i, item in enumerate(eval_set):
        print(f"[{i+1}/{len(eval_set)}] Processing: {item['query'][:60]}...")
        result = resolve_ticket(item["query"], retriever)
        results.append({
            "query": item["query"],
            "truly_answerable": item["answerable"],
            "system_answered": result.status == "auto_answered",
            "retrieval_score": result.retrieval_score,
            "score_gap": result.score_gap,
            "llm_self_confidence": result.llm_self_confidence,
        })
        # Free tier allows only ~5 requests/minute, and each query makes up
        # to 2 API calls (generation + faithfulness check) = 4 calls/min at
        # this pace, safely under the limit.
        time.sleep(30)

    # false answer = system answered a question that wasn't actually answerable
    false_answers = [r for r in results if r["system_answered"] and not r["truly_answerable"]]
    # unnecessary escalation = system escalated something it could have answered
    unnecessary_escalations = [r for r in results if not r["system_answered"] and r["truly_answerable"]]

    total_answerable = sum(1 for r in results if r["truly_answerable"])
    total_unanswerable = sum(1 for r in results if not r["truly_answerable"])

    print("=" * 60)
    print("CALIBRATION REPORT")
    print("=" * 60)
    print(f"Total eval queries: {len(results)}")
    print(f"  Answerable: {total_answerable} | Unanswerable: {total_unanswerable}")
    print()
    print(f"False answer rate (CRITICAL metric): "
          f"{len(false_answers)}/{total_unanswerable} "
          f"= {len(false_answers)/max(total_unanswerable,1):.1%}")
    print(f"Unnecessary escalation rate: "
          f"{len(unnecessary_escalations)}/{total_answerable} "
          f"= {len(unnecessary_escalations)/max(total_answerable,1):.1%}")
    print()

    if false_answers:
        print("Queries that were WRONGLY auto-answered (should have escalated):")
        for r in false_answers:
            print(f"  - '{r['query']}' (rerank={r['retrieval_score']:.2f}, "
                  f"gap={r['score_gap']:.2f}, llm_conf={r['llm_self_confidence']:.2f})")
        print()

    if unnecessary_escalations:
        print("Queries that were unnecessarily escalated (could have been answered):")
        for r in unnecessary_escalations:
            print(f"  - '{r['query']}' (rerank={r['retrieval_score']:.2f}, "
                  f"gap={r['score_gap']:.2f}, llm_conf={r['llm_self_confidence']:.2f})")

    # save raw results for plotting a real precision-recall curve if desired
    out_path = os.path.join(os.path.dirname(__file__), "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out_path} -- use these scores to sweep")
    print("different thresholds in config.py without re-running the LLM.")


if __name__ == "__main__":
    run_calibration()
