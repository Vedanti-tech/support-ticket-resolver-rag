"""
Re-sweep confidence thresholds against ALREADY-COMPUTED calibration results,
without calling the Gemini API again. Saves your free-tier quota while you
tune MIN_RERANK_SCORE / MIN_SCORE_GAP / MIN_SELF_REPORTED_CONFIDENCE.

Usage:
    python eval/resweep.py

Edit the THRESHOLDS list below to try different combinations at once.
Once you find good values, copy them into src/config.py.
"""

import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "calibration_results.json")

# Try several threshold combinations at once to compare tradeoffs.
THRESHOLDS_TO_TRY = [
    {"min_rerank": 0.35, "min_gap": 0.05, "min_llm_conf": 0.6},    # original baseline
    {"min_rerank": 0.005, "min_gap": 0.002, "min_llm_conf": 0.6},  # tuned for small sample KB
    {"min_rerank": 0.3, "min_gap": 0.0, "min_llm_conf": 0.6},      # gap removed, rerank raised
    {"min_rerank": 0.5, "min_gap": 0.0, "min_llm_conf": 0.6},
    {"min_rerank": 0.3, "min_gap": 0.0, "min_llm_conf": 0.8},
]


def load_results():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Corrections discovered via eval/inspect_retrieval.py: the real KB
    # genuinely supports these (weekend hours, crypto payments) -- our
    # original eval labels assumed the small sample KB's content, which
    # didn't cover these cases. Fixing the labels here retroactively
    # avoids re-running the full (slow, rate-limited) calibration.
    CORRECTED_LABELS = {
        "Is customer support available on weekends?": True,
        "Can I pay in Bitcoin or other cryptocurrency?": True,
    }
    for r in results:
        if r["query"] in CORRECTED_LABELS:
            r["truly_answerable"] = CORRECTED_LABELS[r["query"]]

    return results


def evaluate(results, min_rerank, min_gap, min_llm_conf):
    false_answers = []
    unnecessary_escalations = []

    for r in results:
        would_answer = (
            r["retrieval_score"] >= min_rerank
            and r["score_gap"] >= min_gap
            and r["llm_self_confidence"] >= min_llm_conf
        )
        if would_answer and not r["truly_answerable"]:
            false_answers.append(r)
        if not would_answer and r["truly_answerable"]:
            unnecessary_escalations.append(r)

    total_answerable = sum(1 for r in results if r["truly_answerable"])
    total_unanswerable = sum(1 for r in results if not r["truly_answerable"])

    return {
        "false_answer_rate": len(false_answers) / max(total_unanswerable, 1),
        "false_answers": [r["query"] for r in false_answers],
        "unnecessary_escalation_rate": len(unnecessary_escalations) / max(total_answerable, 1),
        "unnecessary_escalations": [r["query"] for r in unnecessary_escalations],
    }


def main():
    results = load_results()

    for t in THRESHOLDS_TO_TRY:
        print("=" * 70)
        print(f"min_rerank={t['min_rerank']}, min_gap={t['min_gap']}, "
              f"min_llm_conf={t['min_llm_conf']}")
        print("=" * 70)
        report = evaluate(results, t["min_rerank"], t["min_gap"], t["min_llm_conf"])

        print(f"False answer rate: {report['false_answer_rate']:.1%} "
              f"({len(report['false_answers'])} queries)")
        for q in report["false_answers"]:
            print(f"    ⚠️  {q}")

        print(f"Unnecessary escalation rate: {report['unnecessary_escalation_rate']:.1%} "
              f"({len(report['unnecessary_escalations'])} queries)")
        for q in report["unnecessary_escalations"]:
            print(f"    - {q}")
        print()


if __name__ == "__main__":
    main()