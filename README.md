# Customer Support Ticket Resolver — RAG with Confidence-Based Abstention

A RAG system that auto-answers customer support tickets when it's confident,
and escalates to a human agent (with context pre-loaded) when it's not.
The core engineering challenge: **calibrating when to say "I don't know"**
instead of hallucinating a plausible-sounding but wrong answer.

Built and calibrated against the real **Bitext Customer Support LLM Chatbot
Training Dataset** (26,872 rows).

## Why this matters

Most RAG tutorials stop at "retrieve chunks, stuff into prompt, generate."
In production support systems, a single confidently-wrong answer costs more
trust than ten honest escalations. This project treats abstention as a
first-class, measurable, calibrated decision — not an afterthought.

## Architecture
Query → Hybrid Retrieval (dense + BM25) → Cross-Encoder Rerank
→ Grounded Generation (self-reports confidence)
→ Multi-Signal Confidence Check (retrieval score, score gap, LLM confidence)
→ [if passing] Faithfulness Check (separate LLM call, skeptical re-read)
→ AUTO-ANSWERED (with citations) or ESCALATED (with context attached)

## Results

Calibrated against a hand-labeled set of 102 queries (72 answerable from
the KB, 30 genuinely out-of-scope) run against the full 26,872-row dataset:

| Threshold config | False answer rate | Unnecessary escalation rate |
|---|---|---|
| Untuned baseline (`min_rerank=0.35, min_gap=0.05`) | 0.0% | 71.6% |
| Tuned (`min_rerank=0.3, min_gap=0.0, min_llm_conf=0.6`) | **0.0%** | **32.4%** |

**False answer rate stayed at 0% throughout tuning** — the priority was
never trading correctness for coverage, only reducing unnecessary
escalations without introducing hallucinated answers.

## Key lessons from calibrating against real data

Moving from a small hand-written KB sample to the full dataset surfaced
issues a toy dataset never would have:

- **The score-gap signal breaks down on large, redundant KBs.** With
  thousands of paraphrased examples per intent, the top-2 retrieved chunks
  are often both correct answers to the same question, so the gap between
  them is near zero — not because retrieval is confused, but because it
  found multiple equally good matches. On a large KB, a low score gap is
  not evidence of ambiguity the way it is on a small one. Removing the
  gap requirement (`min_gap=0.0`) and relying on the raw rerank score
  plus LLM self-confidence more than doubled the "safely answerable"
  auto-answer rate.
- **Eval set labels need to be checked against the real KB, not assumed.**
  Two queries in the eval set ("Is customer support available on
  weekends?" and "Can I pay in Bitcoin?") were manually labeled
  "unanswerable" based on what the small sample KB covered. Once ingested
  against the real dataset, both turned out to have direct, correct
  answers in the KB — the system was right, the eval label was wrong.
  Ground truth for a RAG eval set is only as good as your knowledge of
  what's actually in the KB, which is easy to get wrong by hand at scale.
  `eval/inspect_retrieval.py` (no API calls needed) was built specifically
  to audit these cases against real retrieved content instead of guessing.
- **Real-world knowledge bases contain unresolved template placeholders.**
  Several KB entries in the raw dataset contain unfilled template tags like
  `{{Customer Support Hours}}` instead of real values — an artifact of how
  the dataset was generated. The reranker consistently favored fully-worded
  entries over templated ones when both existed, which avoided the obvious
  failure mode (parroting a literal `{{...}}` tag back to a customer), but
  this is a good reminder that ingested KB content needs a data-quality
  pass in a real deployment.
- **Borderline cases are real, not bugs.** A query like "Can I pay in
  Bitcoin?" scored 0.28 against a 0.3 threshold and was escalated, even
  though a more explicit phrasing of the same question ("...or other
  cryptocurrency?") scored 0.76 and was answered correctly. This is an
  inherent precision/recall trade-off of any threshold-based system, not
  a defect — the alternative (lowering the threshold further) would risk
  false answers elsewhere, which is the one thing calibration was
  explicitly protecting against.

## Project structure

support-rag/
├── data/
│ ├── kb_sample.csv # small sample KB (Bitext schema) -- runs out of the box
│ └── bitext_full.csv # (gitignored) full 27K-row dataset -- see setup below
├── eval/
│ ├── eval_set.json # 102 labeled answerable/unanswerable queries
│ ├── calibrate.py # runs the full pipeline against eval_set.json
│ ├── resweep.py # re-tests threshold combos against saved results, no API calls
│ └── inspect_retrieval.py # debug tool: see retrieved context for any query, no API calls
├── src/
│ ├── config.py # all tunable thresholds and settings
│ ├── ingest.py # builds vector + BM25 indexes from KB
│ ├── retrieve.py # hybrid search + reranking
│ ├── confidence.py # the core abstention logic
│ ├── generate.py # grounded generation + faithfulness check (Gemini API)
│ ├── pipeline.py # orchestrates the full ticket resolution flow
│ └── app.py # Streamlit demo UI
└── requirements.txt

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini API key:
GEMINI_API_KEY=your_key_here

Get a free key at https://aistudio.google.com/apikey. This project uses
`gemini-flash-lite-latest` (the alias with the most generous free-tier
limits at time of writing) — change `LLM_MODEL` in `src/config.py` if
you want a different model.

> ⚠️ Gemini's free tier is rate-limited (a handful of requests/minute) and
> model availability changes frequently — several model names used during
> this project's development were deprecated mid-build. Using the
> `-latest` alias forms (e.g. `gemini-flash-lite-latest`) instead of
> pinned version numbers is more resilient to this.

### 1. Get the real dataset (recommended)

The included `data/kb_sample.csv` is a small hand-written sample (26 rows)
so the pipeline runs out of the box. For real calibration, download the
full dataset (26,872 rows) and point config at it:

https://github.com/bitext/customer-support-llm-chatbot-training-dataset/raw/refs/heads/main/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv

Save it as `data/bitext_full.csv`, then in `src/config.py` change:
```python
KB_CSV_PATH = os.path.join(DATA_DIR, "bitext_full.csv")
```

### 2. Build the indexes

```bash
cd src
python ingest.py
```

Embeds every KB entry and builds a local Qdrant vector index + BM25
keyword index. Takes a few minutes on the small sample; ~25-30 minutes
on the full 27K-row dataset on CPU.

### 3. Calibrate the abstention thresholds (do this before demoing!)

```bash
python eval/calibrate.py
```

Runs the full pipeline against every query in `eval/eval_set.json` and
reports false-answer rate and unnecessary-escalation rate. This calls the
LLM API for every query — on the free tier, budget ~30-45 minutes for 100+
queries due to rate limiting.

Once you have `eval/calibration_results.json`, use `eval/resweep.py` to
test different threshold combinations **without re-calling the API** —
edit the `THRESHOLDS_TO_TRY` list and re-run instantly. Once you land on
good values, copy them into `src/config.py`.

Use `eval/inspect_retrieval.py "your query"` to see exactly what context
was retrieved for any query (also no API call) — essential for debugging
why something was mis-classified before assuming it's a system bug rather
than a mislabeled eval example.

### 4. Run the demo

```bash
streamlit run src/app.py
```

## Extending this further

- Plot an actual precision-recall curve from `eval/calibration_results.json`
  instead of a threshold table.
- Add a real ticket queue backend (e.g. a small FastAPI service) so
  escalations land in an actual agent dashboard instead of a JSONL file.
- Try replacing the LLM self-confidence signal with token-level logprobs
  if your provider exposes them — often more reliable than asking the
  model to self-rate.
- Add category/intent metadata filters to retrieval (e.g. classify the
  query's category first, then restrict search to that category) —
  could help with the "many equally-good paraphrases" issue described
  above without discarding the score-gap signal entirely.
- Add a data-quality pass over the KB to strip or flag unresolved
  template placeholders before indexing.