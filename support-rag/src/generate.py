"""
Generation with:
  1. Grounded answer forced to cite specific retrieved chunks
  2. Structured self-reported confidence (answerable: yes/partial/no)
  3. A SEPARATE follow-up faithfulness check -- this is more reliable than
     trusting the first call's self-rating, because a model can be
     confidently wrong. The faithfulness check re-reads the answer against
     the context with a narrower, more skeptical prompt.

Uses Google's Gemini API (free tier available at https://aistudio.google.com/apikey).
"""

import json
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import config

# Loads variables from a .env file in the project root (if present) into
# the environment. If you'd rather set it manually, you still can --
# load_dotenv() won't override a variable that's already set in the shell.
load_dotenv()

# Reads GEMINI_API_KEY from environment. Set it via a .env file (recommended)
# or manually before running:
#   export GEMINI_API_KEY=your_key_here      (Mac/Linux)
#   set GEMINI_API_KEY=your_key_here         (Windows cmd)
#   $env:GEMINI_API_KEY="your_key_here"      (Windows PowerShell)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(config.LLM_MODEL)


GENERATION_SYSTEM_PROMPT = """You are a customer support assistant. You must answer
ONLY using the provided context chunks. Do not use outside knowledge.

Respond with a JSON object with these exact fields:
{
  "answerable": "yes" | "partial" | "no",
  "confidence": <float 0.0-1.0, your honest self-assessment>,
  "answer": "<your answer to the customer, or empty string if answerable is 'no'>",
  "cited_chunk_ids": [<list of chunk id integers you actually used>]
}

Rules:
- If the context does not contain enough information to answer confidently,
  set answerable to "no" and confidence low (below 0.5). Do NOT guess or
  fill gaps with general knowledge.
- If the context partially covers the question, set answerable to "partial"
  and lower your confidence accordingly.
- Only cite chunk_ids you actually drew the answer from.
- Return ONLY the JSON object, no other text, no markdown fences.
"""

FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given
a CONTEXT and an ANSWER. Determine if every claim in the ANSWER is directly
supported by the CONTEXT, with no invented details.

Respond with ONLY a JSON object:
{
  "faithful": true | false,
  "unsupported_claims": ["<any claim in the answer not backed by context>"]
}

Be strict: if the answer adds specifics (numbers, policies, timeframes) not
present in the context, mark it unfaithful.
"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id={c['id']}] {c['text']}")
    return "\n\n".join(parts)


def _call_gemini(system_prompt: str, user_prompt: str, max_tokens: int, max_retries: int = 5) -> str:
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,  # low temperature: we want consistent, grounded answers
                    response_mime_type="application/json",  # forces valid JSON output
                ),
            )
            return response.text.strip()
        except ResourceExhausted as e:
            # Free tier is limited to ~5 requests/minute. Back off and retry
            # rather than crashing the whole batch on a single rate-limit hit.
            wait_seconds = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s, 100s
            print(f"  Rate limit hit, waiting {wait_seconds}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini API rate limit still exceeded after {max_retries} retries. "
        "Free tier allows only ~5 requests/minute -- wait a minute and try again, "
        "or reduce how many queries you run at once."
    )


def generate_answer(query: str, chunks: list[dict]) -> dict:
    context = _format_context(chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION:\n{query}"

    raw_text = _call_gemini(GENERATION_SYSTEM_PROMPT, user_prompt, config.LLM_MAX_TOKENS)
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # fail safe: if the model didn't return valid JSON, treat as low confidence
        parsed = {
            "answerable": "no",
            "confidence": 0.0,
            "answer": "",
            "cited_chunk_ids": [],
        }
    return parsed


def check_faithfulness(answer: str, chunks: list[dict]) -> dict:
    if not answer:
        return {"faithful": True, "unsupported_claims": []}

    context = _format_context(chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    raw_text = _call_gemini(FAITHFULNESS_SYSTEM_PROMPT, user_prompt, 500)
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {"faithful": False, "unsupported_claims": ["Could not parse faithfulness check."]}
    return parsed
