"""
Streamlit UI: simulates a support ticket queue. Submit a query, see
whether it gets auto-answered (with citations) or escalated to a human
(with retrieved context pre-loaded so the agent doesn't start cold).

Run with: streamlit run src/app.py
"""

import streamlit as st
from ingest import ensure_index_built
from retrieve import HybridRetriever
from pipeline import resolve_ticket

st.set_page_config(page_title="Support Ticket Resolver", layout="wide")


@st.cache_resource
def get_retriever():
    ensure_index_built()
    return HybridRetriever()


st.title("🎫 Customer Support Ticket Resolver")
st.caption("RAG system with confidence-based abstention — auto-answers when sure, escalates when not.")

with st.spinner("Setting up (first load may take a minute)..."):
    retriever = get_retriever()

query = st.text_input("Customer query", placeholder="e.g. How do I get a refund on a damaged item?")
submit = st.button("Resolve Ticket", type="primary")

if submit and query:
    with st.spinner("Retrieving context and generating response..."):
        result = resolve_ticket(query, retriever)

    if result.status == "auto_answered":
        st.success("✅ Auto-answered")
        st.markdown(f"**Answer:** {result.answer}")
        if result.cited_sources:
            with st.expander("📎 Cited sources"):
                for s in result.cited_sources:
                    st.write(f"- {s}")
    else:
        st.warning("🚨 Escalated to human agent")
        st.markdown("**Why it was escalated:**")
        for r in result.escalation_reasons:
            st.write(f"- {r}")
        st.markdown("**Context handed to the human agent (so they don't start cold):**")

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval score", f"{result.retrieval_score:.2f}")
    col2.metric("Score gap", f"{result.score_gap:.2f}")
    col3.metric("LLM self-confidence", f"{result.llm_self_confidence:.2f}")

st.sidebar.header("About")
st.sidebar.write(
    "This demo combines hybrid retrieval (dense + BM25), cross-encoder "
    "reranking, and a multi-signal confidence scorer to decide whether "
    "to auto-answer a ticket or escalate it to a human agent."
)
st.sidebar.write("Run `python eval/calibrate.py` to see the false-answer / "
                  "escalation-rate tradeoff on the labeled eval set.")