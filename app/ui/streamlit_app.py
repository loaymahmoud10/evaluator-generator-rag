"""Streamlit interface for the Evaluator-Generator RAG platform.

Run with::

    streamlit run run_ui.py

The UI lets users:
  * upload PDF / DOCX / TXT / source-code / PPTX / WAV files
  * provide web-page URLs and Wikipedia queries
  * ingest them into the unified external-knowledge layer
  * ask questions and watch the iterative Generator -> Evaluator loop
  * inspect each agent's *isolated* memory and the Redis cache state
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Allow `streamlit run app/ui/streamlit_app.py` to work as well as run_ui.py.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.config import settings
from app.core import build_system
from app.ingestion.document_loader import CODE_EXTENSIONS
from app.ingestion.pipeline import IngestionError, IngestRequest, SourceType
from app.utils.logging import get_logger

logger = get_logger("ui")

UPLOAD_EXTENSIONS = sorted(
    {"pdf", "docx", "txt", "pptx", "wav"}
    | {ext.lstrip(".") for ext in CODE_EXTENSIONS}
)

CRITERIA_LABELS = {
    "accuracy": "Accuracy",
    "relevance": "Relevance",
    "completeness": "Completeness",
    "consistency": "Consistency",
    "grounding": "Grounding",
    "unsupported_claims": "No unsupported claims",
    "overall_quality": "Overall quality",
}


# --------------------------------------------------------------------------- #
# System bootstrap
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Starting the knowledge platform…")
def get_system():
    """Build (and cache) the fully wired platform instance."""
    return build_system()


def _save_uploaded(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def _init_state() -> None:
    st.session_state.setdefault("ingested", [])
    st.session_state.setdefault("history", [])


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(system) -> None:
    st.sidebar.title("⚙️ System status")

    redis_ok = system.redis_cache.enabled
    st.sidebar.markdown(
        f"**LLM** · `{settings.GROQ_MODEL}`  \n"
        f"**API key** · {'🟢 set' if settings.has_groq else '🔴 missing'}  \n"
        f"**Redis cache** · {'🟢 connected' if redis_ok else '🟠 disabled'}  \n"
        f"**Vector store** · `{system.vector_store.collection_name}`  \n"
        f"**KB version** · `v{system.kb_version.current()}`  \n"
        f"**Max iterations** · `{settings.MAX_ITERATIONS}`  \n"
        f"**Top-K retrieval** · `{settings.TOP_K}`"
    )

    if not settings.has_groq:
        st.sidebar.error("Set GROQ_API_KEY in your .env file.")
    if not redis_ok:
        st.sidebar.info(
            "Redis is unreachable — the platform still works, "
            "just without caching."
        )

    st.sidebar.divider()
    st.sidebar.subheader("Maintenance")

    if st.sidebar.button("🧹 Clear knowledge base", use_container_width=True):
        try:
            system.vector_store._store.delete_collection()
            system.kb_version.bump()
            get_system.clear()
            st.session_state["ingested"] = []
            st.sidebar.success("Knowledge base cleared.")
            st.rerun()
        except Exception as exc:  # environment dependent
            st.sidebar.error(f"Could not clear: {exc}")

    if st.sidebar.button("♻️ Clear Redis cache", use_container_width=True):
        system.redis_cache.clear_all()
        st.sidebar.success("Redis cache cleared.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Architecture: ingestion → vector store → Generator ⇄ Evaluator loop "
        "(max 4 iterations), isolated memories, Redis cache."
    )


# --------------------------------------------------------------------------- #
# Tab 1 — Knowledge
# --------------------------------------------------------------------------- #
def render_ingestion(system) -> None:
    st.subheader("📥 Add external knowledge")
    st.caption(
        "Every source — documents, code, web pages, Wikipedia, audio — is "
        "loaded, extracted, chunked, embedded and stored in one unified "
        "knowledge layer."
    )

    col_files, col_web = st.columns(2)

    with col_files:
        uploaded_files = st.file_uploader(
            "Files (PDF, DOCX, TXT, PPTX, source code, WAV)",
            type=UPLOAD_EXTENSIONS,
            accept_multiple_files=True,
        )

    with col_web:
        urls = st.text_area(
            "Web page URLs (one per line)",
            placeholder="https://example.com/article",
            height=110,
        )
        wiki = st.text_input(
            "Wikipedia pages (comma separated)",
            placeholder="Large language model, Retrieval-augmented generation",
        )

    if st.button("🚀 Ingest sources", type="primary"):
        requests: list[IngestRequest] = []
        labels: list[str] = []

        for uf in uploaded_files or []:
            requests.append(IngestRequest(source=_save_uploaded(uf)))
            labels.append(uf.name)

        for line in (urls or "").splitlines():
            line = line.strip()
            if line:
                requests.append(
                    IngestRequest(source=line, source_type=SourceType.URL)
                )
                labels.append(line)

        for query in (wiki or "").split(","):
            query = query.strip()
            if query:
                requests.append(
                    IngestRequest(
                        source=f"wikipedia:{query}",
                        source_type=SourceType.WIKIPEDIA,
                    )
                )
                labels.append(f"wikipedia:{query}")

        if not requests:
            st.warning("Provide at least one source before ingesting.")
            return

        with st.spinner(f"Ingesting {len(requests)} source(s)…"):
            try:
                total = system.pipeline.ingest(requests)
                st.session_state["ingested"].extend(labels)
                st.success(
                    f"Ingested **{total} chunks** from {len(requests)} source(s). "
                    f"Knowledge base is now v{system.kb_version.current()}."
                )
            except IngestionError as exc:
                st.session_state["ingested"].extend(labels)
                st.warning(
                    f"Stored {exc.stored_chunks} chunks, but some sources failed:"
                )
                for err in exc.errors:
                    st.error(err)
            except Exception as exc:
                logger.exception("Ingestion failed from UI")
                st.error(f"Ingestion failed: {exc}")

    ingested = st.session_state.get("ingested", [])
    st.divider()
    st.markdown(f"**Sources ingested this session:** {len(ingested)}")
    if ingested:
        for item in ingested:
            st.markdown(f"- `{item}`")
    else:
        st.info("No sources ingested yet.")


# --------------------------------------------------------------------------- #
# Tab 2 — Ask
# --------------------------------------------------------------------------- #
def _render_result(result: dict, elapsed: float) -> None:
    validated = result["validated"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "Approved" if validated else "Unvalidated")
    c2.metric("Iterations", f"{result['iterations']}/{settings.MAX_ITERATIONS}")
    evaluations = result.get("evaluations") or []
    last_score = evaluations[-1].get("score", "—") if evaluations else "—"
    c3.metric("Final score", last_score)
    c4.metric("Time", f"{elapsed:.1f}s")

    if result.get("reason") == "cache_hit":
        st.info("⚡ Served from the Redis answer cache.")
    elif validated:
        st.success("✅ The Evaluator approved this answer.")
    else:
        st.warning(
            "⚠️ Maximum iterations reached — returning the latest answer, "
            "which the Evaluator could not fully validate."
        )

    st.markdown("### Final answer")
    st.markdown(result["final_answer"])

    st.markdown("### 🔁 Generator ⇄ Evaluator trace")
    if not evaluations:
        st.caption("No new iterations were run (cached result).")

    history = result.get("generation_history") or []
    for idx, ev in enumerate(evaluations, start=1):
        decision = ev.get("decision", "n/a")
        decision = getattr(decision, "value", decision)
        icon = "✅" if decision == "ACCEPT" else "🔁"
        with st.expander(
            f"{icon} Iteration {idx} — {decision} (score {ev.get('score', 'n/a')})"
        ):
            if idx <= len(history):
                st.markdown("**Generator answer**")
                st.markdown(history[idx - 1].get("answer", ""))

            criteria = ev.get("criteria", {})
            if criteria:
                st.markdown("**Evaluator criteria**")
                for key, value in criteria.items():
                    st.progress(
                        min(max(int(value), 0), 100) / 100,
                        text=f"{CRITERIA_LABELS.get(key, key)}: {value}/100",
                    )

            feedback = ev.get("feedback", [])
            if feedback:
                st.markdown("**Feedback sent back to the Generator**")
                for item in feedback:
                    st.markdown(
                        f"- **[{item.get('severity')}] {item.get('criterion')}** — "
                        f"{item.get('issue')}  \n  ↳ *{item.get('suggestion')}*"
                    )

    st.markdown("### 📚 Sources used")
    sources = result.get("sources") or []
    if sources:
        for src in sources:
            with st.expander(
                f"{src.get('source_name') or 'unknown'} "
                f"· {src.get('source_type')} · {src.get('location')}"
            ):
                st.text((src.get("content") or "")[:1500])
    else:
        st.caption("No context was retrieved for this question.")


def render_qa(system) -> None:
    st.subheader("💬 Ask a question")

    if not st.session_state.get("ingested"):
        st.info(
            "Nothing ingested in this session. Questions will still run against "
            "any previously stored knowledge."
        )

    question = st.text_input(
        "Your question",
        placeholder="What do the provided sources say about …?",
    )

    if st.button("🔍 Answer", type="primary") and question.strip():
        started = time.perf_counter()
        with st.spinner("Running the Generator ⇄ Evaluator loop…"):
            try:
                result = system.workflow.run(question)
            except Exception as exc:
                logger.exception("Workflow failed from UI")
                st.error(f"Workflow failed: {exc}")
                return
        elapsed = time.perf_counter() - started
        st.session_state["history"].insert(0, (question, result, elapsed))

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        _render_result(history[0][1], history[0][2])

        if len(history) > 1:
            st.divider()
            st.markdown("### 🕘 Earlier questions")
            for q, res, el in history[1:]:
                with st.expander(f"{q} — {res['iterations']} iteration(s)"):
                    _render_result(res, el)


# --------------------------------------------------------------------------- #
# Tab 3 — Memories
# --------------------------------------------------------------------------- #
def render_memories(system) -> None:
    st.subheader("🧠 Independent agent memories")
    st.caption(
        "The Generator and Evaluator hold separate memory instances. Neither "
        "agent can read the other's memory — enforced in the orchestrator."
    )

    history = st.session_state.get("history", [])
    if not history:
        st.info("Ask a question first to populate the agent memories.")
        return

    question = history[0][0]
    left, right = st.columns(2)

    with left:
        st.markdown("#### Generator memory")
        attempts = system.generator_memory.get_history(question)
        st.caption(f"{len(attempts)} attempt(s) stored for the last question.")
        for attempt in attempts:
            with st.expander(f"Attempt {attempt.get('iteration')}"):
                st.markdown("**Answer**")
                st.markdown(attempt.get("answer", ""))
                if attempt.get("feedback"):
                    st.markdown("**Feedback that triggered this attempt**")
                    st.text(attempt["feedback"])
                st.markdown("**Retrieved context (truncated)**")
                st.text((attempt.get("retrieved_context") or "")[:800])

    with right:
        st.markdown("#### Evaluator memory")
        evaluations = system.evaluator_memory.get_history(question)
        st.caption(f"{len(evaluations)} evaluation(s) stored for the last question.")
        for idx, ev in enumerate(evaluations, start=1):
            decision = getattr(ev.get("decision"), "value", ev.get("decision"))
            with st.expander(f"Evaluation {idx} — {decision}"):
                st.json(
                    {
                        "decision": decision,
                        "score": ev.get("score"),
                        "criteria": ev.get("criteria", {}),
                        "feedback": ev.get("feedback", []),
                    }
                )


# --------------------------------------------------------------------------- #
# Tab 4 — Architecture
# --------------------------------------------------------------------------- #
ARCHITECTURE = """
```
 ┌──────────────────────── External knowledge ────────────────────────┐
 │ PDF · DOCX · TXT · Code · PPTX · URL · Wikipedia · WAV (Whisper)   │
 └───────────────┬────────────────────────────────────────────────────┘
                 │ load → extract → chunk → embed
                 ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  Chroma vector   │◀──────▶│   Redis cache    │
        │      store       │        │ docs·retrieval·  │
        └────────┬─────────┘        │ answers·evals    │
                 │ top-k            └──────────────────┘
                 ▼
   question ─▶ ┌───────────┐  answer  ┌───────────┐
               │ Generator │─────────▶│ Evaluator │
               └─────┬─────┘◀─────────└─────┬─────┘
                     │      feedback        │ ACCEPT → final answer
              Generator Memory        Evaluator Memory
                  (isolated)             (isolated)

            loop ≤ 4 iterations, then return latest answer
```
"""


def render_architecture(system) -> None:
    st.subheader("🏗️ Architecture & workflow")
    st.markdown(ARCHITECTURE)

    st.markdown(
        """
**How a question flows**

1. **Retrieve** – the question hits the Redis retrieval cache; on a miss the
   Chroma vector store returns the top-K chunks.
2. **Generate** – the Generator answers using *only* that context, and says the
   information is unavailable rather than inventing facts.
3. **Evaluate** – the Evaluator scores accuracy, relevance, completeness,
   consistency, grounding, unsupported claims and overall quality, then returns
   `ACCEPT` or `IMPROVE` + actionable feedback.
4. **Loop** – `IMPROVE` feeds the feedback back to the Generator. The loop is
   hard-capped at **4 iterations**; on exhaustion the latest answer is returned
   flagged as *not fully validated*.
5. **Cache** – retrieval and final answers are cached under a key bound to the
   question **and** the knowledge-base version, so ingesting new knowledge
   automatically invalidates stale entries.
"""
    )

    st.markdown("**Live configuration**")
    st.json(
        {
            "llm_model": settings.GROQ_MODEL,
            "whisper_model": settings.WHISPER_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "max_iterations": settings.MAX_ITERATIONS,
            "top_k": settings.TOP_K,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "redis_enabled": settings.REDIS_ENABLED,
            "redis_connected": system.redis_cache.enabled,
            "kb_version": system.kb_version.current(),
        }
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Evaluator–Generator RAG",
        page_icon="🧠",
        layout="wide",
    )
    _init_state()

    st.title("🧠 Evaluator–Generator RAG Platform")
    st.caption(
        "A self-evaluating question-answering system: independent Generator "
        "and Evaluator agents, isolated memories, LCEL orchestration, and "
        "Redis caching."
    )

    system = get_system()
    render_sidebar(system)

    tab_knowledge, tab_ask, tab_memory, tab_arch = st.tabs(
        ["📥 Knowledge", "💬 Ask", "🧠 Memories", "🏗️ Architecture"]
    )
    with tab_knowledge:
        render_ingestion(system)
    with tab_ask:
        render_qa(system)
    with tab_memory:
        render_memories(system)
    with tab_arch:
        render_architecture(system)


if __name__ == "__main__":
    main()
