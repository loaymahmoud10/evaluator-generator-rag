# Evaluator–Generator RAG Platform

A self-evaluating question-answering platform. Instead of answering a question
once and hoping it is right, two independent LLM agents work in a controlled
feedback loop:

* a **Generator** answers using *only* the ingested external knowledge, and
* an **Evaluator** grades that answer and sends back actionable feedback until
  the answer is good enough — or until the hard cap of **4 iterations**.

Knowledge can come from PDFs, DOCX, TXT, source code, PowerPoint, web pages,
Wikipedia articles, and WAV audio (transcribed with Whisper).

---

## 1. Architecture

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

            loop ≤ 4 iterations, then return the latest answer
```

### The workflow, step by step

1. **Ingest** — each source is loaded, its text extracted, split into chunks,
   embedded, and stored in Chroma. Extraction results are cached in Redis so
   re-ingesting the same file is instant. Successful ingestion bumps the
   *knowledge-base version*.
2. **Retrieve** — a question first checks the Redis retrieval cache; on a miss
   the vector store returns the top-K chunks.
3. **Generate** — the Generator answers strictly from that context. If the
   context does not cover the question it replies *"The requested information is
   not available in the provided sources"* rather than inventing an answer.
4. **Evaluate** — the Evaluator scores accuracy, relevance, completeness,
   consistency, grounding, unsupported claims and overall quality, and returns
   `ACCEPT` or `IMPROVE` plus structured feedback.
5. **Loop** — `IMPROVE` sends the feedback back to the Generator, which produces
   an improved answer that is evaluated again. The loop counter is enforced
   strictly at **4**; on exhaustion the latest answer is returned flagged as
   *not fully validated*.
6. **Cache** — the final answer is cached under a key derived from the question,
   the retrieved context, **and** the knowledge-base version, so ingesting new
   knowledge automatically invalidates stale answers.

### Memory isolation

`GeneratorMemory` and `EvaluatorMemory` are different classes and different
instances. The orchestrator refuses to start if the two are the same object or
of the wrong type, and it only ever writes generation data to the Generator
memory and evaluation data to the Evaluator memory. Neither agent receives a
reference to the other's memory.

| Memory | Stores |
|---|---|
| Generator | retrieved context, every answer attempt, the feedback that triggered it |
| Evaluator | every decision, score, per-criterion grades, and feedback issued |

### Project layout

```
app/
  config.py            settings from .env
  core.py              build_system() — wires the entire platform
  cli.py               command-line interface
  ingestion/           loaders (pdf, docx, txt, code, pptx, url, wikipedia, wav) + pipeline
  retrieval/           chunker, embeddings, Chroma vector store, retriever, KB version stamp
  generation/          Generator agent + GeneratorMemory + Groq LLM factory
  evaluator/           Evaluator agent + EvaluatorMemory + evaluation prompt
  workflow/            LCEL chains, graph, and the iterative orchestrator
  cache/               Redis caching layer
  ui/streamlit_app.py  the web interface
scripts/smoke_test.py  end-to-end verification script
tests/                 unit + integration tests
```

### LangChain / LCEL

The Generator and Evaluator are wrapped as LCEL `Runnable`s in
`app/workflow/chains.py`, composed into one iteration with the `|` operator in
`app/workflow/graph.py` (`generator_step | evaluator_step`), and driven by the
orchestrator, which owns the loop counter and the feedback routing. Loading,
chunking, embeddings, and the vector store all use LangChain components.

---

## 2. Setup

```bash
cd evaluator-generator-rag

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env            # then edit it
```

Set at minimum `GROQ_API_KEY` in `.env` (free key from https://console.groq.com).

| Variable | Meaning | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (required) | — |
| `GROQ_MODEL` | Chat model for both agents | `openai/gpt-oss-120b` |
| `WHISPER_MODEL` | Speech-to-text model for WAV | `whisper-large-v3-turbo` |
| `REDIS_URL` / `REDIS_ENABLED` | Cache location / on-off | `redis://localhost:6379` / `true` |
| `MAX_ITERATIONS` | Feedback-loop cap | `4` |
| `TOP_K` | Chunks retrieved per question | `4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking | `1000` / `200` |
| `EMBEDDING_MODEL` | Local HuggingFace embeddings | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` | Vector store | `./chroma_store` / `knowledge` |
| `CACHE_TTL` | Cache expiry in seconds | `3600` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Redis (optional but recommended)

```bash
docker compose up -d          # starts Redis on localhost:6379
```

Without Redis the platform runs normally with caching transparently disabled —
the sidebar and `python -m app.cli status` show which mode you are in.

---

## 3. How to run

### Web interface (main deliverable)

```bash
streamlit run run_ui.py
```

Opens on http://localhost:8501 with four tabs:

* **📥 Knowledge** — upload files, paste URLs, list Wikipedia pages, ingest.
* **💬 Ask** — ask a question and see the final answer, per-iteration trace
  (each Generator attempt, each Evaluator score and feedback item), and the
  source chunks used.
* **🧠 Memories** — side-by-side view of the two isolated agent memories.
* **🏗️ Architecture** — the diagram and the live configuration.

### Command line

```bash
python -m app.cli status                                # config + connectivity
python -m app.cli ingest --path ./docs                  # file or whole folder
python -m app.cli ingest --url https://example.com/page
python -m app.cli ingest --wiki "Large language model"
python -m app.cli ask "What do the sources say about X?"
```

### From Python

```python
from app.core import build_system

system = build_system()
system.pipeline.ingest([IngestRequest(source="notes.pdf")])
result = system.workflow.run("What is in my notes?")
print(result["final_answer"], result["validated"], result["iterations"])
```

---

## 4. How to test

### Unit tests (no API key, no network, no Redis)

```bash
python -m pytest tests/unit -q
```

Covers every loader, the chunker, embeddings, retriever, vector store, both
agents, both memories, the Redis cache, the iteration cap, memory isolation,
and cache invalidation.

### Integration tests (needs `GROQ_API_KEY`)

```bash
python -m pytest tests/integration -q
```

### Everything, with coverage

```bash
python -m pytest tests -q --cov=app --cov-report=term-missing
```

### End-to-end smoke test (real LLM + embeddings + Redis if running)

```bash
python scripts/smoke_test.py
```

It builds the system, ingests a text file and a Wikipedia page, then checks:

1. an **answerable** question produces a grounded, Evaluator-approved answer;
2. the **same question again** is served from the Redis answer cache;
3. an **out-of-scope** question returns *"not available in the provided
   sources"* instead of a hallucination;
4. the two agent memories are **separate instances**.

### Manual test run in the UI (5 minutes)

1. `docker compose up -d` then `streamlit run run_ui.py`.
2. **Knowledge** tab → upload a PDF (or any TXT / DOCX / PPTX / `.py` file),
   optionally add a URL and a Wikipedia page → **Ingest sources**. You should
   see `Ingested N chunks` and the KB version increase in the sidebar.
3. **Ask** tab → ask something the document answers. Expect
   *Status: Approved*, 1–2 iterations, and the source chunks listed.
4. Ask a deliberately vague or demanding question. Expect 2–4 iterations, and
   the trace showing the Evaluator's feedback and the improved answers.
5. Ask something the documents do **not** cover. Expect
   *"The requested information is not available in the provided sources."*
6. Ask the first question again. With Redis running you get
   **⚡ Served from the Redis answer cache** and a near-instant response.
7. Ingest another document, then ask the first question again — the cache is
   invalidated by the new KB version and the loop runs fresh.
8. **Memories** tab → confirm the Generator side shows answer attempts and the
   Evaluator side shows scores/feedback, with no cross-over.
9. Test WAV: upload a `.wav` recording and ask about its content — it is
   transcribed with Whisper and becomes searchable text.

---

## 5. Error handling & notes

* **Bad or unreachable source** — the ingestion batch continues; failed sources
  are reported per source (`IngestionError`) and the good ones are still stored.
* **Redis down** — every cache call degrades to a no-op and the platform keeps
  working; the sidebar shows *disabled*.
* **Unparseable evaluation JSON** — the Evaluator falls back to a safe
  `IMPROVE` verdict rather than crashing the loop.
* **Empty retrieval** — the Generator immediately returns the "not available"
  message instead of guessing.
* **Missing API key** — surfaced in the UI sidebar and `app.cli status`.
* **Logging** — all components log through `app.utils.logging` with the level
  from `LOG_LEVEL`; the workflow logs every iteration, decision, and cache hit.
* **Corporate networks** — Wikipedia/URL ingestion can fail behind an
  SSL-inspecting proxy. That is an environment issue, not a platform bug; file
  ingestion is unaffected.
