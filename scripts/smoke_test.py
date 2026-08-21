"""End-to-end smoke test of the whole platform.

Runs against the *real* LLM, embeddings, vector store and Redis (if running):

    python scripts/smoke_test.py

Steps
-----
1. Build the system from configuration.
2. Ingest a small local text file + a Wikipedia page.
3. Ask a question that IS answerable  -> expects a grounded answer.
4. Ask the same question again        -> expects a Redis cache hit.
5. Ask a question that is NOT covered -> expects an "not available" answer.
6. Assert the Generator and Evaluator memories are separate objects.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import build_system  # noqa: E402
from app.ingestion.pipeline import (  # noqa: E402
    IngestionError,
    IngestRequest,
    SourceType,
)

SAMPLE = """
Photosynthesis is the process used by plants, algae and some bacteria to turn
sunlight into chemical energy. It happens in the chloroplasts and produces
glucose and oxygen from carbon dioxide and water. The light-dependent reactions
occur in the thylakoid membranes, while the Calvin cycle occurs in the stroma.
"""


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and crash on LLM output."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _show(result: dict, elapsed: float) -> None:
    print(f"decision   : {result['decision']} (validated={result['validated']})")
    print(f"iterations : {result['iterations']} | reason: {result['reason']}")
    print(f"elapsed    : {elapsed:.2f}s")
    print(f"sources    : {len(result['sources'])}")
    print("-" * 72)
    print(result["final_answer"])


def main() -> int:
    _force_utf8_stdout()
    _banner("1. Building system")
    system = build_system()
    print(f"redis connected : {system.redis_cache.enabled}")
    print(f"kb version      : {system.kb_version.current()}")

    _banner("2. Ingesting knowledge")
    tmp = Path(tempfile.gettempdir()) / "smoke_photosynthesis.txt"
    tmp.write_text(SAMPLE, encoding="utf-8")

    requests = [
        IngestRequest(source=str(tmp)),
        IngestRequest(
            source="wikipedia:Retrieval-augmented generation",
            source_type=SourceType.WIKIPEDIA,
        ),
    ]
    try:
        chunks = system.pipeline.ingest(requests)
    except IngestionError as exc:
        # Per-source failures (e.g. no network) must not abort the batch.
        chunks = exc.stored_chunks
        for err in exc.errors:
            print(f"  ! source failed (continuing): {err[:160]}")
    print(f"stored {chunks} chunks, kb version now v{system.kb_version.current()}")

    question = "Where do the light-dependent reactions of photosynthesis occur?"

    _banner("3. Answerable question (cold)")
    started = time.perf_counter()
    result = system.workflow.run(question)
    _show(result, time.perf_counter() - started)

    _banner("4. Same question again (expect Redis cache hit)")
    started = time.perf_counter()
    cached = system.workflow.run(question)
    _show(cached, time.perf_counter() - started)
    if system.redis_cache.enabled:
        print(
            "cache behaviour :",
            "HIT ✅" if cached["reason"] == "cache_hit" else "MISS ❌",
        )
    else:
        print("cache behaviour : skipped (Redis not running)")

    _banner("5. Out-of-scope question (expect a refusal, not a hallucination)")
    started = time.perf_counter()
    unknown = system.workflow.run(
        "What was the closing share price of Toyota on 3 March 1997?"
    )
    _show(unknown, time.perf_counter() - started)

    _banner("6. Memory isolation")
    print(f"generator memory : {type(system.generator_memory).__name__}")
    print(f"evaluator memory : {type(system.evaluator_memory).__name__}")
    assert system.generator_memory is not system.evaluator_memory
    assert system.evaluator.memory is system.evaluator_memory
    print("separate instances ✅  evaluator agent shares the workflow memory ✅")

    _banner("SMOKE TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
