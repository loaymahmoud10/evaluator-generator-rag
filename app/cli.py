"""Command-line interface for quick test runs without the UI.

Examples
--------
Ingest a folder of documents, then ask a question::

    python -m app.cli ingest --path ./docs
    python -m app.cli ask "What is machine learning?"

Ingest a URL and a Wikipedia page::

    python -m app.cli ingest --url https://example.com/article
    python -m app.cli ingest --wiki "Large language model"

Ask a question after ingestion::

    python -m app.cli ask "Summarize the sources."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import build_system
from app.ingestion.pipeline import IngestionError, IngestRequest, SourceType
from app.utils.logging import get_logger

logger = get_logger("cli")

def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and crash on LLM output."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass



def _cmd_ingest(args: argparse.Namespace) -> int:
    system = build_system()
    requests: list[IngestRequest] = []

    if args.path:
        base = Path(args.path)
        if base.is_file():
            requests.append(IngestRequest(source=str(base)))
        elif base.is_dir():
            for file in sorted(base.rglob("*")):
                if file.is_file():
                    requests.append(IngestRequest(source=str(file)))
        else:
            logger.error("Path not found: %s", args.path)

    if args.url:
        for url in args.url:
            requests.append(
                IngestRequest(source=url, source_type=SourceType.URL)
            )

    if args.wiki:
        for query in args.wiki:
            requests.append(
                IngestRequest(
                    source=f"wikipedia:{query}",
                    source_type=SourceType.WIKIPEDIA,
                )
            )

    if not requests:
        logger.error("No sources provided to ingest.")
        return 2

    try:
        total = system.pipeline.ingest(requests)
    except IngestionError as exc:
        print(f"Ingested {exc.stored_chunks} chunks, with failures:")
        for err in exc.errors:
            print(f"  ! {err}")
        return 1

    logger.info("Ingested %d chunks.", total)
    print(f"Ingested {total} chunks (knowledge base v{system.kb_version.current()}).")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    system = build_system()
    result = system.workflow.run(args.question)

    print("\n" + "=" * 70)
    print(f"QUESTION: {result['question']}")
    print(f"ITERATIONS: {result['iterations']} | VALIDATED: {result['validated']}")
    print(f"DECISION: {result['decision']} | REASON: {result['reason']}")
    print("-" * 70)
    print("ANSWER:")
    print(result["final_answer"])
    print("-" * 70)
    print("SOURCES:")
    for src in result["sources"]:
        print(f"  - [{src.get('source_type')}] {src.get('source_name')} "
              f"({src.get('location')})")
    print("=" * 70)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from app.config import settings

    system = build_system()
    print("Evaluator-Generator platform status")
    print(f"  LLM model       : {settings.GROQ_MODEL}")
    print(f"  API key set     : {settings.has_groq}")
    print(f"  Redis connected : {system.redis_cache.enabled}")
    print(f"  Vector store    : {system.vector_store.collection_name}")
    print(f"  KB version      : v{system.kb_version.current()}")
    print(f"  Max iterations  : {settings.MAX_ITERATIONS}")
    print(f"  Top-K           : {settings.TOP_K}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.cli", description="Evaluator-Generator RAG CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest knowledge sources")
    ingest.add_argument("--path", help="File or directory to ingest")
    ingest.add_argument("--url", action="append", help="Web page URL")
    ingest.add_argument("--wiki", action="append", help="Wikipedia query")
    ingest.set_defaults(func=_cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", help="The question to answer")
    ask.set_defaults(func=_cmd_ask)

    status = sub.add_parser("status", help="Show platform status")
    status.set_defaults(func=_cmd_status)

    _force_utf8_stdout()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - user-facing errors
        logger.exception("CLI failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
