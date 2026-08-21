"""The answer cache must be invalidated when the knowledge base changes."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.cache.redis_cache import RedisCache
from app.retrieval.kb_version import KnowledgeVersion
from app.schemas.state import EvaluationDecision
from app.workflow.orchestrator import EvaluatorGeneratorWorkflow

from tests.unit.workflow.test_orchestrator import FakeEvaluator, FakeGenerator


class InMemoryCache(RedisCache):
    """RedisCache backed by a plain dict (no server needed)."""

    def __init__(self) -> None:
        super().__init__(enabled=False)
        self._store: dict[str, object] = {}

    def _get_json(self, key):  # type: ignore[override]
        return self._store.get(key)

    def _set_json(self, key, value, ttl=None):  # type: ignore[override]
        self._store[key] = value


def _build(cache, kb_version, retriever):
    return EvaluatorGeneratorWorkflow(
        generator=FakeGenerator(),
        evaluator=FakeEvaluator([EvaluationDecision.ACCEPT]),
        retriever=retriever,
        redis_cache=cache,
        kb_version=kb_version,
    )


def _retriever(text):
    retriever = MagicMock()
    retriever.retrieve.return_value = {
        "retrieved_context": text,
        "sources": [],
    }
    return retriever


def test_identical_question_hits_cache(tmp_path):
    cache = InMemoryCache()
    version = KnowledgeVersion(persist_dir=str(tmp_path))
    workflow = _build(cache, version, _retriever("context"))

    workflow.run("what is x?")
    second = workflow.run("what is x?")

    assert second["reason"] == "cache_hit"


def test_new_ingestion_invalidates_cached_answer(tmp_path):
    cache = InMemoryCache()
    version = KnowledgeVersion(persist_dir=str(tmp_path))
    retriever = _retriever("context")
    workflow = _build(cache, version, retriever)

    workflow.run("what is x?")

    # Simulate ingesting new knowledge.
    version.bump()
    retriever.retrieve.return_value = {
        "retrieved_context": "fresh context",
        "sources": [],
    }

    after = workflow.run("what is x?")

    assert after["reason"] != "cache_hit"
