"""Tests for the ingestion pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.ingestion.pipeline import IngestRequest, IngestionPipeline, SourceType
from app.retrieval.chunker import DocumentChunker
from tests.unit.ingestion.test_document_loader import SAMPLE_PDF


def _fake_loader_factory(document):
    class FakeLoader:
        def load(self, source):
            return [document]

    return FakeLoader


def test_pipeline_ingests_and_stores_chunks():
    vector_store = MagicMock()
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        chunker=DocumentChunker(chunk_size=50, chunk_overlap=10),
    )
    doc = Document(
        page_content="Alpha beta gamma delta epsilon zeta eta theta iota.",
        metadata={"source_type": "pdf", "source_name": "x.pdf"},
    )
    pipeline._select_loader = lambda st: _fake_loader_factory(doc)()

    total = pipeline.ingest([IngestRequest(source="x.pdf", source_type=SourceType.PDF)])
    assert total > 0
    vector_store.add_documents.assert_called_once()


def test_pipeline_uses_real_pdf_loader(monkeypatch):
    vector_store = MagicMock()
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        chunker=DocumentChunker(chunk_size=2000, chunk_overlap=0),
    )
    total = pipeline.ingest(
        [IngestRequest(source=str(SAMPLE_PDF), source_type=SourceType.PDF)]
    )
    assert total >= 1
    vector_store.add_documents.assert_called_once()


def test_pipeline_resolves_auto_type_for_url():
    pipeline = IngestionPipeline(vector_store=MagicMock())
    req = IngestRequest(source="https://example.com/a")
    assert req.resolved_type() == SourceType.URL

    req2 = IngestRequest(source="wikipedia:AI")
    assert req2.resolved_type() == SourceType.WIKIPEDIA

    req3 = IngestRequest(source="script.py")
    assert req3.resolved_type() == SourceType.CODE


def test_pipeline_uses_cache_for_extraction():
    vector_store = MagicMock()
    redis_cache = MagicMock()
    redis_cache.get_processed_document.return_value = [
        {"page_content": "cached text", "metadata": {}}
    ]
    pipeline = IngestionPipeline(
        vector_store=vector_store, redis_cache=redis_cache
    )
    doc = Document(page_content="fresh", metadata={})
    pipeline._select_loader = lambda st: _fake_loader_factory(doc)()

    total = pipeline.ingest([IngestRequest(source="x.txt", source_type=SourceType.TXT)])
    # Extraction came from cache, so the fresh loader was never used.
    redis_cache.get_processed_document.assert_called_once()
    assert total == 1
    vector_store.add_documents.assert_called_once()
