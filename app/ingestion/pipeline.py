"""Ingestion pipeline orchestrator.

This module connects the individual loaders (PDF, DOCX, TXT, code, PPTX,
URL, Wikipedia, WAV) to the chunker and the vector store. It is the single
entry point the UI and CLI use to turn raw user inputs into retrievable
knowledge.

Pipeline stages (per source):
    load  ->  extract content (with optional Redis cache of extraction)
    chunk ->  split into meaningful chunks
    store ->  add chunk embeddings to the vector store

The pipeline never shares memory with the Generator or Evaluator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from app.cache.redis_cache import RedisCache
from app.config import settings
from app.ingestion.audio_loader import AudioLoader
from app.ingestion.base import BaseLoader
from app.ingestion.document_loader import (
    CODE_EXTENSIONS,
    DOCXLoader,
    PDFLoader,
    PPTXLoader,
    TXTLoader,
    CodeLoader,
)
from app.ingestion.web_loader import URLLoader, WikipediaLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.kb_version import KnowledgeVersion
from app.retrieval.vector_store import VectorStore
from app.utils.logging import get_logger

logger = get_logger("ingestion")


class IngestionError(RuntimeError):
    """Raised when one or more sources in a batch failed to ingest."""

    def __init__(self, errors: list[str], stored_chunks: int = 0) -> None:
        self.errors = errors
        self.stored_chunks = stored_chunks
        super().__init__(
            f"{len(errors)} source(s) failed to ingest "
            f"({stored_chunks} chunks stored): " + "; ".join(errors)
        )


class SourceType(str, Enum):
    """Supported external knowledge source types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CODE = "code"
    PPTX = "pptx"
    URL = "url"
    WIKIPEDIA = "wikipedia"
    WAV = "wav"
    AUTO = "auto"


@dataclass
class IngestRequest:
    """A single ingestion request produced by the UI or CLI."""

    source: str
    source_type: SourceType = SourceType.AUTO

    def resolved_type(self) -> SourceType:
        """Resolve an AUTO request to a concrete source type."""
        if self.source_type != SourceType.AUTO:
            return self.source_type

        text = self.source.strip().lower()
        if text.startswith("http://") or text.startswith("https://"):
            return SourceType.URL
        if text.startswith("wikipedia:"):
            return SourceType.WIKIPEDIA

        suffix = Path(self.source).suffix.lower()
        mapping = {
            ".pdf": SourceType.PDF,
            ".docx": SourceType.DOCX,
            ".txt": SourceType.TXT,
            ".pptx": SourceType.PPTX,
            ".wav": SourceType.WAV,
        }
        if suffix in mapping:
            return mapping[suffix]
        if suffix in CODE_EXTENSIONS:
            return SourceType.CODE
        raise ValueError(
            f"Could not auto-detect source type for: {self.source}"
        )


_EXTENSION_LOADERS: dict[SourceType, type[BaseLoader]] = {
    SourceType.PDF: PDFLoader,
    SourceType.DOCX: DOCXLoader,
    SourceType.TXT: TXTLoader,
    SourceType.PPTX: PPTXLoader,
    SourceType.CODE: CodeLoader,
    SourceType.WAV: AudioLoader,
}


class IngestionPipeline:
    """Orchestrate loading, chunking, and storing of knowledge sources."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService | None = None,
        chunker: DocumentChunker | None = None,
        redis_cache: RedisCache | None = None,
        kb_version: KnowledgeVersion | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embeddings = embedding_service or EmbeddingService()
        self._chunker = chunker or DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self._cache = redis_cache
        self._kb_version = kb_version or KnowledgeVersion()
        self._whisper_model = None

    # ------------------------------------------------------------------ #
    # Loader selection
    # ------------------------------------------------------------------ #
    def _select_loader(self, source_type: SourceType) -> BaseLoader:
        if source_type == SourceType.URL:
            return URLLoader()
        if source_type == SourceType.WIKIPEDIA:
            return WikipediaLoader()
        if source_type in _EXTENSION_LOADERS:
            return _EXTENSION_LOADERS[source_type]()
        raise ValueError(f"Unsupported source type: {source_type}")

    # ------------------------------------------------------------------ #
    # Core ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, requests: Iterable[IngestRequest]) -> int:
        """Ingest a collection of requests. Returns number of chunks stored."""
        total_chunks = 0
        errors: list[str] = []
        for request in requests:
            try:
                total_chunks += self._ingest_one(request)
            except Exception as exc:  # one bad source must not kill the batch
                logger.exception("Failed to ingest %s", request.source)
                errors.append(f"{request.source}: {exc}")

        if total_chunks:
            # New knowledge -> invalidate every retrieval/answer cache entry.
            version = self._kb_version.bump()
            logger.info("Knowledge base version bumped to %d", version)

        logger.info("Ingestion complete: %d chunks stored", total_chunks)
        if errors:
            raise IngestionError(errors, total_chunks)
        return total_chunks

    def _ingest_one(self, request: IngestRequest) -> int:
        source_type = request.resolved_type()
        logger.info("Ingesting %s (%s)", request.source, source_type.value)

        documents = self._load_with_cache(request.source, source_type)
        if not documents:
            logger.warning("No content extracted from %s", request.source)
            return 0

        chunks = self._chunker.split(documents)
        self._vector_store.add_documents(chunks)
        logger.info("Stored %d chunks from %s", len(chunks), request.source)
        return len(chunks)

    def _load_with_cache(
        self,
        source: str,
        source_type: SourceType,
    ) -> list[Document]:
        """Load a source, using the Redis cache for extraction when possible."""
        cache_key = self._cache_key(source, source_type)

        if self._cache is not None:
            cached = self._cache.get_processed_document(cache_key)
            if cached is not None:
                logger.info("Cache hit for extraction: %s", source)
                return [
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {}),
                    )
                    for item in cached
                ]

        loader = self._select_loader(source_type)
        documents = loader.load(source)

        if self._cache is not None:
            self._cache.cache_processed_document(
                cache_key,
                [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in documents
                ],
            )

        return documents

    @staticmethod
    def _cache_key(source: str, source_type: SourceType) -> str:
        signature = f"{source_type.value}:{source}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def build_pipeline(
    vector_store: VectorStore,
    redis_cache: RedisCache | None = None,
    kb_version: KnowledgeVersion | None = None,
) -> IngestionPipeline:
    """Convenience factory used by the UI and CLI."""
    return IngestionPipeline(
        vector_store=vector_store,
        redis_cache=redis_cache,
        kb_version=kb_version,
    )
