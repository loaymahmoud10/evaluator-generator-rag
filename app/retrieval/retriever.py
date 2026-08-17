"""Retrieval orchestration for the RAG pipeline."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document

from app.retrieval.vector_store import VectorStore


class RetrievalResult(TypedDict):
    """Normalized result returned by the Retriever."""

    retrieved_context: str
    sources: list[dict[str, str]]


class Retriever:
    """Retrieve relevant documents and normalize their source metadata."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(self, question: str) -> RetrievalResult:
        """Retrieve relevant documents for a question."""
        if not question.strip():
            return {
                "retrieved_context": "",
                "sources": [],
            }

        documents = self._vector_store.similarity_search(
            question,
            k=self._top_k,
        )

        context_parts = [
            document.page_content.strip()
            for document in documents
            if document.page_content.strip()
        ]

        sources = [
            self._build_source_reference(document)
            for document in documents
            if document.page_content.strip()
        ]

        return {
            "retrieved_context": "\n\n".join(context_parts),
            "sources": sources,
        }

    @staticmethod
    def _build_source_reference(
        document: Document,
    ) -> dict[str, str]:
        """Convert Document metadata into the shared source format."""
        metadata = document.metadata

        return {
            "source_id": str(metadata.get("source_id", "")),
            "source_type": str(metadata.get("source_type", "")),
            "source_name": str(metadata.get("source_name", "")),
            "location": str(metadata.get("location", "")),
            "content": document.page_content,
        }