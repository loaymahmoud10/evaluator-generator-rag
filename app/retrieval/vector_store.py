"""Chroma vector-store integration for the retrieval pipeline."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.retrieval.embedding_service import EmbeddingService


class VectorStore:
    """Store and retrieve document chunks using Chroma."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        collection_name: str = "knowledge",
        persist_directory: str | None = None,
    ) -> None:
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_service.embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add document chunks to the vector store."""
        if not documents:
            return

        self._store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Retrieve the most relevant document chunks."""
        if not query.strip():
            return []

        if k <= 0:
            raise ValueError("k must be greater than 0")

        return self._store.similarity_search(
            query,
            k=k,
        )