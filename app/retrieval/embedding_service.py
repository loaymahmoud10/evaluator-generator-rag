"""Embedding service for the retrieval pipeline."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """Generate embeddings using a local Hugging Face model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain embedding model."""
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        if not texts:
            return []

        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""
        return self._embeddings.embed_query(text)