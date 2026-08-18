import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.retrieval.vector_store import VectorStore


def test_vector_store_adds_and_retrieves_documents():
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={
                "source_id": "source-001",
                "source_type": "pdf",
                "source_name": "sample.pdf",
                "location": "page 1",
            },
        )
    ]

    embedding_service = MagicMock()

    fake_embedding_model = MagicMock()
    embedding_service.embeddings = fake_embedding_model

    with patch(
        "app.retrieval.vector_store.Chroma"
    ) as mock_chroma_class:
        mock_store = mock_chroma_class.return_value

        mock_store.similarity_search.return_value = documents

        store = VectorStore(
            embedding_service=embedding_service,
            collection_name="test_collection",
            persist_directory=None,
        )

        store.add_documents(documents)

        results = store.similarity_search(
            "What is machine learning?",
            k=1,
        )

    mock_chroma_class.assert_called_once_with(
        collection_name="test_collection",
        embedding_function=fake_embedding_model,
        persist_directory=None,
    )

    mock_store.add_documents.assert_called_once_with(documents)

    mock_store.similarity_search.assert_called_once_with(
        "What is machine learning?",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert "Machine learning" in result.page_content
    assert result.metadata["source_id"] == "source-001"
    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["source_name"] == "sample.pdf"
    assert result.metadata["location"] == "page 1"


def test_vector_store_ignores_empty_document_list():
    embedding_service = MagicMock()

    fake_embedding_model = MagicMock()
    embedding_service.embeddings = fake_embedding_model

    with patch(
        "app.retrieval.vector_store.Chroma"
    ) as mock_chroma_class:
        mock_store = mock_chroma_class.return_value

        store = VectorStore(
            embedding_service=embedding_service,
            collection_name="test_collection",
            persist_directory=None,
        )

        store.add_documents([])

        mock_store.add_documents.assert_not_called()


def test_vector_store_returns_empty_list_for_empty_query():
    embedding_service = MagicMock()

    fake_embedding_model = MagicMock()
    embedding_service.embeddings = fake_embedding_model

    with patch(
        "app.retrieval.vector_store.Chroma"
    ) as mock_chroma_class:
        mock_store = mock_chroma_class.return_value

        store = VectorStore(
            embedding_service=embedding_service,
            collection_name="test_collection",
            persist_directory=None,
        )

        result = store.similarity_search("   ")

        assert result == []

        mock_store.similarity_search.assert_not_called()


def test_vector_store_rejects_invalid_k():
    embedding_service = MagicMock()

    fake_embedding_model = MagicMock()
    embedding_service.embeddings = fake_embedding_model

    with patch(
        "app.retrieval.vector_store.Chroma"
    ):
        store = VectorStore(
            embedding_service=embedding_service,
            collection_name="test_collection",
            persist_directory=None,
        )

        with pytest.raises(
            ValueError,
            match="k must be greater than 0",
        ):
            store.similarity_search(
                "machine learning",
                k=0,
            )