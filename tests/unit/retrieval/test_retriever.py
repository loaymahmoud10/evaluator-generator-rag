import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.retriever import Retriever


def test_retriever_returns_context_and_source_references():
    documents = [
        Document(
            page_content="Machine learning learns patterns from data.",
            metadata={
                "source_id": "source-001",
                "source_type": "pdf",
                "source_name": "ml.pdf",
                "location": "page 5",
            },
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={
                "source_id": "source-002",
                "source_type": "web",
                "source_name": "example.com",
                "location": "https://example.com/ml",
            },
        ),
    ]

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = documents

    retriever = Retriever(
        vector_store=vector_store,
        top_k=2,
    )

    result = retriever.retrieve("What is machine learning?")

    assert result["retrieved_context"] == (
        "Machine learning learns patterns from data.\n\n"
        "Deep learning uses neural networks."
    )

    assert result["sources"] == [
        {
            "source_id": "source-001",
            "source_type": "pdf",
            "source_name": "ml.pdf",
            "location": "page 5",
            "content": "Machine learning learns patterns from data.",
        },
        {
            "source_id": "source-002",
            "source_type": "web",
            "source_name": "example.com",
            "location": "https://example.com/ml",
            "content": "Deep learning uses neural networks.",
        },
    ]

    vector_store.similarity_search.assert_called_once_with(
        "What is machine learning?",
        k=2,
    )


def test_retriever_returns_empty_result_for_empty_question():
    vector_store = MagicMock()

    retriever = Retriever(
        vector_store=vector_store,
        top_k=2,
    )

    result = retriever.retrieve("   ")

    assert result == {
        "retrieved_context": "",
        "sources": [],
    }

    vector_store.similarity_search.assert_not_called()


def test_retriever_handles_no_retrieved_documents():
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = []

    retriever = Retriever(
        vector_store=vector_store,
        top_k=2,
    )

    result = retriever.retrieve("What is machine learning?")

    assert result == {
        "retrieved_context": "",
        "sources": [],
    }

    vector_store.similarity_search.assert_called_once_with(
        "What is machine learning?",
        k=2,
    )


def test_retriever_rejects_invalid_top_k():
    vector_store = MagicMock()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        Retriever(
            vector_store=vector_store,
            top_k=0,
        )


def test_retriever_ignores_empty_document_content():
    documents = [
        Document(
            page_content="",
            metadata={
                "source_id": "empty-source",
                "source_type": "pdf",
                "source_name": "empty.pdf",
                "location": "page 1",
            },
        ),
        Document(
            page_content="Valid retrieved content.",
            metadata={
                "source_id": "valid-source",
                "source_type": "txt",
                "source_name": "valid.txt",
                "location": "document",
            },
        ),
    ]

    vector_store = MagicMock()
    vector_store.similarity_search.return_value = documents

    retriever = Retriever(
        vector_store=vector_store,
        top_k=2,
    )

    result = retriever.retrieve("test question")

    assert result["retrieved_context"] == "Valid retrieved content."

    assert result["sources"] == [
        {
            "source_id": "valid-source",
            "source_type": "txt",
            "source_name": "valid.txt",
            "location": "document",
            "content": "Valid retrieved content.",
        }
    ]