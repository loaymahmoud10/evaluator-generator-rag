from langchain_core.documents import Document

from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.vector_store import VectorStore


def test_chroma_retrieval_preserves_source_metadata(tmp_path):
    documents = [
        Document(
            page_content=(
                "Machine learning is a subset of artificial intelligence "
                "that allows computers to learn from data."
            ),
            metadata={
                "source_id": "integration-source-001",
                "source_type": "pdf",
                "source_name": "machine-learning.pdf",
                "location": "page 5",
            },
        ),
        Document(
            page_content=(
                "Football is a team sport played between two teams of "
                "eleven players."
            ),
            metadata={
                "source_id": "integration-source-002",
                "source_type": "txt",
                "source_name": "sports.txt",
                "location": "document",
            },
        ),
    ]

    embedding_service = EmbeddingService()

    store = VectorStore(
        embedding_service=embedding_service,
        collection_name="integration_test",
        persist_directory=str(tmp_path),
    )

    store.add_documents(documents)

    results = store.similarity_search(
        "What is machine learning?",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert "Machine learning" in result.page_content

    assert result.metadata["source_id"] == "integration-source-001"
    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["source_name"] == "machine-learning.pdf"
    assert result.metadata["location"] == "page 5"