from unittest.mock import MagicMock, patch

from app.retrieval.embedding_service import EmbeddingService


def test_embed_documents_returns_embeddings():
    fake_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with patch(
        "app.retrieval.embedding_service.HuggingFaceEmbeddings"
    ) as mock_embeddings_class:
        mock_embeddings = mock_embeddings_class.return_value
        mock_embeddings.embed_documents.return_value = fake_embeddings

        service = EmbeddingService(model_name="test-model")

        result = service.embed_documents(
            ["first document", "second document"]
        )

    assert result == fake_embeddings

    mock_embeddings_class.assert_called_once_with(
        model_name="test-model"
    )

    mock_embeddings.embed_documents.assert_called_once_with(
        ["first document", "second document"]
    )


def test_embed_query_returns_embedding():
    fake_embedding = [0.1, 0.2, 0.3]

    with patch(
        "app.retrieval.embedding_service.HuggingFaceEmbeddings"
    ) as mock_embeddings_class:
        mock_embeddings = mock_embeddings_class.return_value
        mock_embeddings.embed_query.return_value = fake_embedding

        service = EmbeddingService(model_name="test-model")

        result = service.embed_query("What is machine learning?")

    assert result == fake_embedding

    mock_embeddings.embed_query.assert_called_once_with(
        "What is machine learning?"
    )


def test_embed_documents_returns_empty_list_for_empty_input():
    with patch(
        "app.retrieval.embedding_service.HuggingFaceEmbeddings"
    ):
        service = EmbeddingService(model_name="test-model")

        result = service.embed_documents([])

    assert result == []