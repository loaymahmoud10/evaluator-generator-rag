from unittest.mock import patch

from langchain_core.documents import Document

from app.ingestion.web_loader import WikipediaLoader


def test_wikipedia_loader_extracts_text_and_source_metadata():
    fake_document = Document(
        page_content="Artificial intelligence is a field of computer science.",
        metadata={},
    )

    with patch(
        "app.ingestion.web_loader.WikipediaBackend"
    ) as mock_backend_class:
        mock_backend = mock_backend_class.return_value
        mock_backend.load.return_value = [fake_document]

        loader = WikipediaLoader()
        documents = loader.load("Artificial intelligence")

    assert len(documents) == 1

    document = documents[0]

    assert (
        "Artificial intelligence is a field of computer science."
        in document.page_content
    )

    assert document.metadata["source_type"] == "wikipedia"
    assert document.metadata["source_name"] == "Artificial intelligence"
    assert document.metadata["location"] == "Wikipedia"
    assert document.metadata["source_id"] == "wikipedia:Artificial intelligence"

    mock_backend.load.assert_called_once_with("Artificial intelligence")