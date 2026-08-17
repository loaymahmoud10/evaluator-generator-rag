from unittest.mock import patch

from langchain_core.documents import Document

from app.ingestion.web_loader import URLLoader


def test_url_loader_extracts_text_and_source_metadata():
    fake_document = Document(
        page_content="Artificial intelligence is a field of computer science.",
        metadata={},
    )

    with patch(
        "app.ingestion.web_loader.WebBaseLoader"
    ) as mock_loader_class:
        mock_loader = mock_loader_class.return_value
        mock_loader.load.return_value = [fake_document]

        loader = URLLoader()
        documents = loader.load("https://example.com/article")

    assert len(documents) == 1

    document = documents[0]

    assert "Artificial intelligence is a field of computer science." in (
        document.page_content
    )

    assert document.metadata["source_type"] == "web"
    assert document.metadata["source_name"] == "example.com"
    assert document.metadata["location"] == "https://example.com/article"
    assert document.metadata["source_id"] == "https://example.com/article"

    mock_loader_class.assert_called_once_with(
        web_paths=["https://example.com/article"]
    )