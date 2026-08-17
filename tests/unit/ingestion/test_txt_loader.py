from pathlib import Path

from app.ingestion.document_loader import TXTLoader


SAMPLE_TXT = Path("tests/data/sample.txt")


def test_txt_loader_extracts_text_and_source_metadata():
    loader = TXTLoader()

    documents = loader.load(SAMPLE_TXT)

    assert len(documents) == 1

    document = documents[0]

    assert "Artificial intelligence is a field of computer science." in document.page_content
    assert "Machine learning is a subset of artificial intelligence." in document.page_content

    assert document.metadata["source_type"] == "txt"
    assert document.metadata["source_name"] == "sample.txt"
    assert document.metadata["location"] == "document"
    assert document.metadata["source_id"]