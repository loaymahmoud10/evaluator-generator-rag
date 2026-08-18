from pathlib import Path

from app.ingestion.document_loader import DOCXLoader


SAMPLE_DOCX = Path("tests/data/sample.docx")


def test_docx_loader_extracts_text_and_source_metadata():
    loader = DOCXLoader()

    documents = loader.load(SAMPLE_DOCX)

    assert len(documents) == 1

    document = documents[0]

    assert "Artificial intelligence is a field of computer science." in document.page_content
    assert "Machine learning is a subset of artificial intelligence." in document.page_content

    assert document.metadata["source_type"] == "docx"
    assert document.metadata["source_name"] == "sample.docx"
    assert document.metadata["location"] == "document"
    assert document.metadata["source_id"]