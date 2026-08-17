from pathlib import Path

from app.ingestion.document_loader import PDFLoader


SAMPLE_PDF = Path("tests/data/sample.pdf")


def test_pdf_loader_extracts_text_and_source_metadata():
    loader = PDFLoader()

    documents = loader.load(SAMPLE_PDF)

    assert len(documents) == 1

    document = documents[0]

    assert "Artificial intelligence is a field of computer science." in document.page_content
    assert "Machine learning is a subset of artificial intelligence." in document.page_content

    assert document.metadata["source_type"] == "pdf"
    assert document.metadata["source_name"] == "sample.pdf"
    assert document.metadata["location"] == "page 1"
    assert document.metadata["source_id"]