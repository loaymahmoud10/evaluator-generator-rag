from pathlib import Path

from app.ingestion.document_loader import CodeLoader


SAMPLE_CODE = Path("tests/data/sample.py")


def test_code_loader_extracts_code_and_source_metadata():
    loader = CodeLoader()

    documents = loader.load(SAMPLE_CODE)

    assert len(documents) == 1

    document = documents[0]

    assert "def calculate_average(values):" in document.page_content
    assert "return sum(values) / len(values)" in document.page_content

    assert document.metadata["source_type"] == "code"
    assert document.metadata["source_name"] == "sample.py"
    assert document.metadata["location"] == "document"
    assert document.metadata["source_id"]