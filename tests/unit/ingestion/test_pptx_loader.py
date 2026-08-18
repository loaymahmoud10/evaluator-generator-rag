from pathlib import Path

from app.ingestion.document_loader import PPTXLoader


SAMPLE_PPTX = Path("tests/data/sample.pptx")


def test_pptx_loader_extracts_text_and_source_metadata():
    loader = PPTXLoader()

    documents = loader.load(SAMPLE_PPTX)

    assert len(documents) == 2

    first_slide = documents[0]
    second_slide = documents[1]

    assert "Artificial Intelligence Test" in first_slide.page_content
    assert "Machine learning is a subset of artificial intelligence." in first_slide.page_content

    assert "Testing" in second_slide.page_content
    assert "This presentation exists only for automated testing." in second_slide.page_content

    assert first_slide.metadata["source_type"] == "pptx"
    assert first_slide.metadata["source_name"] == "sample.pptx"
    assert first_slide.metadata["location"] == "slide 1"
    assert first_slide.metadata["source_id"]