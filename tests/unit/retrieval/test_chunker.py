from langchain_core.documents import Document

from app.retrieval.chunker import DocumentChunker


def test_chunker_preserves_source_metadata():
    document = Document(
        page_content=(
            "Artificial intelligence is a field of computer science. "
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning uses neural networks with multiple layers."
        ),
        metadata={
            "source_id": "test-source-001",
            "source_type": "pdf",
            "source_name": "sample.pdf",
            "location": "page 1",
        },
    )

    chunker = DocumentChunker(
        chunk_size=80,
        chunk_overlap=20,
    )

    chunks = chunker.split([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["source_id"] == "test-source-001"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["source_name"] == "sample.pdf"
        assert chunk.metadata["location"] == "page 1"

        assert chunk.page_content.strip()
    
def test_chunker_returns_empty_list_for_empty_input():
    chunker = DocumentChunker()

    assert chunker.split([]) == []


def test_chunker_handles_short_document():
    document = Document(
        page_content="Short document.",
        metadata={
            "source_id": "short-001",
            "source_type": "txt",
            "source_name": "short.txt",
            "location": "document",
        },
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short document."
    assert chunks[0].metadata["source_id"] == "short-001"


def test_chunker_preserves_metadata_for_multiple_sources():
    documents = [
        Document(
            page_content="Information from the first source.",
            metadata={
                "source_id": "source-001",
                "source_type": "pdf",
                "source_name": "first.pdf",
                "location": "page 1",
            },
        ),
        Document(
            page_content="Information from the second source.",
            metadata={
                "source_id": "source-002",
                "source_type": "web",
                "source_name": "example.com",
                "location": "https://example.com",
            },
        ),
    ]

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split(documents)

    assert len(chunks) == 2

    assert chunks[0].metadata["source_id"] == "source-001"
    assert chunks[0].metadata["source_name"] == "first.pdf"

    assert chunks[1].metadata["source_id"] == "source-002"
    assert chunks[1].metadata["source_name"] == "example.com"