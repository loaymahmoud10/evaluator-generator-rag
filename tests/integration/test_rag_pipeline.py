from app.generation.generator import Generator
from app.generation.llm import create_groq_llm
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

from langchain_core.documents import Document


def test_full_rag_retrieval_to_generation_pipeline(tmp_path):
    # ---------------------------------------------------------
    # 1. Create source documents
    # ---------------------------------------------------------
    documents = [
        Document(
            page_content=(
                "Machine learning is a branch of artificial intelligence "
                "that allows systems to learn patterns from data and make "
                "predictions without being explicitly programmed for every task."
            ),
            metadata={
                "source_id": "rag-source-001",
                "source_type": "pdf",
                "source_name": "machine_learning.pdf",
                "location": "page 1",
            },
        ),
        Document(
            page_content=(
                "Football is a team sport played between two teams of "
                "eleven players."
            ),
            metadata={
                "source_id": "rag-source-002",
                "source_type": "txt",
                "source_name": "football.txt",
                "location": "document",
            },
        ),
    ]

    # ---------------------------------------------------------
    # 2. Create real embedding service
    # ---------------------------------------------------------
    embedding_service = EmbeddingService()

    # ---------------------------------------------------------
    # 3. Create real temporary Chroma vector store
    # ---------------------------------------------------------
    vector_store = VectorStore(
        embedding_service=embedding_service,
        collection_name="full_rag_integration",
        persist_directory=str(tmp_path),
    )

    # ---------------------------------------------------------
    # 4. Add documents to Chroma
    # ---------------------------------------------------------
    vector_store.add_documents(documents)

    # ---------------------------------------------------------
    # 5. Create real Retriever
    # ---------------------------------------------------------
    retriever = Retriever(
        vector_store=vector_store,
        top_k=1,
    )

    # ---------------------------------------------------------
    # 6. Retrieve relevant context
    # ---------------------------------------------------------
    question = "What is machine learning?"

    retrieval_result = retriever.retrieve(question)

    retrieved_context = retrieval_result["retrieved_context"]
    sources = retrieval_result["sources"]

    # Verify retrieval actually found the correct document.
    assert retrieved_context
    assert "Machine learning" in retrieved_context

    assert len(sources) == 1
    assert sources[0]["source_id"] == "rag-source-001"
    assert sources[0]["source_name"] == "machine_learning.pdf"

    # ---------------------------------------------------------
    # 7. Create real Groq Generator
    # ---------------------------------------------------------
    llm = create_groq_llm()

    generator = Generator(
        llm=llm,
    )

    # ---------------------------------------------------------
    # 8. Generate answer using retrieved context
    # ---------------------------------------------------------
    answer = generator.generate(
        question=question,
        retrieved_context=retrieved_context,
    )

    # ---------------------------------------------------------
    # 9. Validate generated answer
    # ---------------------------------------------------------
    assert isinstance(answer, str)
    assert answer.strip()

    print("\nRetrieved context:")
    print(retrieved_context)

    print("\nSources:")
    print(sources)

    print("\nGenerated answer:")
    print(answer)