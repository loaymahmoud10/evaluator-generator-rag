"""System factory: assemble the whole platform from configuration.

The factory is the single place that knows how to wire the LLM, embeddings,
vector store, retriever, agents, isolated memories, Redis cache, ingestion
pipeline, and the workflow. Both the CLI and the Streamlit UI use it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.cache.redis_cache import RedisCache
from app.config import settings
from app.evaluator.evaluator import Evaluator
from app.evaluator.memory import EvaluatorMemory
from app.generation.generator import Generator
from app.generation.llm import create_groq_llm
from app.generation.memory import GeneratorMemory
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.kb_version import KnowledgeVersion
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.workflow.orchestrator import EvaluatorGeneratorWorkflow


@dataclass
class System:
    """A fully wired platform instance."""

    llm: object
    embeddings: EmbeddingService
    vector_store: VectorStore
    retriever: Retriever
    generator: Generator
    evaluator: Evaluator
    generator_memory: GeneratorMemory
    evaluator_memory: EvaluatorMemory
    redis_cache: RedisCache
    pipeline: IngestionPipeline
    workflow: EvaluatorGeneratorWorkflow
    kb_version: KnowledgeVersion


def build_system(
    persist_dir: str | None = None,
    collection: str | None = None,
    redis_cache: RedisCache | None = None,
) -> System:
    """Build and return a fully wired platform instance."""
    persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
    collection = collection or settings.CHROMA_COLLECTION

    llm = create_groq_llm()
    embeddings = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
    vector_store = VectorStore(
        embedding_service=embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    retriever = Retriever(vector_store=vector_store, top_k=settings.TOP_K)

    if redis_cache is None:
        redis_cache = RedisCache()

    # Isolated memories: one instance each, never shared between agents.
    generator_memory = GeneratorMemory()
    evaluator_memory = EvaluatorMemory()

    generator = Generator(llm=llm)
    evaluator = Evaluator(
        llm=llm,
        memory=evaluator_memory,          # the *same* instance the workflow uses
        redis_client=redis_cache.client,  # caches evaluation LLM responses
        cache_ttl=settings.CACHE_TTL,
    )

    kb_version = KnowledgeVersion(persist_dir=persist_dir)

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_service=embeddings,
        redis_cache=redis_cache,
        kb_version=kb_version,
    )

    workflow = EvaluatorGeneratorWorkflow(
        generator=generator,
        evaluator=evaluator,
        retriever=retriever,
        generator_memory=generator_memory,
        evaluator_memory=evaluator_memory,
        redis_cache=redis_cache,
        max_iterations=settings.MAX_ITERATIONS,
        kb_version=kb_version,
    )

    return System(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        generator_memory=generator_memory,
        evaluator_memory=evaluator_memory,
        redis_cache=redis_cache,
        pipeline=pipeline,
        workflow=workflow,
        kb_version=kb_version,
    )
