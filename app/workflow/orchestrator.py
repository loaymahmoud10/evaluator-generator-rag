"""Central Evaluator-Generator orchestrator (the feedback loop).

This is the heart of the platform. It wires together:

* the **Retriever** (external knowledge layer)
* the **Generator** agent + its *isolated* GeneratorMemory
* the **Evaluator** agent + its *isolated* EvaluatorMemory
* an optional **RedisCache** for retrieval/answer caching

Workflow (per question)::

    retrieve context
    loop (max 4 iterations):
        Generator -> answer
        Evaluator -> decision + feedback
        if ACCEPT: stop (validated)
        else: feed feedback back to Generator
    return final answer + metadata

The loop counter is enforced strictly; at iteration 4 without an ACCEPT,
the loop terminates and the answer is returned flagged as not fully
validated.
"""

from __future__ import annotations

from typing import TypedDict

from app.cache.redis_cache import RedisCache
from app.evaluator.evaluator import Evaluator
from app.evaluator.memory import EvaluatorMemory
from app.generation.generator import Generator
from app.generation.memory import GeneratorMemory
from app.retrieval.kb_version import KnowledgeVersion
from app.retrieval.retriever import Retriever
from app.schemas.state import EvaluationDecision, EvaluationResult, SourceReference
from app.utils.logging import get_logger
from app.workflow.graph import WorkflowGraph

logger = get_logger("workflow")


class WorkflowResult(TypedDict, total=False):
    """Final result returned to the caller (UI / CLI)."""

    question: str
    final_answer: str
    decision: str
    validated: bool
    iterations: int
    reason: str
    evaluations: list[EvaluationResult]
    generation_history: list[dict]
    sources: list[SourceReference]


class EvaluatorGeneratorWorkflow:
    """Drive the iterative Evaluator-Generator feedback loop."""

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        retriever: Retriever,
        generator_memory: GeneratorMemory | None = None,
        evaluator_memory: EvaluatorMemory | None = None,
        redis_cache: RedisCache | None = None,
        max_iterations: int = 4,
        kb_version: KnowledgeVersion | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        # Memory isolation: the two agents must never share an instance.
        self._generator_memory = generator_memory or GeneratorMemory()
        self._evaluator_memory = evaluator_memory or EvaluatorMemory()
        if isinstance(self._generator_memory, EvaluatorMemory):
            raise TypeError(
                "Generator must use GeneratorMemory, not EvaluatorMemory"
            )
        if isinstance(self._evaluator_memory, GeneratorMemory):
            raise TypeError(
                "Evaluator must use EvaluatorMemory, not GeneratorMemory"
            )

        if self._generator_memory is self._evaluator_memory:
            raise TypeError("Generator and Evaluator must not share a memory")

        self._generator = generator
        self._evaluator = evaluator
        self._retriever = retriever
        self._redis = redis_cache
        self._max_iterations = max_iterations
        self._kb_version = kb_version or KnowledgeVersion()
        self._graph = WorkflowGraph(generator, evaluator)

    # ------------------------------------------------------------------ #
    @property
    def generator_memory(self) -> GeneratorMemory:
        return self._generator_memory

    @property
    def evaluator_memory(self) -> EvaluatorMemory:
        return self._evaluator_memory

    # ------------------------------------------------------------------ #
    def run(self, question: str) -> WorkflowResult:
        """Run the full feedback loop for a single question."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        logger.info("Workflow started for question: %s", question)

        # Fresh memory for this question so iterations stay isolated.
        self._generator_memory.clear(question)
        self._evaluator_memory.clear(question)

        # ---- Retrieval (with Redis cache) ---------------------------- #
        retrieval = self._retrieve(question)
        context = retrieval["retrieved_context"]
        sources = retrieval["sources"]

        state: dict = {
            "question": question,
            "retrieved_context": context,
            "feedback": None,
        }

        # Short-circuit repeated questions with identical context.
        if self._redis is not None:
            cached = self._redis.get_answer(
                question, self._context_hash(context), self._version()
            )
            if cached is not None:
                logger.info("Answer cache hit for question + context")
                return WorkflowResult(
                    question=question,
                    final_answer=cached.get("final_answer", ""),
                    decision=(
                        EvaluationDecision.ACCEPT.value
                        if cached.get("validated")
                        else EvaluationDecision.IMPROVE.value
                    ),
                    validated=bool(cached.get("validated", False)),
                    iterations=cached.get("iterations", 0),
                    reason="cache_hit",
                    evaluations=[],
                    generation_history=[],
                    sources=sources,
                )

        evaluations: list[EvaluationResult] = []
        reason = "max_iterations_reached"
        validated = False

        for iteration in range(1, self._max_iterations + 1):
            state["iteration"] = iteration
            logger.info("Iteration %d/%d", iteration, self._max_iterations)

            state = self._graph.run_iteration(state)

            answer = state.get("generated_answer", "")
            evaluation: EvaluationResult = state["evaluation"]
            evaluations.append(evaluation)

            # Record in the *correct*, isolated memory only.
            self._generator_memory.record(
                question=question,
                iteration=iteration,
                retrieved_context=context,
                answer=answer,
                feedback=state.get("feedback"),
            )
            # The Evaluator may already share this memory instance and have
            # recorded the result itself; never store it twice.
            if self._evaluator_memory.get_last_evaluation(question) is not evaluation:
                self._evaluator_memory.record(question, evaluation)

            if evaluation["decision"] == EvaluationDecision.ACCEPT:
                logger.info("Evaluator ACCEPTED answer at iteration %d", iteration)
                validated = True
                reason = "evaluator_approved"
                break

            # Build feedback text for the next Generator pass.
            state["feedback"] = self._format_feedback(evaluation)
            logger.info(
                "Evaluator requested IMPROVE (score=%s). Feeding feedback.",
                evaluation.get("score"),
            )

        final_answer = state.get("generated_answer", "")
        if not final_answer:
            final_answer = self._generator.UNAVAILABLE_MESSAGE

        # ---- Cache final answer (question + context bound) ----------- #
        if self._redis is not None:
            self._redis.cache_answer(
                question,
                self._context_hash(context),
                {
                    "final_answer": final_answer,
                    "validated": validated,
                    "iterations": len(evaluations),
                },
                self._version(),
            )

        return WorkflowResult(
            question=question,
            final_answer=final_answer,
            decision=(
                EvaluationDecision.ACCEPT.value
                if validated
                else EvaluationDecision.IMPROVE.value
            ),
            validated=validated,
            iterations=len(evaluations),
            reason=reason,
            evaluations=evaluations,
            generation_history=self._generator_memory.get_history(question),
            sources=sources,
        )

    # ------------------------------------------------------------------ #
    def _retrieve(self, question: str) -> dict:
        if self._redis is not None:
            cached = self._redis.get_retrieval(question, self._version())
            if cached is not None:
                logger.info("Retrieval cache hit for question")
                return cached

        result = self._retriever.retrieve(question)

        if self._redis is not None:
            self._redis.cache_retrieval(question, result, self._version())

        return result

    def _version(self) -> str:
        """Current knowledge-base version, used to invalidate stale cache."""
        return str(self._kb_version.current())

    @staticmethod
    def _format_feedback(evaluation: EvaluationResult) -> str:
        items = evaluation.get("feedback", [])
        if not items:
            return (
                "The evaluator found the answer insufficient. "
                "Please improve accuracy, completeness, and grounding "
                "using only the provided context."
            )

        lines = []
        for item in items:
            criterion = item.get("criterion", "quality")
            severity = item.get("severity", "medium")
            issue = item.get("issue", "")
            suggestion = item.get("suggestion", "")
            lines.append(
                f"- [{severity}] {criterion}: {issue} (suggestion: {suggestion})"
            )
        return "Evaluator feedback:\n" + "\n".join(lines)

    @staticmethod
    def _context_hash(context: str) -> str:
        import hashlib

        return hashlib.sha256(context.encode("utf-8")).hexdigest()
