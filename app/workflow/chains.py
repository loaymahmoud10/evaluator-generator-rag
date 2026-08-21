"""LCEL building blocks for the Evaluator-Generator workflow.

These wrappers turn the Generator and Evaluator agents into LangChain
``Runnable`` steps so the workflow can be composed with LCEL primitives
(``RunnableLambda``, ``RunnablePassthrough``, ``RunnableSequence``).

The agents themselves live in ``app.generation.generator`` and
``app.evaluator.evaluator`` and remain independently testable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from app.evaluator.evaluator import Evaluator
from app.generation.generator import Generator


def generator_step(generator: Generator) -> RunnableLambda:
    """Return an LCEL step that generates an answer from state."""

    def _step(state: dict[str, Any]) -> dict[str, Any]:
        answer = generator.generate(
            question=state["question"],
            retrieved_context=state.get("retrieved_context", ""),
            feedback=state.get("feedback"),
        )
        return {**state, "generated_answer": answer}

    return RunnableLambda(_step)


def evaluator_step(evaluator: Evaluator) -> RunnableLambda:
    """Return an LCEL step that evaluates the current answer."""

    def _step(state: dict[str, Any]) -> dict[str, Any]:
        evaluation = evaluator.evaluate(
            question=state["question"],
            answer=state.get("generated_answer", ""),
            retrieved_context=state.get("retrieved_context", ""),
        )
        return {**state, "evaluation": evaluation}

    return RunnableLambda(_step)
