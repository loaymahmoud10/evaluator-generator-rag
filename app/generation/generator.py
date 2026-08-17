"""LLM-based answer generation for the RAG pipeline."""

from __future__ import annotations

from typing import Any


class Generator:
    """Generate grounded answers from retrieved context."""

    UNAVAILABLE_MESSAGE = (
        "The requested information is not available "
        "in the provided sources."
    )

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def generate(
        self,
        question: str,
        retrieved_context: str,
        feedback: str | None = None,
    ) -> str:
        """Generate an answer using the supplied LLM."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        if not retrieved_context.strip():
            return self.UNAVAILABLE_MESSAGE

        prompt = self._build_prompt(
            question=question,
            retrieved_context=retrieved_context,
            feedback=feedback,
        )

        response = self._llm.invoke(prompt)

        return self._extract_content(response)

    @classmethod
    def _build_prompt(
        cls,
        question: str,
        retrieved_context: str,
        feedback: str | None = None,
    ) -> str:
        """Build a grounding-focused generation prompt."""
        feedback_section = ""

        if feedback and feedback.strip():
            feedback_section = f"""
Evaluator feedback:
{feedback}

Use the evaluator feedback to improve the answer while continuing
to rely ONLY on the provided context.
"""

        return f"""You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer the question,
say that the requested information is not available in the provided sources.
Do not invent facts or use information that is not supported by the context.
{feedback_section}
Context:
{retrieved_context}

Question:
{question}

Answer:"""

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract text from an LLM response."""
        if hasattr(response, "content"):
            return str(response.content)

        return str(response)