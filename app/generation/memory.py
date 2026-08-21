"""Generator-specific memory, fully isolated from the Evaluator memory.

The Generator memory stores only information relevant to producing
answers: conversation context, the retrieved context, every generated
answer attempt, and the evaluator feedback that triggered each
improvement.

This class is intentionally a *different type* from
``app.evaluator.memory.EvaluatorMemory`` and the orchestrator never
exposes one memory to the other agent, preventing accidental
cross-access.
"""

from __future__ import annotations

from typing import TypedDict


class GeneratorAttempt(TypedDict, total=False):
    """A single generation attempt within an iterative workflow."""

    iteration: int
    retrieved_context: str
    answer: str
    feedback: str | None


class GeneratorMemory:
    """Track per-question generation history for the Generator agent."""

    def __init__(self) -> None:
        self._history: dict[str, list[GeneratorAttempt]] = {}

    # ------------------------------------------------------------------ #
    def record(
        self,
        question: str,
        iteration: int,
        retrieved_context: str,
        answer: str,
        feedback: str | None = None,
    ) -> None:
        """Append a generation attempt for the given question."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        attempt: GeneratorAttempt = {
            "iteration": iteration,
            "retrieved_context": retrieved_context,
            "answer": answer,
            "feedback": feedback,
        }
        self._history.setdefault(question.strip().lower(), []).append(attempt)

    def get_history(self, question: str) -> list[GeneratorAttempt]:
        """Return all generation attempts for a question."""
        return list(self._history.get(question.strip().lower(), []))

    def get_last_answer(self, question: str) -> str | None:
        """Return the most recent generated answer, or None."""
        history = self.get_history(question)
        return history[-1]["answer"] if history else None

    def get_iteration_count(self, question: str) -> int:
        """Return how many times a question has been answered."""
        return len(self.get_history(question))

    def format_history_for_prompt(self, question: str) -> str | None:
        """Format previous generation attempts into a string for the prompt.

        Returns None when there is no history.
        """
        history = self.get_history(question)
        if not history:
            return None

        lines: list[str] = []
        for attempt in history:
            iteration = attempt.get("iteration", "?")
            answer = attempt.get("answer", "")
            feedback = attempt.get("feedback")
            lines.append(f"Attempt {iteration}:")
            lines.append(f"  Answer: {answer}")
            if feedback:
                lines.append(f"  Feedback that produced this attempt: {feedback}")

        return "\n".join(lines)

    def clear(self, question: str | None = None) -> None:
        """Clear history, optionally for a single question only."""
        if question is not None:
            self._history.pop(question.strip().lower(), None)
        else:
            self._history.clear()
