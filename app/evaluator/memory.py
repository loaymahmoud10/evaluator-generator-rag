"""Evaluator-specific memory for tracking evaluation history."""

from __future__ import annotations

from app.schemas.state import EvaluationResult, FeedbackItem


class EvaluatorMemory:
    """Maintain evaluation history per question.

    This is separate from Generator memory.  It tracks previous
    evaluations so the Evaluator can avoid repeating itself and
    can detect whether earlier feedback was addressed.

    Storage is in-memory (dict).  Swap the underlying store for Redis
    or a database when persistence is required.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[EvaluationResult]] = {}

    def record(
        self,
        question: str,
        evaluation: EvaluationResult,
    ) -> None:
        """Append an evaluation result for the given question."""
        if not question.strip():
            raise ValueError("question cannot be empty")

        key = question.strip().lower()
        self._history.setdefault(key, []).append(evaluation)

    def get_history(self, question: str) -> list[EvaluationResult]:
        """Return all previous evaluations for a question."""
        key = question.strip().lower()
        return list(self._history.get(key, []))

    def get_last_evaluation(self, question: str) -> EvaluationResult | None:
        """Return the most recent evaluation for a question, or None."""
        history = self.get_history(question)
        return history[-1] if history else None

    def get_iteration_count(self, question: str) -> int:
        """Return how many times a question has been evaluated."""
        return len(self.get_history(question))

    def format_history_for_prompt(self, question: str) -> str | None:
        """Format previous evaluations into a human-readable string.

        Returns None when there is no history, so callers can
        conditionally include the section in the prompt.
        """
        history = self.get_history(question)

        if not history:
            return None

        lines: list[str] = []

        for idx, evaluation in enumerate(history, start=1):
            iteration = f"Iteration {idx}"

            decision = evaluation.get("decision", "UNKNOWN")
            score = evaluation.get("score", "N/A")

            lines.append(f"{iteration}: decision={decision}, score={score}")

            for item in evaluation.get("feedback", []):
                severity = item.get("severity", "")
                issue = item.get("issue", "")
                suggestion = item.get("suggestion", "")
                lines.append(
                    f"  - [{severity}] {issue} -> {suggestion}"
                )

        return "\n".join(lines)

    def clear(self, question: str | None = None) -> None:
        """Clear evaluation history.

        If a question is provided, clear only that question's history.
        Otherwise clear all history.
        """
        if question is not None:
            key = question.strip().lower()
            self._history.pop(key, None)
        else:
            self._history.clear()
