"""LLM-based answer evaluation for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.evaluator.memory import EvaluatorMemory
from app.evaluator.prompts import build_evaluation_prompt
from app.schemas.state import EvaluationDecision, EvaluationResult


class Evaluator:
    """Evaluate generated answers for quality and grounding.

    The Evaluator uses an LLM to score an answer on multiple criteria
    (accuracy, relevance, completeness, consistency, grounding,
    unsupported claims) and decides whether to ACCEPT or IMPROVE.

    Redis caching is optional.  When a Redis client is provided, the
    evaluator caches LLM responses keyed on the hash of the full
    prompt inputs to avoid redundant LLM calls.
    """

    DEFAULT_THRESHOLD = 75

    def __init__(
        self,
        llm: Any,
        memory: EvaluatorMemory | None = None,
        redis_client: Any | None = None,
        cache_ttl: int = 3600,
    ) -> None:
        self._llm = llm
        self._memory = memory or EvaluatorMemory()
        self._redis = redis_client
        self._cache_ttl = cache_ttl

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_context: str,
    ) -> EvaluationResult:
        """Evaluate a generated answer and return a structured result.

        Parameters
        ----------
        question:
            The original user question.
        answer:
            The generated answer to evaluate.
        retrieved_context:
            The context that was provided to the Generator.

        Returns
        -------
        EvaluationResult
            A TypedDict containing decision, score, criteria, and feedback.
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        if not answer.strip():
            return self._empty_answer_result()

        evaluation_history = self._memory.format_history_for_prompt(question)

        prompt = build_evaluation_prompt(
            question=question,
            answer=answer,
            retrieved_context=retrieved_context,
            evaluation_history=evaluation_history,
        )

        raw_response = self._llm_with_cache(prompt)

        evaluation = self._parse_evaluation(raw_response)

        self._memory.record(question, evaluation)

        return evaluation

    def _llm_with_cache(self, prompt: str) -> str:
        """Call the LLM, using Redis cache when available."""
        cache_key = self._build_cache_key(prompt)

        if self._redis is not None and cache_key:
            cached = self._redis.get(cache_key)
            if cached is not None:
                return cached

        response = self._llm.invoke(prompt)
        content = self._extract_content(response)

        if self._redis is not None and cache_key:
            self._redis.setex(
                cache_key,
                self._cache_ttl,
                content,
            )

        return content

    def _parse_evaluation(self, raw_response: str) -> EvaluationResult:
        """Parse the LLM JSON response into an EvaluationResult."""
        try:
            text = raw_response.strip()

            if text.startswith("```"):
                lines = text.split("\n")
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                ]
                text = "\n".join(lines).strip()

            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return self._fallback_evaluation()

        decision_str = data.get("decision", "IMPROVE")

        try:
            decision = EvaluationDecision(decision_str)
        except ValueError:
            decision = EvaluationDecision.IMPROVE

        score = int(data.get("score", 50))
        score = max(0, min(100, score))

        criteria = data.get("criteria", {})
        normalized_criteria = {}
        for key, value in criteria.items():
            normalized_criteria[key] = max(0, min(100, int(value)))

        raw_feedback = data.get("feedback", [])
        feedback = []
        for item in raw_feedback:
            feedback.append({
                "criterion": str(item.get("criterion", "")),
                "severity": str(item.get("severity", "medium")),
                "issue": str(item.get("issue", "")),
                "suggestion": str(item.get("suggestion", "")),
            })

        if score >= self.DEFAULT_THRESHOLD and decision == EvaluationDecision.IMPROVE:
            decision = EvaluationDecision.ACCEPT
        elif score < self.DEFAULT_THRESHOLD and decision == EvaluationDecision.ACCEPT:
            decision = EvaluationDecision.IMPROVE

        return {
            "decision": decision,
            "score": score,
            "criteria": normalized_criteria,
            "feedback": feedback,
        }

    def _empty_answer_result(self) -> EvaluationResult:
        """Return an IMPROVE result when the answer is empty."""
        result: EvaluationResult = {
            "decision": EvaluationDecision.IMPROVE,
            "score": 0,
            "criteria": {},
            "feedback": [
                {
                    "criterion": "completeness",
                    "severity": "high",
                    "issue": "The answer is empty.",
                    "suggestion": "Provide a substantive answer using the retrieved context.",
                },
            ],
        }

        return result

    def _fallback_evaluation(self) -> EvaluationResult:
        """Return a safe fallback when JSON parsing fails."""
        return {
            "decision": EvaluationDecision.IMPROVE,
            "score": 50,
            "criteria": {},
            "feedback": [
                {
                    "criterion": "overall_quality",
                    "severity": "medium",
                    "issue": "Could not parse the evaluation response.",
                    "suggestion": "Retry the evaluation.",
                },
            ],
        }

    @staticmethod
    def _build_cache_key(prompt: str) -> str:
        """Build a SHA-256 cache key from the prompt content."""
        return "eval:" + hashlib.sha256(prompt.encode()).hexdigest()

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract text from an LLM response."""
        if hasattr(response, "content"):
            return str(response.content)

        return str(response)

    @property
    def memory(self) -> EvaluatorMemory:
        """Expose the EvaluatorMemory instance."""
        return self._memory
