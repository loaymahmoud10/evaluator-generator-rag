"""Tests for the Evaluator class."""

import json
from unittest.mock import MagicMock

import pytest

from app.evaluator.evaluator import Evaluator
from app.evaluator.memory import EvaluatorMemory
from app.schemas.state import EvaluationDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(data: dict) -> MagicMock:
    """Create a mock LLM response containing the given JSON data."""
    return MagicMock(content=json.dumps(data))


GOOD_EVALUATION = {
    "decision": "ACCEPT",
    "score": 90,
    "criteria": {
        "accuracy": 95,
        "relevance": 92,
        "completeness": 88,
        "consistency": 90,
        "grounding": 85,
        "unsupported_claims": 90,
        "overall_quality": 90,
    },
    "feedback": [],
}

IMPROVE_EVALUATION = {
    "decision": "IMPROVE",
    "score": 55,
    "criteria": {
        "accuracy": 80,
        "relevance": 85,
        "completeness": 40,
        "consistency": 70,
        "grounding": 60,
        "unsupported_claims": 70,
        "overall_quality": 55,
    },
    "feedback": [
        {
            "criterion": "completeness",
            "severity": "high",
            "issue": "The answer does not explain embeddings.",
            "suggestion": "Add a section on embeddings.",
        }
    ],
}


# ---------------------------------------------------------------------------
# Core evaluation tests
# ---------------------------------------------------------------------------

def test_evaluate_returns_accept_for_good_answer():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI that learns from data.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.ACCEPT
    assert result["score"] == 90
    assert "accuracy" in result["criteria"]
    llm.invoke.assert_called_once()


def test_evaluate_returns_improve_for_weak_answer():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(IMPROVE_EVALUATION)

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is machine learning?",
        answer="It is related to computers.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.IMPROVE
    assert result["score"] == 55
    assert len(result["feedback"]) == 1
    assert result["feedback"][0]["criterion"] == "completeness"


def test_evaluate_rejects_empty_question():
    llm = MagicMock()
    evaluator = Evaluator(llm=llm)

    with pytest.raises(ValueError, match="question cannot be empty"):
        evaluator.evaluate(
            question="   ",
            answer="Some answer.",
            retrieved_context="Some context.",
        )

    llm.invoke.assert_not_called()


def test_evaluate_handles_empty_answer():
    llm = MagicMock()
    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.IMPROVE
    assert result["score"] == 0
    assert result["feedback"][0]["issue"] == "The answer is empty."
    llm.invoke.assert_not_called()


def test_evaluate_handles_invalid_json_response():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="This is not JSON")

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.IMPROVE
    assert result["feedback"][0]["issue"] == "Could not parse the evaluation response."


def test_evaluate_strips_markdown_code_blocks():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="```json\n" + json.dumps(GOOD_EVALUATION) + "\n```"
    )

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.ACCEPT
    assert result["score"] == 90


def test_evaluate_records_in_memory():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    memory = EvaluatorMemory()
    evaluator = Evaluator(llm=llm, memory=memory)

    evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert memory.get_iteration_count("What is ML?") == 1


def test_evaluate_includes_history_in_prompt():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    memory = EvaluatorMemory()
    memory.record(
        "What is ML?",
        IMPROVE_EVALUATION,
    )

    evaluator = Evaluator(llm=llm, memory=memory)

    evaluator.evaluate(
        question="What is ML?",
        answer="Improved answer.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    prompt = llm.invoke.call_args.args[0]
    assert "Previous evaluation history" in prompt
    assert "Iteration 1" in prompt


def test_evaluate_clamps_scores_out_of_range():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response({
        "decision": "ACCEPT",
        "score": 200,
        "criteria": {"accuracy": -10, "relevance": 300},
        "feedback": [],
    })

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["score"] == 100
    assert result["criteria"]["accuracy"] == 0
    assert result["criteria"]["relevance"] == 100


def test_evaluate_fixes_inconsistent_decision():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response({
        "decision": "ACCEPT",
        "score": 30,
        "criteria": {"accuracy": 30},
        "feedback": [],
    })

    evaluator = Evaluator(llm=llm)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.IMPROVE


# ---------------------------------------------------------------------------
# Redis caching tests
# ---------------------------------------------------------------------------

def test_evaluate_uses_redis_cache():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    redis_client = MagicMock()
    redis_client.get.return_value = None

    evaluator = Evaluator(llm=llm, redis_client=redis_client, cache_ttl=600)

    evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    redis_client.setex.assert_called_once()
    args = redis_client.setex.call_args
    assert args.args[0].startswith("eval:")
    assert args.args[1] == 600


def test_evaluate_returns_cached_result():
    llm = MagicMock()
    cached_result = json.dumps(GOOD_EVALUATION)

    redis_client = MagicMock()
    redis_client.get.return_value = cached_result

    evaluator = Evaluator(llm=llm, redis_client=redis_client)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    llm.invoke.assert_not_called()
    assert result["decision"] == EvaluationDecision.ACCEPT
    assert result["score"] == 90


def test_evaluate_works_without_redis():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    evaluator = Evaluator(llm=llm, redis_client=None)

    result = evaluator.evaluate(
        question="What is ML?",
        answer="Machine learning is AI.",
        retrieved_context="Machine learning is a branch of AI.",
    )

    assert result["decision"] == EvaluationDecision.ACCEPT


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

def test_evaluate_prompt_contains_all_inputs():
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response(GOOD_EVALUATION)

    evaluator = Evaluator(llm=llm)

    evaluator.evaluate(
        question="What is machine learning?",
        answer="ML is a subset of AI.",
        retrieved_context="Machine learning is a branch of artificial intelligence.",
    )

    prompt = llm.invoke.call_args.args[0]

    assert "What is machine learning?" in prompt
    assert "ML is a subset of AI." in prompt
    assert "Machine learning is a branch of artificial intelligence." in prompt
    assert "accuracy" in prompt
