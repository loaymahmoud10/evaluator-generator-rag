"""Tests for the EvaluatorMemory class."""

import pytest

from app.evaluator.memory import EvaluatorMemory


def test_record_stores_evaluation():
    memory = EvaluatorMemory()
    evaluation = {
        "decision": "ACCEPT",
        "score": 85,
        "criteria": {"accuracy": 90},
        "feedback": [],
    }

    memory.record("What is ML?", evaluation)

    history = memory.get_history("What is ML?")
    assert len(history) == 1
    assert history[0] == evaluation


def test_record_appends_multiple_evaluations():
    memory = EvaluatorMemory()
    evaluation_1 = {"decision": "IMPROVE", "score": 60, "criteria": {}, "feedback": []}
    evaluation_2 = {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []}

    memory.record("What is ML?", evaluation_1)
    memory.record("What is ML?", evaluation_2)

    history = memory.get_history("What is ML?")
    assert len(history) == 2
    assert history[0]["score"] == 60
    assert history[1]["score"] == 85


def test_record_rejects_empty_question():
    memory = EvaluatorMemory()

    with pytest.raises(ValueError, match="question cannot be empty"):
        memory.record("   ", {"decision": "ACCEPT", "score": 80, "criteria": {}, "feedback": []})


def test_get_history_returns_empty_for_unknown_question():
    memory = EvaluatorMemory()

    assert memory.get_history("unknown question") == []


def test_get_last_evaluation_returns_none_when_empty():
    memory = EvaluatorMemory()

    assert memory.get_last_evaluation("What is ML?") is None


def test_get_last_evaluation_returns_most_recent():
    memory = EvaluatorMemory()
    evaluation_1 = {"decision": "IMPROVE", "score": 60, "criteria": {}, "feedback": []}
    evaluation_2 = {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []}

    memory.record("What is ML?", evaluation_1)
    memory.record("What is ML?", evaluation_2)

    last = memory.get_last_evaluation("What is ML?")
    assert last is not None
    assert last["score"] == 85


def test_get_iteration_count():
    memory = EvaluatorMemory()

    assert memory.get_iteration_count("What is ML?") == 0

    memory.record("What is ML?", {"decision": "IMPROVE", "score": 60, "criteria": {}, "feedback": []})
    assert memory.get_iteration_count("What is ML?") == 1

    memory.record("What is ML?", {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []})
    assert memory.get_iteration_count("What is ML?") == 2


def test_format_history_for_prompt_returns_none_when_empty():
    memory = EvaluatorMemory()

    assert memory.format_history_for_prompt("What is ML?") is None


def test_format_history_for_prompt_formats_correctly():
    memory = EvaluatorMemory()
    evaluation = {
        "decision": "IMPROVE",
        "score": 60,
        "criteria": {},
        "feedback": [
            {
                "criterion": "completeness",
                "severity": "high",
                "issue": "Missing explanation of embeddings.",
                "suggestion": "Add a section on embeddings.",
            }
        ],
    }

    memory.record("What is ML?", evaluation)

    formatted = memory.format_history_for_prompt("What is ML?")

    assert formatted is not None
    assert "Iteration 1" in formatted
    assert "IMPROVE" in formatted
    assert "60" in formatted
    assert "Missing explanation of embeddings." in formatted
    assert "Add a section on embeddings." in formatted


def test_clear_removes_specific_question():
    memory = EvaluatorMemory()
    memory.record("What is ML?", {"decision": "IMPROVE", "score": 60, "criteria": {}, "feedback": []})
    memory.record("What is Python?", {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []})

    memory.clear("What is ML?")

    assert memory.get_history("What is ML?") == []
    assert len(memory.get_history("What is Python?")) == 1


def test_clear_removes_all_history():
    memory = EvaluatorMemory()
    memory.record("What is ML?", {"decision": "IMPROVE", "score": 60, "criteria": {}, "feedback": []})
    memory.record("What is Python?", {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []})

    memory.clear()

    assert memory.get_history("What is ML?") == []
    assert memory.get_history("What is Python?") == []


def test_questions_are_case_insensitive():
    memory = EvaluatorMemory()
    evaluation = {"decision": "ACCEPT", "score": 85, "criteria": {}, "feedback": []}

    memory.record("What is ML?", evaluation)

    assert len(memory.get_history("what is ml?")) == 1
    assert memory.get_iteration_count("WHAT IS ML?") == 1
