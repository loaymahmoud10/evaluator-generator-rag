"""Tests for the isolated Generator memory."""

import pytest

from app.generation.memory import GeneratorMemory


def test_record_and_history():
    mem = GeneratorMemory()
    mem.record("Q?", iteration=1, retrieved_context="ctx", answer="a1", feedback=None)
    mem.record(
        "Q?",
        iteration=2,
        retrieved_context="ctx",
        answer="a2",
        feedback="improve",
    )

    history = mem.get_history("Q?")
    assert len(history) == 2
    assert history[0]["answer"] == "a1"
    assert history[1]["feedback"] == "improve"


def test_get_last_answer():
    mem = GeneratorMemory()
    mem.record("Q?", iteration=1, retrieved_context="c", answer="first")
    mem.record("Q?", iteration=2, retrieved_context="c", answer="second")
    assert mem.get_last_answer("Q?") == "second"
    assert mem.get_last_answer("unknown") is None


def test_iteration_count():
    mem = GeneratorMemory()
    assert mem.get_iteration_count("Q?") == 0
    mem.record("Q?", iteration=1, retrieved_context="c", answer="a")
    assert mem.get_iteration_count("Q?") == 1


def test_rejects_empty_question():
    mem = GeneratorMemory()
    with pytest.raises(ValueError, match="question cannot be empty"):
        mem.record("   ", iteration=1, retrieved_context="c", answer="a")


def test_format_history_returns_none_when_empty():
    mem = GeneratorMemory()
    assert mem.format_history_for_prompt("Q?") is None


def test_format_history_contains_attempts():
    mem = GeneratorMemory()
    mem.record("Q?", iteration=1, retrieved_context="c", answer="a1", feedback="fb")
    formatted = mem.format_history_for_prompt("Q?")
    assert formatted is not None
    assert "Attempt 1" in formatted
    assert "a1" in formatted
    assert "fb" in formatted


def test_clear_specific_and_all():
    mem = GeneratorMemory()
    mem.record("Q1", iteration=1, retrieved_context="c", answer="a")
    mem.record("Q2", iteration=1, retrieved_context="c", answer="b")
    mem.clear("Q1")
    assert mem.get_history("Q1") == []
    assert len(mem.get_history("Q2")) == 1
    mem.clear()
    assert mem.get_history("Q2") == []
