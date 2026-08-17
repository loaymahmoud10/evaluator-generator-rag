import pytest
from unittest.mock import MagicMock

from app.generation.generator import Generator


def test_generator_generates_answer_from_retrieved_context():
    llm = MagicMock()

    llm.invoke.return_value = MagicMock(
        content="Machine learning is a subset of artificial intelligence."
    )

    generator = Generator(llm=llm)

    result = generator.generate(
        question="What is machine learning?",
        retrieved_context=(
            "Machine learning is a subset of artificial intelligence."
        ),
    )

    assert result == (
        "Machine learning is a subset of artificial intelligence."
    )

    llm.invoke.assert_called_once()


def test_generator_returns_unavailable_message_when_context_is_empty():
    llm = MagicMock()

    generator = Generator(llm=llm)

    result = generator.generate(
        question="What is quantum computing?",
        retrieved_context="",
    )

    assert result == (
        "The requested information is not available "
        "in the provided sources."
    )

    llm.invoke.assert_not_called()


def test_generator_rejects_empty_question():
    llm = MagicMock()

    generator = Generator(llm=llm)

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        generator.generate(
            question="   ",
            retrieved_context="Some context.",
        )

    llm.invoke.assert_not_called()


def test_generator_extracts_string_llm_response():
    llm = MagicMock()

    llm.invoke.return_value = "Direct string response"

    generator = Generator(llm=llm)

    result = generator.generate(
        question="What is machine learning?",
        retrieved_context="Machine learning is a field of AI.",
    )

    assert result == "Direct string response"


def test_generator_prompt_contains_question_and_context():
    llm = MagicMock()

    llm.invoke.return_value = MagicMock(
        content="Generated answer"
    )

    generator = Generator(llm=llm)

    generator.generate(
        question="What is machine learning?",
        retrieved_context="Machine learning learns from data.",
    )

    prompt = llm.invoke.call_args.args[0]

    assert "What is machine learning?" in prompt
    assert "Machine learning learns from data." in prompt
    assert "ONLY the provided context" in prompt
    assert "Do not invent facts" in prompt


def test_generator_uses_evaluator_feedback():
    llm = MagicMock()

    llm.invoke.return_value = MagicMock(
        content="Improved answer"
    )

    generator = Generator(llm=llm)

    result = generator.generate(
        question="What is machine learning?",
        retrieved_context="Machine learning learns from data.",
        feedback="Explain the relationship between machine learning and AI.",
    )

    assert result == "Improved answer"

    prompt = llm.invoke.call_args.args[0]

    assert "Explain the relationship between machine learning and AI." in prompt
    assert "evaluator feedback" in prompt.lower()