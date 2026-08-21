"""Tests for the Evaluator-Generator workflow orchestration.

These tests use lightweight fakes for the Generator and Evaluator so the
loop, memory isolation, iteration cap, and caching can be verified without
LLM calls or Redis.
"""

from unittest.mock import MagicMock

import pytest

from app.evaluator.evaluator import Evaluator  # noqa: F401 (type check)
from app.evaluator.memory import EvaluatorMemory
from app.generation.memory import GeneratorMemory
from app.retrieval.retriever import Retriever
from app.schemas.state import EvaluationDecision
from app.workflow.orchestrator import EvaluatorGeneratorWorkflow


class FakeGenerator:
    UNAVAILABLE_MESSAGE = "unavailable"

    def __init__(self):
        self.calls = []

    def generate(self, question, retrieved_context, feedback=None):
        self.calls.append({"question": question, "feedback": feedback})
        if not retrieved_context or not retrieved_context.strip():
            return self.UNAVAILABLE_MESSAGE
        n = len(self.calls)
        return f"answer version {n}"


class FakeEvaluator:
    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._idx = 0

    def evaluate(self, question, answer, retrieved_context):
        decision = self._decisions[
            min(self._idx, len(self._decisions) - 1)
        ]
        self._idx += 1
        score = 90 if decision == EvaluationDecision.ACCEPT else 50
        return {
            "decision": decision,
            "score": score,
            "criteria": {},
            "feedback": [
                {
                    "criterion": "completeness",
                    "severity": "high",
                    "issue": "needs more detail",
                    "suggestion": "add detail",
                }
            ]
            if decision == EvaluationDecision.IMPROVE
            else [],
        }


def _fake_retriever(context="retrieved context"):
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = {
        "retrieved_context": context,
        "sources": [
            {
                "source_id": "s1",
                "source_type": "pdf",
                "source_name": "doc.pdf",
                "location": "page 1",
                "content": context,
            }
        ],
    }
    return retriever


def test_accept_on_second_iteration():
    gen = FakeGenerator()
    eva = FakeEvaluator(
        [EvaluationDecision.IMPROVE, EvaluationDecision.ACCEPT]
    )
    wf = EvaluatorGeneratorWorkflow(
        generator=gen,
        evaluator=eva,
        retriever=_fake_retriever(),
        max_iterations=4,
    )
    result = wf.run("What is ML?")

    assert result["validated"] is True
    assert result["decision"] == "ACCEPT"
    assert result["iterations"] == 2
    assert result["reason"] == "evaluator_approved"
    # Generator was called twice; second call received feedback.
    assert len(gen.calls) == 2
    assert gen.calls[1]["feedback"] is not None
    assert "answer version 2" == result["final_answer"]


def test_max_iterations_enforced():
    gen = FakeGenerator()
    eva = FakeEvaluator([EvaluationDecision.IMPROVE] * 10)
    wf = EvaluatorGeneratorWorkflow(
        generator=gen,
        evaluator=eva,
        retriever=_fake_retriever(),
        max_iterations=4,
    )
    result = wf.run("Explain transformers.")

    assert result["validated"] is False
    assert result["decision"] == "IMPROVE"
    assert result["iterations"] == 4
    assert result["reason"] == "max_iterations_reached"
    assert len(gen.calls) == 4


def test_memory_isolation_is_enforced():
    gen = FakeGenerator()
    eva = FakeEvaluator([EvaluationDecision.ACCEPT])

    # Cannot use EvaluatorMemory as the Generator memory.
    with pytest.raises(TypeError):
        EvaluatorGeneratorWorkflow(
            generator=gen,
            evaluator=eva,
            retriever=_fake_retriever(),
            generator_memory=EvaluatorMemory(),
        )

    # Cannot use GeneratorMemory as the Evaluator memory.
    with pytest.raises(TypeError):
        EvaluatorGeneratorWorkflow(
            generator=gen,
            evaluator=eva,
            retriever=_fake_retriever(),
            evaluator_memory=GeneratorMemory(),
        )


def test_memories_are_populated_and_distinct():
    gen = FakeGenerator()
    eva = FakeEvaluator(
        [EvaluationDecision.IMPROVE, EvaluationDecision.ACCEPT]
    )
    gen_mem = GeneratorMemory()
    eva_mem = EvaluatorMemory()
    wf = EvaluatorGeneratorWorkflow(
        generator=gen,
        evaluator=eva,
        retriever=_fake_retriever(),
        generator_memory=gen_mem,
        evaluator_memory=eva_mem,
    )
    wf.run("Q?")

    assert gen_mem.get_iteration_count("Q?") == 2
    assert eva_mem.get_iteration_count("Q?") == 2
    # The two memories are different objects and different types.
    assert gen_mem is not eva_mem
    assert not isinstance(gen_mem, EvaluatorMemory)


def test_answer_cache_short_circuits():
    gen = FakeGenerator()
    eva = FakeEvaluator([EvaluationDecision.ACCEPT])
    redis = MagicMock()
    redis.get_retrieval.return_value = None
    redis.get_answer.return_value = {
        "final_answer": "cached answer",
        "validated": True,
        "iterations": 1,
    }
    wf = EvaluatorGeneratorWorkflow(
        generator=gen,
        evaluator=eva,
        retriever=_fake_retriever(),
        redis_cache=redis,
    )
    result = wf.run("cached question?")

    assert result["reason"] == "cache_hit"
    assert result["final_answer"] == "cached answer"
    # Generator/Evaluator never ran.
    assert len(gen.calls) == 0


def test_unknown_question_returns_unavailable_when_no_context():
    gen = FakeGenerator()
    eva = FakeEvaluator([EvaluationDecision.ACCEPT])
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = {
        "retrieved_context": "",
        "sources": [],
    }
    wf = EvaluatorGeneratorWorkflow(
        generator=gen,
        evaluator=eva,
        retriever=retriever,
    )
    result = wf.run("Something not in sources?")
    assert result["final_answer"] == "unavailable"
    assert len(gen.calls) == 1
