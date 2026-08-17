"""Shared workflow state contract for the Evaluator-Generator pipeline.

This module defines the interface between the Generator, Evaluator, and
LCEL workflow. Keep the field meanings stable so both contributors can work
independently against the same contract.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class EvaluationDecision(str, Enum):
    """Possible decisions produced by the Evaluator."""

    ACCEPT = "ACCEPT"
    IMPROVE = "IMPROVE"


class SourceReference(TypedDict, total=False):
    """Metadata identifying the origin of retrieved knowledge."""

    source_id: str
    source_type: str
    source_name: str
    location: str
    content: str


class EvaluationResult(TypedDict, total=False):
    """Structured result produced by the Evaluator."""

    decision: EvaluationDecision
    feedback: str


class WorkflowState(TypedDict, total=False):
    """Shared state passed through the Evaluator-Generator workflow.

    Field ownership:
    - question: workflow/application
    - retrieved_context: retrieval/Generator side
    - sources: retrieval/Generator side
    - generated_answer: Generator side
    - evaluation: Evaluator side
    - iteration: workflow/orchestrator
    """

    question: str
    retrieved_context: str
    sources: list[SourceReference]
    generated_answer: str
    evaluation: EvaluationResult
    iteration: int
