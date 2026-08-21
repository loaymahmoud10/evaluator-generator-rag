"""LCEL workflow graph construction.

Builds a single-iteration ``RunnableSequence`` (generate -> evaluate) from
the agent runnables. The iterative loop (max 4 iterations) is driven by the
orchestrator, which repeatedly invokes this graph and feeds the Evaluator
feedback back into the Generator.
"""

from __future__ import annotations

from app.evaluator.evaluator import Evaluator
from app.generation.generator import Generator
from app.workflow.chains import evaluator_step, generator_step


class WorkflowGraph:
    """Compose the generate->evaluate chain using LCEL primitives."""

    def __init__(self, generator: Generator, evaluator: Evaluator) -> None:
        self._generator = generator
        self._evaluator = evaluator
        self._iteration = generator_step(generator) | evaluator_step(evaluator)

    @property
    def iteration_runnable(self):
        """The generate->evaluate ``Runnable`` for one loop iteration."""
        return self._iteration

    def run_iteration(self, state: dict) -> dict:
        """Execute one generate->evaluate pass and return the updated state."""
        return self._iteration.invoke(state)
