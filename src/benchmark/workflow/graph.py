from __future__ import annotations

from benchmark.workflow.agent import BenchmarkAgent
from benchmark.workflow.state import BenchmarkState


def build_graph() -> BenchmarkAgent:
    """Return an invoke-compatible benchmark agent.

    Kept for backward compatibility with older callers/tests that imported a
    "graph" object. The workflow is no longer defined by LangGraph.
    """

    return BenchmarkAgent()


def run_workflow(initial_state: BenchmarkState, *, agent: BenchmarkAgent | None = None) -> BenchmarkState:
    runner = agent or BenchmarkAgent()
    return runner.run(initial_state)
