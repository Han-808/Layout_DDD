from __future__ import annotations

from benchmark.workflow.graph import build_graph, run_workflow
from benchmark.workflow.state import BenchmarkState


def legend_build_graph():
    return build_graph()


def legend_run_workflow(initial_state: BenchmarkState) -> BenchmarkState:
    return run_workflow(initial_state)
