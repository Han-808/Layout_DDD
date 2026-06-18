from __future__ import annotations

from benchmark.workflow.nodes import (
    BuildFeedbackNode,
    ComputeMetricsNode,
    EvaluateLayoutNode,
    GenerateLayoutNode,
    NormalizeInputNode,
    RepairLayoutNode,
    route_after_eval,
)
from benchmark.workflow.state import BenchmarkState


def build_graph():
    """Build the LangGraph state machine.

    If LangGraph is not installed yet, returns a small compatible runner with an
    invoke(state) method. The project dependency list includes LangGraph, so the
    real graph is used after installation.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:  # pragma: no cover - exercised only before deps install
        return SequentialBenchmarkGraph()

    graph = StateGraph(BenchmarkState)
    graph.add_node("normalize_input", NormalizeInputNode())
    graph.add_node("generate_layout", GenerateLayoutNode())
    graph.add_node("evaluate_layout", EvaluateLayoutNode())
    graph.add_node("build_feedback", BuildFeedbackNode())
    graph.add_node("repair_layout", RepairLayoutNode())
    graph.add_node("compute_metrics", ComputeMetricsNode())

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "generate_layout")
    graph.add_edge("generate_layout", "evaluate_layout")
    graph.add_conditional_edges(
        "evaluate_layout",
        route_after_eval,
        {
            "repair": "build_feedback",
            "metrics": "compute_metrics",
        },
    )
    graph.add_edge("build_feedback", "repair_layout")
    graph.add_edge("repair_layout", "evaluate_layout")
    graph.add_edge("compute_metrics", END)
    return graph.compile()


def run_workflow(initial_state: BenchmarkState) -> BenchmarkState:
    return build_graph().invoke(initial_state)


class SequentialBenchmarkGraph:
    """Small invoke-compatible fallback mirroring the intended graph route."""

    def invoke(self, state: BenchmarkState) -> BenchmarkState:
        state = NormalizeInputNode()(state)
        state = GenerateLayoutNode()(state)
        while True:
            state = EvaluateLayoutNode()(state)
            route = route_after_eval(state)
            if route == "metrics":
                return ComputeMetricsNode()(state)
            state = BuildFeedbackNode()(state)
            state = RepairLayoutNode()(state)
