from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

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


AGENT_DONE = "done"
WorkflowCallback = Callable[["WorkflowEvent"], None]
WorkflowAction = Callable[[BenchmarkState], BenchmarkState]


@dataclass(frozen=True)
class WorkflowEvent:
    """Small event object for observing agent execution without coupling to UI."""

    event: str
    action: str
    step_index: int
    state: BenchmarkState
    route: str = ""


class WorkflowPolicy(Protocol):
    def next_action(self, state: BenchmarkState, *, last_action: str | None = None) -> str:
        ...


@dataclass
class DefaultWorkflowPolicy:
    """Default generation-mode policy.

    The policy owns routing decisions, so callers can replace it without
    rebuilding a LangGraph-style fixed state machine.
    """

    route_after_eval: Callable[[BenchmarkState], str] = route_after_eval

    def next_action(self, state: BenchmarkState, *, last_action: str | None = None) -> str:
        if last_action == "compute_metrics":
            return AGENT_DONE
        if "normalized_case" not in state:
            return "normalize_input"
        if "current_layout" not in state:
            return "generate_layout"
        if last_action in {"generate_layout", "repair_layout"} or "current_evaluation" not in state:
            return "evaluate_layout"
        if last_action == "evaluate_layout":
            route = self.route_after_eval(state)
            if route == "metrics":
                return "compute_metrics"
            if route == "repair":
                return "build_feedback"
            raise ValueError(f"Unsupported workflow route '{route}'.")
        if last_action == "build_feedback":
            return "repair_layout"
        return "compute_metrics"


@dataclass
class BenchmarkAgent:
    """Agent-style API for benchmark generation/evaluation workflows.

    This replaces the previous LangGraph-defined control flow with a small,
    explicit runner similar to standalone agent APIs: construct an object,
    optionally pass policy/callbacks, then call run()/invoke().
    """

    policy: WorkflowPolicy = field(default_factory=DefaultWorkflowPolicy)
    callbacks: list[WorkflowCallback] = field(default_factory=list)
    max_steps: int = 100
    actions: dict[str, WorkflowAction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actions:
            return
        self.actions = {
            "normalize_input": NormalizeInputNode(),
            "generate_layout": GenerateLayoutNode(),
            "evaluate_layout": EvaluateLayoutNode(),
            "build_feedback": BuildFeedbackNode(),
            "repair_layout": RepairLayoutNode(),
            "compute_metrics": ComputeMetricsNode(),
        }

    def run(self, initial_state: BenchmarkState) -> BenchmarkState:
        state: BenchmarkState = dict(initial_state)
        last_action: str | None = None
        step_index = 0
        self._emit("start", "workflow", step_index, state)
        while step_index < self.max_steps:
            action_name = self.policy.next_action(state, last_action=last_action)
            if action_name == AGENT_DONE:
                self._emit("done", "workflow", step_index, state)
                return state
            action = self.actions.get(action_name)
            if action is None:
                raise ValueError(f"Unknown benchmark workflow action '{action_name}'.")
            self._emit("before_action", action_name, step_index, state)
            state = action(state)
            step_index += 1
            self._emit("after_action", action_name, step_index, state)
            if action_name == "evaluate_layout":
                route = route_after_eval(state)
                self._emit("route", action_name, step_index, state, route=route)
            last_action = action_name
        raise RuntimeError(f"BenchmarkAgent exceeded max_steps={self.max_steps}.")

    def invoke(self, initial_state: BenchmarkState) -> BenchmarkState:
        """Compatibility method for callers that previously expected graph.invoke."""

        return self.run(initial_state)

    def _emit(self, event: str, action: str, step_index: int, state: BenchmarkState, *, route: str = "") -> None:
        if not self.callbacks:
            return
        payload = WorkflowEvent(event=event, action=action, step_index=step_index, state=state, route=route)
        for callback in self.callbacks:
            callback(payload)
