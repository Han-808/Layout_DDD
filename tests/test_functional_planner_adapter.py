from __future__ import annotations

import pytest

from benchmark.evaluator.scene_quality.functional_planner_adapter import (
    FUNCTIONAL_DISCOVERY_PLANNER_MODE,
    LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE,
    FunctionalEvidencePlannerAdapter,
)


def test_discovery_contract_precedes_legacy_and_normalizes_audit() -> None:
    calls: list[str] = []

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            calls.append("discovery")
            return {
                "reason": "discover first",
                "provenance": {
                    "request_metadata": {"model": "discovery"}
                },
            }

        def plan_functional_evidence(self, request: dict) -> dict:
            calls.append("legacy")
            return {"probe_units": []}

    adapter = FunctionalEvidencePlannerAdapter(Planner())
    execution = adapter.execute(
        {"metric": "functional_consistency"},
        build_plan_from_discovery=lambda discovery: {
            "probe_units": [],
            "source_reason": discovery["reason"],
        },
    )

    assert calls == ["discovery"]
    assert execution.mode == FUNCTIONAL_DISCOVERY_PLANNER_MODE
    assert execution.plan["source_reason"] == "discover first"
    assert execution.request_metadata == {"model": "discovery"}
    assert execution.reason == "discover first"


def test_legacy_contract_uses_the_same_normalized_execution_shape() -> None:
    class Planner:
        def plan_functional_evidence(self, request: dict) -> dict:
            return {
                "probe_units": [],
                "reason": "legacy plan",
                "request_metadata": {"model": "legacy"},
            }

    adapter = FunctionalEvidencePlannerAdapter(Planner())
    execution = adapter.execute(
        {"metric": "functional_consistency"},
        build_plan_from_discovery=lambda discovery: pytest.fail(
            "legacy planning must not use discovery processing"
        ),
    )

    assert execution.mode == LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE
    assert execution.discovery is None
    assert execution.plan["probe_units"] == []
    assert execution.request_metadata == {"model": "legacy"}
    assert execution.reason == "legacy plan"


def test_unconfigured_planner_is_reported_without_invocation() -> None:
    adapter = FunctionalEvidencePlannerAdapter(object())

    assert adapter.configured is False
    with pytest.raises(
        RuntimeError,
        match="functional evidence planner is not configured",
    ):
        adapter.execute(
            {},
            build_plan_from_discovery=lambda discovery: discovery,
        )
