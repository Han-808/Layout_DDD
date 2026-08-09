"""Narrow adapter for current and legacy functional evidence planners.

This module owns capability detection and return-shape normalization only.
Metric-specific discovery processing remains in the active orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


FUNCTIONAL_DISCOVERY_PLANNER_MODE = "functional_discovery_v4"
FUNCTIONAL_DISCOVERY_COMPATIBLE_PLANNER_MODES = frozenset(
    {
        "functional_discovery_v3",
        FUNCTIONAL_DISCOVERY_PLANNER_MODE,
    }
)
LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE = (
    "legacy_functional_probe_plan_v2"
)


def is_functional_discovery_planner_mode(value: Any) -> bool:
    """Whether an audit uses the current or frozen typed discovery route."""

    return str(value or "") in FUNCTIONAL_DISCOVERY_COMPATIBLE_PLANNER_MODES


class FunctionalDiscoveryPlanner(Protocol):
    """Current planner contract used by the functional workflow."""

    def discover_functional_evidence(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


class LegacyFunctionalProbePlanner(Protocol):
    """Compatibility contract retained for existing callers and fixtures."""

    def plan_functional_evidence(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


DiscoveryPlanBuilder = Callable[[Any], Any]


@dataclass(frozen=True)
class FunctionalPlannerExecution:
    """Normalized result of one planner invocation."""

    plan: Any
    discovery: Any | None
    mode: str
    request_metadata: Any | None
    reason: str


class FunctionalEvidencePlannerAdapter:
    """Expose one invocation path over current and legacy planners.

    The current discovery contract takes precedence when an implementation
    exposes both methods, matching the historical inline dispatch.
    """

    def __init__(self, planner: Any) -> None:
        self._discovery_call = _callable_attribute(
            planner,
            "discover_functional_evidence",
        )
        self._legacy_plan_call = _callable_attribute(
            planner,
            "plan_functional_evidence",
        )

    @property
    def configured(self) -> bool:
        """Whether either supported planner contract is available."""

        return (
            self._discovery_call is not None
            or self._legacy_plan_call is not None
        )

    def execute(
        self,
        request: dict[str, Any],
        *,
        build_plan_from_discovery: DiscoveryPlanBuilder,
    ) -> FunctionalPlannerExecution:
        """Invoke the preferred contract and normalize audit-facing fields."""

        if self._discovery_call is not None:
            discovery = self._discovery_call(request)
            plan = build_plan_from_discovery(discovery)
            return FunctionalPlannerExecution(
                plan=plan,
                discovery=discovery,
                mode=FUNCTIONAL_DISCOVERY_PLANNER_MODE,
                request_metadata=_discovery_request_metadata(discovery),
                reason=_mapping_reason(discovery),
            )
        if self._legacy_plan_call is not None:
            plan = self._legacy_plan_call(request)
            return FunctionalPlannerExecution(
                plan=plan,
                discovery=None,
                mode=LEGACY_FUNCTIONAL_PROBE_PLANNER_MODE,
                request_metadata=_mapping_value(
                    plan,
                    "request_metadata",
                ),
                reason=_mapping_reason(plan),
            )
        raise RuntimeError("functional evidence planner is not configured")


def _callable_attribute(
    value: Any,
    attribute: str,
) -> Callable[[dict[str, Any]], Any] | None:
    candidate = getattr(value, attribute, None)
    return candidate if callable(candidate) else None


def _mapping_value(value: Any, key: str) -> Any | None:
    return value.get(key) if isinstance(value, dict) else None


def _mapping_reason(value: Any) -> str:
    return (
        str(value.get("reason") or "")
        if isinstance(value, dict)
        else ""
    )


def _discovery_request_metadata(discovery: Any) -> Any | None:
    if not isinstance(discovery, dict):
        return None
    provenance = discovery.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return provenance.get("request_metadata")
