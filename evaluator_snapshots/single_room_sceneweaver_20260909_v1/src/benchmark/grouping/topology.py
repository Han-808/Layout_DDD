from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.object_grouping import (
    build_object_grouping_report,
)
from benchmark.grouping.interfaces import (
    GroupingRequest,
    GroupingResult,
)
from benchmark.grouping.scene import normalize_grouping_scene


TOPOLOGY_GROUPING_POLICY_ID = "topology_metadata_geometry_v2"


class TopologyGroupingAlgorithm:
    """Deterministic graph partition refined from the existing grouping path.

    Vertices are renderable objects. Semantic regions, explicit support and
    attachment links, visible relations, derived support, metadata locality,
    and scale-aware proximity supply graph edges. Existing group-size and
    footprint limits remain authoritative inside the reused implementation.
    """

    backend = "topology"
    policy_id = TOPOLOGY_GROUPING_POLICY_ID

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is not None and not isinstance(config, dict):
            raise TypeError("topology grouping config must be a JSON object")
        self.config = deepcopy(config or {})

    def group(self, request: GroupingRequest) -> GroupingResult:
        request = GroupingRequest.from_value(request)
        scene = normalize_grouping_scene(request.scene)
        effective_config = _effective_topology_config(
            self.config,
            request.config,
        )
        report = build_object_grouping_report(
            scene.layout(),
            scene.case(request.case),
            {"grouping": effective_config},
        )
        return GroupingResult.create(
            groups=report.get("object_groups") or [],
            expected_object_ids=scene.object_ids,
            backend=self.backend,
            policy_id=self.policy_id,
            reason=(
                "Deterministic connected components over semantic-region, "
                "support, relation, derived-support, metadata, and "
                "scale-aware proximity edges."
            ),
            provenance={
                **scene.provenance(),
                "implementation": (
                    "benchmark.evaluator.object_grouping."
                    "build_object_grouping_report"
                ),
                "legacy_policy_id": "deterministic_metadata_geometry",
                "deterministic": True,
                "model_calls": 0,
            },
            resolved_grouping_config={
                **deepcopy(
                    report.get("resolved_grouping_config") or {}
                ),
                "backend": self.backend,
                "policy_id": self.policy_id,
            },
            omitted_edges=report.get("omitted_edges") or [],
            cross_group_relations=(
                report.get("cross_group_relations") or []
            ),
            object_catalog=scene.object_catalog(),
        )


def _effective_topology_config(
    configured: dict[str, Any],
    request_config: dict[str, Any],
) -> dict[str, Any]:
    combined = _deep_merge(configured, request_config)
    section = combined.get("grouping")
    if isinstance(section, dict):
        combined = deepcopy(section)
    topology = combined.get("topology")
    if isinstance(topology, dict):
        return deepcopy(topology)
    return {
        key: deepcopy(value)
        for key, value in combined.items()
        if key not in {"backend", "anchor", "vlm"}
    }


def _deep_merge(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
