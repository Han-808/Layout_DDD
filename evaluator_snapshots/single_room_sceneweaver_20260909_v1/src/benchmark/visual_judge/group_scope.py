"""Group-scoped camera targets derived from a validated grouping partition.

Grouping answers which objects belong in one downstream evidence scope.
This module performs only the geometric projection of that scope into a camera
target. Metric semantics, camera selection, rendering, and judgement remain in
their respective layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.scene_io.object_normalization import normalize_object
from benchmark.visual_judge.camera_dsl import CAMERA_OBSERVATIONS


GROUP_CAMERA_SCOPE_VERSION = "group_camera_scope_v1"

_METRIC_GROUP_OBSERVATIONS: dict[str, tuple[str, ...]] = {
    "scale_consistency": (
        "joint_visibility",
        "group_context_visible",
        "global_context_preserved",
    ),
    "object_pairing_consistency": (
        "joint_visibility",
        "group_context_visible",
        "limited_local_context",
    ),
    "style_consistency": (
        "joint_visibility",
        "group_context_visible",
        "limited_local_context",
        "global_context_preserved",
    ),
    "functional_consistency": (
        "joint_visibility",
        "group_context_visible",
        "interaction_side_visible",
        "limited_local_context",
    ),
    "semantic_placement_consistency": (
        "target_visible",
        "group_context_visible",
        "limited_local_context",
        "global_context_preserved",
    ),
    "functional_semantic_fidelity": (
        "joint_visibility",
        "group_context_visible",
        "interaction_side_visible",
        "limited_local_context",
    ),
}


@dataclass(frozen=True)
class GroupCameraScope:
    group_id: str
    member_ids: tuple[str, ...]
    target_bounds_min: tuple[float, float, float]
    target_bounds_max: tuple[float, float, float]
    focus_center: tuple[float, float, float]
    extent: tuple[float, float, float]
    required_observations: tuple[str, ...]
    require_global_anchor: bool
    grouping_policy_id: str | None = None
    grouping_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_version": GROUP_CAMERA_SCOPE_VERSION,
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "target_bounds": {
                "min": list(self.target_bounds_min),
                "max": list(self.target_bounds_max),
            },
            "focus_center": list(self.focus_center),
            "extent": list(self.extent),
            "required_observations": list(self.required_observations),
            "require_global_anchor": self.require_global_anchor,
            "grouping_policy_id": self.grouping_policy_id,
            "grouping_backend": self.grouping_backend,
            "scene_access": "read_only",
        }


def build_group_camera_scope(
    scene: dict[str, Any],
    group: dict[str, Any],
    *,
    metric: str,
    include_global_context: bool,
    grouping_report: dict[str, Any] | None = None,
) -> GroupCameraScope:
    """Build a strict AABB camera target for exactly one supplied group."""

    if not isinstance(scene, dict):
        raise TypeError("group camera scope scene must be a JSON object")
    if not isinstance(group, dict):
        raise TypeError("group camera scope group must be a JSON object")
    group_id = str(group.get("group_id") or "").strip()
    if not group_id:
        raise ValueError("group camera scope requires group_id")
    raw_members = group.get("object_ids")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(
            f"group camera scope {group_id!r} requires non-empty object_ids"
        )
    member_ids = tuple(str(item).strip() for item in raw_members)
    if any(not item for item in member_ids) or len(member_ids) != len(
        set(member_ids)
    ):
        raise ValueError(
            f"group camera scope {group_id!r} has malformed or duplicate IDs"
        )

    objects = {
        str(item.get("id") or item.get("object_id")): item
        for item in scene.get("objects") or []
        if isinstance(item, dict)
        and (item.get("id") is not None or item.get("object_id") is not None)
    }
    unknown = [object_id for object_id in member_ids if object_id not in objects]
    if unknown:
        raise ValueError(
            f"group camera scope {group_id!r} references unknown IDs {unknown}"
        )

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for object_id in member_ids:
        try:
            normalized = normalize_object(objects[object_id])
        except ValueError as exc:
            raise ValueError(
                f"group camera scope object {object_id!r} is invalid: {exc}"
            ) from exc
        corners = get_obb_corners(normalized)
        for axis in range(3):
            minimum[axis] = min(
                minimum[axis],
                float(corners[:, axis].min()),
            )
            maximum[axis] = max(
                maximum[axis],
                float(corners[:, axis].max()),
            )

    observations = _METRIC_GROUP_OBSERVATIONS.get(
        str(metric),
        (
            "target_visible",
            "group_context_visible",
            "limited_local_context",
        ),
    )
    unknown_observations = sorted(set(observations) - CAMERA_OBSERVATIONS)
    if unknown_observations:
        raise ValueError(
            "group camera scope contains unknown observations "
            f"{unknown_observations}"
        )
    focus = tuple((minimum[i] + maximum[i]) / 2.0 for i in range(3))
    extent = tuple(maximum[i] - minimum[i] for i in range(3))
    report = grouping_report if isinstance(grouping_report, dict) else {}
    return GroupCameraScope(
        group_id=group_id,
        member_ids=member_ids,
        target_bounds_min=tuple(minimum),
        target_bounds_max=tuple(maximum),
        focus_center=focus,
        extent=extent,
        required_observations=observations,
        require_global_anchor=bool(include_global_context),
        grouping_policy_id=_optional_text(
            report.get("grouping_policy_id") or report.get("policy_id")
        ),
        grouping_backend=_optional_text(
            report.get("grouping_backend") or report.get("backend")
        ),
    )


def group_scope_evidence_goal(scope: GroupCameraScope) -> dict[str, Any]:
    return {
        "view_goal": (
            f"observe group {scope.group_id} as one bounded local evidence scope"
        ),
        "target_ids": list(scope.member_ids),
        "required_observations": list(scope.required_observations),
        "require_global_anchor": scope.require_global_anchor,
        "group_scope": scope.to_dict(),
    }


def _optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None
