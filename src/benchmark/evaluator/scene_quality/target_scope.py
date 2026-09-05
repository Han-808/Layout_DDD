"""Deterministic target-centred evidence scopes.

These scopes are a routing fallback for a localized candidate that cannot be
owned by a trusted multi-object group.  A target scope is deliberately *not* a
group: contextual neighbours are framing aids, never implicit relation
members or defect owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from benchmark.evaluator.generic_validity.geometry import get_obb_corners
from benchmark.scene_io.object_normalization import normalize_object


TARGET_CAMERA_SCOPE_VERSION = "target_camera_scope_v1"
DEFAULT_CONTEXT_NEIGHBOR_LIMIT = 3


@dataclass(frozen=True)
class TargetCameraScope:
    scope_id: str
    target_id: str
    context_ids: tuple[str, ...]
    framing_ids: tuple[str, ...]
    target_bounds_min: tuple[float, float, float]
    target_bounds_max: tuple[float, float, float]
    focus_center: tuple[float, float, float]
    extent: tuple[float, float, float]
    required_observations: tuple[str, ...]
    require_global_anchor: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_version": TARGET_CAMERA_SCOPE_VERSION,
            "scope_kind": "target_centered_context",
            "scope_id": self.scope_id,
            "target_id": self.target_id,
            "context_ids": list(self.context_ids),
            "framing_ids": list(self.framing_ids),
            "target_bounds": {
                "min": list(self.target_bounds_min),
                "max": list(self.target_bounds_max),
            },
            "focus_center": list(self.focus_center),
            "extent": list(self.extent),
            "required_observations": list(self.required_observations),
            "require_global_anchor": self.require_global_anchor,
            "group_identity": None,
            "grouping_role": "none_context_only",
            "context_objects_are_defect_owners": False,
            "default_attribution_target_ids": [self.target_id],
            "scene_access": "read_only",
        }


def build_target_camera_scope(
    scene: dict[str, Any],
    *,
    target_id: str,
    metric: str,
    explicit_context_ids: Iterable[str] = (),
    max_context_neighbors: int = DEFAULT_CONTEXT_NEIGHBOR_LIMIT,
    include_global_context: bool = True,
) -> TargetCameraScope:
    """Build a bounded target-local framing scope without inventing a group."""

    if not isinstance(scene, dict):
        raise TypeError("target camera scope scene must be a JSON object")
    target_id = str(target_id).strip()
    if not target_id:
        raise ValueError("target camera scope requires a target ID")
    if (
        isinstance(max_context_neighbors, bool)
        or not isinstance(max_context_neighbors, int)
        or max_context_neighbors < 0
    ):
        raise ValueError("max_context_neighbors must be an integer >= 0")

    objects = {
        str(item.get("id") or item.get("object_id")): item
        for item in scene.get("objects") or []
        if isinstance(item, dict)
        and (item.get("id") is not None or item.get("object_id") is not None)
    }
    if target_id not in objects:
        raise ValueError(f"unknown target object {target_id!r}")

    explicit = list(
        dict.fromkeys(
            str(item).strip()
            for item in explicit_context_ids
            if str(item).strip() and str(item).strip() != target_id
        )
    )
    unknown = [item for item in explicit if item not in objects]
    if unknown:
        raise ValueError(
            f"target camera scope references unknown context IDs {unknown}"
        )
    context_ids = _bounded_context_ids(
        objects,
        target_id=target_id,
        explicit_context_ids=explicit,
        limit=max_context_neighbors,
    )
    framing_ids = (target_id, *context_ids)
    minimum, maximum = _combined_bounds(objects, framing_ids)
    focus = (minimum + maximum) / 2.0
    extent = np.maximum(maximum - minimum, 0.05)
    return TargetCameraScope(
        scope_id=f"target_scope_{target_id}",
        target_id=target_id,
        context_ids=tuple(context_ids),
        framing_ids=tuple(framing_ids),
        target_bounds_min=tuple(float(item) for item in minimum),
        target_bounds_max=tuple(float(item) for item in maximum),
        focus_center=tuple(float(item) for item in focus),
        extent=tuple(float(item) for item in extent),
        required_observations=_required_observations(metric),
        require_global_anchor=bool(include_global_context),
    )


def localized_target_ids(
    judgement: dict[str, Any],
    *,
    valid_object_ids: Iterable[str],
) -> list[str]:
    """Return explicit routed target IDs; never infer targets from prose."""

    valid = {str(item) for item in valid_object_ids}
    values: list[str] = []
    for defect in judgement.get("defects") or []:
        if not isinstance(defect, dict):
            continue
        values.extend(
            str(item)
            for item in defect.get("target_ids") or []
            if str(item) in valid
        )
    request = judgement.get("evidence_request")
    if isinstance(request, dict):
        values.extend(
            str(item)
            for item in request.get("target_ids") or []
            if str(item) in valid
        )
    return list(dict.fromkeys(values))


def _bounded_context_ids(
    objects: dict[str, dict[str, Any]],
    *,
    target_id: str,
    explicit_context_ids: list[str],
    limit: int,
) -> list[str]:
    if limit == 0:
        return []
    selected = list(explicit_context_ids[:limit])
    remaining = limit - len(selected)
    if remaining <= 0:
        return selected
    target_min, target_max = _object_bounds(objects[target_id])
    ranked: list[tuple[float, float, str]] = []
    for object_id, obj in objects.items():
        if object_id == target_id or object_id in selected:
            continue
        minimum, maximum = _object_bounds(obj)
        delta = np.maximum(
            np.maximum(target_min[:2] - maximum[:2], minimum[:2] - target_max[:2]),
            0.0,
        )
        edge_gap = float(np.linalg.norm(delta))
        center_gap = float(
            np.linalg.norm(
                ((minimum[:2] + maximum[:2]) / 2.0)
                - ((target_min[:2] + target_max[:2]) / 2.0)
            )
        )
        ranked.append((edge_gap, center_gap, object_id))
    ranked.sort()
    selected.extend(item[2] for item in ranked[:remaining])
    return selected


def _combined_bounds(
    objects: dict[str, dict[str, Any]],
    object_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = [_object_bounds(objects[object_id]) for object_id in object_ids]
    return (
        np.min(np.stack([item[0] for item in bounds]), axis=0),
        np.max(np.stack([item[1] for item in bounds]), axis=0),
    )


def _object_bounds(value: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_object(value)
    corners = get_obb_corners(normalized)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _required_observations(metric: str) -> tuple[str, ...]:
    if metric == "object_pairing_consistency":
        return (
            "target_visible",
            "limited_local_context",
            "global_context_preserved",
        )
    if metric == "semantic_placement_consistency":
        return (
            "target_visible",
            "limited_local_context",
            "global_context_preserved",
        )
    raise ValueError(
        "target-centred scope supports only Object Pairing and Placement"
    )
