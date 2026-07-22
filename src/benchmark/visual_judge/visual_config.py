from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_P0B_VISUAL_CONFIG_VERSION = "p0b_metric_visual_config_v1"

# These are production defaults, calibrated from the 7/22 deterministic-camera
# evidence audit. Experiment scripts may explicitly request passthrough evidence
# so their historical arms keep their original meaning.
DEFAULT_P0B_VISUAL_CONFIGS: dict[str, dict[str, Any]] = {
    "collision": {
        "config_id": "collision_local_global_top_budget2_v1",
        "image_budget": 2,
        "image_order": ["deterministic_local", "global_top"],
        "local_view_count": 1,
        "presentation": "highlight_only",
        "global_pose": "global_top",
        "require_flagged_boundary_highlight": False,
    },
    "oob": {
        "config_id": "oob_global_top_local_boundary_highlight_budget2_v1",
        "image_budget": 2,
        "image_order": ["global_top", "deterministic_local"],
        "local_view_count": 1,
        "presentation": "highlight_only",
        "global_pose": "global_top",
        "require_flagged_boundary_highlight": True,
    },
    "support": {
        "config_id": "support_local2_global_top_budget3_v1",
        "image_budget": 3,
        "image_order": ["deterministic_local", "deterministic_local", "global_top"],
        "local_view_count": 2,
        "presentation": "highlight_only",
        "global_pose": "global_top",
        "require_flagged_boundary_highlight": False,
    },
}

_GLOBAL_ROLE = "metric_highlighted_global"
_LOCAL_HIGHLIGHT_ROLES = {"metric_local_highlight", "collision_pair_overlay"}


def is_metric_focus_evidence(items: list[dict[str, Any]]) -> bool:
    """Return whether rich provider output carries the complete global/local bundle."""

    roles = {str(item.get("role") or "") for item in items}
    return _GLOBAL_ROLE in roles and bool(roles & _LOCAL_HIGHLIGHT_ROLES)


def compose_default_p0b_visual_evidence(
    metric: str,
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the exact calibrated evidence bundle for one P0b metric.

    The composer is deliberately fail-closed. Missing highlight images, a
    non-top global pose, or an OOB local overlay without the flagged boundary
    plane must not be silently represented as the calibrated default.
    """

    metric_name = str(metric).strip().lower()
    if metric_name not in DEFAULT_P0B_VISUAL_CONFIGS:
        raise ValueError(f"no default P0b VisualConfig for metric {metric_name!r}")
    if not all(isinstance(item, dict) for item in items):
        raise TypeError("default P0b VisualConfig requires rich evidence objects")

    config = deepcopy(DEFAULT_P0B_VISUAL_CONFIGS[metric_name])
    global_items = [item for item in items if str(item.get("role") or "") == _GLOBAL_ROLE]
    global_top = [item for item in global_items if _view_id(item) == "global_top"]
    if len(global_top) != 1:
        available = [_view_id(item) for item in global_items]
        raise RuntimeError(
            "default P0b VisualConfig requires exactly one highlighted global_top view; "
            f"available global views: {available}"
        )

    local_items = _unique_local_highlights(
        [item for item in items if str(item.get("role") or "") in _LOCAL_HIGHLIGHT_ROLES]
    )
    required_local = int(config["local_view_count"])
    if len(local_items) < required_local:
        raise RuntimeError(
            f"{metric_name} default P0b VisualConfig requires {required_local} highlighted "
            f"local view(s), but only {len(local_items)} are available"
        )
    chosen_local = local_items[:required_local]

    boundary_verified = False
    if bool(config["require_flagged_boundary_highlight"]):
        missing_boundary = [
            _view_id(item) for item in chosen_local if not _has_architecture_plane_legend(item)
        ]
        if missing_boundary:
            raise RuntimeError(
                "OOB local highlight must include the flagged room boundary plane; "
                f"missing for views: {missing_boundary}"
            )
        boundary_verified = True

    local_iter = iter(chosen_local)
    selected: list[dict[str, Any]] = []
    for slot in config["image_order"]:
        if slot == "global_top":
            selected.append(deepcopy(global_top[0]))
        elif slot == "deterministic_local":
            selected.append(deepcopy(next(local_iter)))
        else:  # pragma: no cover - constant-table integrity guard
            raise RuntimeError(f"unsupported P0b VisualConfig slot {slot!r}")

    expected = int(config["image_budget"])
    if len(selected) != expected:  # pragma: no cover - constant-table integrity guard
        raise RuntimeError(
            f"{metric_name} P0b VisualConfig produced {len(selected)} images, expected {expected}"
        )
    policy = {
        "schema_version": DEFAULT_P0B_VISUAL_CONFIG_VERSION,
        "policy": "metric_default",
        **config,
        "actual_image_count": len(selected),
        "selected_view_ids": [_view_id(item) for item in selected],
        "selected_roles": [str(item.get("role") or "") for item in selected],
        "global_perspective_included": False,
        "flagged_boundary_highlight_verified": boundary_verified,
        "selection_source": "deterministic_metric_camera",
    }
    return selected, policy


def _view_id(item: dict[str, Any]) -> str:
    value = item.get("view_id")
    if value is None and isinstance(item.get("pose"), dict):
        value = item["pose"].get("id")
    return str(value or "")


def _unique_local_highlights(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        view_id = _view_id(item)
        key = view_id or str(item.get("path") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _has_architecture_plane_legend(item: dict[str, Any]) -> bool:
    legend = item.get("color_legend")
    return isinstance(legend, list) and any(
        isinstance(entry, dict) and entry.get("role") == "architecture_plane"
        for entry in legend
    )
