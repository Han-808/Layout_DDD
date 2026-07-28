from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_P0B_VISUAL_CONFIG_VERSION = "p0b_metric_visual_config_v2"

# These are production defaults, calibrated from the 7/22 deterministic-camera
# audit and the 7/24 local-highlight ablation. Experiment scripts may explicitly
# request passthrough evidence so their historical arms keep their original
# meaning.
DEFAULT_P0B_VISUAL_CONFIGS: dict[str, dict[str, Any]] = {
    "collision": {
        "config_id": "collision_local_raw_contour_budget2_v2",
        "image_budget": 2,
        "image_order": ["deterministic_local_raw", "deterministic_local_contour"],
        "local_view_count": 1,
        "presentation": "same_pose_raw_plus_segmentation_contour",
        "global_pose": None,
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
        "config_id": "support_local2_raw_global_top_budget3_v2",
        "image_budget": 3,
        "image_order": [
            "deterministic_local_raw",
            "deterministic_local_raw",
            "global_top",
        ],
        "local_view_count": 2,
        "presentation": "distinct_local_raw_plus_global_top",
        "global_pose": "global_top",
        "require_flagged_boundary_highlight": False,
    },
}

_GLOBAL_ROLE = "metric_highlighted_global"
_LOCAL_RAW_ROLES = {"metric_local_rgb", "collision_rgb"}
_LOCAL_HIGHLIGHT_ROLES = {"metric_local_highlight", "collision_pair_overlay"}
_LOCAL_CONTOUR_ROLES = {"metric_local_contour"}


def is_metric_focus_evidence(
    items: list[dict[str, Any]],
    *,
    metric: str | None = None,
) -> bool:
    """Return whether provider output carries role-aware local evidence.

    Collision is recognized only when its new contour role is present; this
    keeps older rich raw/Legacy providers backward compatible. OOB and Support
    retain the historical global-plus-local recognition contract.
    """

    roles = {str(item.get("role") or "") for item in items}
    local_roles = _LOCAL_RAW_ROLES | _LOCAL_HIGHLIGHT_ROLES | _LOCAL_CONTOUR_ROLES
    if str(metric or "").strip().lower() == "collision":
        return bool(roles & _LOCAL_RAW_ROLES) and bool(roles & _LOCAL_CONTOUR_ROLES)
    return _GLOBAL_ROLE in roles and bool(roles & local_roles)


def compose_default_p0b_visual_evidence(
    metric: str,
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the exact calibrated evidence bundle for one P0b metric.

    The composer is deliberately fail-closed. Missing required raw/contour
    images, a non-top required global pose, or an OOB local overlay without the
    flagged boundary plane must not be silently represented as the calibrated
    default.
    """

    metric_name = str(metric).strip().lower()
    if metric_name not in DEFAULT_P0B_VISUAL_CONFIGS:
        raise ValueError(f"no default P0b VisualConfig for metric {metric_name!r}")
    if not all(isinstance(item, dict) for item in items):
        raise TypeError("default P0b VisualConfig requires rich evidence objects")

    config = deepcopy(DEFAULT_P0B_VISUAL_CONFIGS[metric_name])
    global_items = [
        item for item in items if str(item.get("role") or "") == _GLOBAL_ROLE
    ]
    global_top = [item for item in global_items if _view_id(item) == "global_top"]
    if config["global_pose"] == "global_top" and len(global_top) != 1:
        available = [_view_id(item) for item in global_items]
        raise RuntimeError(
            "default P0b VisualConfig requires exactly one highlighted global_top view; "
            f"available global views: {available}"
        )

    local_raw = _unique_local_items(
        [item for item in items if str(item.get("role") or "") in _LOCAL_RAW_ROLES]
    )
    local_highlight = _unique_local_items(
        [item for item in items if str(item.get("role") or "") in _LOCAL_HIGHLIGHT_ROLES]
    )
    local_contour = _unique_local_items(
        [item for item in items if str(item.get("role") or "") in _LOCAL_CONTOUR_ROLES]
    )
    required_local = int(config["local_view_count"])
    required_role = (
        "highlighted"
        if metric_name == "oob"
        else "raw"
    )
    required_items = local_highlight if required_role == "highlighted" else local_raw
    if len(required_items) < required_local:
        raise RuntimeError(
            f"{metric_name} default P0b VisualConfig requires {required_local} {required_role} "
            f"local view(s), but only {len(required_items)} are available"
        )
    chosen_local = required_items[:required_local]

    contour_by_view = {_view_id(item): item for item in local_contour}
    if metric_name == "collision":
        chosen_view_id = _view_id(chosen_local[0])
        if chosen_view_id not in contour_by_view:
            raise RuntimeError(
                "collision default P0b VisualConfig requires a same-pose segmentation "
                f"contour for raw local view {chosen_view_id!r}"
            )

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
        elif slot in {"deterministic_local", "deterministic_local_raw"}:
            selected.append(deepcopy(next(local_iter)))
        elif slot == "deterministic_local_contour":
            raw_view_id = _view_id(selected[-1]) if selected else _view_id(chosen_local[0])
            selected.append(deepcopy(contour_by_view[raw_view_id]))
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


def _unique_local_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
