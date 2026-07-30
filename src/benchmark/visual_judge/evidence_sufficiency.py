from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from benchmark.visual_judge.visual_config import (
    DEFAULT_P0B_VISUAL_CONFIGS,
    compose_default_p0b_visual_evidence,
)


EVIDENCE_SUFFICIENCY_VERSION = "deterministic_visual_evidence_sufficiency_v2"
SUFFICIENT = "sufficient"
INSUFFICIENT = "insufficient"
UNKNOWN = "unknown"

REPAIR_CAMERA = "camera"
REPAIR_PRESENTATION = "presentation"
REPAIR_RERENDER = "rerender"
REPAIR_GEOMETRY = "geometry"
REPAIR_NONVISUAL = "nonvisual"
REPAIR_UNKNOWN = "unknown"
REPAIR_MIXED = "mixed"


def assess_visual_evidence_sufficiency(
    metric: str,
    items: list[Any],
    *,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively assess a final, role-aware P0b evidence packet.

    The result is a routing diagnostic, never a metric verdict.  A fallback is
    recommended only when the insufficiency is measured and camera-repairable.
    Presentation, rerender, geometry-grounding, and nonvisual deficiencies stay
    explicit so camera motion is not used as a generic error retry.
    """

    metric_name = str(metric or "").strip().lower()
    base = _base_result(metric_name)
    if metric_name not in DEFAULT_P0B_VISUAL_CONFIGS:
        return _finish(
            base,
            status=UNKNOWN,
            deficiencies=[
                _deficiency(
                    "unsupported_metric_for_sufficiency_gate",
                    REPAIR_NONVISUAL,
                )
            ],
        )
    if not isinstance(items, list) or not items or not all(
        isinstance(item, dict) for item in items
    ):
        return _finish(
            base,
            status=UNKNOWN,
            deficiencies=[
                _deficiency("unmeasured_or_non_rich_evidence", REPAIR_UNKNOWN)
            ],
        )

    try:
        selected, policy = compose_default_p0b_visual_evidence(metric_name, items)
    except (TypeError, ValueError, RuntimeError):
        deficiencies = _packet_structure_deficiencies(metric_name, items)
        return _finish(
            base,
            status=INSUFFICIENT,
            deficiencies=deficiencies
            or [
                _deficiency(
                    "calibrated_visual_config_incomplete",
                    REPAIR_PRESENTATION,
                )
            ],
        )

    required = int(policy.get("local_view_count") or 0)
    result = {
        **base,
        "required_local_view_count": required,
        "selected_image_count": len(selected),
        "selected_view_ids": list(policy.get("selected_view_ids") or []),
    }
    missing_paths = [
        str(item.get("view_id") or item.get("role") or "view")
        for item in selected
        if not _existing_path(item.get("path"))
    ]
    if missing_paths:
        return _finish(
            {
                **result,
                "missing_evidence_count": len(missing_paths),
            },
            status=INSUFFICIENT,
            deficiencies=[
                _deficiency("selected_evidence_file_missing", REPAIR_RERENDER)
            ],
        )

    containment = (
        _collision_containment_hint(request) if metric_name == "collision" else False
    )
    local_by_view: dict[str, dict[str, Any]] = {}
    for item in selected:
        role = str(item.get("role") or "")
        if role not in {
            "metric_local_rgb",
            "metric_local_highlight",
            "collision_rgb",
        }:
            continue
        view_id = _view_id(item)
        if view_id and (
            view_id not in local_by_view or item.get("visibility") is not None
        ):
            local_by_view[view_id] = item

    per_view = [
        {
            "view_id": view_id,
            "camera_ray": _camera_ray(item),
            **_local_view_status(
                metric_name,
                item,
                containment_hint=containment,
                require_support_focus=True,
                require_collision_focus_pixels=True,
            ),
        }
        for view_id, item in local_by_view.items()
    ]
    result.update(_view_summary(per_view, required=required))
    result["per_view"] = per_view

    if containment:
        return _finish(
            result,
            status=UNKNOWN,
            deficiencies=[
                _deficiency(
                    "intrinsic_collision_occlusion_not_pose_repairable",
                    REPAIR_GEOMETRY,
                )
            ],
        )
    if (
        required > 1
        and int(result["usable_local_view_count"]) >= required
        and _selected_views_redundant(per_view, required=required)
    ):
        return _finish(
            result,
            status=INSUFFICIENT,
            deficiencies=[
                _deficiency("redundant_local_views", REPAIR_CAMERA)
            ],
        )
    if required > 0 and int(result["usable_local_view_count"]) >= required:
        return _finish(
            result,
            status=SUFFICIENT,
            deficiencies=[],
            success_reason="calibrated_packet_and_visibility_sufficient",
        )
    if required > 0 and int(result["measured_local_view_count"]) >= required:
        deficiencies = _view_deficiencies(per_view)
        return _finish(
            result,
            status=INSUFFICIENT,
            deficiencies=deficiencies
            or [
                _deficiency(
                    "measured_local_visibility_insufficient",
                    REPAIR_CAMERA,
                )
            ],
        )
    return _finish(
        result,
        status=UNKNOWN,
        deficiencies=_view_deficiencies(per_view)
        or [
            _deficiency("local_visibility_not_fully_measured", REPAIR_UNKNOWN)
        ],
    )


def assess_preview_selection_sufficiency(
    metric: str,
    selected_view_ids: list[str],
    visibility_by_id: dict[str, dict[str, Any]],
    *,
    request: dict[str, Any] | None = None,
    poses_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assess the local part of one active-search preview selection.

    Global/presentation roles are deliberately excluded here.  They are
    validated again after the fixed-shape final packet is rendered.
    """

    metric_name = str(metric or "").strip().lower()
    base = _base_result(metric_name)
    config = DEFAULT_P0B_VISUAL_CONFIGS.get(metric_name)
    if not isinstance(config, dict):
        return _finish(
            base,
            status=UNKNOWN,
            deficiencies=[
                _deficiency(
                    "unsupported_metric_for_sufficiency_gate",
                    REPAIR_NONVISUAL,
                )
            ],
        )
    required = int(config.get("local_view_count") or 1)
    object_ids = [
        str(value)
        for value in (
            request.get("object_ids", [])
            if isinstance(request, dict)
            and isinstance(request.get("object_ids"), list)
            else []
        )
        if value is not None
    ]
    containment = (
        _collision_containment_hint(request) if metric_name == "collision" else False
    )
    per_view: list[dict[str, Any]] = []
    for view_id in list(dict.fromkeys(str(value) for value in selected_view_ids if value)):
        item = {
            "target_ids": object_ids,
            "visibility": visibility_by_id.get(view_id),
            "pose": (poses_by_id or {}).get(view_id),
        }
        per_view.append(
            {
                "view_id": view_id,
                "camera_ray": _camera_ray(item),
                **_local_view_status(
                    metric_name,
                    item,
                    containment_hint=containment,
                    require_support_focus=True,
                    require_collision_focus_pixels=False,
                ),
            }
        )
    result = {
        **base,
        "required_local_view_count": required,
        "selected_view_ids": list(
            dict.fromkeys(str(value) for value in selected_view_ids if value)
        ),
        "per_view": per_view,
        **_view_summary(per_view, required=required),
    }
    if containment:
        return _finish(
            result,
            status=UNKNOWN,
            deficiencies=[
                _deficiency(
                    "intrinsic_collision_occlusion_not_pose_repairable",
                    REPAIR_GEOMETRY,
                )
            ],
        )
    if len(per_view) < required:
        return _finish(
            result,
            status=INSUFFICIENT,
            deficiencies=[
                _deficiency("required_local_view_count_missing", REPAIR_CAMERA)
            ],
        )
    if (
        required > 1
        and int(result["usable_local_view_count"]) >= required
        and _selected_views_redundant(per_view, required=required)
    ):
        return _finish(
            result,
            status=INSUFFICIENT,
            deficiencies=[
                _deficiency("redundant_local_views", REPAIR_CAMERA)
            ],
        )
    if int(result["usable_local_view_count"]) >= required:
        return _finish(
            result,
            status=SUFFICIENT,
            deficiencies=[],
            success_reason="selected_preview_visibility_sufficient",
        )
    if int(result["measured_local_view_count"]) >= required:
        return _finish(
            result,
            status=INSUFFICIENT,
            deficiencies=_view_deficiencies(per_view)
            or [
                _deficiency(
                    "measured_local_visibility_insufficient",
                    REPAIR_CAMERA,
                )
            ],
        )
    return _finish(
        result,
        status=UNKNOWN,
        deficiencies=[
            _deficiency("local_visibility_not_fully_measured", REPAIR_UNKNOWN)
        ],
    )


def _local_view_status(
    metric: str,
    item: dict[str, Any],
    *,
    containment_hint: bool,
    require_support_focus: bool,
    require_collision_focus_pixels: bool,
) -> dict[str, Any]:
    visibility = item.get("visibility")
    if not isinstance(visibility, dict):
        return _view_status(
            measured=False,
            usable=False,
            utility=0.0,
            deficiency_code="visibility_missing",
            repairability=REPAIR_UNKNOWN,
        )

    if metric == "collision":
        targets = visibility.get("targets")
        if isinstance(targets, dict) and targets:
            target_values = [
                target for target in targets.values() if isinstance(target, dict)
            ]
            pixel_counts = [
                int(target.get("visible_pixels") or 0) for target in target_values
            ]
            normalized = [
                float(
                    target.get("normalized_visibility")
                    if target.get("normalized_visibility") is not None
                    else target.get("visible_fraction")
                    or 0.0
                )
                for target in target_values
            ]
            measured = bool(pixel_counts) and (
                int(visibility.get("image_pixel_count") or 0) > 0
                or any(pixel_counts)
                or str(visibility.get("status") or "")
                in {"ok", "failed", "blank"}
            )
            enough_targets = (
                any(value > 0 for value in pixel_counts)
                if containment_hint
                else len(pixel_counts) >= 2 and all(value > 0 for value in pixel_counts)
            )
            utility = min(normalized) if len(normalized) >= 2 else max(normalized or [0.0])
            focus_in_frame = visibility.get("focus_in_frame")
            focus_fraction = visibility.get("focus_pixel_fraction")
            focus_status = str(
                visibility.get("focus_measurement_status") or ""
            )
            if not require_collision_focus_pixels:
                focus_ok = focus_in_frame is not False
                if focus_ok:
                    utility = min(1.0, utility + 0.05)
                code = (
                    "required_entities_not_jointly_visible"
                    if not enough_targets
                    else "focus_region_out_of_frame"
                    if not focus_ok
                    else None
                )
                return _view_status(
                    measured=measured,
                    usable=bool(measured and enough_targets and focus_ok),
                    utility=utility,
                    deficiency_code=code,
                    repairability=REPAIR_CAMERA if code else None,
                )
            focus_measured = (
                focus_in_frame is not None
                and focus_fraction is not None
                and focus_status not in {
                    "measurement_failed",
                    "unavailable_no_collision_focus_roi",
                }
            )
            if not focus_measured:
                return _view_status(
                    measured=False,
                    usable=False,
                    utility=utility,
                    deficiency_code="collision_focus_roi_unmeasured",
                    repairability=REPAIR_UNKNOWN,
                )
            focus_fraction_value = float(focus_fraction or 0.0)
            focus_ok = bool(focus_in_frame) and focus_fraction_value >= 0.00001
            if focus_ok:
                utility = min(1.0, utility + min(0.05, focus_fraction_value))
            code = (
                "required_entities_not_jointly_visible"
                if not enough_targets
                else "focus_region_out_of_frame"
                if not bool(focus_in_frame)
                else "focus_region_too_small"
                if focus_fraction_value < 0.00001
                else None
            )
            return _view_status(
                measured=measured,
                usable=bool(measured and enough_targets and focus_ok),
                utility=utility,
                deficiency_code=code,
                repairability=REPAIR_CAMERA if code else None,
            )
        legacy = [
            visibility.get("object_a_pixel_fraction"),
            visibility.get("object_b_pixel_fraction"),
        ]
        if any(value is not None for value in legacy):
            values = [float(value or 0.0) for value in legacy]
            enough = all(value >= 0.001 for value in values)
            return _view_status(
                measured=True,
                usable=enough,
                utility=min(1.0, min(values) / 0.001),
                deficiency_code=(
                    None if enough else "required_entities_not_jointly_visible"
                ),
                repairability=None if enough else REPAIR_CAMERA,
            )

    fractions = visibility.get("target_pixel_fractions")
    if not isinstance(fractions, dict) or not fractions:
        return _view_status(
            measured=False,
            usable=False,
            utility=0.0,
            deficiency_code="target_fraction_missing",
            repairability=REPAIR_UNKNOWN,
        )
    target_ids = [
        str(value) for value in item.get("target_ids", []) if value is not None
    ]
    if metric == "collision":
        values = (
            [float(fractions.get(target_id) or 0.0) for target_id in target_ids]
            if target_ids
            else [float(value or 0.0) for value in fractions.values()]
        )
        enough = len(values) >= 2 and all(value >= 0.001 for value in values)
        return _view_status(
            measured=bool(values),
            usable=enough,
            utility=min(1.0, min(values or [0.0]) / 0.001),
            deficiency_code=(
                None if enough else "required_entities_not_jointly_visible"
            ),
            repairability=None if enough else REPAIR_CAMERA,
        )

    primary_id = target_ids[0] if target_ids else str(next(iter(fractions)))
    primary_fraction = float(fractions.get(primary_id) or 0.0)
    primary_visible = primary_fraction >= 0.001
    utility = min(1.0, primary_fraction / 0.001)
    if not primary_visible:
        return _view_status(
            measured=True,
            usable=False,
            utility=utility,
            deficiency_code="target_occluded_or_too_small",
            repairability=REPAIR_CAMERA,
        )
    if metric == "oob":
        region_fractions = visibility.get("region_pixel_fractions")
        if not isinstance(region_fractions, dict) or (
            "architecture_plane" not in region_fractions
        ):
            return _view_status(
                measured=False,
                usable=False,
                utility=utility,
                deficiency_code="architecture_plane_fraction_missing",
                repairability=REPAIR_UNKNOWN,
            )
        plane_fraction = float(
            region_fractions.get("architecture_plane") or 0.0
        )
        utility = min(
            1.0,
            0.5 * utility
            + 0.5 * min(1.0, plane_fraction / 0.00001),
        )
        if plane_fraction <= 0.0:
            return _view_status(
                measured=True,
                usable=False,
                utility=utility,
                deficiency_code="architecture_plane_not_visible",
                repairability=REPAIR_CAMERA,
            )
    if metric == "support" and (
        require_support_focus or _legend_has_role(item, "measured_support_gap")
    ):
        focus_value = visibility.get("focus_pixel_fraction")
        if focus_value is None:
            return _view_status(
                measured=False,
                usable=False,
                utility=utility,
                deficiency_code="support_gap_fraction_missing",
                repairability=REPAIR_PRESENTATION,
            )
        focus = float(focus_value or 0.0)
        utility = min(1.0, 0.5 * utility + 0.5 * min(1.0, focus / 0.00001))
        if focus < 0.00001:
            return _view_status(
                measured=True,
                usable=False,
                utility=utility,
                deficiency_code="focus_region_too_small",
                repairability=REPAIR_CAMERA,
            )
    return _view_status(
        measured=True,
        usable=True,
        utility=utility,
        deficiency_code=None,
        repairability=None,
    )


def _packet_structure_deficiencies(
    metric: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roles = [str(item.get("role") or "") for item in items]
    view_ids = {_view_id(item) for item in items if _view_id(item)}
    deficiencies: list[dict[str, Any]] = []
    if metric == "collision":
        if "collision_rgb" not in roles:
            deficiencies.append(
                _deficiency("collision_raw_local_missing", REPAIR_RERENDER)
            )
        if "metric_local_contour" not in roles:
            deficiencies.append(
                _deficiency("collision_contour_missing", REPAIR_PRESENTATION)
            )
    elif metric == "oob":
        if not any(
            role == "metric_highlighted_global"
            and _view_id(item) == "global_top"
            for role, item in zip(roles, items)
        ):
            deficiencies.append(
                _deficiency("required_global_top_missing", REPAIR_RERENDER)
            )
        highlighted = [
            item
            for item in items
            if str(item.get("role") or "") in {
                "metric_local_highlight",
                "collision_pair_overlay",
            }
        ]
        if not highlighted:
            deficiencies.append(
                _deficiency("oob_local_highlight_missing", REPAIR_PRESENTATION)
            )
        elif not any(_legend_has_role(item, "architecture_plane") for item in highlighted):
            deficiencies.append(
                _deficiency(
                    "oob_architecture_plane_highlight_missing",
                    REPAIR_PRESENTATION,
                )
            )
    elif metric == "support":
        raw_ids = {
            _view_id(item)
            for item in items
            if str(item.get("role") or "") == "metric_local_rgb"
            and _view_id(item)
        }
        required = int(DEFAULT_P0B_VISUAL_CONFIGS["support"]["local_view_count"])
        if len(raw_ids) < required:
            deficiencies.append(
                _deficiency("required_local_view_count_missing", REPAIR_CAMERA)
            )
        if not any(
            str(item.get("role") or "") == "metric_highlighted_global"
            and _view_id(item) == "global_top"
            for item in items
        ):
            deficiencies.append(
                _deficiency("required_global_top_missing", REPAIR_RERENDER)
            )
    if len(view_ids) < 1:
        deficiencies.append(
            _deficiency("rendered_view_identity_missing", REPAIR_RERENDER)
        )
    return _dedupe_deficiencies(deficiencies)


def _view_summary(
    per_view: list[dict[str, Any]],
    *,
    required: int,
) -> dict[str, Any]:
    measured = sum(bool(item.get("measured")) for item in per_view)
    usable = sum(bool(item.get("usable")) for item in per_view)
    utilities = sorted(
        (float(item.get("utility") or 0.0) for item in per_view),
        reverse=True,
    )
    selected_utilities = utilities[: max(1, required)]
    utility = (
        sum(selected_utilities) / max(required, 1)
        if selected_utilities
        else 0.0
    )
    return {
        "measured_local_view_count": measured,
        "usable_local_view_count": usable,
        "evidence_utility": float(max(0.0, min(1.0, utility))),
    }


def _view_deficiencies(per_view: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_deficiencies(
        [
            _deficiency(
                str(item["deficiency_code"]),
                str(item.get("repairability") or REPAIR_UNKNOWN),
            )
            for item in per_view
            if item.get("deficiency_code")
        ]
    )


def _camera_ray(item: dict[str, Any]) -> dict[str, Any] | None:
    pose = item.get("pose")
    if not isinstance(pose, dict):
        return None
    location = pose.get("location")
    target = pose.get("target")
    if (
        not isinstance(location, (list, tuple))
        or not isinstance(target, (list, tuple))
        or len(location) != 3
        or len(target) != 3
    ):
        return None
    try:
        delta = [
            float(location[index]) - float(target[index])
            for index in range(3)
        ]
    except (TypeError, ValueError):
        return None
    distance = math.sqrt(sum(value * value for value in delta))
    if not math.isfinite(distance) or distance <= 1e-9:
        return None
    return {
        "direction": [value / distance for value in delta],
        "distance_m": distance,
    }


def _selected_views_redundant(
    per_view: list[dict[str, Any]],
    *,
    required: int,
) -> bool:
    usable = [item for item in per_view if item.get("usable")]
    if len(usable) < required:
        return False
    selected = usable[:required]
    for left_index, left in enumerate(selected):
        left_ray = left.get("camera_ray")
        if not isinstance(left_ray, dict):
            return False
        for right in selected[left_index + 1 :]:
            right_ray = right.get("camera_ray")
            if not isinstance(right_ray, dict):
                return False
            left_direction = left_ray.get("direction")
            right_direction = right_ray.get("direction")
            if (
                not isinstance(left_direction, list)
                or not isinstance(right_direction, list)
                or len(left_direction) != 3
                or len(right_direction) != 3
            ):
                return False
            cosine = sum(
                float(left_direction[index])
                * float(right_direction[index])
                for index in range(3)
            )
            left_distance = float(left_ray.get("distance_m") or 0.0)
            right_distance = float(right_ray.get("distance_m") or 0.0)
            distance_delta = abs(left_distance - right_distance) / max(
                left_distance,
                right_distance,
                1e-9,
            )
            if cosine < math.cos(math.radians(3.0)) or distance_delta > 0.05:
                return False
    return True


def _view_status(
    *,
    measured: bool,
    usable: bool,
    utility: float,
    deficiency_code: str | None,
    repairability: str | None,
) -> dict[str, Any]:
    return {
        "measured": bool(measured),
        "usable": bool(usable),
        "utility": float(max(0.0, min(1.0, utility))),
        "deficiency_code": deficiency_code,
        "repairability": repairability,
    }


def _base_result(metric: str) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SUFFICIENCY_VERSION,
        "metric": metric,
        "status": UNKNOWN,
        "reason_codes": [],
        "deficiencies": [],
        "repairability": REPAIR_UNKNOWN,
        "camera_repairable": False,
        "trigger_recommended": False,
        "required_local_view_count": 0,
        "measured_local_view_count": 0,
        "usable_local_view_count": 0,
        "evidence_utility": 0.0,
    }


def _finish(
    result: dict[str, Any],
    *,
    status: str,
    deficiencies: list[dict[str, Any]],
    success_reason: str | None = None,
) -> dict[str, Any]:
    normalized = _dedupe_deficiencies(deficiencies)
    repairabilities = {
        str(item.get("repairability") or REPAIR_UNKNOWN) for item in normalized
    }
    repairability = (
        next(iter(repairabilities))
        if len(repairabilities) == 1
        else REPAIR_MIXED
        if repairabilities
        else "none"
    )
    camera_repairable = bool(normalized) and all(
        item.get("repairability") == REPAIR_CAMERA for item in normalized
    )
    reason_codes = [str(item["code"]) for item in normalized]
    if success_reason:
        reason_codes = [success_reason]
    return {
        **result,
        "status": status,
        "reason_codes": reason_codes,
        "deficiencies": normalized,
        "repairability": repairability,
        "camera_repairable": camera_repairable,
        "trigger_recommended": bool(
            status == INSUFFICIENT and camera_repairable
        ),
    }


def _deficiency(code: str, repairability: str) -> dict[str, Any]:
    return {
        "code": str(code),
        "repairability": str(repairability),
    }


def _dedupe_deficiencies(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (
            str(value.get("code") or ""),
            str(value.get("repairability") or REPAIR_UNKNOWN),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append({"code": key[0], "repairability": key[1]})
    return result


def _existing_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)) or not str(value):
        return False
    return Path(value).expanduser().is_file()


def _view_id(item: dict[str, Any]) -> str:
    value = item.get("view_id")
    if value is None and isinstance(item.get("pose"), dict):
        value = item["pose"].get("id")
    return str(value or "")


def _legend_has_role(item: dict[str, Any], role: str) -> bool:
    legend = item.get("color_legend")
    return isinstance(legend, list) and any(
        isinstance(entry, dict) and str(entry.get("role") or "") == role
        for entry in legend
    )


def _collision_containment_hint(request: dict[str, Any] | None) -> bool:
    detector = (
        request.get("detector_evidence")
        if isinstance(request, dict)
        and isinstance(request.get("detector_evidence"), dict)
        else {}
    )
    mesh = detector.get("mesh") if isinstance(detector.get("mesh"), dict) else {}
    return bool(
        mesh.get("containment_a_in_b") is True
        or mesh.get("containment_b_in_a") is True
    )
