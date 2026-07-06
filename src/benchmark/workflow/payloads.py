from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.input_modes import (
    COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
    FULL_METADATA_BUDGETED,
    PROMPT_ONLY,
    get_input_mode_spec,
    resolve_input_representation_mode,
)
from benchmark.object_aliasing import (
    ALIAS_MAP_KEY,
    alias_prompt_object,
    alias_spatial_cue,
    build_object_alias_map,
    prompt_alias_summary,
)


SOURCE_LAYOUT_HINT_FIELDS = {
    "layout_center_hint",
    "layout_center_hint_source",
    "source_floor_position",
    "source_height_position",
}


def build_input_payloads(case: dict) -> dict[str, dict]:
    normalized_case = deepcopy(case)
    mode = resolve_input_representation_mode(normalized_case)
    normalized_case["scene_representation_mode"] = mode
    if mode != PROMPT_ONLY:
        normalized_case[ALIAS_MAP_KEY] = build_object_alias_map(normalized_case)
    if isinstance(normalized_case.get("source"), dict):
        source = dict(normalized_case["source"])
        source.setdefault("input_representation_mode", mode)
        source.setdefault("scene_representation_mode", mode)
        normalized_case["source"] = source

    prompt_payload = build_prompt_payload(normalized_case, mode)
    eval_context = build_eval_context(normalized_case, prompt_payload, mode)
    visibility_audit = eval_context["visibility_audit"]
    input_quality = eval_context["input_quality"]
    return {
        "normalized_case": normalized_case,
        "prompt_payload": prompt_payload,
        "eval_context": eval_context,
        "visibility_audit": visibility_audit,
        "input_quality": input_quality,
    }


def build_prompt_payload(case: dict, mode: str | None = None) -> dict:
    resolved_mode = mode or resolve_input_representation_mode(case)
    spec = get_input_mode_spec(resolved_mode)
    alias_map = case.get(ALIAS_MAP_KEY) if isinstance(case.get(ALIAS_MAP_KEY), dict) else build_object_alias_map(case)
    payload = {
        "case_id": case.get("case_id") or case.get("task_id") or case.get("scene_id"),
        "task_id": case.get("task_id") or case.get("case_id") or case.get("scene_id"),
        "input_level": case.get("input_level"),
        "scene_representation_mode": resolved_mode,
        "description": deepcopy(case.get("description") or {}),
        "source": _prompt_source(case.get("source"), resolved_mode),
    }

    if resolved_mode == PROMPT_ONLY:
        return _drop_empty(payload)

    payload["object_aliasing"] = prompt_alias_summary(alias_map)
    if isinstance(case.get("room"), dict):
        payload["room"] = _prompt_room(case["room"], include_full=spec.includes_full_metadata)
    if spec.includes_required_objects and isinstance(case.get("objects"), list):
        payload["objects"] = [
            _prompt_object(obj, include_full=spec.includes_full_metadata, alias_map=alias_map)
            for obj in case["objects"]
            if isinstance(obj, dict)
        ]

    spatial_cues = _case_spatial_cues(case)
    if spec.includes_spatial_cues and spatial_cues:
        payload["spatial_cues"] = [
            _prompt_spatial_cue(cue, include_full=spec.includes_full_metadata, alias_map=alias_map, index=index)
            for index, cue in enumerate(spatial_cues, start=1)
        ]
        if resolved_mode in {COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS, FULL_METADATA_BUDGETED} and isinstance(case.get("relations"), list):
            # Compatibility with older relation-oriented prompt/eval helpers.
            payload["relations"] = [
                _prompt_spatial_cue(relation, include_full=spec.includes_full_metadata, alias_map=alias_map, index=index)
                for index, relation in enumerate(case["relations"], start=1)
                if isinstance(relation, dict)
            ]
    if spec.includes_full_metadata:
        for key in ["attachments", "reference_layout"]:
            if key in case:
                payload[key] = deepcopy(case[key])
    return _drop_empty(payload)


def build_eval_context(case: dict, prompt_payload: dict, mode: str) -> dict:
    objects = [obj for obj in case.get("objects") or [] if isinstance(obj, dict)]
    objects_by_id = {str(obj.get("id")): deepcopy(obj) for obj in objects if obj.get("id")}
    regions = _regions_from_case(case)
    assigned = [obj for obj in objects if obj.get("source_region_id")]
    source_positions = [obj for obj in objects if isinstance(obj.get("source_position"), list)]
    bbox_available = [obj for obj in objects if isinstance(obj.get("bbox_size"), list)]
    bbox_fallback = [obj for obj in objects if obj.get("bbox_size_source") == "fallback"]
    spatial_cues = _case_spatial_cues(case)
    audit = _visibility_audit(prompt_payload, mode)
    alias_map = case.get(ALIAS_MAP_KEY) if isinstance(case.get(ALIAS_MAP_KEY), dict) else build_object_alias_map(case)
    return {
        "scene_id": case.get("case_id") or case.get("task_id") or case.get("scene_id"),
        "dataset": (case.get("source") or {}).get("dataset") if isinstance(case.get("source"), dict) else None,
        "objects_by_id": objects_by_id,
        ALIAS_MAP_KEY: deepcopy(alias_map),
        "object_aliasing": deepcopy(alias_map.get("diagnostics", {})),
        "regions": {
            "available": bool(regions),
            "items": regions,
            "source": _region_source(case),
        },
        "estimated_spatial_cues": deepcopy(spatial_cues),
        "input_quality": {
            "num_required_objects": len(objects),
            "bbox_size_available_rate": _rate(len(bbox_available), len(objects)),
            "bbox_size_fallback_rate": _rate(len(bbox_fallback), len(objects)),
            "source_pose_available_rate": _rate(len(source_positions), len(objects)),
            "region_info_available": bool(regions),
            "region_assignment_rate": _rate(len(assigned), len(objects)),
            "estimated_spatial_cue_count": len(spatial_cues),
        },
        "visibility_audit": audit,
    }


def eval_context_summary(eval_context: dict) -> dict:
    return {
        "scene_id": eval_context.get("scene_id"),
        "dataset": eval_context.get("dataset"),
        "object_count": len(eval_context.get("objects_by_id", {})) if isinstance(eval_context.get("objects_by_id"), dict) else 0,
        "object_aliasing": eval_context.get("object_aliasing", {}),
        "regions": eval_context.get("regions", {}),
        "input_quality": eval_context.get("input_quality", {}),
        "visibility_audit": eval_context.get("visibility_audit", {}),
        "estimated_spatial_cue_count": len(eval_context.get("estimated_spatial_cues", [])) if isinstance(eval_context.get("estimated_spatial_cues"), list) else 0,
    }


def _prompt_object(obj: dict, *, include_full: bool, alias_map: dict) -> dict:
    obj = alias_prompt_object(obj, alias_map, include_full=include_full)
    keep = {
        "id",
        "category",
        "bbox_size",
        "required",
        "layout_center_hint",
        "layout_center_hint_source",
        "source_floor_position",
        "source_height_position",
        "semantic_category",
    }
    full_keep = keep | {
        "source_region_id",
        "source_region_label",
        "region_assignment_source",
        "region_assignment_confidence",
        "bbox_size_source",
        "source",
        "source_id",
        "source_template_name",
        "source_position",
        "source_rotation",
        "source_non_uniform_scale",
        "source_motion_type",
        "hssd_semantic",
        "source_asset_references",
        "source_object_metadata",
    }
    selected = full_keep if include_full else keep
    return {key: deepcopy(value) for key, value in obj.items() if key in selected and key != "raw_hssd_instance"}


def _prompt_room(room: dict, *, include_full: bool) -> dict:
    keep = {
        "unit",
        "floor_z",
        "boundary",
        "floor_polygon",
        "floor_plan",
        "wall_height",
        "coordinate_note",
        "boundary_source_kind",
        "geometry_fidelity",
        "is_proxy_geometry",
    }
    full_keep = keep | {"regions", "stage_metadata_keys", "stage_asset_references", "semantic_metadata_keys", "supporting_visuals"}
    selected = full_keep if include_full else keep
    compact = {key: deepcopy(value) for key, value in room.items() if key in selected}
    if not include_full:
        compact.pop("regions", None)
        if isinstance(compact.get("floor_plan"), dict):
            floor_plan = dict(compact["floor_plan"])
            floor_plan.pop("regions", None)
            compact["floor_plan"] = floor_plan
    return compact


def _prompt_source(source: object, mode: str) -> dict:
    if not isinstance(source, dict):
        return {"compact": True, "input_representation_mode": mode, "scene_representation_mode": mode}
    keep = {
        "dataset",
        "scene_id",
        "scene_instance",
        "scene_variant",
        "raw_object_instance_count",
        "imported_object_count",
        "max_objects",
        "truncated",
        "mesh_imported",
        "mesh_free_import",
        "room_boundary_source_kind",
        "room_geometry_fidelity",
        "room_is_proxy_geometry",
        "relation_policy",
        "relation_generation_version",
        "relation_counts_by_type",
        "relations_are_ground_truth",
        "relations_source_note",
    }
    compact = {key: deepcopy(value) for key, value in source.items() if key in keep}
    compact["compact"] = True
    compact["input_representation_mode"] = mode
    compact["scene_representation_mode"] = mode
    return compact


def _prompt_spatial_cue(cue: dict, *, include_full: bool, alias_map: dict, index: int) -> dict:
    cue = alias_spatial_cue(cue, alias_map, index=index, include_full=include_full)
    keep = {"id", "relation_id", "type", "subject", "object", "target", "source", "confidence", "hard"}
    full_keep = keep | {"provenance", "evidence", "visible_to_model"}
    selected = full_keep if include_full else keep
    return {key: deepcopy(value) for key, value in cue.items() if key in selected}


def _case_spatial_cues(case: dict) -> list[dict]:
    cues = case.get("spatial_cues")
    if isinstance(cues, list) and cues:
        return [cue for cue in cues if isinstance(cue, dict)]
    relations = case.get("relations")
    return [relation for relation in relations if isinstance(relation, dict)] if isinstance(relations, list) else []


def _regions_from_case(case: dict) -> list[dict]:
    room = case.get("room")
    if not isinstance(room, dict):
        return []
    if isinstance(room.get("regions"), list):
        return [deepcopy(region) for region in room["regions"] if isinstance(region, dict)]
    floor_plan = room.get("floor_plan")
    if isinstance(floor_plan, dict) and isinstance(floor_plan.get("regions"), list):
        return [deepcopy(region) for region in floor_plan["regions"] if isinstance(region, dict)]
    return []


def _region_source(case: dict) -> str:
    room = case.get("room")
    if not isinstance(room, dict):
        return "missing"
    floor_plan = room.get("floor_plan")
    if isinstance(floor_plan, dict) and floor_plan.get("source"):
        return str(floor_plan["source"])
    return str(room.get("room_layout_source") or room.get("boundary_source") or "missing")


def _visibility_audit(prompt_payload: dict, mode: str) -> dict:
    objects = prompt_payload.get("objects")
    prompt_objects = [obj for obj in objects if isinstance(obj, dict)] if isinstance(objects, list) else []
    return {
        "source_layout_hints_visible_to_model": any(any(field in obj for field in SOURCE_LAYOUT_HINT_FIELDS) for obj in prompt_objects),
        "spatial_cues_visible_to_model": isinstance(prompt_payload.get("spatial_cues"), list) and bool(prompt_payload["spatial_cues"]),
        "regions_visible_to_model": _payload_has_regions(prompt_payload),
        "full_metadata_visible_to_model": mode == FULL_METADATA_BUDGETED,
        "prompt_object_list_visible": bool(prompt_objects),
        "prompt_required_object_count": len(prompt_objects),
    }


def _payload_has_regions(prompt_payload: dict) -> bool:
    room = prompt_payload.get("room")
    if not isinstance(room, dict):
        return False
    if isinstance(room.get("regions"), list) and room["regions"]:
        return True
    floor_plan = room.get("floor_plan")
    return bool(isinstance(floor_plan, dict) and isinstance(floor_plan.get("regions"), list) and floor_plan["regions"])


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _drop_empty(value: dict) -> dict:
    return {key: item for key, item in value.items() if item not in (None, {}, [])}
