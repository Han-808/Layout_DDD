from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from benchmark.input_modes import (
    COMPACT_OBJECTS,
    COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
    FULL_METADATA_BUDGETED,
    PROMPT_ONLY,
    get_input_mode_spec,
    prompt_includes_relations,
    resolve_input_representation_mode,
)
from benchmark.models.prompt_budget import PromptSection
from benchmark.object_aliasing import ALIAS_MAP_KEY, alias_for_canonical, get_alias_map


class ModelResponseError(ValueError):
    """Raised when a model response cannot be converted into layout JSON."""


def parse_json_object(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ModelResponseError(f"Expected dict or JSON string, got {type(raw).__name__}.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError("Model response does not contain a JSON object.") from None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"Model response contains malformed JSON: {exc.msg} at char {exc.pos}.") from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError("Model response JSON must be an object.")
    return parsed


@dataclass
class BaseLayoutModel(ABC):
    name: str

    @abstractmethod
    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        """Generate a full layout JSON object for a benchmark instance."""

    @abstractmethod
    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        """Repair a full layout JSON object from deterministic feedback."""


def build_generation_prompt(bm_instance: dict, layout_schema: dict) -> str:
    return _render_prompt_sections(build_generation_prompt_sections(bm_instance, layout_schema))


def build_generation_prompt_sections(bm_instance: dict, layout_schema: dict) -> list[PromptSection]:
    required_ids = _required_object_ids(bm_instance)
    representation_mode = resolve_input_representation_mode(bm_instance)
    mode_spec = get_input_mode_spec(representation_mode)
    object_requirements = _object_requirements_for_prompt(required_ids, mode_spec.includes_required_objects)
    compact_instance = compact_bm_instance_for_model(bm_instance)
    return [
        PromptSection(
            "instruction",
            "Generate an explicit 3D scene layout as JSON only.\n"
            "Input sections may use compact JSON for token efficiency. Your output must not use compact rows.\n"
            "Return exactly one full JSON object matching the normal layout schema, with no Markdown or explanation.\n"
            "Use the input case_id/task_id as scene_id.\n"
            "Use short object_id aliases exactly as provided; the pipeline restores source IDs after parsing.\n"
            "Do not include source IDs, validity fields, hierarchy, or floor_objects.",
        ),
        PromptSection(
            "output_geometry_requirements",
            "Use bbox objects with center [x, y, z], size [width, depth, height], "
            "and yaw in degrees around z/up.",
        ),
        PromptSection("object_requirements", object_requirements, item_count=len(required_ids)),
        PromptSection(
            "coordinate_notes",
            "- HSSD source coordinates are not layout coordinates: source_position is [x, height, z], "
            "while layout center is [x, room_depth_y, height_z]. Never copy source_position into center.\n"
            "- For floor objects without layout_center_hint, center.z must be at least size.height / 2 so the bbox bottom is not below floor.\n"
            "- Relation outputs must use keys source and target, not subject/object or source_category.\n"
            "- If spatial_cues are present, treat them as soft deterministic spatial cues, not HSSD ground-truth relations.\n"
            "- Keep all object centers inside the room boundary when a boundary is provided.",
        ),
        PromptSection(
            "input_mode_description",
            f"Input representation mode: {representation_mode}.\n"
            "The case JSON may preserve more source metadata on disk than this prompt view includes.",
        ),
        PromptSection(
            "compact_benchmark_instance",
            f"Benchmark instance:\n{_compact_json(compact_instance)}",
            item_count=_compact_instance_object_count(compact_instance),
        ),
        PromptSection("output_contract", f"Layout output contract:\n{_layout_output_contract()}"),
    ]


def build_repair_prompt(
    bm_instance: dict,
    current_layout: dict,
    feedback: dict,
    layout_schema: dict,
) -> str:
    return _render_prompt_sections(build_repair_prompt_sections(bm_instance, current_layout, feedback, layout_schema))


def build_repair_prompt_sections(
    bm_instance: dict,
    current_layout: dict,
    feedback: dict,
    layout_schema: dict,
) -> list[PromptSection]:
    required_ids = _required_object_ids(bm_instance)
    compact_instance = compact_bm_instance_for_repair(bm_instance)
    current_layout_compact = compact_layout_for_model(current_layout)
    feedback_compact = compact_feedback_for_model(feedback, bm_instance)
    return [
        PromptSection(
            "repair_instruction",
            "Repair the explicit 3D scene layout.\n"
            "Input may include compact rows for token efficiency. Your output must not use rows.\n"
            "Return exactly one full corrected JSON object matching the normal layout schema, with no Markdown or explanation.\n"
            "Use the same short object aliases exactly; the pipeline restores original HSSD/source IDs after parsing.\n"
            "Preserve object aliases and object count. Preserve valid objects unless necessary. Keep the room "
            "boundary unchanged. Fix only listed violations. Do not include source IDs, hierarchy, or floor_objects.",
        ),
        PromptSection(
            "required_objects_compact",
            "Required object constraints:\n"
            f"- The repaired layout objects array must contain exactly {len(required_ids)} objects.\n"
            "- The repaired object_id set must exactly equal the required object rows below, with no missing, extra, or duplicate ids.\n"
            "- Preserve each required category and bbox_size when present.\n"
            f"{_compact_json(_required_object_rows(bm_instance))}",
            item_count=len(required_ids),
        ),
        PromptSection(
            "scene_constraints",
            "Preserve all non-target objects unless moving them is necessary to resolve global consistency.\n"
            "If repair_targets is non-empty, the repaired layout must change at least one repair_target center, size, yaw, "
            "support_parent, or region_id. A byte-for-byte identical layout is not a valid repair.\n"
            "Treat deterministic feedback.repair_actions as the primary repair plan. For each shown action, make a "
            "meaningful geometric edit unless doing so would create a worse boundary, collision, support, or room-placement "
            "violation. A repair that only rounds numbers or moves every repair target by less than 0.01m is not a valid "
            "repair.\n"
            "The suggested vectors and target centers are advisory. They are not an automatic script. Use them to guide a coherent repaired layout.\n"
            "Do not blindly apply every vector. You may adjust vectors and target centers for global coherence, but do not create new serious collisions.\n"
            "When an action contains suggested_center or suggested_center_for_move_object, apply that center unless a "
            "different concrete center resolves the same violation more plausibly. If an action instead contains "
            "candidate_center_for_reference or candidate_warning, search for a cleaner placement and still make a concrete "
            "change to the affected target.\n"
            "Moving both objects in a colliding pair together usually does not resolve the collision; use suggested_delta_xy "
            "and separation directions to reduce relative overlap. Do not resolve a dense cluster by translating all objects together.\n"
            "Do not fix above-wall by moving objects below the floor. If a height constraint is marked impossible or fallback-derived, "
            "prioritize floor consistency and plausible placement over satisfying approximate fallback wall height.\n"
            "For collision repair actions where candidate_total_overlap_volume_m3 is 0 and candidate_floor_plan_outside_penalty "
            "is 0, strongly prefer the provided suggested_center_for_move_object unless you find an equally clean placement.\n"
            "Prefer repairs that satisfy all of these constraints: keep affected bboxes inside floor_plan regions, avoid "
            "new serious collisions, alleviate large implausible bbox intersections when feasible, "
            "preserve plausible support/room placement, and make minimal changes to valid or locked "
            "objects. Do not blindly follow one suggested center if it creates a new boundary issue, collision, or implausible "
            "placement.\n"
            "Preserve every required object id/category/size. Return full JSON layout, not compact rows.\n"
            "Do not optimize solely for zero bbox overlap. Proxies may overlap for attachment/support cases, "
            "including objects on top of supports, objects contained inside storage, wall-mounted TVs/art/mirrors/shelves/curtains/panels, "
            "beds with headboards, tabletop objects, handles/fixtures, and objects sharing a contact surface.\n"
            "For room_boundary flags, move the target bbox into a suitable floor_plan region. For serious_collision or "
            "soft_collision repair actions, separate the listed objects only when the overlap is implausible penetration; "
            "otherwise preserve plausible attachment, support, containment, and room placement.\n"
            "The room may use dataset floor-plan coordinates, including negative x/y values; do not assume a "
            "0-based origin during repair. Treat layout_center_hint/source positions as historical evidence "
            "only during repair; do not copy them back if they caused violations.",
        ),
        PromptSection(
            "compact_benchmark_instance",
            f"Benchmark instance:\n{_compact_json(compact_instance)}",
            item_count=_compact_instance_object_count(compact_instance),
        ),
        PromptSection(
            "current_layout_compact",
            "Current layout is shown in compact rows for input efficiency. Output must be full JSON layout, not rows.\n"
            f"{_compact_json(current_layout_compact)}",
            item_count=len(current_layout_compact.get("rows", [])) if isinstance(current_layout_compact.get("rows"), list) else None,
        ),
        PromptSection(
            "repair_feedback_compact",
            f"Deterministic feedback:\n{_compact_json(feedback_compact)}",
            item_count=len(feedback_compact.get("repair_actions", [])) if isinstance(feedback_compact.get("repair_actions"), list) else None,
        ),
        PromptSection("output_contract", f"Layout output contract:\n{_layout_output_contract()}"),
    ]


def compact_bm_instance_for_model(bm_instance: dict) -> dict:
    """Keep model-facing prompts compact while preserving full case files on disk."""
    mode = resolve_input_representation_mode(bm_instance)
    compact = {
        key: value
        for key, value in bm_instance.items()
        if key not in {"objects", "source", "reference_layout", "relations", "attachments", ALIAS_MAP_KEY}
    }
    compact["scene_representation_mode"] = mode
    if isinstance(bm_instance.get("objects"), list):
        if mode == FULL_METADATA_BUDGETED:
            compact["objects"] = [_compact_input_object(obj, mode) for obj in bm_instance["objects"] if isinstance(obj, dict)]
        else:
            compact.update(_compact_input_object_rows(bm_instance["objects"], mode))
    if prompt_includes_relations(mode):
        if isinstance(bm_instance.get("spatial_cues"), list):
            compact["spatial_cues"] = [_compact_spatial_cue(cue) for cue in bm_instance["spatial_cues"] if isinstance(cue, dict)]
        if isinstance(bm_instance.get("relations"), list):
            compact["relations"] = [_compact_spatial_cue(relation) for relation in bm_instance["relations"] if isinstance(relation, dict)]
        if isinstance(bm_instance.get("attachments"), list):
            compact["attachments"] = bm_instance["attachments"]
    if isinstance(bm_instance.get("source"), dict):
        compact["source"] = _compact_source_metadata(bm_instance["source"], mode)
    return compact


def compact_bm_instance_for_repair(bm_instance: dict) -> dict:
    compact = compact_bm_instance_for_model(bm_instance)
    columns = compact.get("required_object_rows_columns")
    rows = compact.get("required_object_rows")
    if isinstance(columns, list) and isinstance(rows, list):
        drop_columns = {"layout_center_hint", "layout_center_hint_source", "source_floor_position"}
        kept_indexes = [index for index, column in enumerate(columns) if column not in drop_columns]
        compact["required_object_rows_columns"] = [columns[index] for index in kept_indexes]
        compact["required_object_rows"] = [
            [row[index] for index in kept_indexes if isinstance(row, list) and index < len(row)]
            for row in rows
            if isinstance(row, list)
        ]
    return compact


def _compact_input_object(obj: dict, mode: str = COMPACT_OBJECTS) -> dict:
    if mode == PROMPT_ONLY:
        return {}
    compact_keep = {
        "id",
        "category",
        "bbox_size",
        "required",
        "layout_center_hint",
        "layout_center_hint_source",
        "source_floor_position",
        "source_height_position",
        "semantic_category",
        "source_region_id",
        "source_region_label",
    }
    relation_keep = compact_keep | {"bbox_size_source", "source_collection"}
    full_keep = relation_keep | {
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
    if mode == FULL_METADATA_BUDGETED:
        return {
            key: _budget_value(value)
            for key, value in obj.items()
            if key in full_keep and key not in {"raw_hssd_instance"}
        }
    keep = relation_keep if mode == COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS else compact_keep
    return {key: value for key, value in obj.items() if key in keep}


def _compact_input_object_rows(objects: list[dict], mode: str) -> dict:
    rows = []
    include_hint = any(isinstance(obj, dict) and obj.get("layout_center_hint") is not None for obj in objects)
    include_source_floor = mode == COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS and any(
        isinstance(obj, dict) and obj.get("source_floor_position") is not None for obj in objects
    )
    columns = ["object_id", "category", "bbox_size"]
    if include_hint:
        columns.append("layout_center_hint")
    if include_source_floor:
        columns.append("source_floor_position")

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        compact_obj = _compact_input_object(obj, mode)
        if not compact_obj:
            continue
        row = [compact_obj.get("id"), compact_obj.get("category"), compact_obj.get("bbox_size")]
        if include_hint:
            row.append(compact_obj.get("layout_center_hint"))
        if include_source_floor:
            row.append(compact_obj.get("source_floor_position"))
        rows.append(row)
    return {
        "object_id_policy": "Use these short object_id aliases exactly. Source IDs are restored after parsing.",
        "required_object_rows_columns": columns,
        "required_object_rows": rows,
    }


def _object_requirements_for_prompt(required_ids: list[str], includes_required_objects: bool) -> str:
    if not includes_required_objects:
        return (
            "- No explicit required object list is provided in this input mode; generate a plausible object set for the described scene.\n"
            "- Use stable, descriptive object_id values for any objects you create.\n"
        )
    return (
        "- For each required_object_rows entry, create exactly one layout object.\n"
        f"- The layout objects array must contain exactly {len(required_ids)} objects.\n"
        f"- The layout object_id set must exactly equal this list, with no missing, extra, or duplicate ids: {required_ids}.\n"
        "- Use each short alias as layout object_id exactly; do not output original HSSD/source ids.\n"
        "- Preserve each short model-facing category exactly.\n"
        "- If an input object has bbox_size, use it as layout size.\n"
        "- If an input object has layout_center_hint, use that exact value as layout center.\n"
    )


def _compact_spatial_cue(cue: dict) -> dict:
    keep = {
        "id",
        "relation_id",
        "type",
        "subject",
        "object",
        "target",
        "source",
        "confidence",
        "hard",
        "evidence",
    }
    return {key: value for key, value in cue.items() if key in keep}


def _compact_source_metadata(source: dict, mode: str) -> dict:
    compact_keep = {
        "dataset",
        "scene_instance",
        "scene_id",
        "scene_variant",
        "stage_instance",
        "translation_origin",
        "raw_object_instance_count",
        "imported_object_count",
        "max_objects",
        "truncated",
        "input_representation_mode",
        "scene_representation_mode",
        "mesh_imported",
        "mesh_free_import",
        "mesh_asset_policy",
        "mesh_asset_references_kept",
        "room_boundary_source",
        "room_boundary_source_kind",
        "room_geometry_fidelity",
        "room_is_proxy_geometry",
        "room_region_count",
        "relations_policy",
        "relation_policy",
        "relation_generation_version",
        "relation_counts_by_type",
        "relations_are_ground_truth",
        "relations_source_note",
        "estimated_relations_included",
        "missing_metadata",
    }
    full_keep = compact_keep | {
        "scene_instance_fields",
        "semantic_scene_instance",
        "navmesh_instance",
        "default_lighting",
        "raw_object_collection_counts",
        "preserve_raw_metadata",
        "bbox_from_scale",
        "excluded_asset_extensions",
        "metadata_inclusion",
        "metadata_paths",
        "stage_asset_references",
        "object_asset_reference_count",
    }
    keep = full_keep if mode == FULL_METADATA_BUDGETED else compact_keep
    return {key: _budget_value(value) for key, value in source.items() if key in keep}


def _budget_value(value: Any, *, max_list_items: int = 20, max_dict_items: int = 30, max_string_chars: int = 400) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return value[:max_string_chars] + f"...<truncated {len(value) - max_string_chars} chars>"
    if isinstance(value, list):
        items = [_budget_value(item, max_list_items=max_list_items, max_dict_items=max_dict_items, max_string_chars=max_string_chars) for item in value[:max_list_items]]
        if len(value) > max_list_items:
            items.append({"truncated_items": len(value) - max_list_items})
        return items
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, key in enumerate(sorted(value)):
            if index >= max_dict_items:
                compact["truncated_items"] = len(value) - max_dict_items
                break
            compact[str(key)] = _budget_value(value[key], max_list_items=max_list_items, max_dict_items=max_dict_items, max_string_chars=max_string_chars)
        return compact
    return value


DEFAULT_REPAIR_PROMPT_BUDGET = {
    # These values define scale-aware policy bounds. Effective prompt caps are
    # computed from object and feedback counts in _repair_prompt_limits.
    "min_repair_targets": 20,
    "repair_targets_per_object": 1.0,
    "hard_max_repair_targets": 180,
    "min_repair_actions": 24,
    "repair_actions_per_object": 1.2,
    "hard_max_repair_actions": 180,
    "min_physical_flags": 30,
    "physical_flags_per_object": 1.0,
    "hard_max_physical_flags": 180,
    "min_collision_pairs": 20,
    "collision_pairs_per_object": 0.75,
    "hard_max_collision_pairs": 140,
    "min_render_flags": 10,
    "render_flags_per_group": 2.0,
    "hard_max_render_flags": 80,
    "min_relation_flags": 20,
    "relation_flags_per_object": 0.5,
    "hard_max_relation_flags": 100,
    "min_judge_issues": 12,
    "judge_issues_per_object": 0.25,
    "hard_max_judge_issues": 60,
    "min_group_issues": 8,
    "group_issues_per_object": 0.15,
    "hard_max_group_issues": 40,
    "min_locked_object_ids": 20,
    "locked_object_ids_per_object": 1.0,
    "hard_max_locked_object_ids": 180,
    "max_debug_evidence_chars": 6000,
    "max_feedback_chars": 12000,
    "max_judge_issue_chars_each": 500,
    "max_total_judge_issue_chars": 4000,
    "numeric_precision": 3,
}


def _required_object_ids(bm_instance: dict) -> list[str]:
    return [
        str(obj["id"])
        for obj in bm_instance.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("id"), str)
    ]


def _required_object_rows(bm_instance: dict) -> dict:
    rows = []
    for obj in bm_instance.get("objects", []):
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            continue
        rows.append(
            [
                obj.get("id"),
                obj.get("category"),
                _round_value(obj.get("bbox_size"), int(DEFAULT_REPAIR_PROMPT_BUDGET["numeric_precision"])),
            ]
        )
    return {"columns": ["object_id", "category", "bbox_size"], "rows": rows}


def compact_layout_for_model(layout: dict) -> dict:
    precision = int(DEFAULT_REPAIR_PROMPT_BUDGET["numeric_precision"])
    objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)]
    include_support = any(obj.get("support_parent") is not None for obj in objects)
    include_region = any(obj.get("region_id") is not None for obj in objects)
    columns = ["object_id", "category", "center", "size", "yaw"]
    if include_support:
        columns.append("support_parent")
    if include_region:
        columns.append("region_id")
    rows = []
    for obj in objects:
        row = [
            obj.get("model_object_id") or obj.get("object_id"),
            obj.get("model_category") or obj.get("category"),
            _round_value(obj.get("center"), precision),
            _round_value(obj.get("size"), precision),
            _round_value(obj.get("yaw", 0), precision),
        ]
        if include_support:
            row.append(obj.get("model_support_parent") or obj.get("support_parent"))
        if include_region:
            row.append(obj.get("region_id"))
        rows.append(row)
    compact = {
        "scene_id": layout.get("scene_id"),
        "unit": layout.get("unit"),
        "coordinate_system": layout.get("coordinate_system"),
        "columns": columns,
        "rows": rows,
    }
    return compact


def compact_feedback_for_model(feedback: dict, bm_instance: dict | None = None) -> dict:
    limits = _repair_prompt_limits(feedback, bm_instance)
    repair_targets = _sorted_strings(feedback.get("repair_targets"))
    locked_objects = _sorted_strings(feedback.get("locked_objects"))
    violations = _top_items(feedback.get("violations"), int(limits["max_physical_flags"]) + int(limits["max_judge_issues"]))
    repair_actions = _top_items(feedback.get("repair_actions"), int(limits["max_repair_actions"]))
    compact_debug = (
        _compact_debug_evidence(feedback["debug_evidence"])
        if isinstance(feedback.get("debug_evidence"), dict)
        else _compact_debug_summary(feedback.get("debug_evidence_summary"), limits)
    )
    compact = {
        "task_id": feedback.get("task_id"),
        "iteration": feedback.get("iteration"),
        "repair_targets": repair_targets[: int(limits["max_repair_targets"])],
        "omitted_repair_target_count": max(0, len(repair_targets) - int(limits["max_repair_targets"])),
        "locked_object_ids": locked_objects[: int(limits["max_locked_object_ids"])],
        "omitted_locked_object_count": max(0, len(locked_objects) - int(limits["max_locked_object_ids"])),
        "violations": [_compact_violation(item) for item in violations],
        "omitted_violation_count": _omitted_count(feedback.get("violations"), len(violations)),
        "repair_actions": [_compact_repair_action(item) for item in repair_actions],
        "omitted_repair_action_count": _omitted_count(feedback.get("repair_actions"), len(repair_actions)),
        "physical_evidence_summary": _physical_evidence_summary(feedback, limits),
        "debug_evidence_summary": compact_debug,
        "room_consistency_reason": _short_string(feedback.get("room_consistency_reason", ""), 800),
        "instruction": feedback.get("instruction"),
    }
    if _deterministic_flags_empty(compact) and _judge_invalid_feedback(feedback):
        compact["judge_invalid_fallback_instruction"] = (
            "The layout passed schema/evaluability checks but was judged spatially implausible. "
            "Improve room coherence, support plausibility, spacing, visibility, and semantic organization "
            "while preserving all required object ids/categories/bbox sizes."
        )
    if isinstance(bm_instance, dict):
        compact = _alias_feedback_ids(compact, bm_instance)
    max_list_items = max(
        80,
        int(limits["max_repair_targets"]),
        int(limits["max_repair_actions"]),
        int(limits["max_physical_flags"]) + int(limits["max_judge_issues"]),
    )
    return _budget_value(compact, max_list_items=max_list_items, max_dict_items=max_list_items, max_string_chars=800)


def _repair_prompt_limits(feedback: dict, bm_instance: dict | None) -> dict:
    policy = DEFAULT_REPAIR_PROMPT_BUDGET
    object_count = _input_object_count(bm_instance)
    group_count = _feedback_group_count(feedback)
    limits = dict(policy)
    limits.update(
        {
            "max_repair_targets": _scaled_limit(
                _list_count(feedback.get("repair_targets")),
                object_count,
                minimum=int(policy["min_repair_targets"]),
                per_object=float(policy["repair_targets_per_object"]),
                hard_max=int(policy["hard_max_repair_targets"]),
            ),
            "max_repair_actions": _scaled_limit(
                _list_count(feedback.get("repair_actions")),
                object_count,
                minimum=int(policy["min_repair_actions"]),
                per_object=float(policy["repair_actions_per_object"]),
                hard_max=int(policy["hard_max_repair_actions"]),
            ),
            "max_physical_flags": _scaled_limit(
                _physical_flag_count(feedback),
                object_count,
                minimum=int(policy["min_physical_flags"]),
                per_object=float(policy["physical_flags_per_object"]),
                hard_max=int(policy["hard_max_physical_flags"]),
            ),
            "max_collision_pairs": _scaled_limit(
                _physical_flag_count(feedback),
                object_count,
                minimum=int(policy["min_collision_pairs"]),
                per_object=float(policy["collision_pairs_per_object"]),
                hard_max=int(policy["hard_max_collision_pairs"]),
            ),
            "max_render_flags": _scaled_limit(
                _render_flag_count(feedback),
                max(group_count, object_count),
                minimum=int(policy["min_render_flags"]),
                per_object=float(policy["render_flags_per_group"]),
                hard_max=int(policy["hard_max_render_flags"]),
            ),
            "max_relation_flags": _scaled_limit(
                _list_count(feedback.get("spatial_relation_failures")),
                object_count,
                minimum=int(policy["min_relation_flags"]),
                per_object=float(policy["relation_flags_per_object"]),
                hard_max=int(policy["hard_max_relation_flags"]),
            ),
            "max_judge_issues": _scaled_limit(
                _judge_issue_count(feedback),
                object_count,
                minimum=int(policy["min_judge_issues"]),
                per_object=float(policy["judge_issues_per_object"]),
                hard_max=int(policy["hard_max_judge_issues"]),
            ),
            "max_group_issues": _scaled_limit(
                group_count,
                object_count,
                minimum=int(policy["min_group_issues"]),
                per_object=float(policy["group_issues_per_object"]),
                hard_max=int(policy["hard_max_group_issues"]),
            ),
            "max_locked_object_ids": _scaled_limit(
                _list_count(feedback.get("locked_objects")),
                object_count,
                minimum=int(policy["min_locked_object_ids"]),
                per_object=float(policy["locked_object_ids_per_object"]),
                hard_max=int(policy["hard_max_locked_object_ids"]),
            ),
        }
    )
    return limits


def _scaled_limit(total_count: int, object_count: int, *, minimum: int, per_object: float, hard_max: int) -> int:
    if total_count <= 0:
        return 0
    scaled = max(minimum, int(math.ceil(max(1, object_count) * per_object)))
    return max(1, min(total_count, scaled, hard_max))


def _input_object_count(bm_instance: dict | None) -> int:
    if not isinstance(bm_instance, dict):
        return 0
    count = len(_required_object_ids(bm_instance))
    if count:
        return count
    rows = bm_instance.get("required_object_rows")
    if isinstance(rows, list):
        return len(rows)
    objects = bm_instance.get("objects")
    return len(objects) if isinstance(objects, list) else 0


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _physical_flag_count(feedback: dict) -> int:
    count = sum(1 for item in feedback.get("violations", []) if isinstance(item, dict) and item.get("category") == "physical_debug_flag")
    debug = feedback.get("debug_evidence_summary")
    if isinstance(debug, dict):
        count += _list_count(debug.get("physical_flags"))
    return count


def _render_flag_count(feedback: dict) -> int:
    debug = feedback.get("debug_evidence_summary")
    if isinstance(debug, dict):
        return _list_count(debug.get("view_flags")) + _list_count(debug.get("render_skipped_objects"))
    return 0


def _feedback_group_count(feedback: dict) -> int:
    debug = feedback.get("debug_evidence_summary")
    if not isinstance(debug, dict):
        return 0
    return _list_count(debug.get("selected_groups")) + _list_count(debug.get("omitted_groups"))


def _judge_issue_count(feedback: dict) -> int:
    return sum(1 for item in feedback.get("violations", []) if isinstance(item, dict) and item.get("category") == "vlm_judge_issue")


def _legacy_compact_feedback_for_model(feedback: dict) -> dict:
    compact = {
        key: value
        for key, value in feedback.items()
        if key
        in {
            "task_id",
            "iteration",
            "repair_targets",
            "locked_objects",
            "violations",
            "repair_actions",
            "room_consistency_reason",
            "instruction",
            "debug_evidence_summary",
        }
    }
    if "debug_evidence_summary" not in compact and isinstance(feedback.get("debug_evidence"), dict):
        compact["debug_evidence_summary"] = _compact_debug_evidence(feedback["debug_evidence"])
    return compact


def _render_prompt_sections(sections: list[PromptSection]) -> str:
    return "\n\n".join(section.text.strip() for section in sections if section.text.strip()) + "\n"


def _sorted_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if isinstance(item, str) and item)


def _top_items(value: object, limit: int) -> list[dict]:
    if not isinstance(value, list) or limit <= 0:
        return []
    indexed = [(index, item) for index, item in enumerate(value) if isinstance(item, dict)]
    indexed.sort(key=lambda pair: _priority_key(pair[1], pair[0]))
    return [item for _, item in indexed[:limit]]


def _priority_key(item: dict, index: int) -> tuple:
    object_ids = _item_object_ids(item)
    return (
        -_severity_rank(item),
        -len(object_ids),
        ",".join(object_ids),
        index,
    )


def _severity_rank(item: dict) -> int:
    item_type = str(item.get("type") or item.get("issue_type") or item.get("action") or "").lower()
    category = str(item.get("category") or "").lower()
    severity = str(item.get("severity") or "").lower()
    if severity == "critical" or "missing" in item_type or "hard" in item_type or "schema" in category:
        return 100
    if "dense_collision_cluster" in item_type or "spread_dense_collision_cluster" in item_type:
        return 86
    if item_type in {"invalid_bbox", "no_renderable_object", "no_renderable_objects"}:
        return 90
    if "serious_collision" in item_type or "separate_collision" in item_type or "reduce_collisions" in item_type:
        return 80
    if "boundary" in item_type or "out_of_bounds" in item_type:
        return 70
    if "above_wall" in item_type or "below_floor" in item_type:
        return 60
    if "floating" in item_type or "support" in item_type:
        return 50
    if "vlm" in category or "judge" in category:
        return 40
    if "relation" in category or "attachment" in category:
        return 30
    if "view" in item_type or "render" in item_type:
        return 20
    return 10


def _item_object_ids(item: dict) -> list[str]:
    values = []
    for key in ["objects", "object_ids"]:
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw if isinstance(value, str))
    for key in ["object_id", "move_object_id", "keep_object_id", "move_object", "anchor_object"]:
        value = item.get(key)
        if isinstance(value, str):
            values.append(value)
    return sorted(set(values))


def _omitted_count(value: object, shown_count: int) -> int:
    return max(0, len(value) - shown_count) if isinstance(value, list) else 0


def _compact_violation(item: dict) -> dict:
    compact = {
        key: item[key]
        for key in [
            "category",
            "type",
            "code",
            "severity",
            "confidence",
            "source_kind",
            "source_confidence",
            "blocking",
            "objects",
            "target_category",
            "hard",
            "group_id",
        ]
        if key in item
    }
    if "message" in item:
        compact["message"] = _short_string(item.get("message"), 500)
    if "repair_hint" in item:
        compact["repair_hint"] = _short_string(item.get("repair_hint"), 300)
    return compact


def _compact_repair_action(item: dict) -> dict:
    keep = {
        "action",
        "code",
        "advisory",
        "confidence",
        "object_id",
        "object_ids",
        "move_object",
        "anchor_object",
        "move_object_id",
        "keep_object_id",
        "suggested_center",
        "suggested_delta",
        "suggested_delta_xy",
        "suggested_center_for_move_object",
        "candidate_center_for_reference",
        "candidate_warning",
        "candidate_strategy",
        "candidate_total_overlap_volume_m3",
        "candidate_floor_plan_outside_penalty",
        "separation_axis",
        "minimum_delta_m",
        "min_separation_distance",
        "overlap_axis",
        "overlap_depth",
        "target_region",
        "distance_outside",
        "boundary_source_kind",
        "boundary_source_confidence",
        "wall_height",
        "floor_z",
        "current_center_z",
        "target_center_z",
        "min_center_z",
        "max_center_z",
        "object_height",
        "source_kind",
        "blocking",
        "collision_count",
        "top_partners",
        "contributing_pair_count",
        "cluster_id",
        "objects",
        "anchor_objects",
        "movable_objects",
        "suggested_strategy",
        "top_pair_count",
        "top_pairs",
        "omitted_pair_count",
        "suggested_delta_xy_by_object",
        "fallback_note",
        "bottom_z",
        "nearest_support_top_z",
        "vertical_gap",
        "message",
        "reason_code",
        "reason",
        "issue_type",
        "severity",
        "repair_hint",
        "soft_collision",
    }
    compact = {key: item[key] for key in keep if key in item}
    return _round_value(_budget_value(compact, max_list_items=10, max_dict_items=20, max_string_chars=300), int(DEFAULT_REPAIR_PROMPT_BUDGET["numeric_precision"]))


def _compact_debug_summary(value: object, limits: dict | None = None) -> dict:
    if not isinstance(value, dict):
        return {}
    limits = limits or _repair_prompt_limits({}, None)
    return {
        "sanity_flags": _compact_flags(value.get("sanity_flags"), limit=10),
        "physical_flags": _compact_flags(value.get("physical_flags"), limit=int(limits["max_physical_flags"])),
        "view_flags": _compact_flags(value.get("view_flags"), limit=int(limits["max_render_flags"])),
        "render_skipped_objects": _compact_flags(value.get("render_skipped_objects"), limit=10),
        "selected_groups": _compact_group_manifest(value.get("selected_groups"), limit=int(limits["max_group_issues"])),
        "omitted_groups": _compact_group_manifest(value.get("omitted_groups"), limit=int(limits["max_group_issues"])),
    }


def _physical_evidence_summary(feedback: dict, limits: dict | None = None) -> dict:
    limits = limits or _repair_prompt_limits(feedback, None)
    flags = []
    debug = feedback.get("debug_evidence_summary")
    if isinstance(debug, dict) and isinstance(debug.get("physical_flags"), list):
        flags.extend(item for item in debug["physical_flags"] if isinstance(item, dict))
    if isinstance(feedback.get("violations"), list):
        flags.extend(item for item in feedback["violations"] if isinstance(item, dict) and item.get("category") == "physical_debug_flag")
    collision_pairs = []
    collision_object_counts: dict[str, int] = {}
    for item in flags:
        if item.get("type") != "serious_collision":
            continue
        object_ids = _item_object_ids(item)
        if len(object_ids) < 2:
            continue
        pair = object_ids[:2]
        collision_pairs.append(pair)
        for object_id in pair:
            collision_object_counts[object_id] = collision_object_counts.get(object_id, 0) + 1
    unique_pairs = []
    seen = set()
    for pair in sorted(collision_pairs):
        key = tuple(pair)
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
    shown_pairs = unique_pairs[: int(limits["max_collision_pairs"])]
    top_objects = [
        {"object_id": object_id, "collision_count": count}
        for object_id, count in sorted(collision_object_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    ]
    counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in flags:
        key = str(item.get("type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
        confidence = str(item.get("confidence") or "unknown")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        source_kind = str(item.get("source_kind") or "unknown")
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
    return {
        "flag_counts": dict(sorted(counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "serious_collision_count": len(collision_pairs),
        "shown_collision_pair_count": len(shown_pairs),
        "omitted_collision_pair_count": max(0, len(unique_pairs) - len(shown_pairs)),
        "top_collision_objects": top_objects,
        "shown_collision_pairs": shown_pairs,
    }


def _deterministic_flags_empty(compact_feedback: dict) -> bool:
    summary = compact_feedback.get("physical_evidence_summary")
    if isinstance(summary, dict) and summary.get("flag_counts"):
        return False
    return not compact_feedback.get("violations") and not compact_feedback.get("repair_actions")


def _judge_invalid_feedback(feedback: dict) -> bool:
    for item in feedback.get("violations", []) if isinstance(feedback.get("violations"), list) else []:
        if isinstance(item, dict) and item.get("category") == "vlm_judge_issue":
            return True
    return bool(feedback.get("room_consistency_reason"))


def _short_string(value: object, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _round_value(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [_round_value(item, precision) for item in value]
    if isinstance(value, dict):
        return {key: _round_value(item, precision) for key, item in value.items()}
    return value


def _compact_layout_object(obj: dict) -> dict:
    keep = {"object_id", "category", "center", "size", "yaw", "support_parent", "region_id"}
    return {key: value for key, value in obj.items() if key in keep}


def _compact_debug_evidence(debug: dict) -> dict:
    manifest = debug.get("judge_input_manifest") if isinstance(debug.get("judge_input_manifest"), dict) else {}
    return {
        "sanity_flags": _compact_flags(debug.get("sanity_flags"), limit=20),
        "physical_flags": _compact_flags(debug.get("physical_flags"), limit=40),
        "view_flags": _compact_flags(debug.get("view_flags"), limit=20),
        "render_skipped_objects": _compact_flags(debug.get("render_skipped_objects"), limit=20),
        "selected_groups": _compact_group_manifest(manifest.get("selected_groups"), limit=10),
        "omitted_groups": _compact_group_manifest(manifest.get("omitted_groups"), limit=20),
    }


def _compact_flags(value: object, *, limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    compact_flags = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        compact_flags.append(
            {
                key: item[key]
                for key in [
                    "type",
                    "code",
                    "severity",
                    "confidence",
                    "source_kind",
                    "source_confidence",
                    "blocking",
                    "suppressed",
                    "repair_relevant",
                    "objects",
                    "object_ids",
                    "object_id",
                    "message",
                    "group_id",
                    "projection",
                    "view_id",
                    "overlap_ratio",
                    "threshold",
                    "vertical_gap",
                ]
                if key in item
            }
        )
    if len(value) > limit:
        compact_flags.append({"type": "truncated", "message": f"{len(value) - limit} additional flags omitted."})
    return compact_flags


def _compact_group_manifest(value: object, *, limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    groups = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        groups.append(
            {
                key: item[key]
                for key in ["group_id", "object_ids", "reason", "selection_score", "selection_reasons"]
                if key in item
            }
        )
    if len(value) > limit:
        groups.append({"group_id": "truncated", "reason": f"{len(value) - limit} additional groups omitted."})
    return groups


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _layout_output_contract() -> str:
    return _compact_json(
        {
            "scene_id": "string matching case_id/task_id",
            "unit": "meter",
            "coordinate_system": {
                "origin": "case floor-plan frame",
                "x_axis": "floor-plan x",
                "y_axis": "floor-plan y/depth",
                "z_axis": "height",
                "rotation_unit": "degree",
            },
            "objects": [
                {
                    "object_id": "short alias such as o001",
                    "category": "short category",
                    "center": ["x", "y", "z"],
                    "size": ["width", "depth", "height"],
                    "yaw": "deg around z",
                }
            ],
        }
    )


def _compact_instance_object_count(compact_instance: dict) -> int | None:
    rows = compact_instance.get("required_object_rows")
    if isinstance(rows, list):
        return len(rows)
    objects = compact_instance.get("objects")
    return len(objects) if isinstance(objects, list) else None


def _alias_feedback_ids(value: Any, bm_instance: dict) -> Any:
    if not isinstance(bm_instance.get(ALIAS_MAP_KEY), dict):
        return value
    alias_map = get_alias_map(bm_instance)
    return _alias_feedback_value(value, alias_map)


def _alias_feedback_value(value: Any, alias_map: dict) -> Any:
    if isinstance(value, str):
        return _alias_text(value, alias_map)
    if isinstance(value, list):
        return [_alias_feedback_value(item, alias_map) for item in value]
    if isinstance(value, dict):
        aliased = {}
        for key, item in value.items():
            if key in {
                "object_id",
                "move_object_id",
                "keep_object_id",
                "source",
                "target",
                "subject",
                "object",
                "child",
                "parent",
            }:
                aliased[key] = _alias_feedback_value(item, alias_map)
            elif key in {
                "object_ids",
                "objects",
                "repair_targets",
                "locked_object_ids",
                "shown_collision_pairs",
                "top_collision_objects",
            }:
                aliased[key] = _alias_feedback_value(item, alias_map)
            elif isinstance(item, (list, dict)):
                aliased[key] = _alias_feedback_value(item, alias_map)
            elif isinstance(item, str):
                aliased[key] = _alias_text(item, alias_map)
            else:
                aliased[key] = item
        return aliased
    return value


def _alias_text(text: str, alias_map: dict) -> str:
    exact = alias_for_canonical(alias_map, text)
    if exact:
        return exact
    canonical_to_alias = alias_map.get("canonical_to_alias") if isinstance(alias_map, dict) else {}
    if not isinstance(canonical_to_alias, dict) or not canonical_to_alias:
        return text
    result = text
    for canonical_id, alias in sorted(canonical_to_alias.items(), key=lambda item: len(str(item[0])), reverse=True):
        result = result.replace(str(canonical_id), str(alias))
    return result
