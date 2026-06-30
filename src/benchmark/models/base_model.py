from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
    required_ids = _required_object_ids(bm_instance)
    return (
        "Generate an explicit 3D scene layout as JSON only.\n"
        "Use bbox objects with center [x, y, z], size [width, depth, height], "
        "and yaw in degrees around z/up. Do not include validity fields.\n\n"
        "Requirements:\n"
        "- Return exactly one JSON object and no Markdown.\n"
        "- Use the input case_id/task_id as scene_id.\n"
        "- For each input object in bm_instance.objects, create exactly one layout object.\n"
        f"- The layout objects array must contain exactly {len(required_ids)} objects.\n"
        f"- The layout object_id set must exactly equal this list, with no missing, extra, or duplicate ids: {required_ids}.\n"
        "- Preserve each input object id as layout object_id exactly.\n"
        "- Preserve each input object category exactly.\n"
        "- If an input object has bbox_size, use it as layout size.\n"
        "- If an input object has layout_center_hint, use that exact value as layout center.\n"
        "- HSSD source coordinates are not layout coordinates: source_position is [x, height, z], "
        "while layout center is [x, room_depth_y, height_z]. Never copy source_position into center.\n"
        "- For floor objects without layout_center_hint, center.z must be at least size.height / 2 so the bbox bottom is not below floor.\n"
        "- Relation outputs must use keys source and target, not subject/object or source_category.\n"
        "- Keep all object centers inside the room boundary when a boundary is provided.\n\n"
        f"Benchmark instance:\n{_compact_json(compact_bm_instance_for_model(bm_instance))}\n\n"
        f"Layout output contract:\n{_layout_output_contract()}\n"
    )


def build_repair_prompt(
    bm_instance: dict,
    current_layout: dict,
    feedback: dict,
    layout_schema: dict,
) -> str:
    required_ids = _required_object_ids(bm_instance)
    return (
        "Repair the explicit 3D scene layout. Return full corrected layout JSON only, "
        "with no explanation.\n"
        "Preserve object IDs and object count. Preserve valid objects unless necessary. Keep the room "
        "boundary unchanged. Fix only listed violations.\n\n"
        f"The repaired layout objects array must contain exactly {len(required_ids)} objects.\n"
        f"The repaired object_id set must exactly equal this list, with no missing, extra, or duplicate ids: {required_ids}.\n"
        "Do not delete locked_objects; copy locked object records verbatim unless a listed violation explicitly names that object.\n\n"
        "If repair_targets is non-empty, the repaired layout must change at least one repair_target center, size, yaw, "
        "support_parent, or region_id. A byte-for-byte identical layout is not a valid repair.\n"
        "Use deterministic feedback.repair_actions as advisory hints, not mandatory edits. When an action contains "
        "suggested_center or suggested_center_for_move_object, treat it as a locally valid anchor, but you may choose "
        "different centers or yaws when that produces a more globally plausible layout. If an action instead contains "
        "candidate_center_for_reference or candidate_warning, do not blindly copy that center; search for a cleaner "
        "placement.\n"
        "For collision repair actions where candidate_total_overlap_volume_m3 is 0 and candidate_floor_plan_outside_penalty "
        "is 0, strongly prefer the provided suggested_center_for_move_object unless you find an equally clean placement.\n"
        "Prefer repairs that satisfy all of these constraints: keep affected bboxes inside floor_plan regions, avoid "
        "new serious collisions, alleviate large implausible bbox intersections when feasible, "
        "preserve plausible support/room placement, and make minimal changes to valid or locked "
        "objects. Do not blindly follow one suggested center if it creates a new boundary issue, collision, or implausible "
        "placement.\n"
        "Do not optimize solely for zero bbox overlap. Bbox proxies may reasonably overlap for attachment/support cases, "
        "including objects on top of supports, objects contained inside storage, wall-mounted TVs/art/mirrors/shelves/curtains/panels, "
        "beds with headboards, tabletop objects, handles/fixtures, and objects sharing a contact surface.\n"
        "For room_boundary flags, move the target bbox into a suitable floor_plan region. For serious_collision or "
        "soft_collision repair actions, separate the listed objects only when the overlap is implausible penetration; "
        "otherwise preserve plausible attachment, support, containment, and room placement.\n"
        "The benchmark room may use dataset floor-plan coordinates, including negative x/y values; do not assume a "
        "0-based front-left origin during repair. Treat layout_center_hint/source positions as historical evidence "
        "only during repair; do not copy them back if they caused violations.\n\n"
        f"Benchmark instance:\n{_compact_json(compact_bm_instance_for_repair(bm_instance))}\n\n"
        f"Current layout:\n{_compact_json(compact_layout_for_model(current_layout))}\n\n"
        f"Deterministic feedback:\n{_compact_json(compact_feedback_for_model(feedback))}\n\n"
        f"Layout output contract:\n{_layout_output_contract()}\n"
    )


def compact_bm_instance_for_model(bm_instance: dict) -> dict:
    """Keep model-facing prompts compact while preserving full case files on disk."""
    compact = {
        key: value
        for key, value in bm_instance.items()
        if key not in {"objects", "source", "reference_layout"}
    }
    if isinstance(bm_instance.get("objects"), list):
        compact["objects"] = [_compact_input_object(obj) for obj in bm_instance["objects"] if isinstance(obj, dict)]
    if isinstance(bm_instance.get("source"), dict):
        compact["source"] = {
            key: value
            for key, value in bm_instance["source"].items()
            if key
            in {
                "dataset",
                "scene_instance",
                "stage_instance",
                "translation_origin",
                "raw_object_instance_count",
                "imported_object_count",
                "truncated",
            }
        }
    return compact


def compact_bm_instance_for_repair(bm_instance: dict) -> dict:
    compact = compact_bm_instance_for_model(bm_instance)
    objects = compact.get("objects")
    if isinstance(objects, list):
        compact["objects"] = [
            {
                key: value
                for key, value in obj.items()
                if key not in {"layout_center_hint", "layout_center_hint_source", "source_floor_position"}
            }
            for obj in objects
            if isinstance(obj, dict)
        ]
    return compact


def _compact_input_object(obj: dict) -> dict:
    keep = {
        "id",
        "category",
        "bbox_size",
        "bbox_size_source",
        "required",
        "source",
        "layout_center_hint",
        "layout_center_hint_source",
        "source_floor_position",
        "source_template_name",
        "semantic_category",
        "hssd_semantic",
    }
    return {key: value for key, value in obj.items() if key in keep}


def _required_object_ids(bm_instance: dict) -> list[str]:
    return [
        str(obj["id"])
        for obj in bm_instance.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("id"), str)
    ]


def compact_layout_for_model(layout: dict) -> dict:
    compact = {
        key: value
        for key, value in layout.items()
        if key in {"scene_id", "unit", "coordinate_system", "relations", "hierarchy"}
    }
    if isinstance(layout.get("objects"), list):
        compact["objects"] = [_compact_layout_object(obj) for obj in layout["objects"] if isinstance(obj, dict)]
    return compact


def compact_feedback_for_model(feedback: dict) -> dict:
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
                    "severity",
                    "objects",
                    "message",
                    "group_id",
                    "projection",
                    "view_id",
                    "overlap_ratio",
                    "threshold",
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
                "origin": "case floor-plan coordinate frame; HSSD cases may use negative x/y values",
                "x_axis": "floor-plan x coordinate",
                "y_axis": "floor-plan y/depth coordinate",
                "z_axis": "height",
                "rotation_unit": "degree",
            },
            "objects": [
                {
                    "object_id": "input object id",
                    "category": "input object category",
                    "center": ["x", "y", "z"],
                    "size": ["width", "depth", "height"],
                    "yaw": "degrees around z/up",
                    "support_parent": "floor or object_id",
                    "region_id": "optional string",
                }
            ],
            "relations": [{"type": "near/facing/etc", "source": "object_id", "target": "object_id", "hard": False}],
            "hierarchy": {"regions": [], "floor_objects": ["object_id"], "supported_objects": []},
        }
    )
