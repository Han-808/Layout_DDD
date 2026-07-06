from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any


DEFAULT_TEXT_BUDGET = {
    "max_total_chars": 12000,
    "max_scene_summary_chars": 1500,
    "max_layout_summary_chars": 2500,
    "max_selected_group_objects": 40,
    "max_objects_per_group_in_prompt": 12,
    "max_flag_examples_per_type": 8,
    "numeric_precision": 2,
}

COORDINATE_CONVENTION = {
    "origin": "case floor-plan coordinate frame; HSSD-derived cases may use negative x/y values",
    "x_axis": "floor-plan x coordinate",
    "y_axis": "floor-plan y/depth coordinate",
    "z_axis": "height",
    "unit": "meters",
    "yaw": "degrees around z/up axis",
    "bbox": "center [x,y,z] + size [width,depth,height]",
}

TEMPORARY_RUBRIC = {
    "target": "Evaluate explicit 3D bbox-proxy layout quality, not photorealistic reconstruction.",
    "do_not_penalize": [
        "missing meshes",
        "missing textures",
        "missing real wall geometry",
        "missing doors/windows",
        "lack of photorealism",
    ],
    "criteria": [
        {"id": "parseability", "description": "Parseability and structural usability of the layout JSON."},
        {"id": "completeness", "description": "Explicit task/object completeness relative to the benchmark instance."},
        {"id": "boundary", "description": "Room-boundary and containment plausibility within the proxy room."},
        {"id": "height", "description": "Floor-height and vertical plausibility."},
        {"id": "collision", "description": "Severe collision or overlap between bbox objects."},
        {"id": "support", "description": "Support and local spatial plausibility."},
        {"id": "evidence", "description": "Evidence sufficiency from rendered views and structured summaries."},
    ],
}

JUDGE_OUTPUT_SCHEMA = {
    "valid": "boolean",
    "score": "integer 0..4",
    "confidence": "low | medium | high",
    "judgement_status": "valid_judgement | insufficient_evidence | judge_error",
    "brief_reasoning": "short string; do not include chain-of-thought",
    "issues": [
        {
            "group_id": "string or null",
            "issue_type": "parseability | completeness | boundary | height | collision | support | spatial_relation | evidence",
            "severity": "minor | major | critical",
            "object_ids": ["string"],
            "evidence": "string",
            "repair_hint": "string",
        }
    ],
    "insufficient_evidence": "boolean",
}


def text_budget_config(benchmark_config: dict | None) -> dict:
    config = dict(DEFAULT_TEXT_BUDGET)
    vlm_judge = (benchmark_config or {}).get("vlm_judge")
    text_budget = vlm_judge.get("text_budget") if isinstance(vlm_judge, dict) else {}
    if isinstance(text_budget, dict):
        config.update(text_budget)
    return {
        "max_total_chars": _positive_int(config.get("max_total_chars"), DEFAULT_TEXT_BUDGET["max_total_chars"]),
        "max_scene_summary_chars": _positive_int(config.get("max_scene_summary_chars"), DEFAULT_TEXT_BUDGET["max_scene_summary_chars"]),
        "max_layout_summary_chars": _positive_int(config.get("max_layout_summary_chars"), DEFAULT_TEXT_BUDGET["max_layout_summary_chars"]),
        "max_selected_group_objects": _positive_int(config.get("max_selected_group_objects"), DEFAULT_TEXT_BUDGET["max_selected_group_objects"]),
        "max_objects_per_group_in_prompt": _positive_int(config.get("max_objects_per_group_in_prompt"), DEFAULT_TEXT_BUDGET["max_objects_per_group_in_prompt"]),
        "max_flag_examples_per_type": _positive_int(config.get("max_flag_examples_per_type"), DEFAULT_TEXT_BUDGET["max_flag_examples_per_type"]),
        "numeric_precision": _positive_int(config.get("numeric_precision"), DEFAULT_TEXT_BUDGET["numeric_precision"]),
    }


def build_scene_summary(case: dict, input_level: str, text_budget: dict) -> dict:
    room = case.get("room") if isinstance(case, dict) else {}
    source = case.get("source") if isinstance(case, dict) else {}
    summary = {
        "case_id": case.get("case_id") if isinstance(case, dict) else None,
        "task_id": case.get("task_id") if isinstance(case, dict) else None,
        "scene_id": (
            _source_value(source, ["scene_id", "scene_instance", "scene_instance_id"])
            or (case.get("scene_id") if isinstance(case, dict) else None)
            or (case.get("case_id") if isinstance(case, dict) else None)
            or (case.get("task_id") if isinstance(case, dict) else None)
        ),
        "scene_prompt": _description_text(case),
        "input_level": input_level,
        "input_mode": case.get("input_mode") if isinstance(case, dict) else None,
        "room_proxy": _room_proxy_summary(room, text_budget),
        "num_input_objects": _input_object_count(case),
        "dataset_source": _dataset_source(case, source),
        "hssd_source_path": _source_value(source, ["scene_instance", "scene_instance_path", "source_path", "path"]),
        "notes": [
            "Room boundary is bbox-evaluation evidence. If floor_plan.regions exist, use their polygon union as the primary floor plan.",
            "Aggregate room rectangles are compatibility proxies only when multi-region floor plans are available.",
            "Do not treat missing meshes/textures/doors/windows as errors.",
        ],
    }
    return _cap_scene_summary(_round_numbers(summary, text_budget), text_budget)


def build_layout_summary(
    *,
    layout: dict,
    renderable_layout: dict | None = None,
    layout_normalization_summary: dict | None = None,
    object_groups: list[dict] | None = None,
    sanity_flags: list[dict] | None = None,
    physical_flags: list[dict] | None = None,
    view_flags: list[dict] | None = None,
    render_skipped_objects: list[dict] | None = None,
    evidence_selection: dict | None = None,
    judge_input_manifest: dict | None = None,
    text_budget: dict | None = None,
) -> dict:
    budget = text_budget or DEFAULT_TEXT_BUDGET
    groups = [group for group in (object_groups or []) if isinstance(group, dict)]
    selection = evidence_selection if isinstance(evidence_selection, dict) else {}
    manifest = judge_input_manifest if isinstance(judge_input_manifest, dict) else {}
    budgeting_enabled = bool(selection.get("budgeting_enabled") or manifest.get("budgeting_enabled"))
    selected_ids = _selected_group_ids(selection, manifest, groups, budgeting_enabled)
    omitted_ids = _omitted_group_ids(selection, manifest)
    layout_objects = [obj for obj in layout.get("objects", []) if isinstance(obj, dict)] if isinstance(layout, dict) else []
    renderable_objects = [obj for obj in (renderable_layout or layout).get("objects", []) if isinstance(obj, dict)] if isinstance(renderable_layout or layout, dict) else []
    selected_details = _selected_group_details(groups, layout_objects, selected_ids, budgeting_enabled, budget)
    evidence_budgeting_policy = _evidence_budgeting_policy(budgeting_enabled, len(groups), len(selected_ids), len(omitted_ids))
    summary = {
        "num_layout_objects": len(layout_objects),
        "num_renderable_objects": len(renderable_objects),
        "num_groups": len(groups),
        "coordinate_convention": COORDINATE_CONVENTION,
        "layout_normalization_summary": layout_normalization_summary or {},
        "skipped_render_objects_count": len(render_skipped_objects or []),
        "selected_groups_count": len(selected_ids) if budgeting_enabled else len(groups),
        "omitted_groups_count": len(omitted_ids) if budgeting_enabled else 0,
        "budgeting_enabled": budgeting_enabled,
        "evidence_budgeting_policy": evidence_budgeting_policy,
        "selected_group_details": selected_details,
        "omitted_groups_summary": _omitted_group_summary(groups, omitted_ids) if budgeting_enabled else [],
        "flag_summary": build_flag_summary(
            sanity_flags=sanity_flags or [],
            physical_flags=physical_flags or [],
            view_flags=view_flags or [],
            render_skipped_objects=render_skipped_objects or [],
            text_budget=budget,
        ),
    }
    if not budgeting_enabled:
        summary["full_group_details_included"] = True
    return _cap_layout_summary(_round_numbers(summary, budget), budget)


def build_flag_summary(
    *,
    sanity_flags: list[dict],
    physical_flags: list[dict],
    view_flags: list[dict],
    render_skipped_objects: list[dict],
    text_budget: dict,
) -> dict:
    return {
        "schema_flags": _summarize_flags(sanity_flags, text_budget),
        "physical_flags": _summarize_flags(physical_flags, text_budget),
        "view_flags": _summarize_flags(view_flags, text_budget),
        "render_skipped_objects": _summarize_flags(render_skipped_objects, text_budget),
    }


def build_evidence_manifest(evidence_selection: dict, image_manifest: list[dict]) -> list[dict]:
    records = []
    for item in image_manifest:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "id": item.get("id"),
                "scope": item.get("scope"),
                "path": item.get("path"),
                "meaning": _image_meaning(str(item.get("id") or ""), str(item.get("scope") or "")),
                "included_in_prompt": bool(item.get("included_in_prompt")),
            }
        )
    return records


def build_judge_prompt_payload(
    *,
    case: dict,
    layout: dict,
    renderable_layout: dict | None,
    input_level: str,
    layout_normalization_summary: dict | None,
    object_groups: list[dict],
    sanity_flags: list[dict],
    physical_flags: list[dict],
    view_flags: list[dict],
    render_skipped_objects: list[dict],
    relation_specs: list[dict],
    attachment_specs: list[dict],
    evidence_selection: dict,
    image_manifest: list[dict],
    benchmark_config: dict | None,
) -> dict:
    budget = text_budget_config(benchmark_config)
    scene_summary = build_scene_summary(case, input_level, budget)
    layout_summary = build_layout_summary(
        layout=layout,
        renderable_layout=renderable_layout,
        layout_normalization_summary=layout_normalization_summary,
        object_groups=object_groups,
        sanity_flags=sanity_flags,
        physical_flags=physical_flags,
        view_flags=view_flags,
        render_skipped_objects=render_skipped_objects,
        evidence_selection=evidence_selection,
        text_budget=budget,
    )
    payload = {
        "role": "independent evaluator of explicit 3D bbox-proxy layouts",
        "coordinate_convention": COORDINATE_CONVENTION,
        "evaluation_policy": {
            "parseable_layouts": "Run VLM judge; overall_valid is set from vlm_judgement.valid.",
            "unparseable_layouts": "Skip VLM judge; overall_valid=false; judgement_status=unparseable_layout.",
            "deterministic_flags": "Schema, physical, view, skipped render object, and grouping diagnostics are evidence only.",
            "physical_evidence_semantics": (
                "High-confidence serious collisions should strongly affect validity. "
                "Fallback-derived room boundary or wall-height flags are lower-confidence evidence, not hard invalidity. "
                "floating_or_vertical_inconsistency is evidence of possible spatial implausibility but is not hard invalidity by itself."
            ),
            "fallback_metadata": (
                "Some room geometry constraints may be fallback-derived and approximate. Treat source_kind=fallback_default "
                "or object_position_extent_fallback as lower-confidence evidence."
            ),
            "use_images_and_summaries_together": True,
            "evidence_budgeting": (
                "If evidence budgeting is enabled, omitted groups/images are omitted for prompt budget only. "
                "Do not treat budget-omitted groups as missing layout objects or task incompleteness."
            ),
            "completeness_source": "Judge object completeness from num_layout_objects versus num_input_objects, not from selected/omitted evidence groups.",
        },
        "scene_summary": scene_summary,
        "layout_summary": layout_summary,
        "evidence_manifest": build_evidence_manifest(evidence_selection, image_manifest),
        "flag_summary": layout_summary.get("flag_summary", {}),
        "rubric": TEMPORARY_RUBRIC,
        "score_scale": {
            "0": "Unusable, invalid, or impossible to judge because evidence is insufficient.",
            "1": "Severe bbox layout or task-completeness problems.",
            "2": "Partially plausible with multiple important issues.",
            "3": "Mostly coherent with minor bbox layout issues.",
            "4": "Coherent, plausible, task-complete bbox-proxy layout.",
        },
        "relations_to_judge": _compact_specs(relation_specs),
        "attachments_to_judge": _compact_specs(attachment_specs),
        "required_output_schema": JUDGE_OUTPUT_SCHEMA,
        "output_instruction": "Return only one valid JSON object. Do not include Markdown or chain-of-thought.",
    }
    payload = _enforce_total_budget(payload, budget)
    prompt_text = _json_text(payload)
    return {
        "prompt_payload": payload,
        "scene_summary": payload["scene_summary"],
        "layout_summary": payload["layout_summary"],
        "text_budget_used": {
            "max_total_chars": budget["max_total_chars"],
            "max_scene_summary_chars": budget["max_scene_summary_chars"],
            "max_layout_summary_chars": budget["max_layout_summary_chars"],
            "scene_summary_chars": len(_json_text(payload["scene_summary"])),
            "layout_summary_chars": len(_json_text(payload["layout_summary"])),
            "prompt_chars": len(prompt_text),
            "truncated": _contains_truncation(payload),
        },
    }


def _room_proxy_summary(room: object, text_budget: dict) -> dict:
    if not isinstance(room, dict):
        return {"type": "unknown", "source": "missing"}
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    valid_regions = [region for region in regions if isinstance(region, dict)]
    if valid_regions:
        width, depth = _room_width_depth(room)
        return {
            "type": "multi_region_floor_plan",
            "primary_representation": floor_plan.get("primary_representation") or "regions",
            "source": floor_plan.get("source") or room.get("room_layout_source") or room.get("boundary_source"),
            "coordinate_mapping": floor_plan.get("coordinate_mapping"),
            "boundary_check_source": "floor_plan.regions polygon union",
            "region_count": len(valid_regions),
            "region_labels": _region_labels(valid_regions, limit=16),
            "aggregate_boundary_role": floor_plan.get("aggregate_boundary_role") or "compatibility_proxy",
            "aggregate_extent": {"width": width, "depth": depth},
            "height": _first_present(room, ["wall_height", "height", "height_m"]),
        }
    width, depth = _room_width_depth(room)
    return {
        "type": "synthetic_proxy_rectangle" if width is not None and depth is not None else "proxy_polygon",
        "width": width,
        "depth": depth,
        "height": _first_present(room, ["wall_height", "height", "height_m"]),
        "source": room.get("source") or "case.room boundary/floor_polygon proxy",
    }


def _region_labels(regions: list[dict], *, limit: int) -> list[str]:
    labels = []
    for region in regions[:limit]:
        label = region.get("label") or region.get("name") or region.get("id")
        if label is not None:
            labels.append(str(label))
    if len(regions) > limit:
        labels.append(f"... {len(regions) - limit} more")
    return labels


def _evidence_budgeting_policy(budgeting_enabled: bool, group_count: int, selected_count: int, omitted_count: int) -> dict:
    if not budgeting_enabled:
        return {
            "mode": "full",
            "all_groups_sent_to_judge": True,
            "omitted_groups_are_missing_objects": False,
        }
    return {
        "mode": "budgeted",
        "all_groups_rendered_for_humans": True,
        "selected_groups_sent_to_judge": selected_count,
        "omitted_groups_not_sent_to_judge": omitted_count,
        "total_groups_in_layout": group_count,
        "omitted_groups_are_missing_objects": False,
        "omitted_groups_reason": "prompt/image budget, not object absence",
        "judge_instruction": "Do not mark completeness invalid solely because groups/images were omitted by evidence budgeting.",
    }


def _room_width_depth(room: dict) -> tuple[float | None, float | None]:
    width = _to_float(_first_present(room, ["width", "room_width", "width_m"]))
    depth = _to_float(_first_present(room, ["depth", "room_depth", "depth_m"]))
    if width is not None and depth is not None:
        return width, depth
    polygon = room.get("floor_polygon") or room.get("boundary")
    if isinstance(polygon, list):
        points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
        if points:
            try:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
            except (TypeError, ValueError):
                return width, depth
            return max(xs) - min(xs), max(ys) - min(ys)
    return width, depth


def _input_object_count(case: dict) -> int:
    for key in ["objects", "required_objects"]:
        value = case.get(key) if isinstance(case, dict) else None
        if isinstance(value, list):
            return len(value)
    return 0


def _dataset_source(case: dict, source: object) -> str:
    if isinstance(source, dict):
        for key in ["dataset", "dataset_source", "source_dataset"]:
            if source.get(key):
                return str(source[key])
    for key in ["dataset", "dataset_source"]:
        if isinstance(case, dict) and case.get(key):
            return str(case[key])
    return ""


def _source_value(source: object, keys: list[str]) -> str:
    if not isinstance(source, dict):
        return ""
    for key in keys:
        value = source.get(key)
        if value:
            return str(value)
    return ""


def _selected_group_ids(selection: dict, manifest: dict, groups: list[dict], budgeting_enabled: bool) -> list[str]:
    if not budgeting_enabled:
        return [str(group.get("group_id")) for group in groups if group.get("group_id")]
    selected = selection.get("selected_groups") if isinstance(selection.get("selected_groups"), list) else manifest.get("selected_groups", [])
    return [str(item.get("group_id")) for item in selected if isinstance(item, dict) and item.get("group_id")]


def _omitted_group_ids(selection: dict, manifest: dict) -> list[str]:
    omitted = selection.get("omitted_groups") if isinstance(selection.get("omitted_groups"), list) else manifest.get("omitted_groups", [])
    return [str(item.get("group_id")) for item in omitted if isinstance(item, dict) and item.get("group_id")]


def _selected_group_details(groups: list[dict], layout_objects: list[dict], selected_ids: list[str], budgeting_enabled: bool, budget: dict) -> list[dict]:
    object_by_id = {str(obj.get("object_id") or obj.get("id")): obj for obj in layout_objects if obj.get("object_id") or obj.get("id")}
    selected_set = set(selected_ids)
    details = []
    shown_objects_total = 0
    for group in groups:
        group_id = str(group.get("group_id"))
        if budgeting_enabled and group_id not in selected_set:
            continue
        object_ids = [str(item) for item in group.get("object_ids", []) if item is not None]
        object_limit = min(
            budget["max_objects_per_group_in_prompt"],
            max(0, budget["max_selected_group_objects"] - shown_objects_total),
        )
        shown_ids = object_ids[:object_limit]
        shown_objects_total += len(shown_ids)
        record = {
            "group_id": group_id,
            "num_objects": len(object_ids),
            "object_ids": object_ids,
            "edge_reasons": group.get("edge_reasons", []),
            "formation_edges": group.get("formation_edges", []),
            "objects": [_compact_layout_object(object_by_id.get(object_id, {"object_id": object_id})) for object_id in shown_ids],
        }
        if len(shown_ids) < len(object_ids):
            record["objects_truncated"] = {"truncated": True, "shown": len(shown_ids), "total": len(object_ids)}
        details.append(record)
        if shown_objects_total >= budget["max_selected_group_objects"]:
            remaining_groups = [item for item in groups if str(item.get("group_id")) not in {detail["group_id"] for detail in details}]
            if remaining_groups:
                details.append({"truncated": True, "shown_groups": len(details), "total_groups": len(groups), "reason": "max_selected_group_objects"})
            break
    return details


def _omitted_group_summary(groups: list[dict], omitted_ids: list[str]) -> list[dict]:
    omitted_set = set(omitted_ids)
    return [
        {
            "group_id": group.get("group_id"),
            "num_objects": group.get("num_objects", len(group.get("object_ids", []))),
            "object_ids_count": len(group.get("object_ids", [])) if isinstance(group.get("object_ids"), list) else 0,
            "edge_reasons": group.get("edge_reasons", []),
        }
        for group in groups
        if str(group.get("group_id")) in omitted_set
    ]


def _compact_layout_object(obj: dict) -> dict:
    compact = {
        "object_id": obj.get("model_object_id") or obj.get("object_id"),
        "category": obj.get("model_category") or obj.get("category"),
    }
    if obj.get("model_object_id"):
        compact["canonical_object_id"] = obj.get("canonical_object_id") or obj.get("object_id")
    for key in ["center", "size", "yaw", "support_parent", "region_id"]:
        if key in obj:
            compact[key] = _round_value(obj.get(key), 2)
    return compact


def _summarize_flags(flags: list[dict], text_budget: dict) -> dict:
    by_type = Counter(str(flag.get("type") or "unknown") for flag in flags if isinstance(flag, dict))
    examples_by_type: dict[str, list[dict]] = defaultdict(list)
    limit = text_budget["max_flag_examples_per_type"]
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        flag_type = str(flag.get("type") or "unknown")
        if len(examples_by_type[flag_type]) >= limit:
            continue
        examples_by_type[flag_type].append(
            {
                key: flag[key]
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
                    "group_id",
                    "projection",
                    "message",
                    "reason",
                    "vertical_gap",
                ]
                if key in flag
            }
        )
    return {
        "total": len(flags),
        "by_type": dict(sorted(by_type.items())),
        "examples_by_type": {
            key: {
                "shown": len(value),
                "total": by_type[key],
                "truncated": by_type[key] > len(value),
                "examples": value,
            }
            for key, value in sorted(examples_by_type.items())
        },
    }


def _compact_specs(specs: list[dict]) -> list[dict]:
    return [
        {
            key: spec.get(key)
            for key in ["id", "type", "subject", "object", "source", "target", "child", "parent", "hard"]
            if key in spec
        }
        for spec in specs
        if isinstance(spec, dict)
    ]


def _cap_scene_summary(summary: dict, text_budget: dict) -> dict:
    cap = text_budget["max_scene_summary_chars"]
    if len(_json_text(summary)) <= cap:
        return summary
    capped = dict(summary)
    capped["scene_prompt"] = _truncate_string(str(capped.get("scene_prompt", "")), 240)
    capped["hssd_source_path"] = _truncate_string(str(capped.get("hssd_source_path", "")), 180)
    capped["truncated"] = True
    capped["truncation"] = {"cap_chars": cap, "strategy": "truncate long scene_prompt and source path"}
    return capped


def _cap_layout_summary(summary: dict, text_budget: dict) -> dict:
    cap = text_budget["max_layout_summary_chars"]
    if len(_json_text(summary)) <= cap:
        return summary
    capped = dict(summary)
    details = capped.get("selected_group_details", [])
    if isinstance(details, list) and details:
        minimal = []
        for item in details[:2]:
            if not isinstance(item, dict):
                continue
            minimal.append(
                {
                    "group_id": item.get("group_id"),
                    "num_objects": item.get("num_objects"),
                    "object_ids": item.get("object_ids", [])[: text_budget["max_objects_per_group_in_prompt"]],
                    "objects_truncated": {
                        "truncated": True,
                        "shown": min(len(item.get("object_ids", [])), text_budget["max_objects_per_group_in_prompt"]),
                        "total": len(item.get("object_ids", [])),
                    },
                }
            )
        capped["selected_group_details"] = minimal
    capped["truncated"] = True
    capped["truncation"] = {"cap_chars": cap, "strategy": "compact selected group details"}
    if len(_json_text(capped)) > cap:
        capped["selected_group_details"] = [
            {
                "truncated": True,
                "shown": 0,
                "total": len(details) if isinstance(details, list) else 0,
                "reason": "layout summary text budget",
            }
        ]
        capped["truncation"] = {"cap_chars": cap, "strategy": "drop selected group details after recording counts"}
    return capped


def _enforce_total_budget(payload: dict, text_budget: dict) -> dict:
    cap = text_budget["max_total_chars"]
    if len(_json_text(payload)) <= cap:
        return payload
    result = dict(payload)
    layout_summary = dict(result.get("layout_summary", {}))
    selected_group_details = layout_summary.get("selected_group_details", [])
    layout_summary["selected_group_details"] = [
        {
            "truncated": True,
            "shown": 0,
            "total": len(selected_group_details) if isinstance(selected_group_details, list) else 0,
            "reason": "total prompt text budget",
        }
    ]
    layout_summary["total_budget_truncation"] = {
        "truncated": True,
        "cap_chars": cap,
        "strategy": "removed selected group details from prompt payload only",
    }
    result["layout_summary"] = layout_summary
    if len(_json_text(result)) <= cap:
        return result
    scene_summary = dict(result.get("scene_summary", {}))
    scene_summary["scene_prompt"] = _truncate_string(str(scene_summary.get("scene_prompt", "")), 120)
    scene_summary["total_budget_truncation"] = {"truncated": True, "cap_chars": cap}
    result["scene_summary"] = scene_summary
    if len(_json_text(result)) <= cap:
        return result
    result["relations_to_judge"] = []
    result["attachments_to_judge"] = []
    result["total_budget_truncation"] = {"truncated": True, "cap_chars": cap, "strategy": "dropped relation lists last"}
    if len(_json_text(result)) <= cap:
        return result
    result["rubric"] = {
        "target": TEMPORARY_RUBRIC["target"],
        "do_not_penalize": TEMPORARY_RUBRIC["do_not_penalize"],
        "criteria": [item["id"] for item in TEMPORARY_RUBRIC["criteria"]],
        "truncated": True,
    }
    result["required_output_schema"] = {
        "valid": "boolean",
        "score": "integer 0..4",
        "confidence": "low|medium|high",
        "judgement_status": "valid_judgement|insufficient_evidence|judge_error",
        "brief_reasoning": "string",
        "issues": "list of issue objects",
        "insufficient_evidence": "boolean",
        "truncated": True,
    }
    if len(_json_text(result)) <= cap:
        return result
    result["flag_summary"] = {
        key: {"total": value.get("total", 0), "by_type": value.get("by_type", {}), "truncated": True}
        for key, value in result.get("flag_summary", {}).items()
        if isinstance(value, dict)
    }
    result["score_scale"] = {"0": "unusable", "1": "severe", "2": "partial", "3": "mostly coherent", "4": "coherent", "truncated": True}
    if len(_json_text(result)) <= cap:
        return result
    result["coordinate_convention"] = {"unit": "meters", "bbox": "center+size", "truncated": True}
    result["evaluation_policy"] = {
        "overall_valid": "vlm_judgement.valid",
        "deterministic_flags": "evidence_only",
        "budgeted_omissions": "not_missing_objects",
        "truncated": True,
    }
    result["scene_summary"] = {
        "case_id": result.get("scene_summary", {}).get("case_id") if isinstance(result.get("scene_summary"), dict) else None,
        "num_input_objects": result.get("scene_summary", {}).get("num_input_objects") if isinstance(result.get("scene_summary"), dict) else None,
        "dataset_source": result.get("scene_summary", {}).get("dataset_source") if isinstance(result.get("scene_summary"), dict) else None,
        "truncated": True,
    }
    prior_layout_summary = result.get("layout_summary", {}) if isinstance(result.get("layout_summary"), dict) else {}
    prior_group_details = prior_layout_summary.get("selected_group_details", [])
    prior_group_total = 0
    if isinstance(prior_group_details, list) and prior_group_details:
        first_group = prior_group_details[0] if isinstance(prior_group_details[0], dict) else {}
        prior_group_total = int(first_group.get("total") or first_group.get("num_objects") or len(prior_group_details))
    result["layout_summary"] = {
        "num_layout_objects": prior_layout_summary.get("num_layout_objects"),
        "num_groups": prior_layout_summary.get("num_groups"),
        "selected_groups_count": prior_layout_summary.get("selected_groups_count"),
        "flag_summary": prior_layout_summary.get("flag_summary", {}),
        "selected_group_details": [{"truncated": True, "shown": 0, "total": prior_group_total, "reason": "total prompt text budget"}],
        "truncated": True,
    }
    result["total_budget_truncation"] = {"truncated": True, "cap_chars": cap, "strategy": "final compact payload"}
    return result


def _contains_truncation(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("truncated") is True:
            return True
        return any(_contains_truncation(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_truncation(item) for item in value)
    return False


def _description_text(case: dict) -> str:
    if not isinstance(case, dict):
        return ""
    description = case.get("description")
    if isinstance(description, dict):
        return str(description.get("text", ""))
    if isinstance(description, str):
        return description
    return str(case.get("scene_prompt", ""))


def _image_meaning(artifact_id: str, scope: str) -> str:
    if scope == "global" or artifact_id.startswith("topdown_global"):
        return "global top-down proxy room view"
    if artifact_id.endswith("_xy"):
        return "selected group xy top-down view"
    if artifact_id.endswith("_yz"):
        return "selected group yz side elevation view"
    if artifact_id.endswith("_xz"):
        return "selected group xz front elevation view"
    return "rendered bbox evidence view"


def _first_present(value: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_numbers(value: Any, text_budget: dict) -> Any:
    return _round_value(value, int(text_budget.get("numeric_precision", 2)))


def _round_value(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [_round_value(item, precision) for item in value]
    if isinstance(value, dict):
        return {key: _round_value(item, precision) for key, item in value.items()}
    return value


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _truncate_string(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
