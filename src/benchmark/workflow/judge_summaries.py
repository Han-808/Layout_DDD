from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from benchmark.data.scene_adapters import layout_to_scene, normalize_scene


DEFAULT_TEXT_BUDGET = {
    "max_total_chars": 12000,
    "max_scene_summary_chars": 1500,
    "max_layout_summary_chars": 2500,
    "max_scene_assets": 80,
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
    "asset_geometry": "asset.placement.position [x,y,z], asset.dimensions [width,depth,height], and yaw_degrees",
}

TEMPORARY_RUBRIC = {
    "target": "Evaluate explicit 3D scene asset placement quality, not photorealistic reconstruction.",
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
        {"id": "collision", "description": "Severe collision or overlap between placed assets."},
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
        "max_scene_assets": _positive_int(config.get("max_scene_assets"), DEFAULT_TEXT_BUDGET["max_scene_assets"]),
        "max_selected_group_objects": _positive_int(config.get("max_selected_group_objects"), DEFAULT_TEXT_BUDGET["max_selected_group_objects"]),
        "max_objects_per_group_in_prompt": _positive_int(config.get("max_objects_per_group_in_prompt"), DEFAULT_TEXT_BUDGET["max_objects_per_group_in_prompt"]),
        "max_flag_examples_per_type": _positive_int(config.get("max_flag_examples_per_type"), DEFAULT_TEXT_BUDGET["max_flag_examples_per_type"]),
        "numeric_precision": _positive_int(config.get("numeric_precision"), DEFAULT_TEXT_BUDGET["numeric_precision"]),
    }


def build_compact_scene_payload(scene: dict, layout: dict, text_budget: dict) -> dict:
    normalized = normalize_scene(scene)
    assets = [asset for asset in normalized.get("assets", []) if isinstance(asset, dict)]
    max_assets = int(text_budget.get("max_scene_assets", DEFAULT_TEXT_BUDGET["max_scene_assets"]))
    shown_assets = assets[:max_assets]
    omitted_assets = assets[max_assets:]
    compact = {
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "unit": normalized.get("unit", "meter"),
        "room": _compact_room(normalized.get("room"), text_budget),
        "assets": [_compact_scene_asset(asset, text_budget) for asset in shown_assets],
        "asset_count": len(assets),
        "geometry_asset_count": sum(1 for asset in assets if _asset_has_geometry(asset)),
        "non_geometric_asset_count": sum(1 for asset in assets if not _asset_has_geometry(asset)),
    }
    if isinstance(normalized.get("scene_ref"), dict):
        compact["scene_ref"] = _compact_scene_ref(normalized["scene_ref"])
    if isinstance(normalized.get("relations"), list):
        compact["relations"] = _compact_specs(normalized["relations"])
    if isinstance(normalized.get("attachments"), list):
        compact["attachments"] = _compact_specs(normalized["attachments"])
    if omitted_assets:
        omitted_ids = [str(asset.get("asset_id") or asset.get("object_id") or f"asset_{idx + max_assets + 1:03d}") for idx, asset in enumerate(omitted_assets)]
        compact["omitted_asset_count"] = len(omitted_assets)
        compact["omitted_asset_ids"] = omitted_ids[:200]
        compact["omitted_asset_ids_truncated"] = len(omitted_ids) > 200
        compact["assets_truncated"] = True
        compact["truncation"] = {"reason": "max_scene_assets prompt budget", "shown": len(shown_assets), "total": len(assets)}
    non_geometric_assets = layout.get("_non_geometric_assets") if isinstance(layout, dict) else None
    if not isinstance(non_geometric_assets, list) and isinstance(layout, dict):
        non_geometric_assets = layout.get("_non_bbox_assets")
    if isinstance(non_geometric_assets, list) and non_geometric_assets:
        compact["non_geometric_assets"] = [
            {
                key: item.get(key)
                for key in ["asset_id", "object_id", "category", "reason"]
                if isinstance(item, dict) and key in item
            }
            for item in non_geometric_assets[:max_assets]
            if isinstance(item, dict)
        ]
    return _round_numbers(compact, text_budget)


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
            "Room boundary is geometry-proxy evidence. If floor_plan.regions exist, use their polygon union as the primary floor plan.",
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
    scene: dict | None = None,
    renderable_layout: dict | None,
    input_level: str,
    judge_input_mode: str = "json_only",
    render_evidence_used: bool = False,
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
    compact_scene = build_compact_scene_payload(scene or layout_to_scene(layout, case), layout, budget)
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
    geometry_rate = _geometry_available_rate(compact_scene)
    payload = {
        "task": "Evaluate whether this 3D scene/layout placement is plausible.",
        "role": "independent evaluator of 3D scene/layout placement",
        "vlm_judge_input_mode": judge_input_mode,
        "json_scene_used": True,
        "render_evidence_used": bool(render_evidence_used),
        "coordinate_convention": COORDINATE_CONVENTION,
        "evaluation_policy": {
            "input_modes": "Input may be JSON-only or JSON plus rendered views.",
            "json_only": "When rendered views are absent, judge from structured scene JSON and evidence.",
            "json_plus_render": "Rendered geometry-proxy views are legacy/full visual evidence, not a mesh dependency.",
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
        "scene": compact_scene,
        "structured_evidence": {
            "object_completeness": {
                "num_input_objects": scene_summary.get("num_input_objects"),
                "num_scene_assets": compact_scene.get("asset_count"),
                "num_layout_objects": layout_summary.get("num_layout_objects"),
                "num_renderable_objects": layout_summary.get("num_renderable_objects"),
                "geometry_available_rate": geometry_rate,
                "omitted_asset_count": compact_scene.get("omitted_asset_count", 0),
            },
            "physical_flags": _compact_flags(physical_flags, budget),
            "schema_flags": _compact_flags(sanity_flags, budget),
            "render_flags": _compact_flags(view_flags, budget),
            "geometry_missing": _compact_flags(_flags_of_type(render_skipped_objects, "geometry_missing_asset"), budget),
            "fallback_confidence": _fallback_confidence_summary(physical_flags),
            "render_available": bool(image_manifest),
            "render_evidence_used": bool(render_evidence_used),
        },
        "scene_summary": scene_summary,
        "layout_summary": layout_summary,
        "evidence_manifest": build_evidence_manifest(evidence_selection, image_manifest),
        "flag_summary": layout_summary.get("flag_summary", {}),
        "rubric": {
            "judge_geometry": True,
            "judge_object_relations": True,
            "judge_room_coherence": True,
            "physical_flags_are_evidence_not_rules": True,
            "relation_aware_overlap_guidance": (
                "Some geometry-proxy overlaps may be valid depending on object relation: chair under table, "
                "object on table, object inside cabinet, or wall-mounted object. Other intersections "
                "are likely invalid: bed through table or large furniture penetrating unrelated large furniture. "
                "Consider category, relation, support, containment, and overall room coherence."
            ),
            **TEMPORARY_RUBRIC,
        },
        "score_scale": {
            "0": "Unusable, invalid, or impossible to judge because evidence is insufficient.",
            "1": "Severe asset placement or task-completeness problems.",
            "2": "Partially plausible with multiple important issues.",
            "3": "Mostly coherent with minor asset placement issues.",
            "4": "Coherent, plausible, task-complete scene asset placement.",
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


def _compact_room(room: object, text_budget: dict) -> dict:
    if not isinstance(room, dict):
        return {}
    return _room_proxy_summary(room, text_budget)


def _compact_scene_asset(asset: dict, text_budget: dict) -> dict:
    compact = {
        "asset_id": asset.get("asset_id"),
        "category": asset.get("category"),
    }
    if asset.get("object_id") and asset.get("object_id") != asset.get("asset_id"):
        compact["object_id"] = asset.get("object_id")
    placement = asset.get("placement")
    if isinstance(placement, dict):
        compact["placement"] = {
            key: _round_value(placement.get(key), int(text_budget.get("numeric_precision", 2)))
            for key in ["position", "yaw_degrees", "region_id", "support_parent"]
            if key in placement
        }
    if isinstance(asset.get("dimensions"), list):
        compact["dimensions"] = _round_value(asset["dimensions"], int(text_budget.get("numeric_precision", 2)))
    geometry = asset.get("geometry")
    if "placement" not in compact and isinstance(geometry, dict):
        compact["geometry"] = {
            key: _round_value(geometry.get(key), int(text_budget.get("numeric_precision", 2)))
            for key in ["position", "center", "dimensions", "size", "yaw_degrees", "yaw"]
            if key in geometry
        }
    legend_bbox = asset.get("bbox")
    if isinstance(legend_bbox, dict) and "placement" not in compact and "dimensions" not in compact:
        compact["legend_geometry"] = {
            key: _round_value(legend_bbox.get(key), int(text_budget.get("numeric_precision", 2)))
            for key in ["center", "size", "yaw"]
            if key in legend_bbox
        }
    for key in ["support_parent", "support_surface", "parent_id", "region_id"]:
        if asset.get(key) is not None:
            compact[key] = asset.get(key)
    if isinstance(asset.get("asset_ref"), dict):
        compact["asset_ref"] = _compact_asset_ref(asset["asset_ref"])
    if isinstance(asset.get("metadata"), dict):
        metadata = _compact_metadata(asset["metadata"])
        if metadata:
            compact["metadata"] = metadata
    return compact


def _compact_scene_ref(scene_ref: dict) -> dict:
    compact = {
        key: _truncate_string(str(scene_ref[key]), 240) if isinstance(scene_ref.get(key), str) else scene_ref.get(key)
        for key in ["source", "collection", "scene_id", "repo_path", "scene_json_path", "scene_type", "asset_count", "scene_height"]
        if key in scene_ref and scene_ref.get(key) is not None
    }
    metadata = scene_ref.get("metadata")
    if isinstance(metadata, dict) and metadata:
        compact["metadata_keys"] = sorted(str(key) for key in metadata.keys())[:12]
        compact["metadata_key_count"] = len(metadata)
    return compact


def _compact_asset_ref(asset_ref: dict) -> dict:
    compact = {
        key: _truncate_string(str(asset_ref[key]), 240) if isinstance(asset_ref.get(key), str) else asset_ref.get(key)
        for key in ["source", "collection", "asset_id", "template_id", "repo_path", "mesh_path", "pointcloud_path", "metadata_path", "mesh_uri", "category", "caption_en", "license", "model_id", "asset_type"]
        if key in asset_ref and asset_ref.get(key) is not None
    }
    metadata = asset_ref.get("metadata")
    if isinstance(metadata, dict) and metadata:
        compact["metadata_keys"] = sorted(str(key) for key in metadata.keys())[:12]
        compact["metadata_key_count"] = len(metadata)
    for key, value in asset_ref.items():
        if key in compact or key == "metadata":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = _truncate_string(value, 240) if isinstance(value, str) else value
        elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value[:3]):
            compact[key] = _round_value(value, 3)
        if len(compact) >= 12:
            break
    return compact


def _compact_metadata(metadata: dict) -> dict:
    primitive_fields = {}
    for key, value in metadata.items():
        if key == "source_layout_object":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            primitive_fields[str(key)] = _truncate_string(value, 180) if isinstance(value, str) else value
        if len(primitive_fields) >= 6:
            break
    result: dict[str, Any] = {
        "keys": sorted(str(key) for key in metadata if key != "source_layout_object")[:12],
        "key_count": len([key for key in metadata if key != "source_layout_object"]),
    }
    if primitive_fields:
        result["primitive_fields"] = primitive_fields
    return result if result["key_count"] or primitive_fields else {}


def _asset_has_geometry(asset: dict) -> bool:
    placement = asset.get("placement")
    dimensions = asset.get("dimensions")
    if isinstance(placement, dict) and isinstance(placement.get("position"), list) and isinstance(dimensions, list):
        return len(placement["position"]) >= 3 and len(dimensions) >= 3
    geometry = asset.get("geometry")
    if isinstance(geometry, dict):
        position = geometry.get("position") or geometry.get("center")
        size = geometry.get("dimensions") or geometry.get("size")
        return isinstance(position, list) and isinstance(size, list) and len(position) >= 3 and len(size) >= 3
    legend_bbox = asset.get("bbox")
    return isinstance(legend_bbox, dict) and all(key in legend_bbox for key in ["center", "size", "yaw"])


def _geometry_available_rate(compact_scene: dict) -> float | None:
    asset_count = compact_scene.get("asset_count")
    geometry_count = compact_scene.get("geometry_asset_count")
    if not isinstance(asset_count, int) or asset_count <= 0 or not isinstance(geometry_count, int):
        return None
    return float(geometry_count) / float(asset_count)


def _compact_flags(flags: list[dict], text_budget: dict) -> list[dict]:
    limit = text_budget["max_flag_examples_per_type"]
    compact = []
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        item = {
            key: flag[key]
            for key in [
                "type",
                "code",
                "severity",
                "confidence",
                "source_kind",
                "source_confidence",
                "blocking",
                "objects",
                "object_ids",
                "object_id",
                "asset_id",
                "category",
                "message",
                "reason",
            ]
            if key in flag
        }
        compact.append(item)
        if len(compact) >= limit:
            break
    return compact


def _flags_of_type(flags: list[dict], flag_type: str) -> list[dict]:
    return [
        flag
        for flag in flags
        if isinstance(flag, dict) and (flag.get("type") == flag_type or flag.get("code") == flag_type)
    ]


def _fallback_confidence_summary(flags: list[dict]) -> dict:
    confidence = Counter(str(flag.get("confidence") or "unknown") for flag in flags if isinstance(flag, dict))
    source_kind = Counter(str(flag.get("source_kind") or "unknown") for flag in flags if isinstance(flag, dict))
    return {
        "confidence_counts": dict(sorted(confidence.items())),
        "source_kind_counts": dict(sorted(source_kind.items())),
        "has_low_confidence_or_fallback": any(
            str(flag.get("confidence") or "").lower() == "low"
            or str(flag.get("source_kind") or "").lower() in {"fallback_default", "object_position_extent_fallback", "unknown"}
            for flag in flags
            if isinstance(flag, dict)
        ),
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
    result["coordinate_convention"] = {"unit": "meters", "asset_geometry": "placement+dimensions", "truncated": True}
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
    if len(_json_text(result)) <= cap:
        return result
    scene = result.get("scene") if isinstance(result.get("scene"), dict) else {}
    shown_assets = scene.get("assets") if isinstance(scene.get("assets"), list) else []
    result["scene"] = {
        "scene_id": scene.get("scene_id"),
        "unit": scene.get("unit"),
        "asset_count": scene.get("asset_count"),
        "geometry_asset_count": scene.get("geometry_asset_count"),
        "non_geometric_asset_count": scene.get("non_geometric_asset_count"),
        "assets": [{"truncated": True, "shown": 0, "total": len(shown_assets), "reason": "total prompt text budget"}],
        "truncated": True,
    }
    structured = result.get("structured_evidence") if isinstance(result.get("structured_evidence"), dict) else {}
    result["structured_evidence"] = {
        "object_completeness": structured.get("object_completeness", {}),
        "physical_flags": structured.get("physical_flags", [])[:1] if isinstance(structured.get("physical_flags"), list) else [],
        "geometry_missing": structured.get("geometry_missing", [])[:1] if isinstance(structured.get("geometry_missing"), list) else [],
        "render_available": structured.get("render_available"),
        "render_evidence_used": structured.get("render_evidence_used"),
        "truncated": True,
    }
    result["total_budget_truncation"] = {"truncated": True, "cap_chars": cap, "strategy": "final compact payload plus compact scene"}
    if len(_json_text(result)) <= cap:
        return result
    result["rubric"] = {"target": TEMPORARY_RUBRIC["target"], "physical_flags_are_evidence_not_rules": True, "truncated": True}
    result["output_instruction"] = "Return one JSON object."
    result["total_budget_truncation"] = {"truncated": True, "cap_chars": cap, "strategy": "minimal final payload"}
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
    return "rendered geometry evidence view"


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
