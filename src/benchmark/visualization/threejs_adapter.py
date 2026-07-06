from __future__ import annotations


def export_viewer_scene(
    bm_instance: dict,
    layout: dict,
    evaluation_report: dict,
    history: list[dict] | None = None,
    benchmark_config: dict | None = None,
) -> dict:
    room = bm_instance.get("room") or {}
    viewer_options = _viewer_options(benchmark_config)
    scene = _scene_payload(bm_instance, layout, evaluation_report, history or [], viewer_options)
    scene.update(
        {
            "scene": {
                "task_id": bm_instance.get("task_id") or bm_instance.get("case_id") or layout.get("scene_id"),
                "unit": room.get("unit", layout.get("unit", "meter")),
                "coordinate_system": layout.get("coordinate_system", {}),
            },
            "metadata": {
                "task_id": bm_instance.get("task_id") or bm_instance.get("case_id") or layout.get("scene_id"),
                "iteration": evaluation_report.get("iteration", 0),
                "overall_valid": evaluation_report.get("overall_valid"),
                "summary": evaluation_report.get("summary", {}),
            },
            "coordinate_conversion": {
                "project_xyz_to_threejs": "[x, y, z] -> [x, z, y]",
                "project_z_is_height": True,
                "yaw_mapping": "project yaw rotates around z/up; Three.js yaw rotates around its vertical y axis",
            },
            "iterations": _iteration_scenes(bm_instance, history or [], viewer_options),
            "workflow_steps": [],
            "artifacts": [],
            "judge_evidence": _judge_evidence(evaluation_report),
            "viewer_options": viewer_options,
        }
    )
    return scene


def _scene_payload(
    bm_instance: dict,
    layout: dict,
    evaluation_report: dict,
    history: list[dict],
    viewer_options: dict | None = None,
) -> dict:
    invalid_objects, violations = _violations(evaluation_report)
    room = bm_instance.get("room") or {}
    group_evidence = _group_evidence(evaluation_report)
    group_by_object = _group_lookup(group_evidence)
    return {
        "task_id": bm_instance.get("task_id") or bm_instance.get("case_id") or layout.get("scene_id"),
        "iteration": evaluation_report.get("iteration", 0),
        "label": _iteration_label(evaluation_report.get("iteration", 0)),
        "overall_valid": evaluation_report.get("overall_valid"),
        "summary": evaluation_report.get("summary", {}),
        "room": {
            "floor_polygon": room.get("floor_polygon") or room.get("boundary") or [],
            "floor_plan": _floor_plan(room),
            "wall_height": room.get("wall_height"),
            "floor_z": room.get("floor_z", 0.0),
            "unit": room.get("unit", layout.get("unit", "meter")),
            "boundary_role": room.get("boundary_role"),
            "boundary_source": room.get("boundary_source"),
        },
        "objects": [
            {
                "object_id": obj.get("object_id"),
                "canonical_object_id": obj.get("canonical_object_id") or obj.get("object_id"),
                "model_object_id": obj.get("model_object_id"),
                "model_category": obj.get("model_category"),
                "label": _object_label(obj),
                "category": obj.get("category"),
                "center": obj.get("center"),
                "three_center": _project_to_three(obj.get("center", [0, 0, 0])),
                "size": obj.get("size"),
                "yaw": obj.get("yaw", 0),
                "validity_status": "invalid" if obj.get("object_id") in invalid_objects else "valid",
                **_object_group_fields(obj.get("object_id"), group_by_object),
            }
            for obj in layout.get("objects", [])
            if isinstance(obj, dict)
        ],
        "groups": group_evidence,
        "relations": [
            {
                "type": rel.get("type"),
                "source": rel.get("source"),
                "target": rel.get("target"),
                "hard": bool(rel.get("hard", False)),
            }
            for rel in layout.get("relations", [])
            if isinstance(rel, dict)
        ],
        "global_views": evaluation_report.get("room_consistency", {}).get("view_artifacts", []),
        "scene_summary": evaluation_report.get("scene_summary", {}),
        "layout_summary": evaluation_report.get("layout_summary", {}),
        "judgement_status": evaluation_report.get("judgement_status"),
        "insufficient_evidence": bool(evaluation_report.get("insufficient_evidence", False)),
        "group_evidence": group_evidence,
        "skipped_objects": _skipped_objects(evaluation_report),
        "vlm_judge_artifacts": evaluation_report.get("vlm_judge_artifacts", {}),
        "judge_input_manifest": evaluation_report.get("debug_evidence", {}).get("judge_input_manifest", {}),
        "judge_evidence": _judge_evidence(evaluation_report),
        "evaluation_policy": evaluation_report.get("evaluation_policy", {}),
        "layout_normalization": evaluation_report.get("debug_evidence", {}).get("layout_normalization", {}),
        "object_set_normalization": evaluation_report.get("debug_evidence", {}).get("layout_normalization", {}).get("object_set_normalization", {}),
        "runtime_evidence_config": evaluation_report.get("debug_evidence", {}).get("runtime_evidence_config", {}),
        "resolved_grouping_config": evaluation_report.get("debug_evidence", {}).get("resolved_grouping_config", {}),
        "omitted_grouping_edges": evaluation_report.get("debug_evidence", {}).get("omitted_grouping_edges", []),
        "cross_group_relations": evaluation_report.get("debug_evidence", {}).get("cross_group_relations", []),
        "violations": violations,
        "history": _history_summary(history),
        "viewer_options": viewer_options or DEFAULT_VIEWER_OPTIONS,
    }


def _object_label(obj: dict) -> str:
    alias = obj.get("model_object_id")
    category = obj.get("model_category") or obj.get("category")
    if alias and category:
        return f"{alias} {category}"
    return str(category or obj.get("object_id") or "object")


def _iteration_scenes(bm_instance: dict, history: list[dict], viewer_options: dict) -> list[dict]:
    scenes = []
    previous_layout = None
    initial_layout = None
    for item in history:
        layout = item.get("layout")
        evaluation = item.get("evaluation")
        if not isinstance(layout, dict) or not isinstance(evaluation, dict):
            continue
        scene = _scene_payload(bm_instance, layout, evaluation, [], viewer_options)
        scene["layout_path"] = item.get("layout_path", "")
        scene["evaluation_path"] = item.get("evaluation_path", "")
        scene["evaluation_report_path"] = item.get("evaluation_path", "")
        scene["feedback_path"] = item.get("feedback_path", "")
        scene["label"] = "initial" if int(item.get("iteration", 0)) == 0 else f"repair_{item.get('iteration')}"
        if initial_layout is None:
            initial_layout = layout
        scene["diff_from_previous"] = _layout_diff(previous_layout, layout, viewer_options) if previous_layout else _empty_diff()
        scene["diff_from_initial"] = _layout_diff(initial_layout, layout, viewer_options) if initial_layout and layout is not initial_layout else _empty_diff()
        scenes.append(scene)
        previous_layout = layout
    return scenes


def _history_summary(history: list[dict]) -> list[dict]:
    summary = []
    for item in history:
        compact = {
            "iteration": item.get("iteration"),
            "layout_path": item.get("layout_path", ""),
            "evaluation_path": item.get("evaluation_path", ""),
            "feedback_path": item.get("feedback_path", ""),
            "schema_valid": item.get("schema_valid"),
            "physical_valid": item.get("physical_valid"),
            "spatial_relation_valid": item.get("spatial_relation_valid"),
            "overall_valid": item.get("overall_valid"),
            "num_schema_errors": item.get("num_schema_errors"),
            "num_physical_errors": item.get("num_physical_errors"),
            "num_spatial_relation_errors": item.get("num_spatial_relation_errors"),
        }
        summary.append(compact)
    return summary


def _violations(evaluation_report: dict) -> tuple[set[str], list[dict]]:
    invalid_objects = set()
    violations = []
    for category, key in [
        ("schema", "schema_failures"),
        ("physical", "physical_failures"),
        ("spatial_relation", "spatial_relation_failures"),
    ]:
        for failure in evaluation_report.get(key, []):
            objects = list(failure.get("objects", []))
            invalid_objects.update(objects)
            violations.append(
                {
                    "category": category,
                    "type": failure.get("type", "unknown"),
                    "objects": objects,
                    "message": failure.get("message", ""),
                }
            )
    return invalid_objects, violations


def _group_evidence(evaluation_report: dict) -> list[dict]:
    debug = evaluation_report.get("debug_evidence", {})
    groups = debug.get("object_groups", []) if isinstance(debug, dict) else []
    view_records = debug.get("group_view_artifacts", []) if isinstance(debug, dict) else []
    view_by_group = {
        item.get("group_id"): item
        for item in view_records
        if isinstance(item, dict)
    }
    relation_results = evaluation_report.get("specified_relations", {}).get("results", [])
    skipped = _skipped_objects(evaluation_report)
    evidence = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        object_ids = list(group.get("object_ids", [])) if isinstance(group.get("object_ids"), list) else []
        object_set = set(object_ids)
        views = view_by_group.get(group.get("group_id"), {})
        record = {
            "group_id": group.get("group_id"),
            "group_index": index,
            "group_color_key": group.get("group_id") or f"group_{index:03d}",
            "object_ids": object_ids,
            "num_objects": group.get("num_objects"),
            "group_footprint_diameter_m": group.get("group_footprint_diameter_m"),
            "group_label": group.get("group_label") or group.get("group_id"),
            "formation_edges": group.get("formation_edges", []),
            "edge_reasons": group.get("edge_reasons", []),
            "views": views.get("views", {}),
            "diagnostics": views.get("diagnostics", {}),
            "view_flags": views.get("view_flags", []),
            "relations": [
                relation
                for relation in relation_results
                if relation.get("subject") in object_set or relation.get("object") in object_set
            ],
            "skipped_objects": [
                item
                for item in skipped
                if item.get("object_id") in object_set
            ],
        }
        if "sent_to_judge" in group:
            record["sent_to_judge"] = group.get("sent_to_judge")
            record["selection_score"] = group.get("selection_score")
            record["selection_reasons"] = group.get("selection_reasons", [])
        evidence.append(record)
    return evidence


DEFAULT_VIEWER_OPTIONS = {
    "default_focus": "visual",
    "json_preview": {"lazy_load": True, "hidden_by_default": True, "truncate_chars": 20000},
    "group_coloring": {"enabled_by_default": False, "toggle_available": True},
    "overlays": {
        "show_floor_grid": True,
        "show_axes": True,
        "show_room_proxy": False,
        "show_floor_plan_regions": True,
        "show_relation_edges_by_default": False,
        "allow_relation_edge_toggle": True,
        "show_judge_status_markers": True,
    },
    "replay": {"enabled": True, "autoplay_default": False, "step_duration_ms": 1200},
    "diff": {
        "enabled": True,
        "position_tolerance_m": 0.01,
        "size_tolerance_m": 0.01,
        "yaw_tolerance_degrees": 1.0,
    },
}


def _viewer_options(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    viewer = config.get("viewer") if isinstance(config.get("viewer"), dict) else {}
    return _deep_merge(DEFAULT_VIEWER_OPTIONS, viewer)


def _deep_merge(base: dict, patch: dict) -> dict:
    merged = {key: _copy_value(value) for key, value in base.items()}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _copy_value(value)
    return merged


def _copy_value(value):
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _group_lookup(group_evidence: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for group in group_evidence:
        for object_id in group.get("object_ids", []):
            if isinstance(object_id, str):
                lookup[object_id] = group
    return lookup


def _object_group_fields(object_id: object, group_by_object: dict[str, dict]) -> dict:
    if not isinstance(object_id, str):
        return {}
    group = group_by_object.get(object_id)
    if not group:
        return {}
    return {
        "group_id": group.get("group_id"),
        "group_index": group.get("group_index"),
        "group_color_key": group.get("group_color_key"),
        "sent_to_judge": group.get("sent_to_judge") if "sent_to_judge" in group else None,
    }


def _floor_plan(room: dict) -> dict:
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    return {
        "source": floor_plan.get("source") or room.get("room_layout_source") or room.get("boundary_source"),
        "primary_representation": floor_plan.get("primary_representation") or ("regions" if regions else "floor_polygon"),
        "aggregate_boundary": floor_plan.get("aggregate_boundary") or room.get("floor_polygon") or room.get("boundary") or [],
        "aggregate_boundary_role": floor_plan.get("aggregate_boundary_role") or room.get("boundary_role"),
        "regions": [region for region in regions if isinstance(region, dict)],
        "region_count": len([region for region in regions if isinstance(region, dict)]),
    }


def _judge_evidence(evaluation_report: dict) -> dict:
    debug = evaluation_report.get("debug_evidence", {}) if isinstance(evaluation_report, dict) else {}
    return {
        "manifest": debug.get("judge_input_manifest", {}) if isinstance(debug, dict) else {},
        "artifacts": evaluation_report.get("vlm_judge_artifacts", {}) if isinstance(evaluation_report, dict) else {},
        "global_views": evaluation_report.get("room_consistency", {}).get("view_artifacts", []),
        "group_views": debug.get("group_view_artifacts", []) if isinstance(debug, dict) else [],
    }


def _iteration_label(iteration: object) -> str:
    try:
        value = int(iteration)
    except (TypeError, ValueError):
        return "iteration"
    return "initial" if value == 0 else f"repair_{value}"


def _layout_diff(previous: dict | None, current: dict, viewer_options: dict) -> dict:
    if not previous:
        return _empty_diff()
    tolerances = (viewer_options.get("diff") or {}) if isinstance(viewer_options, dict) else {}
    position_tol = float(tolerances.get("position_tolerance_m", 0.01))
    size_tol = float(tolerances.get("size_tolerance_m", 0.01))
    yaw_tol = float(tolerances.get("yaw_tolerance_degrees", 1.0))
    prev_objects = _objects_by_id(previous)
    curr_objects = _objects_by_id(current)
    added = sorted(set(curr_objects) - set(prev_objects))
    removed = sorted(set(prev_objects) - set(curr_objects))
    changed = []
    for object_id in sorted(set(prev_objects) & set(curr_objects)):
        before = prev_objects[object_id]
        after = curr_objects[object_id]
        fields = []
        if _vector_changed(before.get("center"), after.get("center"), position_tol):
            fields.append("center")
        if _vector_changed(before.get("size"), after.get("size"), size_tol):
            fields.append("size")
        if abs(float(before.get("yaw", 0) or 0) - float(after.get("yaw", 0) or 0)) > yaw_tol:
            fields.append("yaw")
        if before.get("support_parent") != after.get("support_parent"):
            fields.append("support_parent")
        if fields:
            changed.append({"object_id": object_id, "fields": fields})
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "changed_object_ids": added + removed + [item["object_id"] for item in changed],
    }


def _empty_diff() -> dict:
    return {"added": [], "removed": [], "changed": [], "changed_object_ids": []}


def _objects_by_id(layout: dict) -> dict[str, dict]:
    return {
        obj.get("object_id"): obj
        for obj in layout.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    }


def _vector_changed(left: object, right: object, tolerance: float) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return left != right
    try:
        return any(abs(float(a) - float(b)) > tolerance for a, b in zip(left, right))
    except (TypeError, ValueError):
        return left != right


def _skipped_objects(evaluation_report: dict) -> list[dict]:
    debug = evaluation_report.get("debug_evidence", {})
    skipped = debug.get("render_skipped_objects", []) if isinstance(debug, dict) else []
    records = []
    for item in skipped:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "object_id": item.get("object_id") or (item.get("objects") or [None])[0],
                "category": item.get("category"),
                "reason": item.get("reason") or item.get("message", ""),
                "raw_object": item.get("raw_object"),
                "object_index": item.get("object_index"),
            }
        )
    return records


def _project_to_three(center: list[float]) -> list[float]:
    if not isinstance(center, list) or len(center) != 3:
        return [0.0, 0.0, 0.0]
    return [center[0], center[2], center[1]]
