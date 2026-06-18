from __future__ import annotations


def export_viewer_scene(
    bm_instance: dict,
    layout: dict,
    evaluation_report: dict,
    history: list[dict] | None = None,
) -> dict:
    room = bm_instance.get("room") or {}
    scene = _scene_payload(bm_instance, layout, evaluation_report, history or [])
    scene.update(
        {
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
            "iterations": _iteration_scenes(bm_instance, history or []),
        }
    )
    return scene


def _scene_payload(
    bm_instance: dict,
    layout: dict,
    evaluation_report: dict,
    history: list[dict],
) -> dict:
    invalid_objects, violations = _violations(evaluation_report)
    room = bm_instance.get("room") or {}
    return {
        "task_id": bm_instance.get("task_id") or bm_instance.get("case_id") or layout.get("scene_id"),
        "iteration": evaluation_report.get("iteration", 0),
        "overall_valid": evaluation_report.get("overall_valid"),
        "summary": evaluation_report.get("summary", {}),
        "room": {
            "floor_polygon": room.get("floor_polygon") or room.get("boundary") or [],
            "wall_height": room.get("wall_height"),
            "floor_z": room.get("floor_z", 0.0),
            "unit": room.get("unit", layout.get("unit", "meter")),
        },
        "objects": [
            {
                "object_id": obj.get("object_id"),
                "label": obj.get("category"),
                "category": obj.get("category"),
                "center": obj.get("center"),
                "three_center": _project_to_three(obj.get("center", [0, 0, 0])),
                "size": obj.get("size"),
                "yaw": obj.get("yaw", 0),
                "validity_status": "invalid" if obj.get("object_id") in invalid_objects else "valid",
            }
            for obj in layout.get("objects", [])
            if isinstance(obj, dict)
        ],
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
        "violations": violations,
        "history": _history_summary(history),
    }


def _iteration_scenes(bm_instance: dict, history: list[dict]) -> list[dict]:
    scenes = []
    for item in history:
        layout = item.get("layout")
        evaluation = item.get("evaluation")
        if not isinstance(layout, dict) or not isinstance(evaluation, dict):
            continue
        scene = _scene_payload(bm_instance, layout, evaluation, [])
        scene["layout_path"] = item.get("layout_path", "")
        scene["evaluation_path"] = item.get("evaluation_path", "")
        scene["feedback_path"] = item.get("feedback_path", "")
        scenes.append(scene)
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


def _project_to_three(center: list[float]) -> list[float]:
    if not isinstance(center, list) or len(center) != 3:
        return [0.0, 0.0, 0.0]
    return [center[0], center[2], center[1]]
