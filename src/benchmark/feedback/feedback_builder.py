from __future__ import annotations


def build_feedback(evaluation_report: dict, current_layout: dict, bm_instance: dict) -> dict:
    repair_targets = sorted(evaluation_report.get("repair_targets", []))
    repair_target_set = set(repair_targets)
    object_ids = sorted(
        obj.get("object_id")
        for obj in current_layout.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    )
    locked_objects = [object_id for object_id in object_ids if object_id not in repair_target_set]

    violations = []
    violations.extend(_failures_to_violations("schema", evaluation_report.get("schema_failures", [])))
    violations.extend(_failures_to_violations("physical", evaluation_report.get("physical_failures", [])))
    violations.extend(_failures_to_violations("spatial_relation", evaluation_report.get("spatial_relation_failures", [])))
    room_consistency = evaluation_report.get("room_consistency", {})
    if room_consistency.get("short_reason") and room_consistency.get("score", 4) <= 2:
        violations.append(
            {
                "category": "room_consistency",
                "type": "vlm_room_judge",
                "message": room_consistency["short_reason"],
                "objects": [],
            }
        )

    return {
        "task_id": evaluation_report.get("task_id", bm_instance.get("task_id", "unknown_task")),
        "iteration": int(evaluation_report.get("iteration", 0)),
        "repair_targets": repair_targets,
        "locked_objects": locked_objects,
        "violations": violations,
        "debug_evidence": evaluation_report.get("debug_evidence", {}),
        "room_consistency_reason": room_consistency.get("short_reason", ""),
        "instruction": "Fix only the listed violations. Preserve valid objects. Return corrected layout JSON only.",
    }


def _failures_to_violations(category: str, failures: list[dict]) -> list[dict]:
    violations = []
    for failure in failures:
        violation = {
            "category": category,
            "type": failure.get("type", "unknown"),
            "message": failure.get("message", ""),
        }
        if "objects" in failure:
            violation["objects"] = list(failure["objects"])
        if "category" in failure and category == "spatial_relation":
            violation["target_category"] = failure["category"]
        if "hard" in failure:
            violation["hard"] = bool(failure["hard"])
        violations.append(violation)
    return violations
