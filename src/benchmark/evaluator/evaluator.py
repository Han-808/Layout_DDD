from __future__ import annotations

from benchmark.evaluator.physical_check import check_physical_validity
from benchmark.evaluator.schema_check import check_layout_schema
from benchmark.evaluator.spatial_check import check_spatial_relations
from benchmark.evaluator.geometry import sorted_unique


class LayoutEvaluator:
    def __init__(self, config: dict | None = None, layout_schema: dict | None = None) -> None:
        self.config = config or {}
        self.layout_schema = layout_schema or {}

    def evaluate(self, bm_instance: dict, layout: dict | str, iteration: int = 0) -> dict:
        schema_result = check_layout_schema(layout, self.layout_schema)
        schema_valid = schema_result.valid
        task_id = bm_instance.get("task_id", "unknown_task")

        physical_valid = False
        spatial_valid = False
        physical_failures: list[dict] = []
        spatial_failures: list[dict] = []
        repair_targets: list[str] = []

        if schema_result.layout is not None and schema_valid:
            physical_valid, physical_failures, physical_targets = check_physical_validity(
                schema_result.layout,
                bm_instance,
                self.config,
            )
            spatial_valid, spatial_failures, spatial_targets = check_spatial_relations(
                schema_result.layout,
                bm_instance,
                self.config,
            )
            repair_targets.extend(physical_targets)
            repair_targets.extend(spatial_targets)
        else:
            for failure in schema_result.failures:
                repair_targets.extend(failure.get("objects", []))

        overall_valid = schema_valid and physical_valid and spatial_valid
        metrics = {
            "schema_validity": int(bool(schema_valid)),
            "physical_validity": int(bool(physical_valid)),
            "spatial_relation_validity": int(bool(spatial_valid)),
        }
        return {
            "task_id": task_id,
            "iteration": int(iteration),
            "overall_valid": bool(overall_valid),
            "metrics": metrics,
            "summary": {
                "schema_valid": bool(schema_valid),
                "physical_valid": bool(physical_valid),
                "spatial_relation_valid": bool(spatial_valid),
                "num_schema_errors": len(schema_result.failures),
                "num_physical_errors": len(physical_failures),
                "num_spatial_relation_errors": len(spatial_failures),
            },
            "schema_failures": schema_result.failures,
            "physical_failures": physical_failures,
            "spatial_relation_failures": spatial_failures,
            "physical_diagnostics": _physical_diagnostics(schema_result.layout, physical_failures),
            "repair_targets": sorted_unique(repair_targets),
        }


def _physical_diagnostics(layout: dict | None, physical_failures: list[dict]) -> dict:
    object_ids = []
    if isinstance(layout, dict):
        object_ids = [
            obj.get("object_id")
            for obj in layout.get("objects", [])
            if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
        ]
    collision_counts = {object_id: 0 for object_id in object_ids}
    for failure in physical_failures:
        if failure.get("type") != "collision":
            continue
        for object_id in failure.get("objects", []):
            collision_counts[object_id] = collision_counts.get(object_id, 0) + 1
    return {
        "scene_has_collision": any(count > 0 for count in collision_counts.values()),
        "object_collision_counts": collision_counts,
    }
