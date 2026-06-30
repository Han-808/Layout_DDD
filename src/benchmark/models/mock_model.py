from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from benchmark.models.base_model import BaseLayoutModel


DEFAULT_DIMS: dict[str, list[float]] = {
    "bed": [2.0, 1.4, 0.6],
    "desk": [1.2, 0.6, 0.75],
    "chair": [0.6, 0.6, 0.9],
    "wardrobe": [1.0, 0.6, 2.0],
    "nightstand": [0.45, 0.45, 0.6],
    "sofa": [2.0, 0.9, 0.8],
    "coffee_table": [1.0, 0.6, 0.45],
    "tv_stand": [1.4, 0.45, 0.6],
    "floor_lamp": [0.35, 0.35, 1.7],
    "bookshelf": [1.0, 0.35, 1.8],
}


@dataclass
class MockModel(BaseLayoutModel):
    """Deterministic model adapter for tests and no-API benchmark runs.

    behavior="valid" returns a conservative non-colliding layout.
    behavior="colliding_then_repair" deliberately collides the first two objects
    during generation and returns a valid layout during repair.
    """

    behavior: str = "valid"

    def __init__(self, name: str = "mock", behavior: str = "valid", judge_evidence_budgeting: bool = False) -> None:
        super().__init__(name=name)
        self.behavior = behavior
        self.judge_evidence_budgeting = judge_evidence_budgeting

    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        layout = self._make_layout(bm_instance)
        if self.behavior == "colliding_then_repair" and len(layout["objects"]) >= 2:
            layout["objects"][1]["center"] = list(layout["objects"][0]["center"])
        return layout

    def repair_layout(
        self,
        bm_instance: dict,
        current_layout: dict,
        feedback: dict,
        layout_schema: dict,
    ) -> dict:
        if current_layout.get("objects"):
            categories = [obj.get("category", f"object_{i}") for i, obj in enumerate(current_layout["objects"])]
            object_ids = [obj.get("object_id", f"{cat}_{i + 1}") for i, (cat, obj) in enumerate(zip(categories, current_layout["objects"]))]
            return self._make_layout(bm_instance, categories=categories, object_ids=object_ids)
        return self._make_layout(bm_instance)

    def _make_layout(
        self,
        bm_instance: dict,
        categories: Iterable[str] | None = None,
        object_ids: Iterable[str] | None = None,
    ) -> dict:
        scene_id = bm_instance.get("task_id") or bm_instance.get("case_id") or "mock_scene"
        object_specs = [item for item in bm_instance.get("objects", []) if isinstance(item, dict)]
        categories = list(
            categories
            or bm_instance.get("required_objects")
            or [item.get("category", "box") for item in object_specs]
            or ["box"]
        )
        object_ids = list(
            object_ids
            or [item.get("id") for item in object_specs if item.get("id")]
            or [f"{category}_{idx + 1}" for idx, category in enumerate(categories)]
        )
        room = bm_instance.get("room") or {
            "floor_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
            "floor_z": 0.0,
            "wall_height": 2.8,
        }
        floor_polygon = room.get("floor_polygon") or room.get("boundary") or [[0, 0], [5, 0], [5, 4], [0, 4]]
        xs = [point[0] for point in floor_polygon]
        ys = [point[1] for point in floor_polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        objects = []
        for idx, (category, object_id) in enumerate(zip(categories, object_ids)):
            size = list(DEFAULT_DIMS.get(category, [0.8, 0.8, 0.8]))
            center = self._deterministic_center(category, idx, size, min_x, min_y, max_x, max_y)
            yaw = self._deterministic_yaw(category)
            objects.append(
                {
                    "object_id": object_id,
                    "category": category,
                    "center": center,
                    "size": size,
                    "yaw": yaw,
                    "support_parent": "floor",
                    "region_id": self._region_for(category),
                }
            )

        relations = self._relations_for(bm_instance, objects)
        floor_object_ids = [obj["object_id"] for obj in objects if obj.get("support_parent") == "floor"]
        regions = sorted({obj["region_id"] for obj in objects if obj.get("region_id")})
        return {
            "scene_id": scene_id,
            "unit": "meter",
            "coordinate_system": {
                "origin": "front-left floor corner",
                "x_axis": "room width",
                "y_axis": "room depth",
                "z_axis": "height",
                "rotation_unit": "degree",
            },
            "objects": objects,
            "relations": relations,
            "hierarchy": {
                "regions": regions,
                "floor_objects": floor_object_ids,
                "supported_objects": [],
            },
        }

    @staticmethod
    def _deterministic_center(
        category: str,
        idx: int,
        size: list[float],
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> list[float]:
        width, depth, height = size
        presets = {
            "bed": [min_x + width / 2 + 0.10, min_y + depth / 2 + 0.05, height / 2],
            "desk": [max_x - width / 2 - 0.25, min_y + depth / 2 + 0.30, height / 2],
            "chair": [max_x - width / 2 - 0.65, min_y + depth / 2 + 1.10, height / 2],
            "wardrobe": [max_x - width / 2 - 0.15, max_y - depth / 2 - 0.15, height / 2],
            "nightstand": [min_x + 2.55, min_y + depth / 2 + 0.10, height / 2],
            "sofa": [min_x + width / 2 + 0.20, min_y + depth / 2 + 1.30, height / 2],
            "coffee_table": [min_x + width / 2 + 1.65, min_y + depth / 2 + 1.45, height / 2],
            "tv_stand": [max_x - width / 2 - 0.15, min_y + depth / 2 + 1.35, height / 2],
            "floor_lamp": [min_x + 0.45, max_y - 0.45, height / 2],
            "bookshelf": [max_x - width / 2 - 0.10, max_y - depth / 2 - 0.10, height / 2],
        }
        if category in presets:
            return [round(v, 4) for v in presets[category]]

        columns = 3
        usable_w = max(width, (max_x - min_x) - width)
        usable_d = max(depth, (max_y - min_y) - depth)
        col = idx % columns
        row = idx // columns
        x = min_x + width / 2 + 0.2 + (usable_w / max(1, columns - 1)) * col
        y = min_y + depth / 2 + 0.2 + min(usable_d, 1.2 * row)
        return [round(min(max_x - width / 2 - 0.05, x), 4), round(min(max_y - depth / 2 - 0.05, y), 4), height / 2]

    @staticmethod
    def _deterministic_yaw(category: str) -> float:
        if category in {"chair"}:
            return 180.0
        if category in {"sofa"}:
            return 90.0
        return 0.0

    @staticmethod
    def _region_for(category: str) -> str:
        if category in {"bed", "nightstand", "wardrobe"}:
            return "sleeping_zone"
        if category in {"desk", "chair", "bookshelf"}:
            return "work_zone"
        if category in {"sofa", "coffee_table", "tv_stand", "floor_lamp"}:
            return "living_zone"
        return "main_zone"

    @staticmethod
    def _relations_for(bm_instance: dict, objects: list[dict]) -> list[dict]:
        by_category = {obj["category"]: obj["object_id"] for obj in objects}
        by_id = {obj["object_id"]: obj["object_id"] for obj in objects}
        relations = []
        for constraint in bm_instance.get("relations") or []:
            source_id = by_id.get(constraint.get("subject")) or by_category.get(constraint.get("subject"))
            target_id = by_id.get(constraint.get("object")) or by_category.get(constraint.get("object"))
            if not source_id or not target_id:
                continue
            relations.append(
                {
                    "type": constraint["type"],
                    "source": source_id,
                    "target": target_id,
                    "hard": True,
                }
            )
        for constraint in bm_instance.get("spatial_constraints") or []:
            source_id = by_category.get(constraint.get("source_category"))
            if not source_id:
                continue
            relation = {
                "type": constraint["type"],
                "source": source_id,
                "hard": bool(constraint.get("hard", False)),
            }
            if "target_category" in constraint:
                target_id = by_category.get(constraint["target_category"])
                if not target_id:
                    continue
                relation["target"] = target_id
            elif "target" in constraint:
                relation["target"] = constraint["target"]
            relations.append(relation)
        return relations
