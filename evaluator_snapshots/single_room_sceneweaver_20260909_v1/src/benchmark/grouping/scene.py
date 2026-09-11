from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


DERIVED_OBJECT_ID_POLICY = "scene_order_v1"


@dataclass(frozen=True)
class NormalizedGroupingScene:
    scene_id: str
    scene_type: str
    boundary: tuple[tuple[float, float], ...]
    objects: tuple[dict[str, Any], ...]
    omitted_objects: tuple[dict[str, Any], ...]
    explicit_object_id_count: int
    derived_object_id_count: int

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(str(item["object_id"]) for item in self.objects)

    def layout(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "boundary": [list(point) for point in self.boundary],
            "objects": [
                {
                    "object_id": item["object_id"],
                    "id": item["object_id"],
                    "category": item["category"],
                    "center": list(item["center"]),
                    "size": list(item["size"]),
                    "rotation": list(item["rotation"]),
                    "yaw": float(item["rotation"][2]),
                    **(
                        {"support_parent": item["support_parent"]}
                        if item.get("support_parent")
                        else {}
                    ),
                    **(
                        {"region_id": item["region_id"]}
                        if item.get("region_id")
                        else {}
                    ),
                    **(
                        {"is_anchor": True}
                        if item.get("is_anchor") is True
                        else {}
                    ),
                }
                for item in self.objects
            ],
        }

    def case(self, value: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(value)
        if not isinstance(result.get("room"), dict):
            result["room"] = {}
        if self.boundary and not (
            result["room"].get("boundary")
            or result["room"].get("floor_polygon")
        ):
            result["room"]["boundary"] = [
                list(point) for point in self.boundary
            ]
        if self.scene_type and not result.get("scene_type"):
            result["scene_type"] = self.scene_type
        return result

    def object_catalog(
        self,
        *,
        description_chars: int = 160,
        include_rotation: bool = True,
    ) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for item in self.objects:
            record = {
                "object_id": item["object_id"],
                "source_index": item["source_index"],
                "category": item["category"],
                "description": str(item["description"])[
                    : max(1, int(description_chars))
                ],
                "center": list(item["center"]),
                "size": list(item["size"]),
                "support_parent": item.get("support_parent"),
                "region_id": item.get("region_id"),
            }
            if include_rotation:
                record["rotation"] = list(item["rotation"])
            catalog.append(record)
        return catalog

    def provenance(self) -> dict[str, Any]:
        payload = {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "boundary": [list(point) for point in self.boundary],
            "objects": self.object_catalog(),
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "normalization_version": "grouping_scene_v1",
            "derived_object_id_policy": DERIVED_OBJECT_ID_POLICY,
            "renderable_object_count": len(self.objects),
            "omitted_object_count": len(self.omitted_objects),
            "omitted_objects": list(deepcopy(self.omitted_objects)),
            "explicit_object_id_count": self.explicit_object_id_count,
            "derived_object_id_count": self.derived_object_id_count,
            "normalized_scene_sha256": digest,
        }


def normalize_grouping_scene(
    scene: dict[str, Any],
) -> NormalizedGroupingScene:
    if not isinstance(scene, dict):
        raise TypeError("grouping scene must be a JSON object")
    raw_objects = scene.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("grouping scene objects must be a list")

    objects: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    explicit_count = 0
    derived_count = 0
    for index, value in enumerate(raw_objects):
        if not isinstance(value, dict):
            omitted.append(
                {"source_index": index, "reason": "object_not_json"}
            )
            continue
        center = _finite_vector(value.get("center"), length=3)
        size = _finite_vector(value.get("size"), length=3)
        if center is None or size is None or any(item <= 0.0 for item in size):
            omitted.append(
                {
                    "source_index": index,
                    "reason": "non_renderable_geometry",
                }
            )
            continue

        explicit_id = _first_text(
            value.get("object_id"),
            value.get("id"),
        )
        if explicit_id is not None:
            if explicit_id in seen_ids:
                raise ValueError(
                    "grouping scene contains duplicate explicit object ID "
                    f"{explicit_id!r}"
                )
            object_id = explicit_id
            explicit_count += 1
        else:
            object_id = _available_derived_id(index, seen_ids)
            derived_count += 1
        seen_ids.add(object_id)

        category = _first_text(
            value.get("category"),
            value.get("class_name"),
            value.get("class"),
            value.get("type"),
            value.get("short_desc"),
            value.get("name"),
        ) or "unknown_object"
        description = _first_text(
            value.get("short_desc"),
            value.get("description"),
            value.get("desc"),
            value.get("name"),
            category,
        ) or category
        rotation = _rotation(value)
        support_parent = _first_text(value.get("support_parent"))
        region_id = _first_text(
            value.get("region_id"),
            value.get("source_region_id"),
        )
        objects.append(
            {
                "object_id": object_id,
                "source_index": index,
                "category": category,
                "description": description,
                "center": tuple(_round(item) for item in center),
                "size": tuple(_round(item) for item in size),
                "rotation": tuple(_round(item) for item in rotation),
                "support_parent": support_parent,
                "region_id": region_id,
                "is_anchor": value.get("is_anchor") is True
                or str(value.get("role") or "").strip().lower() == "anchor",
            }
        )

    return NormalizedGroupingScene(
        scene_id=_first_text(
            scene.get("scene_id"),
            scene.get("id"),
        )
        or "scene",
        scene_type=_first_text(
            scene.get("scene_type"),
            scene.get("room_type"),
            scene.get("category"),
        )
        or "",
        boundary=_boundary(scene),
        objects=tuple(objects),
        omitted_objects=tuple(omitted),
        explicit_object_id_count=explicit_count,
        derived_object_id_count=derived_count,
    )


def floor_box(item: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(value) for value in item["center"])
    width, depth, _ = (float(value) for value in item["size"])
    return (
        x - width / 2.0,
        x + width / 2.0,
        y - depth / 2.0,
        y + depth / 2.0,
    )


def floor_gap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    left_box = floor_box(left)
    right_box = floor_box(right)
    dx = max(
        0.0,
        right_box[0] - left_box[1],
        left_box[0] - right_box[1],
    )
    dy = max(
        0.0,
        right_box[2] - left_box[3],
        left_box[2] - right_box[3],
    )
    return math.hypot(dx, dy)


def scene_diagonal(scene: NormalizedGroupingScene) -> float:
    if scene.boundary:
        xs = [point[0] for point in scene.boundary]
        ys = [point[1] for point in scene.boundary]
        return max(
            1.0,
            math.hypot(max(xs) - min(xs), max(ys) - min(ys)),
        )
    if not scene.objects:
        return 1.0
    boxes = [floor_box(item) for item in scene.objects]
    return max(
        1.0,
        math.hypot(
            max(box[1] for box in boxes) - min(box[0] for box in boxes),
            max(box[3] for box in boxes) - min(box[2] for box in boxes),
        ),
    )


def footprint_area(item: dict[str, Any]) -> float:
    return max(0.0, float(item["size"][0]) * float(item["size"][1]))


def _available_derived_id(index: int, seen: set[str]) -> str:
    base = f"scene_object_{index:03d}"
    candidate = base
    suffix = 1
    while candidate in seen:
        candidate = f"{base}_derived_{suffix:02d}"
        suffix += 1
    return candidate


def _finite_vector(value: Any, *, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        result.append(numeric)
    return tuple(result)


def _rotation(value: dict[str, Any]) -> tuple[float, float, float]:
    rotation = _finite_vector(
        value.get("rotation")
        if value.get("rotation") is not None
        else value.get("rotation_degrees"),
        length=3,
    )
    if rotation is not None:
        return rotation
    yaw = value.get("yaw")
    if isinstance(yaw, (int, float)) and not isinstance(yaw, bool):
        numeric = float(yaw)
        if math.isfinite(numeric):
            return (0.0, 0.0, numeric)
    return (0.0, 0.0, 0.0)


def _boundary(scene: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    candidate = scene.get("boundary")
    if candidate is None and isinstance(scene.get("room"), dict):
        candidate = (
            scene["room"].get("boundary")
            or scene["room"].get("floor_polygon")
        )
    if not isinstance(candidate, list):
        return ()
    points: list[tuple[float, float]] = []
    for value in candidate:
        point = _finite_vector(value, length=2)
        if point is None:
            return ()
        points.append((_round(point[0]), _round(point[1])))
    return tuple(points) if len(points) >= 3 else ()


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return None


def _round(value: float) -> float:
    return round(float(value), 6)
