from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from benchmark.utils.io import write_json


def iter_scene_instance_paths(hssd_root: str | Path) -> Iterable[Path]:
    root = Path(hssd_root)
    scenes = root / "scenes"
    if scenes.exists():
        yield from sorted(scenes.rglob("*.scene_instance.json"))
    else:
        yield from sorted(root.rglob("*.scene_instance.json"))


def convert_hssd_hab(
    *,
    hssd_root: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    levels: list[str] | None = None,
    max_objects: int | None = None,
    compact_object_ids: bool = False,
) -> list[Path]:
    selected_levels = levels or ["prompt_only", "structured_basic"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, path in enumerate(iter_scene_instance_paths(hssd_root)):
        if limit is not None and index >= limit:
            break
        scene = json.loads(path.read_text(encoding="utf-8"))
        scene_id = _scene_id(scene, path)
        objects = _object_specs(scene, max_objects=max_objects, compact_object_ids=compact_object_ids)
        boundary = _room_boundary_from_objects(objects)

        if "prompt_only" in selected_levels:
            written.append(
                write_json(
                    out / f"{scene_id}_prompt_only.json",
                    {
                        "case_id": f"{scene_id}_prompt_only",
                        "schema_version": "2.0",
                        "input_level": "prompt_only",
                        "description": {
                            "text": f"Generate a plausible 3D room layout for HSSD-HAB scene {scene_id}.",
                            "room_type": "room",
                            "tags": ["hssd-hab"],
                        },
                        "source": {"dataset": "hssd-hab", "scene_instance": str(path)},
                    },
                )
            )

        if "structured_basic" in selected_levels:
            written.append(
                write_json(
                    out / f"{scene_id}_structured_basic.json",
                    {
                        "case_id": f"{scene_id}_structured_basic",
                        "schema_version": "2.0",
                        "input_level": "structured_basic",
                        "description": {
                            "text": f"Generate a plausible 3D room layout containing the specified HSSD-HAB objects for scene {scene_id}.",
                            "room_type": "room",
                            "tags": ["hssd-hab"],
                        },
                        "room": {
                            "unit": "meter",
                            "boundary": boundary,
                            "floor_z": 0.0,
                            "wall_height": 3.0,
                        },
                        "objects": objects,
                        "source": {"dataset": "hssd-hab", "scene_instance": str(path)},
                    },
                )
            )

        if "structured_relation" in selected_levels:
            relations = _near_relations(objects)
            if relations:
                written.append(
                    write_json(
                        out / f"{scene_id}_structured_relation.json",
                        {
                            "case_id": f"{scene_id}_structured_relation",
                            "schema_version": "2.0",
                            "input_level": "structured_relation",
                            "description": {
                                "text": f"Generate a plausible 3D room layout with the specified objects and visible relations for HSSD-HAB scene {scene_id}.",
                                "room_type": "room",
                                "tags": ["hssd-hab"],
                            },
                            "room": {
                                "unit": "meter",
                                "boundary": boundary,
                                "floor_z": 0.0,
                                "wall_height": 3.0,
                            },
                            "objects": objects,
                            "relations": relations,
                            "source": {"dataset": "hssd-hab", "scene_instance": str(path)},
                        },
                    )
                )
    return written


def _scene_id(scene: dict, path: Path) -> str:
    return str(scene.get("scene_id") or scene.get("name") or path.stem.replace(".scene_instance", ""))


def _object_specs(scene: dict, *, max_objects: int | None = None, compact_object_ids: bool = False) -> list[dict]:
    raw_objects = scene.get("object_instances") or scene.get("objects") or []
    specs = []
    for index, item in enumerate(raw_objects):
        if max_objects is not None and len(specs) >= max_objects:
            break
        if not isinstance(item, dict):
            continue
        explicit_id = item.get("id") or item.get("object_id")
        template_name = item.get("template_name") or item.get("object_template")
        source_id = str(explicit_id or f"{template_name or 'object'}_{index + 1:03d}")
        object_id = f"object_{len(specs) + 1:03d}" if compact_object_ids else source_id
        category = f"hssd_object_{len(specs) + 1:03d}" if compact_object_ids else _category(item)
        spec = {
            "id": object_id,
            "category": category,
            "bbox_size": _bbox_size(item),
            "required": True,
            "source": "hssd-hab",
        }
        if compact_object_ids:
            spec["source_id"] = source_id
            if template_name:
                spec["source_template_name"] = str(template_name)
        translation = item.get("translation") or item.get("position")
        if isinstance(translation, list) and len(translation) >= 3:
            spec["source_position"] = translation[:3]
            spec["source_floor_position"] = [translation[0], translation[2]]
        specs.append(spec)
    if specs:
        return specs
    return [
        {
            "id": "object_001",
            "category": "furniture",
            "bbox_size": [0.8, 0.8, 0.8],
            "required": True,
            "source": "estimated",
        }
    ]


def _category(item: dict) -> str:
    for key in ["category", "semantic_category", "class_name", "template_name", "object_template"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value.split("/")[-1].split(".")[0].lower()
    return "furniture"


def _bbox_size(item: dict) -> list[float]:
    for key in ["bbox_size", "size", "dimensions"]:
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    return [0.8, 0.8, 0.8]


def _room_boundary_from_objects(objects: list[dict]) -> list[list[float]]:
    points = [
        obj.get("source_floor_position") or obj.get("source_position")
        for obj in objects
        if isinstance(obj.get("source_floor_position") or obj.get("source_position"), list)
    ]
    if not points:
        return [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs) - 1.5, max(xs) + 1.5
    min_y, max_y = min(ys) - 1.5, max(ys) + 1.5
    return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


def _near_relations(objects: list[dict]) -> list[dict]:
    with_positions = [obj for obj in objects if isinstance(obj.get("source_position"), list)]
    relations = []
    for index in range(min(len(with_positions) - 1, 3)):
        first = with_positions[index]
        second = with_positions[index + 1]
        relations.append(
            {
                "id": f"rel_{index + 1:03d}",
                "type": "near",
                "subject": first["id"],
                "object": second["id"],
                "visible_to_model": True,
                "source": "estimated",
            }
        )
    return relations
