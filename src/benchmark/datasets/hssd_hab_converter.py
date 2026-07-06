from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from benchmark.datasets.estimated_relations import build_estimated_spatial_cues, compatibility_relations, relation_policy_metadata
from benchmark.input_modes import (
    ACCEPTED_INPUT_REPRESENTATION_MODES,
    canonicalize_input_mode,
    representation_mode_for_level,
)
from benchmark.utils.io import write_json

MESH_ASSET_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply"}
SUPPORTING_VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}
SCENE_DATASET_CONFIG_NAMES = [
    "hssd-hab.scene_dataset_config.json",
    "hssd-hab-uncluttered.scene_dataset_config.json",
    "hssd-hab-articulated.scene_dataset_config.json",
]


def iter_scene_instance_paths(hssd_root: str | Path) -> Iterable[Path]:
    root = Path(hssd_root)
    yielded: set[Path] = set()
    for dirname in ["scenes", "scenes-uncluttered", "scenes-articulated", "scenes_articulated"]:
        folder = root / dirname
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.scene_instance.json")):
            yielded.add(path.resolve())
            yield path
    for path in sorted(root.rglob("*.scene_instance.json")):
        if path.resolve() not in yielded:
            yield path


def convert_hssd_hab(
    *,
    hssd_root: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    scene_paths: list[str | Path] | None = None,
    levels: list[str] | None = None,
    max_objects: int | None = None,
    compact_object_ids: bool = False,
    preserve_raw_metadata: bool = False,
    bbox_from_scale: bool = False,
    include_estimated_relations: bool = True,
    input_representation_mode: str | None = None,
) -> list[Path]:
    if input_representation_mode is not None and input_representation_mode not in ACCEPTED_INPUT_REPRESENTATION_MODES:
        available = ", ".join(sorted(ACCEPTED_INPUT_REPRESENTATION_MODES))
        raise ValueError(f"Unsupported input_representation_mode '{input_representation_mode}'. Available: {available}")
    input_representation_mode = canonicalize_input_mode(input_representation_mode) if input_representation_mode else None
    selected_levels = levels or ["prompt_only", "structured_basic"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    paths = [Path(path) for path in scene_paths] if scene_paths else list(iter_scene_instance_paths(hssd_root))
    context = HSSDMetadataContext(Path(hssd_root))
    for index, path in enumerate(paths):
        if limit is not None and index >= limit:
            break
        scene = json.loads(path.read_text(encoding="utf-8"))
        scene_id = _scene_id(scene, path)
        scene_metadata = context.scene_metadata(scene, path)
        objects = _object_specs(
            scene,
            context=context,
            max_objects=max_objects,
            compact_object_ids=compact_object_ids,
            preserve_raw_metadata=preserve_raw_metadata,
            bbox_from_scale=bbox_from_scale,
        )
        room = _room_metadata(scene_metadata, objects)
        room.update(
            {
                "unit": "meter",
                "floor_z": 0.0,
                "coordinate_note": "HSSD translation [x, y, z] is adapted to benchmark floor [x, z] with y as height.",
            }
        )
        _assign_object_regions(objects, room)
        estimated_spatial_cues = build_estimated_spatial_cues(objects) if include_estimated_relations else []
        source_base = _source_metadata(
            scene,
            path,
            objects=objects,
            metadata=scene_metadata,
            room=room,
            estimated_spatial_cues=estimated_spatial_cues,
            preserve_raw_metadata=preserve_raw_metadata,
            bbox_from_scale=bbox_from_scale,
            max_objects=max_objects,
            include_estimated_relations=include_estimated_relations,
        )

        if "prompt_only" in selected_levels:
            mode = representation_mode_for_level("prompt_only", input_representation_mode)
            written.append(
                write_json(
                    out / f"{scene_id}_prompt_only.json",
                    {
                        "case_id": f"{scene_id}_prompt_only",
                        "schema_version": "2.0",
                        "input_level": "prompt_only",
                        "scene_representation_mode": mode,
                        "description": {
                            "text": f"Generate a plausible 3D room layout for HSSD-HAB scene {scene_id}.",
                            "room_type": "room",
                            "tags": ["hssd-hab"],
                        },
                        "source": _source_for_mode(source_base, mode),
                    },
                )
            )

        if "structured_basic" in selected_levels:
            mode = representation_mode_for_level("structured_basic", input_representation_mode)
            written.append(
                write_json(
                    out / f"{scene_id}_structured_basic.json",
                    {
                        "case_id": f"{scene_id}_structured_basic",
                        "schema_version": "2.0",
                        "input_level": "structured_basic",
                        "scene_representation_mode": mode,
                        "description": {
                            "text": f"Generate a plausible 3D room layout containing the specified HSSD-HAB objects for scene {scene_id}.",
                            "room_type": "room",
                            "tags": ["hssd-hab"],
                        },
                        "room": room,
                        "objects": objects,
                        "source": _source_for_mode(source_base, mode),
                    },
                )
            )

        if "structured_relation" in selected_levels:
            relations = compatibility_relations(estimated_spatial_cues) if include_estimated_relations else []
            if estimated_spatial_cues:
                mode = representation_mode_for_level("structured_relation", input_representation_mode)
                case = {
                    "case_id": f"{scene_id}_structured_relation",
                    "schema_version": "2.0",
                    "input_level": "structured_relation",
                    "scene_representation_mode": mode,
                    "description": {
                        "text": f"Generate a plausible 3D room layout with the specified objects and deterministic spatial cues for HSSD-HAB scene {scene_id}.",
                        "room_type": "room",
                        "tags": ["hssd-hab"],
                    },
                    "room": room,
                    "objects": objects,
                    "spatial_cues": estimated_spatial_cues,
                    "source": _source_for_mode(source_base, mode),
                }
                if relations:
                    case["relations"] = relations
                written.append(write_json(out / f"{scene_id}_structured_relation.json", case))
    return written


def _scene_id(scene: dict, path: Path) -> str:
    return str(scene.get("scene_id") or scene.get("name") or path.stem.replace(".scene_instance", ""))


def _object_specs(
    scene: dict,
    *,
    context: "HSSDMetadataContext",
    max_objects: int | None = None,
    compact_object_ids: bool = False,
    preserve_raw_metadata: bool = False,
    bbox_from_scale: bool = False,
) -> list[dict]:
    specs = []
    for index, (collection_name, item) in enumerate(_raw_scene_objects(scene)):
        if max_objects is not None and len(specs) >= max_objects:
            break
        if not isinstance(item, dict):
            continue
        explicit_id = item.get("id") or item.get("object_id")
        template_name = item.get("template_name") or item.get("object_template")
        object_metadata = context.object_metadata(str(template_name or ""))
        source_id = str(explicit_id or f"{template_name or 'object'}_{index + 1:03d}")
        object_id = f"object_{len(specs) + 1:03d}" if compact_object_ids else source_id
        semantic_category = _semantic_category(item, object_metadata)
        category = f"hssd_object_{len(specs) + 1:03d}" if compact_object_ids else semantic_category
        bbox_size, bbox_source = _bbox_size(item, object_metadata=object_metadata, bbox_from_scale=bbox_from_scale)
        spec = {
            "id": object_id,
            "category": category,
            "semantic_category": semantic_category,
            "bbox_size": bbox_size,
            "bbox_size_source": bbox_source,
            "required": True,
            "source": "hssd-hab",
            "source_collection": collection_name,
        }
        if compact_object_ids:
            spec["source_id"] = source_id
            if template_name:
                spec["source_template_name"] = str(template_name)
        elif template_name:
            spec["source_template_name"] = str(template_name)
        semantic = _semantic_payload(object_metadata)
        if semantic:
            spec["hssd_semantic"] = semantic
        translation = item.get("translation") or item.get("position")
        if isinstance(translation, list) and len(translation) >= 3:
            spec["source_position"] = translation[:3]
            spec["source_floor_position"] = [translation[0], translation[2]]
            spec["source_height_position"] = translation[1]
            spec["layout_center_hint"] = _layout_center_hint(translation, bbox_size)
            spec["layout_center_hint_source"] = "hssd_translation_xz_plus_height_center_hint"
        if preserve_raw_metadata:
            spec["source_index"] = index
            spec["source_rotation"] = item.get("rotation")
            spec["source_non_uniform_scale"] = item.get("non_uniform_scale")
            spec["source_motion_type"] = item.get("motion_type")
            spec["source_object_metadata"] = object_metadata
            spec["raw_hssd_instance"] = item
        else:
            source_asset_refs = _asset_references(object_metadata)
            if source_asset_refs:
                spec["source_asset_references"] = source_asset_refs
        specs.append(spec)
    if specs:
        return specs
    return [
        {
            "id": "object_001",
            "category": "furniture",
            "bbox_size": [0.8, 0.8, 0.8],
            "bbox_size_source": "fallback",
            "required": True,
            "source": "estimated",
        }
    ]


def _raw_scene_objects(scene: dict) -> list[tuple[str, dict]]:
    raw: list[tuple[str, dict]] = []
    for collection_name in ["object_instances", "articulated_object_instances", "objects"]:
        value = scene.get(collection_name)
        if not isinstance(value, list):
            continue
        raw.extend((collection_name, item) for item in value if isinstance(item, dict))
    return raw


def _semantic_category(item: dict, object_metadata: dict | None = None) -> str:
    for key in ["category", "semantic_category", "class_name"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value.split("/")[-1].split(".")[0].lower()
    metadata = object_metadata or {}
    for value in _metadata_strings(metadata):
        if value:
            return value.split("/")[-1].split(".")[0].lower()
    for key in ["template_name", "object_template"]:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value.split("/")[-1].split(".")[0].lower()
    return "furniture"


def _bbox_size(item: dict, *, object_metadata: dict | None = None, bbox_from_scale: bool = False) -> tuple[list[float], str]:
    for key in ["bbox_size", "size", "dimensions"]:
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])], key
    metadata = object_metadata or {}
    for key in ["bbox_size", "size", "dimensions", "scale"]:
        value = metadata.get(key)
        if isinstance(value, list) and len(value) >= 3:
            return [_positive_extent(value[0]), _positive_extent(value[2]), _positive_extent(value[1])], f"object_config.{key}"
    for key in ["aligned.dims", "dims"]:
        parsed = _parse_extent_string(metadata.get(key))
        if parsed:
            return [_positive_extent(parsed[0]), _positive_extent(parsed[2]), _positive_extent(parsed[1])], f"metadata.{key}"
    scale = item.get("non_uniform_scale")
    if bbox_from_scale and isinstance(scale, list) and len(scale) >= 3:
        # HSSD/Habitat uses y as height. The benchmark uses z as height, so
        # floor-plane depth maps from source z.
        return [_positive_extent(scale[0]), _positive_extent(scale[2]), _positive_extent(scale[1])], "non_uniform_scale_abs"
    return [0.8, 0.8, 0.8], "fallback"


def _parse_extent_string(value: object) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return []
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) < 3:
        return []
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError:
        return []


def _layout_center_hint(translation: list, bbox_size: list[float]) -> list[float]:
    source_height = _safe_float(translation[1], 0.0)
    bbox_height = _safe_float(bbox_size[2] if len(bbox_size) >= 3 else 0.0, 0.0)
    # HSSD translation is [x, height/up, z]. In this benchmark center is
    # [x, floor-depth-y, height-z]. If an asset origin sits on the floor,
    # lift the proxy bbox by half its height so it does not start below floor.
    center_z = max(source_height, bbox_height / 2.0)
    return [_safe_float(translation[0], 0.0), _safe_float(translation[2], 0.0), center_z]


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _positive_extent(value: object) -> float:
    try:
        numeric = abs(float(value))
    except (TypeError, ValueError):
        return 0.8
    return max(0.05, numeric)


def _source_metadata(
    scene: dict,
    path: Path,
    *,
    objects: list[dict],
    metadata: dict,
    room: dict,
    estimated_spatial_cues: list[dict],
    preserve_raw_metadata: bool,
    bbox_from_scale: bool,
    max_objects: int | None,
    include_estimated_relations: bool,
) -> dict:
    raw_objects = _raw_scene_objects(scene)
    raw_count = len(raw_objects)
    room_layout = metadata.get("room_layout") if isinstance(metadata.get("room_layout"), dict) else {}
    stage_asset_refs = _asset_references(metadata.get("stage_config"))
    object_asset_ref_count = sum(1 for obj in objects if isinstance(obj.get("source_asset_references"), dict) and obj["source_asset_references"])
    mesh_asset_references_kept = bool(stage_asset_refs or object_asset_ref_count)
    relation_metadata = relation_policy_metadata(estimated_spatial_cues) if include_estimated_relations else {
        "relation_policy": "none",
        "relation_generation_version": None,
        "relation_counts_by_type": {},
        "relations_are_ground_truth": False,
        "relations_source_note": "No deterministic spatial cues generated for this case.",
    }
    return {
        "dataset": "hssd-hab",
        "dataset_url": "https://huggingface.co/datasets/hssd/hssd-hab",
        "scene_instance": str(path),
        "scene_variant": path.parent.name,
        "scene_id": _scene_id(scene, path),
        "scene_instance_fields": sorted(scene.keys()),
        "stage_instance": scene.get("stage_instance"),
        "semantic_scene_instance": scene.get("semantic_scene_instance"),
        "navmesh_instance": scene.get("navmesh_instance"),
        "default_lighting": scene.get("default_lighting"),
        "translation_origin": scene.get("translation_origin"),
        "raw_object_instance_count": raw_count,
        "raw_object_collection_counts": _raw_object_collection_counts(scene),
        "imported_object_count": len(objects),
        "max_objects": max_objects,
        "truncated": max_objects is not None and len(objects) < raw_count,
        "preserve_raw_metadata": preserve_raw_metadata,
        "bbox_from_scale": bbox_from_scale,
        "estimated_relations_included": include_estimated_relations,
        "relations_policy": relation_metadata["relation_policy"],
        "relation_policy": relation_metadata["relation_policy"],
        "relation_generation_version": relation_metadata["relation_generation_version"],
        "relation_counts_by_type": relation_metadata["relation_counts_by_type"],
        "relations_are_ground_truth": relation_metadata["relations_are_ground_truth"],
        "relations_source_note": relation_metadata["relations_source_note"],
        "mesh_imported": False,
        "mesh_free_import": True,
        "mesh_asset_policy": "metadata_references_only",
        "mesh_asset_references_kept": mesh_asset_references_kept,
        "object_asset_reference_count": object_asset_ref_count,
        "excluded_asset_extensions": sorted(MESH_ASSET_EXTENSIONS),
        "metadata_inclusion": {
            "scene_dataset_configs": metadata.get("scene_dataset_config_count", 0),
            "stage_config": bool(metadata.get("stage_config")),
            "semantic_scene_config": bool(metadata.get("semantic_scene_config")),
            "scene_filter": bool(metadata.get("scene_filter")),
            "object_configs_indexed": metadata.get("object_configs_indexed", 0),
            "articulated_object_configs_indexed": metadata.get("articulated_object_configs_indexed", 0),
            "metadata_tables_indexed": metadata.get("metadata_tables_indexed", 0),
            "semantic_lexicon": bool(metadata.get("semantic_lexicon")),
            "supporting_visuals": len(metadata.get("supporting_visuals", [])),
        },
        "metadata_paths": metadata.get("metadata_paths", {}),
        "stage_asset_references": stage_asset_refs,
        "room_layout_source": room_layout.get("source"),
        "room_boundary_source": room.get("boundary_source"),
        "room_boundary_source_kind": room.get("boundary_source_kind"),
        "room_geometry_fidelity": room.get("geometry_fidelity"),
        "room_is_proxy_geometry": room.get("is_proxy_geometry"),
        "room_region_count": room_layout.get("region_count", 0),
        "supporting_visuals": metadata.get("supporting_visuals", []),
        "missing_metadata": metadata.get("missing_metadata", []),
    }


def _source_for_mode(source: dict, mode: str) -> dict:
    resolved = dict(source)
    resolved["input_representation_mode"] = mode
    resolved["scene_representation_mode"] = mode
    return resolved


def _raw_object_collection_counts(scene: dict) -> dict[str, int]:
    counts = {}
    for key in ["object_instances", "articulated_object_instances", "objects"]:
        value = scene.get(key)
        if isinstance(value, list):
            counts[key] = len(value)
    return counts


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


def _room_metadata(metadata: dict, objects: list[dict]) -> dict:
    semantic_config = metadata.get("semantic_scene_config") if isinstance(metadata.get("semantic_scene_config"), dict) else {}
    stage_config = metadata.get("stage_config") if isinstance(metadata.get("stage_config"), dict) else {}
    room_layout = metadata.get("room_layout") if isinstance(metadata.get("room_layout"), dict) else {}
    semantic_floor_polygon = room_layout.get("aggregate_boundary") if isinstance(room_layout.get("aggregate_boundary"), list) else []
    stage_floor_polygon = _extract_polygon(stage_config)
    floor_polygon = semantic_floor_polygon or stage_floor_polygon
    if semantic_floor_polygon:
        boundary_source = "hssd_semantic_config.region_annotations.poly_loop"
        boundary_source_kind = "hssd_semantic_region_polygon"
        geometry_fidelity = "semantic_floor_plan"
    elif stage_floor_polygon:
        boundary_source = "hssd_stage_config.floor_polygon"
        boundary_source_kind = "hssd_stage_floor_polygon"
        geometry_fidelity = "stage_floor_polygon"
    else:
        boundary_source = "hssd_object_position_extent"
        boundary_source_kind = "object_position_extent_fallback"
        geometry_fidelity = "proxy_rectangle"
    boundary = floor_polygon or _room_boundary_from_objects(objects)
    regions = room_layout.get("regions") if isinstance(room_layout.get("regions"), list) else _extract_regions(semantic_config)
    room = {
        "boundary": boundary,
        "floor_polygon": boundary,
        "floor_plan": {
            "source": boundary_source,
            "source_kind": boundary_source_kind,
            "geometry_fidelity": geometry_fidelity,
            "coordinate_mapping": "HSSD semantic poly_loop [x, y, z] is imported as benchmark floor polygon [x, z].",
            "regions": regions,
            "aggregate_boundary": boundary,
            "primary_representation": "regions" if regions else "aggregate_boundary",
            "aggregate_boundary_role": "compatibility_proxy",
        },
        "wall_height": _wall_height(stage_config) or _semantic_wall_height(regions) or 3.0,
        "boundary_source": boundary_source,
        "boundary_source_kind": boundary_source_kind,
        "room_layout_source": boundary_source,
        "geometry_fidelity": geometry_fidelity,
        "is_proxy_geometry": geometry_fidelity == "proxy_rectangle",
        "mesh_floor_geometry_imported": False,
        "boundary_role": "aggregate_proxy" if regions else "primary_boundary",
    }
    if regions:
        room["regions"] = regions
    if stage_config:
        room["stage_metadata_keys"] = sorted(stage_config.keys())
        room["stage_asset_references"] = _asset_references(stage_config)
    if semantic_config:
        room["semantic_metadata_keys"] = sorted(semantic_config.keys())
    supporting_visuals = metadata.get("supporting_visuals")
    if isinstance(supporting_visuals, list):
        room["supporting_visuals"] = supporting_visuals
    return room


def _assign_object_regions(objects: list[dict], room: dict) -> None:
    regions = _room_regions(room)
    if not regions:
        return
    for obj in objects:
        point = obj.get("source_floor_position")
        if not isinstance(point, list) or len(point) < 2:
            continue
        assignment = _region_for_point(point, regions)
        if not assignment:
            continue
        region, confidence = assignment
        obj["source_region_id"] = region.get("id")
        obj["source_region_label"] = region.get("label") or region.get("name")
        obj["region_assignment_source"] = "semantic_region_polygon"
        obj["region_assignment_confidence"] = confidence


def _room_regions(room: dict) -> list[dict]:
    direct = room.get("regions")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    floor_plan = room.get("floor_plan")
    nested = floor_plan.get("regions") if isinstance(floor_plan, dict) else []
    return [item for item in nested if isinstance(item, dict)]


def _region_for_point(point: list, regions: list[dict]) -> tuple[dict, float] | None:
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    containing = []
    for region in regions:
        polygon = region.get("floor_polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        if _point_in_polygon(x, y, polygon):
            containing.append((abs(_polygon_area(polygon)), region))
    if not containing:
        return None
    containing.sort(key=lambda item: (item[0], str(item[1].get("id") or "")))
    return containing[0][1], 1.0


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    points = []
    for item in polygon:
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    if len(points) < 3:
        return False
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-12) + xi):
            inside = not inside
        j = i
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float, *, eps: float = 1.0e-9) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps


def _polygon_area(polygon: list) -> float:
    points = []
    for item in polygon:
        if not isinstance(item, list) or len(item) < 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


class HSSDMetadataContext:
    """Best-effort, mesh-free metadata index for local HSSD-HAB files."""

    def __init__(self, hssd_root: Path) -> None:
        self.root = Path(hssd_root)
        self.scene_dataset_configs = self._load_scene_dataset_configs()
        self.object_configs = self._load_object_configs()
        self.articulated_object_configs = self._load_articulated_object_configs()
        self.metadata_tables = self._load_metadata_tables()
        self.semantic_lexicon = self._load_json_optional(self.root / "semantics" / "hssd-hab_semantic_lexicon.json")
        self.supporting_visuals = self._index_supporting_visuals()

    def scene_metadata(self, scene: dict, scene_path: Path) -> dict:
        missing = []
        stage_config, stage_config_path = self._stage_config(scene)
        if not stage_config:
            missing.append("stages/*.stage_config.json")
        semantic_scene_config, semantic_scene_config_path = self._semantic_scene_config(scene, scene_path)
        if not semantic_scene_config:
            missing.append("semantics/scenes/*.semantic_config.json")
        scene_filter, scene_filter_path = self._scene_filter(scene, scene_path)
        if not scene_filter:
            missing.append("scene_filter_files/*.rec_filter.json")
        if not self.object_configs:
            missing.append("objects/**/*.object_config.json")
        if scene_path.parent.name == "scenes-articulated" and not self.articulated_object_configs:
            missing.append("urdf/**/*.ao_config.json")
        if not self.metadata_tables:
            missing.append("metadata/*.csv/json")
        if not self.semantic_lexicon:
            missing.append("semantics/hssd-hab_semantic_lexicon.json")
        visuals = self._scene_supporting_visuals(scene_path)
        room_layout = _room_layout_from_semantic_config(semantic_scene_config, semantic_scene_config_path)
        return {
            "stage_config": stage_config,
            "semantic_scene_config": semantic_scene_config,
            "scene_filter": scene_filter,
            "semantic_lexicon": self.semantic_lexicon,
            "room_layout": room_layout,
            "object_configs_indexed": len(self.object_configs),
            "articulated_object_configs_indexed": len(self.articulated_object_configs),
            "metadata_tables_indexed": len(self.metadata_tables),
            "scene_dataset_config_count": len(self.scene_dataset_configs),
            "supporting_visuals": visuals,
            "metadata_paths": {
                "scene_dataset_configs": [_relative_path(self.root, path) for path in self.scene_dataset_configs.keys()],
                "stage_config": _relative_path(self.root, stage_config_path) if stage_config_path else None,
                "semantic_scene_config": _relative_path(self.root, semantic_scene_config_path) if semantic_scene_config_path else None,
                "scene_filter": _relative_path(self.root, scene_filter_path) if scene_filter_path else None,
            },
            "missing_metadata": sorted(set(missing)),
        }

    def object_metadata(self, template_name: str) -> dict:
        if not template_name:
            return {}
        key = Path(template_name).stem
        direct = self.object_configs.get(key) or self.object_configs.get(template_name) or {}
        articulated = self.articulated_object_configs.get(key) or self.articulated_object_configs.get(template_name) or {}
        table_entry = self.metadata_tables.get(key) or self.metadata_tables.get(template_name) or {}
        merged = {}
        if isinstance(table_entry, dict):
            merged.update(table_entry)
            merged["metadata_table"] = table_entry
        if isinstance(direct, dict):
            merged.update(direct)
            merged["object_config"] = direct
        if isinstance(articulated, dict):
            merged.update(articulated)
            merged["articulated_object_config"] = articulated
        return merged

    def _stage_config(self, scene: dict) -> tuple[dict, Path | None]:
        template = (scene.get("stage_instance") or {}).get("template_name")
        if not isinstance(template, str) or not template:
            return {}, None
        stem = Path(template).stem
        candidates = [
            self.root / "stages" / f"{stem}.stage_config.json",
            self.root / "stages" / f"{stem}.json",
            self.root / f"{template}.stage_config.json",
        ]
        return self._first_json_with_path(candidates)

    def _semantic_scene_config(self, scene: dict, scene_path: Path) -> tuple[dict, Path | None]:
        semantic_name = scene.get("semantic_scene_instance")
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        candidates = []
        for stem in [semantic_name, scene_id]:
            if isinstance(stem, str) and stem:
                candidates.extend(
                    [
                        self.root / "semantics" / "scenes" / f"{stem}.semantic_config.json",
                        self.root / "semantics" / "scenes" / f"{stem}.json",
                        self.root / "semantics" / f"{stem}.semantic_config.json",
                        self.root / "semantics" / f"{stem}.json",
                    ]
                )
        return self._first_json_with_path(candidates)

    def _scene_filter(self, scene: dict, scene_path: Path) -> tuple[dict, Path | None]:
        rel = (scene.get("user_defined") or {}).get("scene_filter_file")
        candidates = []
        if isinstance(rel, str) and rel:
            candidates.append(self.root / rel)
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        candidates.extend(
            [
                self.root / "scene_filter_files" / f"{scene_id}.rec_filter.json",
                self.root / "scene_filter_files" / f"{scene_id}_rec_filter.json",
                self.root / "scene_filter_files" / "articulated_scene_filter_files" / f"{scene_id}.rec_filter.json",
            ]
        )
        return self._first_json_with_path(candidates)

    def _load_object_configs(self) -> dict[str, dict]:
        configs: dict[str, dict] = {}
        for path in self.root.rglob("*.object_config.json"):
            parsed = self._load_json_optional(path)
            if isinstance(parsed, dict):
                parsed.setdefault("source_config_path", _relative_path(self.root, path))
                configs[path.name.replace(".object_config.json", "")] = parsed
                configs[path.stem] = parsed
        return configs

    def _load_articulated_object_configs(self) -> dict[str, dict]:
        configs: dict[str, dict] = {}
        for path in self.root.rglob("*.ao_config.json"):
            parsed = self._load_json_optional(path)
            if isinstance(parsed, dict):
                parsed.setdefault("source_config_path", _relative_path(self.root, path))
                configs[path.name.replace(".ao_config.json", "")] = parsed
                configs[path.stem] = parsed
                configs[path.parent.name] = parsed
        return configs

    def _load_scene_dataset_configs(self) -> dict[Path, dict]:
        configs: dict[Path, dict] = {}
        for name in SCENE_DATASET_CONFIG_NAMES:
            path = self.root / name
            parsed = self._load_json_optional(path)
            if isinstance(parsed, dict) and parsed:
                configs[path] = parsed
        return configs

    def _load_metadata_tables(self) -> dict[str, dict]:
        metadata_root = self.root / "metadata"
        tables: dict[str, dict] = {}
        if not metadata_root.exists():
            return tables
        for path in metadata_root.rglob("*.json"):
            parsed = self._load_json_optional(path)
            if isinstance(parsed, dict):
                _index_metadata_records(parsed, tables)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        _index_metadata_record(item, tables)
        for path in metadata_root.rglob("*.csv"):
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        _index_metadata_record(dict(row), tables)
            except OSError:
                continue
        return tables

    def _first_json(self, candidates: list[Path]) -> dict:
        parsed, _ = self._first_json_with_path(candidates)
        return parsed

    def _first_json_with_path(self, candidates: list[Path]) -> tuple[dict, Path | None]:
        for path in candidates:
            parsed = self._load_json_optional(path)
            if isinstance(parsed, dict) and parsed:
                return parsed, path
        return {}, None

    def _index_supporting_visuals(self) -> list[dict]:
        visuals = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTING_VISUAL_EXTENSIONS:
                continue
            visuals.append(_visual_record(self.root, path))
        return visuals

    def _scene_supporting_visuals(self, scene_path: Path) -> list[dict]:
        scene_id = scene_path.name.replace(".scene_instance.json", "")
        selected = []
        for record in self.supporting_visuals:
            rel = str(record.get("path") or "")
            stem = Path(rel).stem
            if scene_id in rel or scene_id in stem:
                selected.append(record)
        return selected

    @staticmethod
    def _load_json_optional(path: Path) -> Any:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}


def _index_metadata_records(value: dict, tables: dict[str, dict]) -> None:
    _index_metadata_record(value, tables)
    for item in value.values():
        if isinstance(item, dict):
            _index_metadata_record(item, tables)
        elif isinstance(item, list):
            for nested in item:
                if isinstance(nested, dict):
                    _index_metadata_record(nested, tables)


def _index_metadata_record(record: dict, tables: dict[str, dict]) -> None:
    for key in ["template_name", "object_template", "handle", "id", "object_id", "name"]:
        value = record.get(key)
        if isinstance(value, str) and value:
            tables.setdefault(Path(value).stem, record)
            tables.setdefault(value, record)


def _metadata_strings(metadata: dict) -> list[str]:
    preferred_keys = [
        "category",
        "semantic_category",
        "class_name",
        "object_category",
        "object_class",
        "clean_category",
        "main_category",
        "super_category",
        "wnsynsetkey",
        "synset",
        "label",
        "name",
    ]
    values = []
    for key in preferred_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    user_defined = metadata.get("user_defined")
    if isinstance(user_defined, dict):
        values.extend(_metadata_strings(user_defined))
    metadata_table = metadata.get("metadata_table")
    if isinstance(metadata_table, dict):
        values.extend(_metadata_strings(metadata_table))
    return values


def _semantic_payload(metadata: dict) -> dict:
    payload = {}
    for key in [
        "category",
        "semantic_category",
        "class_name",
        "object_category",
        "object_class",
        "clean_category",
        "main_category",
        "super_category",
        "wnsynsetkey",
        "synset",
        "label",
        "name",
        "foundIn",
        "support",
        "floorplanner-category-tags",
        "isArticulatable",
    ]:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            payload[key] = value
    user_defined = metadata.get("user_defined")
    if isinstance(user_defined, dict):
        for key in ["category", "semantic_category", "class_name", "object_category", "object_class", "synset", "label", "name"]:
            value = user_defined.get(key)
            if isinstance(value, str) and value:
                payload[f"user_defined.{key}"] = value
    metadata_table = metadata.get("metadata_table")
    if isinstance(metadata_table, dict):
        for key, value in _semantic_payload(metadata_table).items():
            payload.setdefault(f"metadata_table.{key}", value)
    return payload


def _room_layout_from_semantic_config(semantic_config: dict, semantic_path: Path | None) -> dict:
    regions = _extract_regions(semantic_config)
    aggregate_boundary = _aggregate_region_boundary(regions)
    if not regions and not aggregate_boundary:
        return {}
    return {
        "source": "semantics/scenes/*.semantic_config.json region_annotations",
        "path": str(semantic_path) if semantic_path else None,
        "region_count": len(regions),
        "regions": regions,
        "aggregate_boundary": aggregate_boundary,
    }


def _aggregate_region_boundary(regions: list[dict]) -> list[list[float]]:
    points = []
    for region in regions:
        polygon = region.get("floor_polygon")
        if isinstance(polygon, list):
            points.extend(point for point in polygon if isinstance(point, list) and len(point) >= 2)
    if not points:
        return []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]


def _semantic_wall_height(regions: list[dict]) -> float | None:
    heights = []
    for region in regions:
        value = region.get("extrusion_height")
        if isinstance(value, (int, float)):
            heights.append(float(value))
    return max(heights) if heights else None


def _asset_references(metadata: Any) -> dict:
    if not isinstance(metadata, dict):
        return {}
    refs = {}
    for key in [
        "render_asset",
        "collision_asset",
        "semantic_asset",
        "semantic_descriptor",
        "source_config_path",
        "urdf_file",
        "urdf_path",
    ]:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            refs[key] = value
    object_config = metadata.get("object_config")
    if isinstance(object_config, dict):
        for key, value in _asset_references(object_config).items():
            refs.setdefault(f"object_config.{key}", value)
    articulated_config = metadata.get("articulated_object_config")
    if isinstance(articulated_config, dict):
        for key, value in _asset_references(articulated_config).items():
            refs.setdefault(f"articulated_object_config.{key}", value)
    if refs:
        refs["mesh_files_not_imported"] = True
    return refs


def _visual_record(root: Path, path: Path) -> dict:
    rel = _relative_path(root, path)
    return {
        "path": rel,
        "kind": _visual_kind(rel),
        "extension": path.suffix.lower(),
        "source": "hssd-hab",
    }


def _visual_kind(path: str) -> str:
    lowered = path.lower()
    if "floor" in lowered or "plan" in lowered or "map" in lowered or "topdown" in lowered:
        return "floor_plan_or_map"
    if "rgb" in lowered or "render" in lowered:
        return "rgb_or_render"
    return "supporting_visual"


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _extract_polygon(value: Any) -> list[list[float]]:
    if isinstance(value, dict):
        for key in ["floor_polygon", "boundary", "floorplan", "room_boundary", "polygon"]:
            polygon = _as_polygon(value.get(key))
            if polygon:
                return polygon
        for item in value.values():
            polygon = _extract_polygon(item)
            if polygon:
                return polygon
    elif isinstance(value, list):
        polygon = _as_polygon(value)
        if polygon:
            return polygon
        for item in value:
            polygon = _extract_polygon(item)
            if polygon:
                return polygon
    return []


def _as_polygon(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        return []
    points = []
    for item in value:
        if not isinstance(item, list) or len(item) < 2:
            return []
        try:
            if len(item) >= 3:
                points.append([float(item[0]), float(item[2])])
            else:
                points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            return []
    return points


def _extract_regions(value: Any) -> list[dict]:
    regions = []
    if isinstance(value, dict):
        candidate = value.get("regions") or value.get("region_annotations")
        if isinstance(candidate, list):
            for index, item in enumerate(candidate):
                if isinstance(item, dict):
                    regions.append(_region_record(item, index))
        for item in value.values():
            if isinstance(item, (dict, list)):
                nested = _extract_regions(item)
                if nested:
                    regions.extend(nested)
                    break
    return regions


def _region_record(item: dict, index: int) -> dict:
    min_bounds = item.get("min_bounds")
    max_bounds = item.get("max_bounds")
    polygon = _as_polygon(item.get("poly_loop")) or _extract_polygon(item)
    record = {
        "id": str(item.get("id") or item.get("region_id") or item.get("name") or f"region_{index:03d}"),
        "name": item.get("name"),
        "label": item.get("label") or item.get("category") or item.get("name"),
        "floor_polygon": polygon,
        "source_fields": [key for key in ["poly_loop", "min_bounds", "max_bounds", "floor_height", "extrusion_height", "label", "name"] if key in item],
    }
    for key in ["floor_height", "extrusion_height"]:
        value = item.get(key)
        if isinstance(value, (int, float)):
            record[key] = float(value)
    if isinstance(min_bounds, list) and len(min_bounds) >= 3:
        record["min_bounds"] = [float(min_bounds[0]), float(min_bounds[1]), float(min_bounds[2])]
    if isinstance(max_bounds, list) and len(max_bounds) >= 3:
        record["max_bounds"] = [float(max_bounds[0]), float(max_bounds[1]), float(max_bounds[2])]
    return record


def _wall_height(stage_config: dict) -> float | None:
    for key in ["wall_height", "height", "ceiling_height"]:
        value = stage_config.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
