from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.input_modes import representation_mode_for_level, resolve_input_representation_mode
from benchmark.utils.io import read_json


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    path: Path
    source_type: str


class DatasetAdapter(Protocol):
    def discover_cases(self, dataset_config: dict) -> list[CaseRef]:
        ...

    def load_case(self, case_ref: CaseRef) -> dict:
        ...

    def normalize_case(self, raw_case: dict, dataset_config: dict) -> dict:
        ...


class JsonFolderAdapter:
    source_type = "json_folder"

    def discover_cases(self, dataset_config: dict) -> list[CaseRef]:
        root = _configured_path(dataset_config)
        pattern = str(dataset_config.get("pattern") or "*.json")
        paths = [root] if root.is_file() else sorted(root.glob(pattern))
        return [CaseRef(_case_id_from_path(path), path, self.source_type) for path in paths if path.is_file()]

    def load_case(self, case_ref: CaseRef) -> dict:
        raw = read_json(case_ref.path)
        if not isinstance(raw, dict):
            raise ValueError(f"Case at {case_ref.path} must be a JSON object.")
        return raw

    def normalize_case(self, raw_case: dict, dataset_config: dict) -> dict:
        case = dict(raw_case)
        case_id = case.get("case_id") or case.get("task_id") or dataset_config.get("case_id")
        if case_id:
            case.setdefault("case_id", str(case_id))
            case.setdefault("task_id", str(case_id))
        input_level = case.get("input_level") or dataset_config.get("input_level")
        if input_level:
            case.setdefault("input_level", str(input_level))
        mode = resolve_input_representation_mode(
            {
                **dataset_config,
                **case,
                "scene_representation_mode": case.get("scene_representation_mode") or dataset_config.get("scene_representation_mode"),
            },
            default=representation_mode_for_level(input_level),
        )
        case.setdefault("scene_representation_mode", mode)
        source = case.get("source")
        if isinstance(source, dict):
            source = dict(source)
            source.setdefault("input_representation_mode", mode)
            source.setdefault("scene_representation_mode", mode)
            case["source"] = source
        return case


class LegendHSSDSceneInstanceAdapter(JsonFolderAdapter):
    source_type = "legend_hssd_scene_instance_json"

    def discover_cases(self, dataset_config: dict) -> list[CaseRef]:
        root = _configured_path(dataset_config)
        pattern = str(dataset_config.get("pattern") or "**/*.scene_instance.json")
        paths = [root] if root.is_file() else sorted(root.glob(pattern))
        return [CaseRef(_case_id_from_path(path), path, self.source_type) for path in paths if path.is_file()]

    def normalize_case(self, raw_case: dict, dataset_config: dict) -> dict:
        if _looks_like_bm_instance(raw_case):
            return super().normalize_case(raw_case, dataset_config)

        scene_id = _hssd_scene_id(raw_case, dataset_config)
        objects = _hssd_objects(raw_case)
        input_level = dataset_config.get("input_level", "structured_basic")
        mode = resolve_input_representation_mode(
            dataset_config,
            default=representation_mode_for_level(input_level, dataset_config.get("scene_representation_mode")),
        )
        room = _room_for_hssd_case(objects, dataset_config)
        raw_objects = _hssd_raw_objects(raw_case)
        return {
            "case_id": scene_id,
            "task_id": scene_id,
            "input_level": input_level,
            "scene_representation_mode": mode,
            "description": {
                "text": dataset_config.get(
                    "description",
                    "LEGEND HSSD scene-instance metadata normalized to legacy bbox layout benchmark input.",
                )
            },
            "room": room,
            "objects": objects,
            "relations": [],
            "attachments": [],
            "source": {
                "dataset": "hssd-hab",
                "source_type": self.source_type,
                "input_chain": "legend",
                "current_input_chain": "natural_language",
                "scene_instance_fields": sorted(raw_case.keys()),
                "stage_instance": raw_case.get("stage_instance"),
                "translation_origin": raw_case.get("translation_origin"),
                "raw_object_instance_count": len(raw_objects),
                "imported_object_count": len(objects),
                "truncated": False,
                "input_representation_mode": mode,
                "scene_representation_mode": mode,
                "mesh_imported": False,
                "mesh_free_import": True,
                "mesh_asset_policy": "metadata_references_only",
                "room_boundary_source_kind": room.get("boundary_source_kind"),
                "room_geometry_fidelity": room.get("geometry_fidelity"),
                "room_is_proxy_geometry": room.get("is_proxy_geometry"),
            },
        }


DATASET_ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "json_folder": JsonFolderAdapter,
    "legend_hssd_scene_instance_json": LegendHSSDSceneInstanceAdapter,
    # Backward-compatible alias for old configs; HSSD remains a legend input chain.
    "hssd_scene_instance_json": LegendHSSDSceneInstanceAdapter,
}


def create_dataset_adapter(source_type: str) -> DatasetAdapter:
    adapter_cls = DATASET_ADAPTERS.get(source_type)
    if adapter_cls is None:
        available = ", ".join(sorted(DATASET_ADAPTERS))
        raise ValueError(f"Unsupported dataset source_type '{source_type}'. Available: {available}")
    return adapter_cls()


def discover_and_normalize_cases(dataset_config: dict) -> list[tuple[CaseRef, dict]]:
    source_type = _source_type(dataset_config)
    adapter = create_dataset_adapter(source_type)
    cases = []
    for case_ref in adapter.discover_cases(dataset_config):
        raw_case = adapter.load_case(case_ref)
        normalized = adapter.normalize_case(raw_case, {**dataset_config, "case_id": case_ref.case_id})
        cases.append((case_ref, normalized))
    if not cases:
        raise ValueError(f"No cases discovered for dataset source_type '{source_type}'.")
    return cases


def _source_type(dataset_config: dict) -> str:
    source_type = dataset_config.get("source_type") or dataset_config.get("adapter")
    if not source_type:
        raise ValueError("Dataset config requires source_type.")
    return str(source_type)


def _configured_path(dataset_config: dict) -> Path:
    value = (
        dataset_config.get("path")
        or dataset_config.get("root")
        or dataset_config.get("cases_dir")
        or dataset_config.get("case")
        or dataset_config.get("source_path")
    )
    if not value:
        raise ValueError("Dataset config requires path/root/cases_dir/case.")
    return Path(value)


def _case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".scene_instance.json"):
        return name[: -len(".scene_instance.json")]
    return path.stem


def _looks_like_bm_instance(raw_case: dict) -> bool:
    return isinstance(raw_case.get("objects"), list) and isinstance(raw_case.get("room"), dict)


def _hssd_scene_id(raw_case: dict, dataset_config: dict) -> str:
    return str(
        dataset_config.get("case_id")
        or raw_case.get("scene_id")
        or raw_case.get("name")
        or raw_case.get("template_name")
        or raw_case.get("stage_instance", {}).get("template_name")
        or "hssd_scene"
    )


def _hssd_objects(raw_case: dict) -> list[dict]:
    objects = []
    for index, (collection_name, obj) in enumerate(_hssd_raw_objects(raw_case)):
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("name") or obj.get("object_id") or f"object_{index:03d}")
        template = str(obj.get("template_name") or object_id)
        category = str(obj.get("category") or obj.get("semantic_category") or _category_from_template(template))
        translation = _vector3(obj.get("translation"), default=[0.0, 0.0, 0.0])
        bbox_size = _bbox_size_from_object(obj)
        objects.append(
            {
                "id": object_id,
                "category": category,
                "required": True,
                "bbox_size": bbox_size,
                "bbox_size_source": "hssd_scale_or_default",
                "source": "hssd_scene_instance",
                "source_collection": collection_name,
                "source_template_name": template,
                "source_position": translation,
                "source_floor_position": [translation[0], translation[2]],
                "source_height_position": translation[1],
                "layout_center_hint": [translation[0], translation[2], max(translation[1], bbox_size[2] / 2.0)],
                "layout_center_hint_source": "hssd_translation_xz_plus_height_center_hint",
            }
        )
    return objects


def _hssd_raw_objects(raw_case: dict) -> list[tuple[str, dict]]:
    raw: list[tuple[str, dict]] = []
    for collection_name in ["object_instances", "articulated_object_instances", "objects"]:
        value = raw_case.get(collection_name)
        if isinstance(value, list):
            raw.extend((collection_name, item) for item in value if isinstance(item, dict))
    return raw


def _bbox_size_from_object(obj: dict) -> list[float]:
    for key in ["bbox_size", "size", "scale", "non_uniform_scale"]:
        value = obj.get(key)
        vector = _vector3(value)
        if vector is not None:
            return [max(0.05, abs(float(vector[0]))), max(0.05, abs(float(vector[2]))), max(0.05, abs(float(vector[1])))]
    return [1.0, 1.0, 1.0]


def _vector3(value: Any, default: list[float] | None = None) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return default
    return default


def _category_from_template(template: str) -> str:
    clean = template.split("/")[-1].split(".")[0]
    return clean.split("_")[0] if clean else "object"


def _default_room() -> dict:
    return {
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "floor_z": 0.0,
        "wall_height": 3.0,
        "boundary_source": "dataset_config_default",
        "boundary_source_kind": "dataset_config_default",
        "geometry_fidelity": "configured_proxy",
        "is_proxy_geometry": True,
        "mesh_floor_geometry_imported": False,
    }


def _room_for_hssd_case(objects: list[dict], dataset_config: dict) -> dict:
    configured = dataset_config.get("room") or dataset_config.get("default_room")
    if isinstance(configured, dict) and configured:
        room = dict(configured)
        room.setdefault("boundary_source", "dataset_config_default")
        room.setdefault("boundary_source_kind", "dataset_config_default")
        room.setdefault("geometry_fidelity", "configured_proxy")
        room.setdefault("is_proxy_geometry", True)
        room.setdefault("mesh_floor_geometry_imported", False)
        return room
    return {
        "boundary": _boundary_from_hssd_objects(objects),
        "floor_polygon": _boundary_from_hssd_objects(objects),
        "floor_z": 0.0,
        "wall_height": 3.0,
        "boundary_source": "hssd_object_position_extent",
        "boundary_source_kind": "object_position_extent_fallback",
        "geometry_fidelity": "proxy_rectangle",
        "is_proxy_geometry": True,
        "mesh_floor_geometry_imported": False,
    }


def _boundary_from_hssd_objects(objects: list[dict]) -> list[list[float]]:
    points = [obj.get("source_floor_position") for obj in objects if isinstance(obj.get("source_floor_position"), list)]
    if not points:
        return _default_room()["boundary"]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs) - 1.5, max(xs) + 1.5
    min_y, max_y = min(ys) - 1.5, max(ys) + 1.5
    return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
