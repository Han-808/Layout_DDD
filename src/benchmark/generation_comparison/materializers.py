"""Generation-side materializers for one logical catalog snapshot."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    converter_asset_manifest,
    physical_dimensions,
)
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.protocol import ComparisonProtocol, FROZEN_ASSETS
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


MATERIALIZATION_SCHEMA_VERSION = "comparison_catalog_materialization_v1"
SUPPORTED_METHODS = {
    "catalog_placement",
    "layout_gpt",
    "direct_layout",
    "layout_vlm",
    "respace",
    "scene_weaver",
}


def materialize_method_catalog(
    *,
    adapter_name: str,
    catalog: CanonicalAssetCatalog,
    protocol: ComparisonProtocol,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write an auditable, method-native view without changing catalog meaning."""

    if adapter_name not in SUPPORTED_METHODS:
        raise ArtifactValidationError(
            f"no comparison catalog materializer for adapter {adapter_name!r}"
        )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = write_json(root / "canonical_catalog.json", catalog.as_dict())
    converter_manifest_path = write_json(
        root / "converter_exact_manifest.json",
        converter_asset_manifest(catalog),
    )
    slot_map = logical_to_native_slot_map(protocol, adapter_name)
    payload = _method_payload(
        adapter_name=adapter_name,
        catalog=catalog,
        protocol=protocol,
        slot_map=slot_map,
        root=root,
    )
    payload_path = write_json(root / f"{adapter_name}_catalog_input.json", payload)
    control = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "adapter": adapter_name,
        "protocol_id": protocol.as_dict()["protocol_id"],
        "protocol_version": protocol.as_dict()["protocol_version"],
        "protocol_mode": protocol.mode,
        "case_id": protocol.case_id,
        "architecture": protocol.architecture,
        "architecture_sha256": protocol.architecture_hash,
        "catalog": catalog.identity,
        "linear_unit": "meter",
        "object_inventory_policy": protocol.inventory_policy,
        "object_inventory_sha256": protocol.inventory_sha256,
        "asset_binding_sha256": protocol.binding_sha256,
        "scale_policy": protocol.scale_policy,
        "retrieval_policy": protocol.as_dict()["retrieval_policy"],
        "logical_to_native_slot": slot_map,
        "catalog_path": catalog_path.resolve().as_posix(),
        "method_catalog_path": payload_path.resolve().as_posix(),
    }
    if protocol.mode == FROZEN_ASSETS:
        control["frozen_asset_bindings"] = protocol.bindings
    if adapter_name == "direct_layout":
        control["method_asset_root"] = (
            root / "direct_layout_asset_library"
        ).resolve().as_posix()
    control_path = write_json(root / "comparison_control.json", control)
    return {
        **control,
        "comparison_control_path": control_path.resolve().as_posix(),
        "converter_asset_manifest_path": converter_manifest_path.resolve().as_posix(),
        "materialized_catalog_sha256": catalog.sha256,
        "method_payload_sha256": canonical_json_sha256(payload),
        "method_payload_file_sha256": _file_sha256(payload_path),
        "catalog_file_sha256": _file_sha256(catalog_path),
        "control_file_sha256": _file_sha256(control_path),
        "converter_manifest_file_sha256": _file_sha256(converter_manifest_path),
    }


def logical_to_native_slot_map(
    protocol: ComparisonProtocol,
    adapter_name: str,
) -> dict[str, str]:
    if adapter_name != "layout_gpt":
        return {str(item["slot_id"]): str(item["slot_id"]) for item in protocol.objects}
    counts: Counter[str] = Counter()
    result: dict[str, str] = {}
    for item in protocol.objects:
        category = str(item["category"])
        counts[category] += 1
        result[str(item["slot_id"])] = f"{_slug(category)}_{counts[category]}"
    if len(set(result.values())) != len(result):
        raise ArtifactValidationError(
            "LayoutGPT comparison slot normalization produced duplicate native IDs"
        )
    return result


def load_materialized_control(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("materialized comparison control must be an object")
    return dict(value)


def architecture_from_native_input(
    adapter_name: str,
    value: Any,
) -> dict[str, Any]:
    """Reverse only the room portion of each released runner input."""

    if adapter_name == "catalog_placement":
        comparison = (
            value.get("generation_comparison")
            if isinstance(value, Mapping)
            else None
        )
        if not isinstance(comparison, Mapping) or not isinstance(
            comparison.get("architecture"), Mapping
        ):
            raise ArtifactValidationError(
                "catalog_placement method input lacks generation_comparison architecture"
            )
        return dict(comparison["architecture"])
    if adapter_name == "layout_gpt":
        dimensions = _vector(value.get("room_dimensions_m"), "LayoutGPT room dimensions")
        return _architecture_from_dimensions(*dimensions)
    if adapter_name == "direct_layout":
        if not (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
            and isinstance(value[1], Sequence)
            and value[1]
        ):
            raise ArtifactValidationError("DirectLayout native input lacks room dimensions")
        dimensions = _vector(value[1][0], "DirectLayout room dimensions")
        return _architecture_from_dimensions(*dimensions)
    if adapter_name == "layout_vlm":
        boundary = value.get("boundary") if isinstance(value, Mapping) else None
        boundary = boundary if isinstance(boundary, Mapping) else {}
        vertices = boundary.get("floor_vertices")
        if not isinstance(vertices, Sequence):
            raise ArtifactValidationError("LayoutVLM native input lacks floor vertices")
        points = [[point[0], point[1]] for point in vertices]
        return {
            "room_model": "single_room",
            "boundary_model": "axis_aligned_rectangle",
            "room": {
                "boundary": points,
                "height": boundary.get("wall_height"),
                "unit": "meter",
            },
        }
    if adapter_name == "respace":
        scene = value.get("scene") if isinstance(value, Mapping) else None
        scene = scene if isinstance(scene, Mapping) else {}
        bottom = scene.get("bounds_bottom")
        top = scene.get("bounds_top")
        if not isinstance(bottom, Sequence) or not isinstance(top, Sequence):
            raise ArtifactValidationError("ReSpace native input lacks room bounds")
        points = [[float(point[0]), -float(point[2])] for point in bottom]
        bottom_y = {float(point[1]) for point in bottom}
        top_y = {float(point[1]) for point in top}
        if len(bottom_y) != 1 or len(top_y) != 1:
            raise ArtifactValidationError("ReSpace native input room bounds are not planar")
        return {
            "room_model": "single_room",
            "boundary_model": "axis_aligned_rectangle",
            "room": {
                "boundary": points,
                "height": next(iter(top_y)) - next(iter(bottom_y)),
                "unit": "meter",
            },
        }
    if adapter_name == "scene_weaver":
        room = value.get("benchmark_room") if isinstance(value, Mapping) else None
        room = room if isinstance(room, Mapping) else {}
        size = room.get("roomsize")
        if not (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(size) == 2
        ):
            raise ArtifactValidationError("SceneWeaver native input lacks roomsize")
        return _architecture_from_dimensions(size[0], size[1], room.get("height"))
    raise ArtifactValidationError(
        f"no native architecture reader for adapter {adapter_name!r}"
    )


def _method_payload(
    *,
    adapter_name: str,
    catalog: CanonicalAssetCatalog,
    protocol: ComparisonProtocol,
    slot_map: Mapping[str, str],
    root: Path,
) -> dict[str, Any]:
    common = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "adapter": adapter_name,
        "catalog": catalog.identity,
        "mode": protocol.mode,
        "scale_policy": protocol.scale_policy,
        "logical_to_native_slot": dict(slot_map),
    }
    assets_by_id = {asset["asset_id"]: asset for asset in catalog.assets}
    if adapter_name == "catalog_placement":
        return {
            **common,
            "frozen_selected_assets": {
                slot_map[slot_id]: _scene_weaver_asset(catalog.get(asset_id))
                for slot_id, asset_id in protocol.bindings.items()
            },
        }
    if adapter_name == "layout_gpt":
        return {
            **common,
            "dataset_asset_index": {
                asset_id: _layout_gpt_asset(asset)
                for asset_id, asset in assets_by_id.items()
            },
            "frozen_asset_ids": {
                slot_map[slot_id]: asset_id
                for slot_id, asset_id in protocol.bindings.items()
            },
        }
    if adapter_name == "direct_layout":
        return {
            **common,
            "asset_library": _direct_layout_assets(catalog, root),
            "frozen_asset_bindings": {
                slot_map[slot_id]: asset_id
                for slot_id, asset_id in protocol.bindings.items()
            },
        }
    if adapter_name == "layout_vlm":
        candidates = {
            asset_id: _layout_vlm_asset(asset)
            for asset_id, asset in assets_by_id.items()
        }
        selected = {
            slot_map[slot_id]: _layout_vlm_asset(catalog.get(asset_id))
            for slot_id, asset_id in protocol.bindings.items()
        }
        return {**common, "catalog_candidates": candidates, "frozen_assets": selected}
    if adapter_name == "respace":
        return {
            **common,
            "asset_metadata": {
                asset_id: _respace_asset(asset)
                for asset_id, asset in assets_by_id.items()
            },
            "frozen_scene_objects": [
                _respace_frozen_object(item, slot_map, catalog)
                for item in protocol.objects
                if item.get("asset_id")
            ],
        }
    return {
        **common,
        "asset_source": {
            asset_id: _scene_weaver_asset(asset)
            for asset_id, asset in assets_by_id.items()
        },
        "frozen_asset_bindings": {
            slot_map[slot_id]: _scene_weaver_asset(catalog.get(asset_id))
            for slot_id, asset_id in protocol.bindings.items()
        },
    }


def _layout_gpt_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "jid": asset["asset_id"],
        "source_db": asset["source_db"],
        "category": asset["category"],
        "description": asset["description"],
        "mesh_uri": asset.get("mesh_uri"),
        "bbox_size_local": asset["bbox_size_local"],
        "bbox_center_local": asset["bbox_center_local"],
        "canonical_front": asset.get("canonical_front"),
        "native_scale": asset["native_scale"],
        "physical_dimensions": asset["physical_dimensions"],
    }


def _direct_layout_assets(
    catalog: CanonicalAssetCatalog,
    root: Path,
) -> list[dict[str, Any]]:
    library = root / "direct_layout_asset_library"
    library.mkdir(exist_ok=True)
    used_names: set[str] = set()
    result = []
    for asset in catalog.assets:
        mesh = asset.get("mesh_uri")
        materialized: str | None = None
        if mesh:
            path = _local_path(str(mesh))
            if path is not None and path.is_file():
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(asset["asset_id"]))
                filename = f"{safe}{path.suffix.lower()}"
                if filename in used_names:
                    raise ArtifactValidationError(
                        "DirectLayout materialization produced duplicate mesh filename"
                    )
                used_names.add(filename)
                link = library / filename
                link.symlink_to(path)
                materialized = f"direct_layout_asset_library/{filename}"
        result.append(
            {
                "new_object_id": asset["asset_id"],
                "source_db": asset["source_db"],
                "category": asset["category"],
                "description": asset["description"],
                "source_mesh_uri": mesh,
                "materialized_mesh_path": materialized,
                "bbox_size_local": asset["bbox_size_local"],
                "bbox_center_local": asset["bbox_center_local"],
                "canonical_front": asset.get("canonical_front"),
                "native_scale": asset["native_scale"],
                "physical_dimensions": asset["physical_dimensions"],
            }
        )
    return result


def _layout_vlm_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "uid": asset["asset_id"],
        "category": asset["category"],
        "description": asset["description"],
        "assetMetadata": {
            "boundingBox": {
                "x": asset["bbox_size_local"][0],
                "y": asset["bbox_size_local"][1],
                "z": asset["bbox_size_local"][2],
            },
            "boundingBoxCenter": list(asset["bbox_center_local"]),
            "nativeScale": list(asset["native_scale"]),
            "physicalDimensions": list(asset["physical_dimensions"]),
        },
    }
    if asset.get("mesh_uri"):
        result["path"] = asset["mesh_uri"]
    if asset.get("canonical_front") is not None:
        result["frontView"] = asset["canonical_front"]
        result["canonical_front"] = asset["canonical_front"]
    return result


def _respace_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "jid": asset["asset_id"],
        "source_db": asset["source_db"],
        "category": asset["category"],
        "desc": asset["description"],
        "mesh_uri": asset.get("mesh_uri"),
        "sampled_asset_size": [
            asset["bbox_size_local"][0],
            asset["bbox_size_local"][2],
            asset["bbox_size_local"][1],
        ],
        "sampled_asset_center": asset["bbox_center_local"],
        "canonical_front": asset.get("canonical_front"),
        "scale": [
            asset["native_scale"][0],
            asset["native_scale"][2],
            asset["native_scale"][1],
        ],
        "physical_dimensions": asset["physical_dimensions"],
    }


def _respace_frozen_object(
    slot: Mapping[str, Any],
    slot_map: Mapping[str, str],
    catalog: CanonicalAssetCatalog,
) -> dict[str, Any]:
    asset = catalog.get(str(slot["asset_id"]))
    physical = physical_dimensions(asset)
    return {
        "id": slot_map[str(slot["slot_id"])],
        "category": slot["category"],
        "desc": slot["description"],
        "sampled_asset_jid": asset["asset_id"],
        "sampled_asset_size": asset["bbox_size_local"],
        "scale": [
            asset["native_scale"][0],
            asset["native_scale"][2],
            asset["native_scale"][1],
        ],
        "size": [physical[0], physical[2], physical[1]],
    }


def _scene_weaver_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_key": asset["asset_id"],
        "source_db": asset["source_db"],
        "category": asset["category"],
        "description": asset["description"],
        "mesh_uri": asset.get("mesh_uri"),
        "bbox_size": asset["bbox_size_local"],
        "bbox_center_local": asset["bbox_center_local"],
        "canonical_front": asset.get("canonical_front"),
        "native_scale": asset["native_scale"],
        "physical_dimensions": asset["physical_dimensions"],
    }


def _architecture_from_dimensions(width: Any, depth: Any, height: Any) -> dict[str, Any]:
    width_value = float(width)
    depth_value = float(depth)
    return {
        "room_model": "single_room",
        "boundary_model": "axis_aligned_rectangle",
        "room": {
            "boundary": [
                [0.0, 0.0],
                [width_value, 0.0],
                [width_value, depth_value],
                [0.0, depth_value],
            ],
            "height": float(height),
            "unit": "meter",
        },
    }


def _vector(value: Any, path: str) -> list[float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        raise ArtifactValidationError(f"{path} must be a three-vector")
    return [float(component) for component in value]


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "object"


def _local_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return None
    return Path(parsed.path if parsed.scheme == "file" else value).expanduser().resolve()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "MATERIALIZATION_SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "architecture_from_native_input",
    "load_materialized_control",
    "logical_to_native_slot_map",
    "materialize_method_catalog",
]
