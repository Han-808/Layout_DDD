from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.native_input import (
    public_asset_selection,
    public_instruction,
    public_room,
)
from benchmark.adapters.layout_vlm.converter import convert_layout_vlm
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


class LayoutVLMAdapter(HarnessConverterAdapter):
    """Adapter for LayoutVLM layout.json plus its input asset table."""

    name = "layout_vlm"
    output_schema = "layoutvlm_layout_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    executable_integration = True
    native_input_filename = "layoutvlm_scene_config.json"
    native_input_schema = "layoutvlm_scene_config_v1"
    default_native_artifact = "{upstream_output_dir}/layout.json"
    converter = staticmethod(convert_layout_vlm)

    def build_native_input(self, method_input: dict, config: dict) -> dict:
        scene_config = _configured_scene_config(config)
        room = public_room(method_input)
        floor_z = float(room.get("floor_z", 0.0))
        boundary = room.get("boundary")
        if not isinstance(boundary, list):
            raise ArtifactValidationError(
                "LayoutVLM public room requires a boundary"
            )
        scene_config["task_description"] = public_instruction(method_input)
        scene_config.setdefault(
            "layout_criteria",
            str(
                config.get("layout_criteria")
                or "follow the public task description and room contract"
            ),
        )
        scene_config["boundary"] = {
            "floor_vertices": [
                [float(point[0]), float(point[1]), floor_z]
                for point in boundary
            ],
            "wall_height": float(room["height"]),
        }
        if not isinstance(scene_config.get("assets"), Mapping):
            scene_config["assets"] = _assets_from_public_selection(
                public_asset_selection(method_input)
            )
        return scene_config

    def enrich_conversion_config(self, config: dict) -> dict:
        cfg = super().enrich_conversion_config(config)
        if cfg.get("scene_config") is not None or cfg.get("scene_config_path"):
            return cfg
        metadata = getattr(self, "last_preparation_metadata", None)
        path = metadata.get("native_input_path") if isinstance(metadata, dict) else None
        if path:
            cfg["scene_config_path"] = str(path)
        return cfg


def _configured_scene_config(config: Mapping[str, object]) -> dict:
    value = config.get("layout_vlm_scene_config")
    if value is None:
        value = config.get("scene_config")
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    path_value = config.get("layout_vlm_scene_config_path")
    if path_value is None:
        path_value = config.get("scene_config_path")
    if path_value is None:
        return {}
    path = Path(str(path_value)).expanduser()
    execution = config.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    if not path.is_absolute() and execution.get("repo_path"):
        path = Path(str(execution["repo_path"])).expanduser() / path
    loaded = read_json(path)
    if not isinstance(loaded, Mapping):
        raise ArtifactValidationError(
            "LayoutVLM configured scene input must be a JSON object"
        )
    return deepcopy(dict(loaded))


def _assets_from_public_selection(selection: dict | None) -> dict:
    if not isinstance(selection, Mapping):
        return {}
    objects = selection.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        return {}
    assets: dict[str, dict] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            continue
        selected = item.get("selected_asset")
        if not isinstance(selected, Mapping):
            continue
        object_id = str(item.get("object_id") or f"asset_{index}")
        asset_ref = selected.get("asset_ref")
        asset_ref = asset_ref if isinstance(asset_ref, Mapping) else {}
        proxy = selected.get("asset_proxy")
        proxy = proxy if isinstance(proxy, Mapping) else {}
        asset_key = str(
            selected.get("jid")
            or asset_ref.get("asset_key")
            or object_id
        )
        size = selected.get("size") or proxy.get("bbox_size")
        if not (
            isinstance(size, Sequence)
            and not isinstance(size, (str, bytes))
            and len(size) == 3
        ):
            continue
        entry = {
            "uid": asset_key,
            "category": str(selected.get("category") or "unknown"),
            "description": str(
                selected.get("description")
                or selected.get("desc")
                or selected.get("category")
                or "unknown"
            ),
            "assetMetadata": {
                "boundingBox": {
                    "x": float(size[0]),
                    "y": float(size[1]),
                    "z": float(size[2]),
                }
            },
        }
        if asset_ref.get("mesh_uri"):
            entry["path"] = str(asset_ref["mesh_uri"])
        assets[object_id] = entry
    return assets


__all__ = ["LayoutVLMAdapter"]
