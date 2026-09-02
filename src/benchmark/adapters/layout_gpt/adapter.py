from collections.abc import Mapping

from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.common.native_input import (
    public_asset_selection,
    public_instruction,
    public_request_id,
    public_room,
    public_room_dimensions,
    public_scene_type,
)
from benchmark.adapters.layout_gpt.converter import convert_layout_gpt
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


class LayoutGPTAdapter(HarnessConverterAdapter):
    """Adapter for LayoutGPT's released parsed 3D layout JSON."""

    name = "layout_gpt"
    output_schema = "layoutgpt_3d_output_v1"
    executable_integration = True
    native_input_filename = "layoutgpt_request.json"
    native_input_schema = "layoutgpt_public_runner_request_v1"
    default_native_artifact_glob = "{upstream_output_dir}/*.json"
    converter = staticmethod(convert_layout_gpt)

    def build_native_input(self, method_input: dict, config: dict) -> dict:
        width, depth, height = public_room_dimensions(method_input)
        return {
            "schema_version": self.native_input_schema,
            "query_id": public_request_id(method_input),
            "prompt": public_instruction(method_input),
            "room_type": public_scene_type(method_input),
            "room": public_room(method_input),
            "room_dimensions_m": [width, depth, height],
            "requested_output_unit": str(
                config.get("layoutgpt_unit") or "px"
            ),
            "asset_selection": public_asset_selection(method_input),
        }

    def enrich_conversion_config(self, config: dict) -> dict:
        cfg = super().enrich_conversion_config(config)
        if isinstance(cfg.get("asset_ids"), Mapping):
            return cfg
        path_value = cfg.get("asset_ids_path")
        run_metadata = getattr(self, "last_run_metadata", None)
        auxiliary = (
            run_metadata.get("preserved_auxiliary_artifacts")
            if isinstance(run_metadata, dict)
            else None
        )
        if not path_value and isinstance(auxiliary, Mapping):
            item = auxiliary.get("asset_ids")
            if isinstance(item, Mapping):
                path_value = item.get("path")
        if not path_value:
            return cfg
        loaded = read_json(path_value)
        if isinstance(loaded, Mapping) and isinstance(
            loaded.get("asset_ids"), Mapping
        ):
            loaded = loaded["asset_ids"]
        if not isinstance(loaded, Mapping):
            raise ArtifactValidationError(
                "LayoutGPT asset_ids_path must contain an asset-id mapping"
            )
        cfg["asset_ids"] = dict(loaded)
        cfg["asset_ids_path"] = str(path_value)
        return cfg


__all__ = ["LayoutGPTAdapter"]
