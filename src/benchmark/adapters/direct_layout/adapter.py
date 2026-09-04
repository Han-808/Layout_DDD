from collections.abc import Mapping

from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.native_input import (
    public_instruction,
    public_room_dimensions,
)
from benchmark.adapters.direct_layout.converter import convert_direct_layout
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


class DirectLayoutAdapter(HarnessConverterAdapter):
    """Adapter for the released DirectLayout object-array JSON."""

    name = "direct_layout"
    output_schema = "directlayout_output_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    executable_integration = True
    native_input_filename = "directlayout_input.json"
    native_input_schema = "directlayout_batch_input_v1"
    default_native_artifact_glob = "{upstream_output_dir}/*.json"
    converter = staticmethod(convert_direct_layout)

    def build_native_input(self, method_input: dict, config: dict) -> list:
        del config
        width, depth, height = public_room_dimensions(method_input)
        return [
            [public_instruction(method_input)],
            [[width, depth, height]],
        ]

    def enrich_conversion_config(self, config: dict) -> dict:
        cfg = super().enrich_conversion_config(config)
        if isinstance(cfg.get("asset_bindings"), Mapping):
            return cfg
        path_value = cfg.get("asset_bindings_path")
        run_metadata = getattr(self, "last_run_metadata", None)
        auxiliary = (
            run_metadata.get("preserved_auxiliary_artifacts")
            if isinstance(run_metadata, dict)
            else None
        )
        if not path_value and isinstance(auxiliary, Mapping):
            item = auxiliary.get("asset_bindings")
            if isinstance(item, Mapping):
                path_value = item.get("path")
        if not path_value:
            return cfg
        loaded = read_json(path_value)
        if isinstance(loaded, Mapping) and isinstance(
            loaded.get("asset_bindings"), Mapping
        ):
            loaded = loaded["asset_bindings"]
        if not isinstance(loaded, Mapping):
            raise ArtifactValidationError(
                "DirectLayout asset_bindings_path must contain a binding mapping"
            )
        cfg["asset_bindings"] = dict(loaded)
        cfg["asset_bindings_path"] = str(path_value)
        return cfg


__all__ = ["DirectLayoutAdapter"]
