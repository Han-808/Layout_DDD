from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.native_input import (
    public_asset_selection,
    public_instruction,
    public_room,
    public_scene_type,
)
from benchmark.adapters.respace.converter import convert_respace
from benchmark.scene_io.validate import ArtifactValidationError


class ReSpaceAdapter(HarnessConverterAdapter):
    """Adapter for ReSpace Structured Scene Representation (SSR) JSON."""

    name = "respace"
    output_schema = "respace_ssr_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    executable_integration = True
    native_input_filename = "respace_request.json"
    native_input_schema = "respace_public_runner_request_v1"
    default_native_artifact = "{upstream_output_dir}/scene.json"
    converter = staticmethod(convert_respace)

    def build_native_input(self, method_input: dict, config: dict) -> dict:
        del config
        room = public_room(method_input)
        boundary = room.get("boundary")
        if not isinstance(boundary, list):
            raise ArtifactValidationError("ReSpace public room requires a boundary")
        floor_y = float(room.get("floor_z", 0.0))
        height = float(room["height"])
        bounds_bottom = [
            [float(point[0]), floor_y, -float(point[1])]
            for point in boundary
        ]
        bounds_top = [
            [point[0], floor_y + height, point[2]]
            for point in bounds_bottom
        ]
        return {
            "schema_version": self.native_input_schema,
            "operation": "full_scene_generation",
            "prompt": public_instruction(method_input),
            "scene": {
                "room_type": public_scene_type(method_input),
                "bounds_bottom": bounds_bottom,
                "bounds_top": bounds_top,
                "objects": [],
            },
            "asset_selection": public_asset_selection(method_input),
        }


__all__ = ["ReSpaceAdapter"]
