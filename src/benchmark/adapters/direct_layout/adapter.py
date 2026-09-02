from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.native_input import (
    public_instruction,
    public_room_dimensions,
)
from benchmark.adapters.direct_layout.converter import convert_direct_layout


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


__all__ = ["DirectLayoutAdapter"]
