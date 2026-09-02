from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.layout_vlm.converter import convert_layout_vlm


class LayoutVLMAdapter(HarnessConverterAdapter):
    """Adapter for LayoutVLM layout.json plus its input asset table."""

    name = "layout_vlm"
    output_schema = "layoutvlm_layout_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    converter = staticmethod(convert_layout_vlm)


__all__ = ["LayoutVLMAdapter"]
