from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.layout_gpt.converter import convert_layout_gpt


class LayoutGPTAdapter(HarnessConverterAdapter):
    """Adapter for LayoutGPT's released parsed 3D layout JSON."""

    name = "layout_gpt"
    output_schema = "layoutgpt_3d_output_v1"
    converter = staticmethod(convert_layout_gpt)


__all__ = ["LayoutGPTAdapter"]
