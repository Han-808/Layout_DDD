from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.direct_layout.converter import convert_direct_layout


class DirectLayoutAdapter(HarnessConverterAdapter):
    """Adapter for the released DirectLayout object-array JSON."""

    name = "direct_layout"
    output_schema = "directlayout_output_v1"
    converter = staticmethod(convert_direct_layout)


__all__ = ["DirectLayoutAdapter"]
