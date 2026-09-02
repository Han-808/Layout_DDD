from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.respace.converter import convert_respace


class ReSpaceAdapter(HarnessConverterAdapter):
    """Adapter for ReSpace Structured Scene Representation (SSR) JSON."""

    name = "respace"
    output_schema = "respace_ssr_v1"
    converter = staticmethod(convert_respace)


__all__ = ["ReSpaceAdapter"]
