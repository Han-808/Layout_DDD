from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.respace.converter import convert_respace


class ReSpaceAdapter(HarnessConverterAdapter):
    """Adapter for ReSpace Structured Scene Representation (SSR) JSON."""

    name = "respace"
    output_schema = "respace_ssr_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    converter = staticmethod(convert_respace)


__all__ = ["ReSpaceAdapter"]
