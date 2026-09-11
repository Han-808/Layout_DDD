from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.holodeck.converter import convert_holodeck


class HolodeckAdapter(HarnessConverterAdapter):
    """Adapter for raw ProcTHOR JSON or Holodeck's SceneState export."""

    name = "holodeck"
    output_schema = "holodeck_procthor_or_scene_state_v1"
    converter = staticmethod(convert_holodeck)


__all__ = ["HolodeckAdapter"]
