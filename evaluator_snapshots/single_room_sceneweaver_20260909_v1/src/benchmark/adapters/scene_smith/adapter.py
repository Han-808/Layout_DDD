from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.scene_smith.converter import convert_scene_smith


class SceneSmithAdapter(HarnessConverterAdapter):
    """Adapter for SceneSmith state JSON or its official SceneEval export."""

    name = "scene_smith"
    output_schema = "scenesmith_state_or_scene_state_v1"
    converter = staticmethod(convert_scene_smith)


__all__ = ["SceneSmithAdapter"]
