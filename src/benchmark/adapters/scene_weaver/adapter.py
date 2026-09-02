from benchmark.adapters.common.adapter import HarnessConverterAdapter
from benchmark.adapters.scene_weaver.converter import convert_scene_weaver


class SceneWeaverAdapter(HarnessConverterAdapter):
    """Adapter for SceneWeaver's final record_scene/layout_<iter>.json."""

    name = "scene_weaver"
    output_schema = "sceneweaver_layout_v1"
    converter = staticmethod(convert_scene_weaver)


__all__ = ["SceneWeaverAdapter"]
