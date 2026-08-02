from __future__ import annotations

from benchmark.adapters.base import GenerationAdapter
from benchmark.adapters.catalog_placement.adapter import CatalogPlacementAdapter
from benchmark.adapters.layout_json.adapter import LayoutJsonAdapter
from benchmark.adapters.object_state.adapter import ObjectStateAdapter
from benchmark.adapters.scene_package.adapter import ScenePackageAdapter
from benchmark.adapters.scene_program.adapter import SceneProgramAdapter


_ADAPTERS: dict[str, type[GenerationAdapter]] = {
    "catalog_placement": CatalogPlacementAdapter,
    "layout_json": LayoutJsonAdapter,
    "object_state": ObjectStateAdapter,
    "scene_package": ScenePackageAdapter,
    "scene_program": SceneProgramAdapter,
}


def get_adapter(name: str) -> GenerationAdapter:
    key = str(name or "").strip().lower()
    adapter_cls = _ADAPTERS.get(key)
    if adapter_cls is None:
        raise KeyError(f"Unknown generation adapter {name!r}. Available adapters: {', '.join(list_adapters())}")
    return adapter_cls()


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)
