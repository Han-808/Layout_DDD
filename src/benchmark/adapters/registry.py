from __future__ import annotations

from collections.abc import Callable, Mapping

from benchmark.adapters.base import GenerationAdapter
from benchmark.adapters.catalog_placement.adapter import CatalogPlacementAdapter
from benchmark.adapters.direct_layout.adapter import DirectLayoutAdapter
from benchmark.adapters.holodeck.adapter import HolodeckAdapter
from benchmark.adapters.layout_json.adapter import LayoutJsonAdapter
from benchmark.adapters.layout_gpt.adapter import LayoutGPTAdapter
from benchmark.adapters.layout_vlm.adapter import LayoutVLMAdapter
from benchmark.adapters.object_state.adapter import ObjectStateAdapter
from benchmark.adapters.respace.adapter import ReSpaceAdapter
from benchmark.adapters.scene_package.adapter import ScenePackageAdapter
from benchmark.adapters.scene_program.adapter import SceneProgramAdapter
from benchmark.adapters.scene_smith.adapter import SceneSmithAdapter
from benchmark.adapters.scene_weaver.adapter import SceneWeaverAdapter


AdapterFactory = Callable[[], GenerationAdapter]


def _adapter_key(name: str) -> str:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("generation adapter name must not be empty")
    return key


class AdapterRegistry:
    """Factory registry for built-in and external harness adapters."""

    def __init__(self, factories: Mapping[str, AdapterFactory] | None = None) -> None:
        self._factories: dict[str, AdapterFactory] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a lazy adapter factory under its stable harness name."""

        key = _adapter_key(name)
        if not callable(factory):
            raise TypeError("adapter factory must be callable")
        if key in self._factories and not replace:
            raise ValueError(f"Generation adapter {key!r} is already registered")
        self._factories[key] = factory

    def create(self, name: str) -> GenerationAdapter:
        """Instantiate one adapter without sharing per-run adapter state."""

        key = _adapter_key(name)
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(
                f"Unknown generation adapter {name!r}. "
                f"Available adapters: {', '.join(self.names())}"
            )
        adapter = factory()
        if not isinstance(adapter, GenerationAdapter):
            raise TypeError(
                f"Adapter factory {key!r} returned {type(adapter).__name__}, "
                "expected GenerationAdapter"
            )
        if adapter.name != key:
            raise ValueError(
                f"Adapter factory registered as {key!r} returned adapter named "
                f"{adapter.name!r}"
            )
        # Resolve the route now so an incomplete plugin fails at selection,
        # before any generator or evaluator work starts.
        adapter.scene_output_route()
        return adapter

    def names(self) -> list[str]:
        return sorted(self._factories)


_BUILTIN_ADAPTERS: dict[str, AdapterFactory] = {
    "catalog_placement": CatalogPlacementAdapter,
    "direct_layout": DirectLayoutAdapter,
    "holodeck": HolodeckAdapter,
    "layout_json": LayoutJsonAdapter,
    "layout_gpt": LayoutGPTAdapter,
    "layout_vlm": LayoutVLMAdapter,
    "object_state": ObjectStateAdapter,
    "respace": ReSpaceAdapter,
    "scene_package": ScenePackageAdapter,
    "scene_program": SceneProgramAdapter,
    "scene_smith": SceneSmithAdapter,
    "scene_weaver": SceneWeaverAdapter,
}
_REGISTRY = AdapterRegistry(_BUILTIN_ADAPTERS)


def get_adapter(name: str) -> GenerationAdapter:
    return _REGISTRY.create(name)


def register_adapter(
    name: str,
    factory: AdapterFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a harness adapter factory in the process-wide registry."""

    _REGISTRY.register(name, factory, replace=replace)


def list_adapters() -> list[str]:
    return _REGISTRY.names()


__all__ = [
    "AdapterFactory",
    "AdapterRegistry",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
