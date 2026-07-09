from __future__ import annotations

from benchmark.adapters.base import GenerationAdapter
from benchmark.adapters.manual.adapter import ManualAdapter
from benchmark.adapters.passthrough.adapter import PassthroughAdapter


_ADAPTERS: dict[str, type[GenerationAdapter]] = {
    "manual": ManualAdapter,
    "passthrough": PassthroughAdapter,
}


def get_adapter(name: str) -> GenerationAdapter:
    key = str(name or "").strip().lower()
    adapter_cls = _ADAPTERS.get(key)
    if adapter_cls is None:
        raise KeyError(f"Unknown generation adapter {name!r}. Available adapters: {', '.join(list_adapters())}")
    return adapter_cls()


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)
