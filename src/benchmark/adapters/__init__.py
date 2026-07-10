"""Generation method adapter registry."""

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.adapters.registry import get_adapter, list_adapters

__all__ = ["AdapterCapabilities", "GenerationAdapter", "get_adapter", "list_adapters"]
