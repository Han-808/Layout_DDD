"""Generation method adapter registry."""

from benchmark.adapters.base import GenerationAdapter
from benchmark.adapters.registry import get_adapter, list_adapters

__all__ = ["GenerationAdapter", "get_adapter", "list_adapters"]
