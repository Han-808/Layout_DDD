"""Generation method adapter registry and I/O contracts."""

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter, OutputMaterializationRequired
from benchmark.adapters.catalog_placement import CatalogPlacementAdapter
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
    O2_SCENE_PROGRAM,
    O3_SCENE_PACKAGE,
    GeneratorIOContract,
)
from benchmark.adapters.registry import get_adapter, list_adapters

__all__ = [
    "AdapterCapabilities",
    "CatalogPlacementAdapter",
    "GenerationAdapter",
    "GeneratorIOContract",
    "I1_NATURAL_LANGUAGE",
    "I2_NATURAL_LANGUAGE_STRUCTURE",
    "O1_OBJECT_STATE",
    "O2_SCENE_PROGRAM",
    "O3_SCENE_PACKAGE",
    "OutputMaterializationRequired",
    "get_adapter",
    "list_adapters",
]
