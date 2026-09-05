"""Generation harness registry and canonical-output compatibility contracts.

Every adapter selects one output route before evaluation: an existing loader
for a supported representation, or a harness-specific converter for a foreign
representation. Concrete external harness adapters can be registered without
adding source-specific branches to the evaluator.
"""

from benchmark.adapters.base import (
    AdapterCapabilities,
    GenerationAdapter,
    OutputMaterializationRequired,
    SceneCompatibilityRequirements,
)
from benchmark.adapters.catalog_placement import CatalogPlacementAdapter
from benchmark.adapters.common import (
    ASSET_RESOLUTION_ALLOW_RETRIEVAL,
    ASSET_RESOLUTION_EXACT_ONLY,
    AssetProvider,
    DatasetRetrievalAssetProvider,
    ExternalExecutionError,
    MappingAssetProvider,
)
from benchmark.adapters.direct_layout import DirectLayoutAdapter
from benchmark.adapters.holodeck import HolodeckAdapter
from benchmark.adapters.layout_gpt import LayoutGPTAdapter
from benchmark.adapters.layout_vlm import LayoutVLMAdapter
from benchmark.adapters.output_routing import (
    OUTPUT_CONVERTER,
    OUTPUT_LOADER,
    OutputIngestionKind,
    SceneOutputRoute,
)
from benchmark.adapters.registry import (
    AdapterRegistry,
    get_adapter,
    list_adapters,
    register_adapter,
)
from benchmark.adapters.respace import ReSpaceAdapter
from benchmark.adapters.scene_smith import SceneSmithAdapter
from benchmark.adapters.scene_weaver import SceneWeaverAdapter
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
    O2_SCENE_PROGRAM,
    O3_SCENE_PACKAGE,
    GeneratorIOContract,
)

__all__ = [
    "ASSET_RESOLUTION_ALLOW_RETRIEVAL",
    "ASSET_RESOLUTION_EXACT_ONLY",
    "AdapterCapabilities",
    "AdapterRegistry",
    "AssetProvider",
    "DatasetRetrievalAssetProvider",
    "ExternalExecutionError",
    "CatalogPlacementAdapter",
    "DirectLayoutAdapter",
    "GenerationAdapter",
    "GeneratorIOContract",
    "I1_NATURAL_LANGUAGE",
    "I2_NATURAL_LANGUAGE_STRUCTURE",
    "HolodeckAdapter",
    "LayoutGPTAdapter",
    "LayoutVLMAdapter",
    "MappingAssetProvider",
    "O1_OBJECT_STATE",
    "O2_SCENE_PROGRAM",
    "O3_SCENE_PACKAGE",
    "OUTPUT_CONVERTER",
    "OUTPUT_LOADER",
    "OutputIngestionKind",
    "OutputMaterializationRequired",
    "ReSpaceAdapter",
    "SceneCompatibilityRequirements",
    "SceneOutputRoute",
    "SceneSmithAdapter",
    "SceneWeaverAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
