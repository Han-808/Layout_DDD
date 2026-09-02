"""Shared primitives for external scene-generation harness adapters."""

from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.assets import (
    ASSET_RESOLUTION_ALLOW_RETRIEVAL,
    ASSET_RESOLUTION_EXACT_ONLY,
    AssetProvider,
    DatasetRetrievalAssetProvider,
    MappingAssetProvider,
    asset_fields,
    asset_resolution_policy,
    load_asset_provider,
    resolve_asset_record,
)
from benchmark.adapters.common.scene_state import convert_scene_state

__all__ = [
    "ASSET_RESOLUTION_ALLOW_RETRIEVAL",
    "ASSET_RESOLUTION_EXACT_ONLY",
    "AssetProvider",
    "DatasetRetrievalAssetProvider",
    "HarnessConverterAdapter",
    "MappingAssetProvider",
    "SINGLE_ROOM_HARNESS_CAPABILITIES",
    "asset_fields",
    "asset_resolution_policy",
    "convert_scene_state",
    "load_asset_provider",
    "resolve_asset_record",
]
