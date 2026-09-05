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
from benchmark.adapters.common.execution import (
    ExternalExecutionError,
    artifact_sha256,
    execute_external_harness,
    preserve_native_artifact,
)
from benchmark.adapters.common.scene_state import convert_scene_state

__all__ = [
    "ASSET_RESOLUTION_ALLOW_RETRIEVAL",
    "ASSET_RESOLUTION_EXACT_ONLY",
    "AssetProvider",
    "DatasetRetrievalAssetProvider",
    "ExternalExecutionError",
    "HarnessConverterAdapter",
    "MappingAssetProvider",
    "SINGLE_ROOM_HARNESS_CAPABILITIES",
    "asset_fields",
    "artifact_sha256",
    "asset_resolution_policy",
    "convert_scene_state",
    "execute_external_harness",
    "load_asset_provider",
    "preserve_native_artifact",
    "resolve_asset_record",
]
