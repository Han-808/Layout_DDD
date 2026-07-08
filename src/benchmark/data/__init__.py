"""Benchmark case loading utilities."""

from benchmark.data.adapters import (
    DATASET_ADAPTERS,
    CaseRef,
    create_dataset_adapter,
    discover_and_normalize_cases,
)
from benchmark.data.load_cases import iter_case_paths, load_case, load_cases
from benchmark.data.local_assets import LOCAL_ASSET_SOURCE, load_local_asset_index, resolve_local_asset_ref
from benchmark.data.local_scenes import LOCAL_SCENE_SOURCE, load_local_scene, load_local_scene_index, resolve_local_scene_ref
from benchmark.data.scene_adapters import (
    layout_to_scene,
    legend_layout_to_scene,
    normalize_scene,
    scene_to_case,
    scene_to_layout,
    scene_to_legend_layout,
)

__all__ = [
    "DATASET_ADAPTERS",
    "CaseRef",
    "create_dataset_adapter",
    "discover_and_normalize_cases",
    "iter_case_paths",
    "layout_to_scene",
    "load_local_asset_index",
    "load_local_scene",
    "load_local_scene_index",
    "LOCAL_ASSET_SOURCE",
    "LOCAL_SCENE_SOURCE",
    "legend_layout_to_scene",
    "load_case",
    "load_cases",
    "normalize_scene",
    "resolve_local_asset_ref",
    "resolve_local_scene_ref",
    "scene_to_case",
    "scene_to_layout",
    "scene_to_legend_layout",
]
