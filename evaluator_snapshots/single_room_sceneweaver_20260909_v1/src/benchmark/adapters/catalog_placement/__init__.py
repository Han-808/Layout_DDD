"""Fixed-catalog placement adapter."""

from benchmark.adapters.catalog_placement.adapter import CatalogPlacementAdapter
from benchmark.adapters.catalog_placement.converter import (
    build_catalog_instance_registry,
    convert_catalog_placement_to_scene,
    extract_catalog_placement,
    public_slot_ids_from_generation_input,
    public_task_slots_from_generation_input,
    selected_asset_ids_from_generation_input,
    validate_catalog_placement,
)
from benchmark.adapters.catalog_placement.prompt import (
    CATALOG_PLACEMENT_VERSION,
    build_catalog_placement_method_input,
)

__all__ = [
    "CATALOG_PLACEMENT_VERSION",
    "CatalogPlacementAdapter",
    "build_catalog_instance_registry",
    "build_catalog_placement_method_input",
    "convert_catalog_placement_to_scene",
    "extract_catalog_placement",
    "public_slot_ids_from_generation_input",
    "public_task_slots_from_generation_input",
    "selected_asset_ids_from_generation_input",
    "validate_catalog_placement",
]
