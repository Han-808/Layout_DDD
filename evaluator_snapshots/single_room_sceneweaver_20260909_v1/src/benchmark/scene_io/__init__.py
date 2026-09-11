"""Canonical scene-construction artifact I/O helpers."""

from benchmark.scene_io.normalize import bind_scene_to_generation_request, normalize_object, normalize_scene
from benchmark.scene_io.object_normalization import NormalizedObject, normalize_object as normalize_geometry_object, normalize_objects
from benchmark.scene_io.validate import (
    ArtifactValidationError,
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_package,
    validate_scene_request,
)

__all__ = [
    "ArtifactValidationError",
    "NormalizedObject",
    "bind_scene_to_generation_request",
    "normalize_geometry_object",
    "normalize_object",
    "normalize_objects",
    "normalize_scene",
    "validate_asset_selection",
    "validate_generated_scene",
    "validate_generation_input",
    "validate_object_plan",
    "validate_scene_package",
    "validate_scene_request",
]
