"""Canonical scene-construction artifact I/O helpers."""

from benchmark.scene_io.normalize import normalize_object, normalize_scene
from benchmark.scene_io.validate import (
    ArtifactValidationError,
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)

__all__ = [
    "ArtifactValidationError",
    "normalize_object",
    "normalize_scene",
    "validate_asset_selection",
    "validate_generated_scene",
    "validate_generation_input",
    "validate_object_plan",
    "validate_scene_request",
]
