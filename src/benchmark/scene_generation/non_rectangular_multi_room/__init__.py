"""Additive global generation mode for non-rectangular multi-room layouts."""

from benchmark.scene_generation.non_rectangular_multi_room.contracts import (
    GENERATION_MODE,
    GENERATION_MODE_V2,
    GLOBAL_PLACEMENT_SCHEMA_VERSION,
    NonRectangularGenerationContractError,
    build_global_retrieval_plan,
    build_stage_a_user_value,
    build_stage_a_user_value_v2,
    build_stage_c_user_value,
    build_stage_c_user_value_v2,
    group_asset_selection,
    materialize_generated_scene,
    validate_global_placement,
    validate_stage_a_artifacts,
)
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    COMPILED_ARCHITECTURE_SCHEMA_VERSION,
    NonRectangularArchitectureError,
    build_polygon_architecture,
)
from benchmark.scene_generation.non_rectangular_multi_room.artifacts import (
    NonRectangularGenerationArtifactError,
    NonRectangularGenerationArtifacts,
)
from benchmark.scene_generation.non_rectangular_multi_room.runtime import (
    NonRectangularGenerationRuntimeError,
    run_non_rectangular_generation,
    run_non_rectangular_generation_v2,
)


__all__ = [
    "GENERATION_MODE",
    "GENERATION_MODE_V2",
    "GLOBAL_PLACEMENT_SCHEMA_VERSION",
    "COMPILED_ARCHITECTURE_SCHEMA_VERSION",
    "NonRectangularArchitectureError",
    "NonRectangularGenerationArtifactError",
    "NonRectangularGenerationArtifacts",
    "NonRectangularGenerationRuntimeError",
    "NonRectangularGenerationContractError",
    "build_global_retrieval_plan",
    "build_polygon_architecture",
    "build_stage_a_user_value",
    "build_stage_a_user_value_v2",
    "build_stage_c_user_value",
    "build_stage_c_user_value_v2",
    "group_asset_selection",
    "materialize_generated_scene",
    "run_non_rectangular_generation",
    "run_non_rectangular_generation_v2",
    "validate_global_placement",
    "validate_stage_a_artifacts",
]
