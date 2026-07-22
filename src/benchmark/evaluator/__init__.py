"""Current canonical scene evaluators and alignment infrastructure."""

from benchmark.evaluator.OAR.evaluator import DEFAULT_OAR_CONFIG, evaluate_oar
from benchmark.evaluator.OOR.evaluator import DEFAULT_OOR_CONFIG, DETERMINISTIC_ONLY, evaluate_oor
from benchmark.evaluator.generic_validity.evaluator import DEFAULT_GENERIC_VALIDITY_CONFIG, evaluate_generic_validity, evaluate_scene_validity
from benchmark.evaluator.object_alignment import evaluate_object_alignment
from benchmark.evaluator.object_mapping import (
    DEFAULT_OBJECT_MAPPING_CONFIG,
    evaluate_object_mapping,
    route_relationship_intents,
)
from benchmark.evaluator.profile import (
    COARSE_GRAINED_MODE,
    DEFAULT_EVALUATION_PROFILE,
    EVALUATION_MODES,
    FINE_GRAINED_MODE,
    build_evaluation_plan,
    evaluation_mode_for_prompt_granularity,
    resolve_evaluation_profile,
)
from benchmark.evaluator.spatial_fidelity import (
    DEFAULT_SPATIAL_FIDELITY_CONFIG,
    evaluate_spatial_fidelity,
)

__all__ = [
    "DEFAULT_GENERIC_VALIDITY_CONFIG",
    "DEFAULT_OAR_CONFIG",
    "DEFAULT_OBJECT_MAPPING_CONFIG",
    "DEFAULT_OOR_CONFIG",
    "DEFAULT_SPATIAL_FIDELITY_CONFIG",
    "DETERMINISTIC_ONLY",
    "DEFAULT_EVALUATION_PROFILE",
    "COARSE_GRAINED_MODE",
    "EVALUATION_MODES",
    "FINE_GRAINED_MODE",
    "build_evaluation_plan",
    "evaluate_generic_validity",
    "evaluate_oar",
    "evaluate_object_alignment",
    "evaluate_object_mapping",
    "evaluate_oor",
    "evaluate_scene_validity",
    "evaluate_spatial_fidelity",
    "evaluation_mode_for_prompt_granularity",
    "route_relationship_intents",
    "resolve_evaluation_profile",
]
