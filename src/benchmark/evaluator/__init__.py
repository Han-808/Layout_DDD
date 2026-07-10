"""Current deterministic bbox/OBB proxy evaluators."""

from benchmark.evaluator.OAR.evaluator import DEFAULT_OAR_CONFIG, evaluate_oar
from benchmark.evaluator.OOR.evaluator import DEFAULT_OOR_CONFIG, DETERMINISTIC_ONLY, evaluate_oor
from benchmark.evaluator.generic_validity.evaluator import DEFAULT_GENERIC_VALIDITY_CONFIG, evaluate_generic_validity, evaluate_scene_validity

__all__ = [
    "DEFAULT_GENERIC_VALIDITY_CONFIG",
    "DEFAULT_OAR_CONFIG",
    "DEFAULT_OOR_CONFIG",
    "DETERMINISTIC_ONLY",
    "evaluate_generic_validity",
    "evaluate_oar",
    "evaluate_oor",
    "evaluate_scene_validity",
]
