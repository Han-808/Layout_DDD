"""Deprecated import shim for the former coarse-specification interface.

The only active implementation is
``benchmark.evaluator.specification_fidelity.functional_semantic``.  This
module exists solely so historical experiment code can still import its old
symbols; it does not define or select a second evaluator workflow.
"""

from benchmark.evaluator.specification_fidelity.functional_semantic import (
    DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG,
    FUNCTIONAL_SEMANTIC_FIDELITY,
    FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE,
    FUNCTIONAL_SEMANTIC_INTERFACE_VERSION,
    FUNCTIONAL_SEMANTIC_METRICS,
    FunctionalSemanticConfigError,
    evaluate_functional_semantic_fidelity,
    resolve_functional_semantic_config,
)

COARSE_SPECIFICATION_INTERFACE_VERSION = FUNCTIONAL_SEMANTIC_INTERFACE_VERSION
COARSE_SPECIFICATION_INTERFACE_NAMESPACE = "coarse_specification_interfaces"
COARSE_SPECIFICATION_METRICS = FUNCTIONAL_SEMANTIC_METRICS
COARSE_METRIC_ALIASES = {
    "room_scene_type": FUNCTIONAL_SEMANTIC_FIDELITY,
    "broad_semantic_intent": FUNCTIONAL_SEMANTIC_FIDELITY,
    "visual_functional_intent": FUNCTIONAL_SEMANTIC_FIDELITY,
    "required_functional_areas": FUNCTIONAL_SEMANTIC_FIDELITY,
    "required_zones": FUNCTIONAL_SEMANTIC_FIDELITY,
    "local_functionality": FUNCTIONAL_SEMANTIC_FIDELITY,
}
COARSE_ALIASES_BY_CANONICAL = {
    FUNCTIONAL_SEMANTIC_FIDELITY: list(COARSE_METRIC_ALIASES)
}
DEFAULT_COARSE_SPECIFICATION_CONFIG = DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG
CoarseSpecificationConfigError = FunctionalSemanticConfigError
resolve_coarse_specification_config = resolve_functional_semantic_config
evaluate_coarse_specification_interfaces = evaluate_functional_semantic_fidelity


def normalize_coarse_metric_name(name: str) -> str:
    """Deprecated helper; canonical config resolution never calls it."""

    return COARSE_METRIC_ALIASES.get(str(name), str(name))


# Historical import name retained only inside this deprecated shim.
normalize_functional_semantic_metric_name = normalize_coarse_metric_name

__all__ = [
    "COARSE_ALIASES_BY_CANONICAL",
    "COARSE_METRIC_ALIASES",
    "COARSE_SPECIFICATION_INTERFACE_NAMESPACE",
    "COARSE_SPECIFICATION_INTERFACE_VERSION",
    "COARSE_SPECIFICATION_METRICS",
    "CoarseSpecificationConfigError",
    "DEFAULT_COARSE_SPECIFICATION_CONFIG",
    "DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG",
    "FUNCTIONAL_SEMANTIC_FIDELITY",
    "FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE",
    "FUNCTIONAL_SEMANTIC_INTERFACE_VERSION",
    "FUNCTIONAL_SEMANTIC_METRICS",
    "FunctionalSemanticConfigError",
    "evaluate_coarse_specification_interfaces",
    "evaluate_functional_semantic_fidelity",
    "normalize_coarse_metric_name",
    "normalize_functional_semantic_metric_name",
    "resolve_coarse_specification_config",
    "resolve_functional_semantic_config",
]
