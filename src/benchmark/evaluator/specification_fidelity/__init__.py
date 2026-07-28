"""Canonical L2 Specification Fidelity contracts and runtime.

OOR, OAR, and ``functional_semantic_fidelity`` are the only canonical scoring
families. Room/scene type, broad intent, required areas, and explicitly
requested local functionality are components of the implemented
functional-semantic family rather than independent weighted metrics.
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
from benchmark.evaluator.specification_fidelity.contract import (
    ACCEPTED_SPECIFICATION_CLAIM_FAMILIES,
    FINE_DETAIL_CLAIM_FAMILIES,
    HIGH_LEVEL_CLAIM_FAMILIES,
    SPECIFICATION_CLAIM_FAMILIES,
    SPECIFICATION_CONTRACT_VERSION,
    TRUSTED_CONTRACT_SOURCES,
    SpecificationContractError,
    build_specification_fidelity_report,
    compile_specification_evaluation_plan,
    specification_contract_from_reference_annotation,
    validate_specification_contract,
)

__all__ = [
    "DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG",
    "ACCEPTED_SPECIFICATION_CLAIM_FAMILIES",
    "FUNCTIONAL_SEMANTIC_FIDELITY",
    "FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE",
    "FUNCTIONAL_SEMANTIC_INTERFACE_VERSION",
    "FUNCTIONAL_SEMANTIC_METRICS",
    "FunctionalSemanticConfigError",
    "FINE_DETAIL_CLAIM_FAMILIES",
    "HIGH_LEVEL_CLAIM_FAMILIES",
    "SPECIFICATION_CLAIM_FAMILIES",
    "SPECIFICATION_CONTRACT_VERSION",
    "TRUSTED_CONTRACT_SOURCES",
    "SpecificationContractError",
    "build_specification_fidelity_report",
    "compile_specification_evaluation_plan",
    "evaluate_functional_semantic_fidelity",
    "resolve_functional_semantic_config",
    "specification_contract_from_reference_annotation",
    "validate_specification_contract",
]
