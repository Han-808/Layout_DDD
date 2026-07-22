"""Coarse-grained Spatial Fidelity evaluator.

Scale and co-occurrence use SceneOnto-style statistical evidence with
conservative coverage and candidate routing. Functional Grouping is retained
as an explicit, zero-weight placeholder.
"""

from benchmark.evaluator.spatial_fidelity.cooccurrence import (
    COOCCURRENCE_EVALUATOR_VERSION,
    DEFAULT_COOCCURRENCE_CONFIG,
    evaluate_cooccurrence,
)
from benchmark.evaluator.spatial_fidelity.evaluator import (
    DEFAULT_SPATIAL_FIDELITY_CONFIG,
    SPATIAL_FIDELITY_EVALUATOR_VERSION,
    evaluate_spatial_fidelity,
)
from benchmark.evaluator.spatial_fidelity.functional_grouping import (
    DEFAULT_FUNCTIONAL_GROUPING_CONFIG,
    FUNCTIONAL_GROUPING_EVALUATOR_VERSION,
    evaluate_functional_grouping,
)
from benchmark.evaluator.spatial_fidelity.ontology import (
    DEFAULT_CATEGORY_ALIASES,
    CategoryResolution,
    CooccurrenceRecord,
    OntologyIndex,
    load_ontology,
    normalize_category_label,
)
from benchmark.evaluator.spatial_fidelity.scale import (
    DEFAULT_SCALE_CONFIG,
    SCALE_EVALUATOR_VERSION,
    evaluate_scale,
)

__all__ = [
    "COOCCURRENCE_EVALUATOR_VERSION",
    "DEFAULT_CATEGORY_ALIASES",
    "DEFAULT_COOCCURRENCE_CONFIG",
    "DEFAULT_FUNCTIONAL_GROUPING_CONFIG",
    "DEFAULT_SCALE_CONFIG",
    "DEFAULT_SPATIAL_FIDELITY_CONFIG",
    "FUNCTIONAL_GROUPING_EVALUATOR_VERSION",
    "SCALE_EVALUATOR_VERSION",
    "SPATIAL_FIDELITY_EVALUATOR_VERSION",
    "CategoryResolution",
    "CooccurrenceRecord",
    "OntologyIndex",
    "evaluate_cooccurrence",
    "evaluate_functional_grouping",
    "evaluate_scale",
    "evaluate_spatial_fidelity",
    "load_ontology",
    "normalize_category_label",
]

