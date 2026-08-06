"""Canonical L3 Scene Quality evaluator.

Scene Quality reframes the former narrow "Visual Quality" layer into two
subfamilies: Semantic Coherence (``scale_consistency``,
``object_pairing_consistency``) and Perceptual Visual Quality
(``style_consistency``), with opt-in Functional Validity
(``functional_consistency``) and Semantic Placement
(``semantic_placement_consistency``). The optional functional and placement
metrics first judge the scene-global scope, including cross-group relations,
then review every non-singleton group locally. The evaluator consumes
scope-correct prepared visual evidence, the canonical grouping report,
asset-policy applicability, and an injected strict VLM judge. It scores
complete applicable metrics and preserves missing evidence or malformed
judgements as unresolved.
"""

from benchmark.evaluator.scene_quality.authorized_deviations import (
    AUTHORIZED_DEVIATION_SOURCES,
    AuthorizedDeviationError,
    deviation_matches,
    deviations_for_metric,
    validate_authorized_deviations,
)
from benchmark.evaluator.scene_quality.interfaces import (
    CAMERA_MODES,
    CAMERA_SCOPES,
    DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG,
    EVIDENCE_SELECTORS,
    FUNCTIONAL_VALIDITY,
    FUNCTIONAL_VALIDITY_METRICS,
    IMAGE_ORDER_TOKENS,
    JUDGMENT_SCOPE_BY_METRIC,
    PERCEPTUAL_VISUAL_QUALITY,
    PERCEPTUAL_VISUAL_QUALITY_METRICS,
    PRESENTATIONS,
    SCENE_QUALITY_INTERFACE_METRICS,
    SCENE_QUALITY_INTERFACE_NAMESPACE,
    SCENE_QUALITY_INTERFACE_VERSION,
    SEMANTIC_PLACEMENT,
    SEMANTIC_PLACEMENT_METRICS,
    SEMANTIC_COHERENCE,
    SEMANTIC_COHERENCE_METRICS,
    SUBFAMILY_BY_METRIC,
    SceneQualityInterfaceConfigError,
    evaluate_scene_quality_interfaces,
    resolve_scene_quality_config,
)

__all__ = [
    "AUTHORIZED_DEVIATION_SOURCES",
    "AuthorizedDeviationError",
    "CAMERA_MODES",
    "CAMERA_SCOPES",
    "DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG",
    "EVIDENCE_SELECTORS",
    "FUNCTIONAL_VALIDITY",
    "FUNCTIONAL_VALIDITY_METRICS",
    "IMAGE_ORDER_TOKENS",
    "JUDGMENT_SCOPE_BY_METRIC",
    "PERCEPTUAL_VISUAL_QUALITY",
    "PERCEPTUAL_VISUAL_QUALITY_METRICS",
    "PRESENTATIONS",
    "SCENE_QUALITY_INTERFACE_METRICS",
    "SCENE_QUALITY_INTERFACE_NAMESPACE",
    "SCENE_QUALITY_INTERFACE_VERSION",
    "SEMANTIC_PLACEMENT",
    "SEMANTIC_PLACEMENT_METRICS",
    "SEMANTIC_COHERENCE",
    "SEMANTIC_COHERENCE_METRICS",
    "SUBFAMILY_BY_METRIC",
    "SceneQualityInterfaceConfigError",
    "deviation_matches",
    "deviations_for_metric",
    "evaluate_scene_quality_interfaces",
    "resolve_scene_quality_config",
    "validate_authorized_deviations",
]
