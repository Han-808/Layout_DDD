"""Canonical L3 Scene Quality evaluator.

Scene Quality reframes the former narrow "Visual Quality" layer into two
subfamilies: Semantic Coherence (``scale_consistency``,
``object_pairing_consistency``) and Perceptual Visual Quality
(``style_consistency``), Functional Validity (``functional_consistency``),
and Semantic Placement (``semantic_placement_consistency``). All five are
active benchmark metrics in the current profile. Functional and placement
first judge the scene-global scope. Functional consistency then routes
each discovered cross-group correspondence through its own Judge episode.
Both metrics next review every ordinarily eligible group plus any singleton
group that owns an explicit typed functional or placement check. The evaluator consumes
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
from benchmark.evaluator.scene_quality.functional_prejudgement import (
    DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG,
    DisabledFunctionalPrejudgementEvidenceSource,
    FrozenFunctionalPrejudgementEvidenceSource,
    FunctionalPrejudgementEvidenceRequest,
    FunctionalPrejudgementEvidenceResult,
    FunctionalPrejudgementEvidenceSource,
    RuntimeFunctionalPrejudgementEvidenceSource,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    ATYPICAL,
    CLEAR_SEMANTIC_MISPLACEMENT,
    IMPLAUSIBLE,
    LEGACY_PLACEMENT_SEVERITY_LEVELS,
    MATERIAL_CONTEXTUAL_MISMATCH,
    PLACEMENT_SEVERITY_LEVELS,
)

__all__ = [
    "ATYPICAL",
    "AUTHORIZED_DEVIATION_SOURCES",
    "AuthorizedDeviationError",
    "CAMERA_MODES",
    "CAMERA_SCOPES",
    "CLEAR_SEMANTIC_MISPLACEMENT",
    "DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG",
    "DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG",
    "DisabledFunctionalPrejudgementEvidenceSource",
    "EVIDENCE_SELECTORS",
    "FUNCTIONAL_VALIDITY",
    "FUNCTIONAL_VALIDITY_METRICS",
    "FrozenFunctionalPrejudgementEvidenceSource",
    "FunctionalPrejudgementEvidenceRequest",
    "FunctionalPrejudgementEvidenceResult",
    "FunctionalPrejudgementEvidenceSource",
    "IMAGE_ORDER_TOKENS",
    "IMPLAUSIBLE",
    "JUDGMENT_SCOPE_BY_METRIC",
    "LEGACY_PLACEMENT_SEVERITY_LEVELS",
    "MATERIAL_CONTEXTUAL_MISMATCH",
    "PERCEPTUAL_VISUAL_QUALITY",
    "PERCEPTUAL_VISUAL_QUALITY_METRICS",
    "PLACEMENT_SEVERITY_LEVELS",
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
    "RuntimeFunctionalPrejudgementEvidenceSource",
    "deviation_matches",
    "deviations_for_metric",
    "evaluate_scene_quality_interfaces",
    "resolve_scene_quality_config",
    "validate_authorized_deviations",
]
