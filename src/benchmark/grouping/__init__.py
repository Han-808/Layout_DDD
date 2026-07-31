"""Replaceable object grouping backends for evidence localization.

Grouping produces a complete object partition. It does not score a benchmark
metric, adjudicate functional plausibility, or mutate the scene.
"""

from benchmark.grouping.anchor import (
    ANCHOR_GROUPING_POLICY_ID,
    DEFAULT_ANCHOR_GROUPING_CONFIG,
    AnchorGroupingAlgorithm,
)
from benchmark.grouping.factory import (
    ACTIVE_GROUPING_BACKENDS,
    DEFAULT_GROUPING_BACKEND,
    DEPRECATED_GROUPING_BACKENDS,
    GROUPING_BACKENDS,
    build_grouping_algorithm,
    group_scene,
)
from benchmark.grouping.interfaces import (
    GROUPING_ROLE,
    GroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
    ObjectGroup,
)
from benchmark.grouping.scene import (
    DERIVED_OBJECT_ID_POLICY,
    NormalizedGroupingScene,
    normalize_grouping_scene,
)
from benchmark.grouping.topology import (
    TOPOLOGY_GROUPING_POLICY_ID,
    TopologyGroupingAlgorithm,
)
from benchmark.grouping.vlm import (
    DEFAULT_VLM_GROUPING_CONFIG,
    VLM_GROUPING_POLICY_ID,
    VLM_GROUPING_PROMPT_VERSION,
    VLMGroupingAlgorithm,
)


__all__ = [
    "ANCHOR_GROUPING_POLICY_ID",
    "ACTIVE_GROUPING_BACKENDS",
    "DEFAULT_ANCHOR_GROUPING_CONFIG",
    "DEFAULT_GROUPING_BACKEND",
    "DEPRECATED_GROUPING_BACKENDS",
    "DEFAULT_VLM_GROUPING_CONFIG",
    "DERIVED_OBJECT_ID_POLICY",
    "GROUPING_BACKENDS",
    "GROUPING_ROLE",
    "GroupingAlgorithm",
    "GroupingRequest",
    "GroupingResult",
    "NormalizedGroupingScene",
    "ObjectGroup",
    "TOPOLOGY_GROUPING_POLICY_ID",
    "VLM_GROUPING_POLICY_ID",
    "VLM_GROUPING_PROMPT_VERSION",
    "AnchorGroupingAlgorithm",
    "TopologyGroupingAlgorithm",
    "VLMGroupingAlgorithm",
    "build_grouping_algorithm",
    "group_scene",
    "normalize_grouping_scene",
]
