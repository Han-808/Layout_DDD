"""Legend HSSD-HAB input adapters.

HSSD-HAB conversion is retained only as a legacy/benchmark adapter. The active
repo input path is natural-language scene construction under
``benchmark.nl_scene``.
"""

from __future__ import annotations

from benchmark.legend.hssd.estimated_relations import (
    build_estimated_spatial_cues,
    compatibility_relations,
    cue_counts_by_type,
    relation_policy_metadata,
)
from benchmark.legend.hssd.hssd_hab_converter import convert_hssd_hab, iter_scene_instance_paths
from benchmark.legend.hssd.hssd_small_selector import (
    HSSDSmallSceneCandidate,
    convert_selected_small_hssd_scene,
    iter_hssd_scene_candidates,
    select_natural_small_hssd_scene,
)

__all__ = [
    "HSSDSmallSceneCandidate",
    "build_estimated_spatial_cues",
    "compatibility_relations",
    "convert_hssd_hab",
    "convert_selected_small_hssd_scene",
    "cue_counts_by_type",
    "iter_hssd_scene_candidates",
    "iter_scene_instance_paths",
    "relation_policy_metadata",
    "select_natural_small_hssd_scene",
]
