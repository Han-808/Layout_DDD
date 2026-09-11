"""Compatibility import for the deprecated deterministic grouping replay.

The active evaluator uses :mod:`benchmark.grouping.vlm`. This old path remains
available only to historical ``benchmark.legend`` callers without maintaining
a second legacy implementation.
"""

from benchmark.evaluator.object_grouping import (
    ResolvedGroupingConfig,
    build_object_grouping_report,
    build_object_groups,
    resolve_grouping_config,
)

__all__ = [
    "ResolvedGroupingConfig",
    "build_object_grouping_report",
    "build_object_groups",
    "resolve_grouping_config",
]
