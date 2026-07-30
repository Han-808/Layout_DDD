"""Compatibility import for the canonical object-grouping implementation.

New evaluator code must import :mod:`benchmark.evaluator.object_grouping`.
The old path remains available to historical callers without maintaining a
second implementation.
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
