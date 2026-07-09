"""Compatibility wrapper for legend HSSD relation-cue heuristics.

These helpers belong to the legacy HSSD input adapter. The current input path
is natural language via ``benchmark.nl_scene``.
"""

from __future__ import annotations

import warnings

from benchmark.legend.hssd import estimated_relations as _legend
from benchmark.legend.hssd.estimated_relations import *  # noqa: F401,F403

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"


def _warn_legend_import() -> None:
    warnings.warn(
        "benchmark.datasets.estimated_relations is a legend compatibility wrapper; "
        "use benchmark.legend.hssd.estimated_relations for HSSD adapters or "
        "benchmark.nl_scene for the current natural-language input path.",
        DeprecationWarning,
        stacklevel=2,
    )


def build_estimated_spatial_cues(*args, **kwargs):
    _warn_legend_import()
    return _legend.build_estimated_spatial_cues(*args, **kwargs)


def cue_counts_by_type(*args, **kwargs):
    _warn_legend_import()
    return _legend.cue_counts_by_type(*args, **kwargs)


def relation_policy_metadata(*args, **kwargs):
    _warn_legend_import()
    return _legend.relation_policy_metadata(*args, **kwargs)


def compatibility_relations(*args, **kwargs):
    _warn_legend_import()
    return _legend.compatibility_relations(*args, **kwargs)
