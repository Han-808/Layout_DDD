"""Compatibility wrapper for the legend HSSD-HAB small-scene selector.

The current input path is natural language via ``benchmark.nl_scene``. Import
``benchmark.legend.hssd.hssd_small_selector`` for legacy HSSD selection code.
"""

from __future__ import annotations

import warnings

from benchmark.legend.hssd import hssd_small_selector as _legend
from benchmark.legend.hssd.hssd_small_selector import *  # noqa: F401,F403

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"


def _warn_legend_import() -> None:
    warnings.warn(
        "benchmark.datasets.hssd_small_selector is a legend compatibility wrapper; "
        "use benchmark.legend.hssd.hssd_small_selector for HSSD adapters or "
        "benchmark.nl_scene for the current natural-language input path.",
        DeprecationWarning,
        stacklevel=2,
    )


def iter_hssd_scene_candidates(*args, **kwargs):
    _warn_legend_import()
    return _legend.iter_hssd_scene_candidates(*args, **kwargs)


def select_natural_small_hssd_scene(*args, **kwargs):
    _warn_legend_import()
    return _legend.select_natural_small_hssd_scene(*args, **kwargs)


def convert_selected_small_hssd_scene(*args, **kwargs):
    _warn_legend_import()
    return _legend.convert_selected_small_hssd_scene(*args, **kwargs)
