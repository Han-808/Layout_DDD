"""Compatibility wrapper for the legend HSSD-HAB input converter.

The current input path is natural language via ``benchmark.nl_scene``. Import
``benchmark.legend.hssd.hssd_hab_converter`` for legacy HSSD conversion code.
"""

from __future__ import annotations

import warnings

from benchmark.legend.hssd import hssd_hab_converter as _legend
from benchmark.legend.hssd.hssd_hab_converter import *  # noqa: F401,F403

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"


def _warn_legend_import() -> None:
    warnings.warn(
        "benchmark.datasets.hssd_hab_converter is a legend compatibility wrapper; "
        "use benchmark.legend.hssd.hssd_hab_converter for HSSD adapters or "
        "benchmark.nl_scene for the current natural-language input path.",
        DeprecationWarning,
        stacklevel=2,
    )


def iter_scene_instance_paths(*args, **kwargs):
    _warn_legend_import()
    return _legend.iter_scene_instance_paths(*args, **kwargs)


def convert_hssd_hab(*args, **kwargs):
    _warn_legend_import()
    return _legend.convert_hssd_hab(*args, **kwargs)
