"""Compatibility namespace for legacy dataset adapters.

New input work should enter through ``benchmark.nl_scene``. HSSD-HAB and other
pre-structured benchmark case adapters are retained under ``benchmark.legend``.
"""

from __future__ import annotations

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"
