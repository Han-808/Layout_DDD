"""Default and replay-only adapter routing.

The fixed-catalog generator path owns only instance identity, asset selection,
and rigid placement.  ``layout_json`` remains available for byte-for-byte
historical replay, but is intentionally not the default for new generation.
"""

DEFAULT_GENERATION_ADAPTER = "catalog_placement"
LEGACY_LAYOUT_REPLAY_ADAPTER = "layout_json"


__all__ = [
    "DEFAULT_GENERATION_ADAPTER",
    "LEGACY_LAYOUT_REPLAY_ADAPTER",
]
