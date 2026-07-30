"""Deprecated compatibility package for the former "Visual Quality" L3 layer.

The canonical L3 layer is now :mod:`benchmark.evaluator.scene_quality`. This
package re-exports the canonical API under the historical ``visual_quality``
names so that existing imports continue to work. Prefer ``scene_quality`` in new
code.
"""

from benchmark.evaluator.visual_quality.interfaces import (
    CAMERA_MODES,
    CAMERA_SCOPES,
    DEFAULT_VISUAL_QUALITY_INTERFACE_CONFIG,
    EVIDENCE_SELECTORS,
    IMAGE_ORDER_TOKENS,
    PRESENTATIONS,
    VISUAL_QUALITY_INTERFACE_METRICS,
    VISUAL_QUALITY_INTERFACE_NAMESPACE,
    VISUAL_QUALITY_INTERFACE_VERSION,
    VisualQualityInterfaceConfigError,
    evaluate_visual_quality_interfaces,
    resolve_visual_quality_config,
)

__all__ = [
    "CAMERA_MODES",
    "CAMERA_SCOPES",
    "DEFAULT_VISUAL_QUALITY_INTERFACE_CONFIG",
    "EVIDENCE_SELECTORS",
    "IMAGE_ORDER_TOKENS",
    "PRESENTATIONS",
    "VISUAL_QUALITY_INTERFACE_METRICS",
    "VISUAL_QUALITY_INTERFACE_NAMESPACE",
    "VISUAL_QUALITY_INTERFACE_VERSION",
    "VisualQualityInterfaceConfigError",
    "evaluate_visual_quality_interfaces",
    "resolve_visual_quality_config",
]
