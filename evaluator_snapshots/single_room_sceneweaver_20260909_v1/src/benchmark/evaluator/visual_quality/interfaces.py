"""Deprecated compatibility shim for the former "Visual Quality" L3 interfaces.

The L3 layer was reframed from narrow "Visual Quality" to "Scene Quality" with
two subfamilies (Semantic Coherence and Perceptual Visual Quality). The canonical
implementation now lives in :mod:`benchmark.evaluator.scene_quality.interfaces`.

This module re-exports the canonical API under the historical ``visual_quality``
names so existing imports keep working. Prefer the ``scene_quality`` names in new
code. ``object_pairing_consistency`` is canonical; ``object_coexistence_consistency``
is a backward-compatible metric alias.
"""

from __future__ import annotations

from benchmark.evaluator.scene_quality.interfaces import (
    CAMERA_MODES,
    CAMERA_SCOPES,
    DEFAULT_SCENE_QUALITY_INTERFACE_CONFIG as DEFAULT_VISUAL_QUALITY_INTERFACE_CONFIG,
    EVIDENCE_SELECTORS,
    IMAGE_ORDER_TOKENS,
    PRESENTATIONS,
    SCENE_QUALITY_INTERFACE_METRICS as VISUAL_QUALITY_INTERFACE_METRICS,
    SCENE_QUALITY_INTERFACE_NAMESPACE as VISUAL_QUALITY_INTERFACE_NAMESPACE,
    SCENE_QUALITY_INTERFACE_VERSION as VISUAL_QUALITY_INTERFACE_VERSION,
    SceneQualityInterfaceConfigError as VisualQualityInterfaceConfigError,
    evaluate_scene_quality_interfaces as evaluate_visual_quality_interfaces,
    resolve_scene_quality_config as resolve_visual_quality_config,
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
