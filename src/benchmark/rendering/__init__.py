"""Rendering backends for canonical generated scenes."""

from benchmark.rendering.blender import CYCLES_DEVICES, RENDER_ENGINES, BlenderRenderError, BlenderRenderer
from benchmark.rendering.camera_pose import (
    CAMERA_ACTIONS,
    CAMERA_POSE_MODES,
    DEFAULT_CAMERA_MODE_BY_METRIC,
    P0B_CAMERA_METRICS,
    parse_metric_camera_modes,
)

__all__ = [
    "CAMERA_ACTIONS",
    "CAMERA_POSE_MODES",
    "DEFAULT_CAMERA_MODE_BY_METRIC",
    "P0B_CAMERA_METRICS",
    "CYCLES_DEVICES",
    "RENDER_ENGINES",
    "BlenderRenderError",
    "BlenderRenderer",
    "parse_metric_camera_modes",
]
