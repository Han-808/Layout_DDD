"""Rendering backends for canonical generated scenes."""

from benchmark.rendering.blender import CYCLES_DEVICES, RENDER_ENGINES, BlenderRenderError, BlenderRenderer
from benchmark.rendering.browser import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
    CONTROLLED_CAMERA_VIEW_FAMILY,
    DEFAULT_VIEWS as BROWSER_DEFAULT_VIEWS,
    BrowserRenderError,
    FrozenBrowserCaptureRenderer,
    HeadlessBrowserRenderer,
    UnsupportedBrowserRenderPipelineError,
)
from benchmark.rendering.camera_pose import (
    CAMERA_ACTIONS,
    CAMERA_ACTION_PARAMETERS,
    CAMERA_ACTION_PROTOCOL_VERSION,
    CAMERA_POSE_MODES,
    DEFAULT_CAMERA_MODE_BY_METRIC,
    P0B_CAMERA_METRICS,
    parse_metric_camera_modes,
)

__all__ = [
    "CAMERA_ACTIONS",
    "CAMERA_ACTION_PARAMETERS",
    "CAMERA_ACTION_PROTOCOL_VERSION",
    "CAMERA_POSE_MODES",
    "DEFAULT_CAMERA_MODE_BY_METRIC",
    "P0B_CAMERA_METRICS",
    "BROWSER_DEFAULT_VIEWS",
    "BROWSER_RENDER_BACKEND",
    "CONTROLLED_CAMERA_APPEARANCE_FIDELITY",
    "CONTROLLED_CAMERA_VIEW_FAMILY",
    "CYCLES_DEVICES",
    "RENDER_ENGINES",
    "BlenderRenderError",
    "BlenderRenderer",
    "BrowserRenderError",
    "FrozenBrowserCaptureRenderer",
    "HeadlessBrowserRenderer",
    "UnsupportedBrowserRenderPipelineError",
    "parse_metric_camera_modes",
]
