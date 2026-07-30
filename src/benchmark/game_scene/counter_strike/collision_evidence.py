"""Pair-grounded collision evidence derived from one frozen browser capture.

The generic Game route can reuse a browser capture for rendering and can expose
its regional bank to the Scene Quality evaluator, but it is intentionally not a
P0b local-view provider.  Counter-Strike collision candidates therefore need a
small adapter:

* choose one already-frozen original-runtime regional view;
* keep that raw image unchanged;
* draw the two canonical object OBBs as a deterministic 2D annotation;
* return the same-pose raw/overlay pair to the existing collision judge.

The overlay is not described as a segmentation contour.  It is a projected OBB
wireframe and carries that limitation explicitly.  Pixels, scene geometry, and
camera poses all remain bound to the same hash-verified browser capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from benchmark.evaluator.generic_validity.geometry import (
    get_obb_corners,
    normalize_objects,
)
from benchmark.rendering.browser import FrozenBrowserCaptureRenderer

from .evidence import load_counter_strike_frozen_evidence
from .loader import CounterStrikeBenchmarkConfig


COUNTER_STRIKE_COLLISION_EVIDENCE_VERSION = (
    "counter_strike_frozen_collision_evidence_v1"
)
_OBB_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
_PAIR_COLORS = ((224, 55, 62), (40, 117, 224))


class CounterStrikeCollisionEvidenceError(RuntimeError):
    """Raised when a collision event cannot be grounded in the frozen bank."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(
            f"Counter-Strike collision evidence failed [{code}]: {message}"
        )


@dataclass(frozen=True)
class _ViewProjection:
    view_id: str
    path: Path
    pose: dict[str, Any]
    declared_target_ids: frozenset[str]
    scope: str


class CounterStrikeFrozenCaptureRenderer:
    """Trusted renderer plus a callable CS collision-evidence provider."""

    def __init__(
        self,
        *,
        capture_dir: str | Path,
        evidence_out_dir: str | Path,
        benchmark_config: CounterStrikeBenchmarkConfig,
    ) -> None:
        if not isinstance(benchmark_config, CounterStrikeBenchmarkConfig):
            raise TypeError(
                "benchmark_config must come from "
                "load_counter_strike_benchmark_config"
            )
        self._renderer = FrozenBrowserCaptureRenderer(capture_dir=capture_dir)
        # This validates every capture artifact and the exact frozen view bank.
        self._evidence = load_counter_strike_frozen_evidence(
            capture_dir,
            benchmark_config=benchmark_config,
        )
        self.capture_dir = self._evidence.capture_dir
        self.evidence_out_dir = Path(evidence_out_dir).expanduser().resolve()
        self.evidence_out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._renderer.manifest
        self.exported_scene = self._renderer.exported_scene
        self._objects, object_errors = normalize_objects(self.exported_scene)
        if object_errors:
            raise CounterStrikeCollisionEvidenceError(
                "canonical_objects_invalid",
                f"frozen exported scene has invalid objects: "
                f"{sorted(object_errors)}",
            )
        self._objects_by_id = {str(item.id): item for item in self._objects}
        self._views = _frozen_projections(
            self.manifest,
            capture_dir=self.capture_dir,
        )
        self.policy_config = {
            "provider_version": COUNTER_STRIKE_COLLISION_EVIDENCE_VERSION,
            "capture_manifest_sha256": self._evidence.manifest_sha256,
            "collision_view_source": "frozen_original_runtime_regional_bank",
            "collision_annotation": "canonical_obb_projection",
            "segmentation_contour_claimed": False,
            "same_capture_required": True,
        }

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        return self._renderer.render_scene(
            scene_path=scene_path,
            out_dir=out_dir,
            asset_root=asset_root,
        )

    def provide_scene_quality_evidence(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._renderer.provide_scene_quality_evidence(request)

    def __call__(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one same-pose raw/OBB-overlay pair for Collision only."""

        if not isinstance(request, dict):
            raise CounterStrikeCollisionEvidenceError(
                "request_invalid",
                "local evidence request must be a JSON object",
            )
        if request.get("metric") != "collision":
            return []
        raw_ids = request.get("object_ids")
        if (
            not isinstance(raw_ids, list)
            or len(raw_ids) != 2
            or len({str(value) for value in raw_ids}) != 2
        ):
            raise CounterStrikeCollisionEvidenceError(
                "target_pair_invalid",
                "collision request must contain two distinct object_ids",
            )
        object_ids = tuple(str(value) for value in raw_ids)
        missing = [
            object_id
            for object_id in object_ids
            if object_id not in self._objects_by_id
        ]
        if missing:
            raise CounterStrikeCollisionEvidenceError(
                "target_object_missing",
                f"collision targets are absent from the frozen scene: {missing}",
            )
        focus = _event_focus_point(
            request,
            objects=[self._objects_by_id[object_id] for object_id in object_ids],
        )
        chosen, projections = self._choose_view(object_ids, focus=focus)
        overlay_path = self._render_overlay(
            chosen,
            object_ids=object_ids,
            projections=projections,
            focus=focus,
        )
        pair_id = hashlib.sha256(
            (
                self._evidence.manifest_sha256
                + "\0"
                + chosen.view_id
                + "\0"
                + "\0".join(sorted(object_ids))
            ).encode("utf-8")
        ).hexdigest()[:16]
        shared = {
            "metric": "collision",
            "view_id": chosen.view_id,
            "pair_id": pair_id,
            "target_object_ids": list(object_ids),
            "pose": {
                "id": chosen.view_id,
                "camera_type": chosen.pose["camera_type"],
                "location": list(chosen.pose["location"]),
                "target": list(chosen.pose["target"]),
                "vertical_fov_degrees": chosen.pose["vertical_fov_degrees"],
            },
            "selection_source": "deterministic_projected_target_coverage",
            "appearance_fidelity": "original_runtime_direct_webgl",
            "focus_point_canonical": [float(value) for value in focus],
        }
        return [
            {
                **shared,
                "path": chosen.path.as_posix(),
                "role": "collision_rgb",
                "presentation": "raw",
                "representation": "original_runtime_rgb",
            },
            {
                **shared,
                "path": overlay_path.as_posix(),
                "role": "collision_pair_overlay",
                "presentation": "highlight",
                "representation": "same_pose_projected_canonical_obb_wireframe",
                "color_legend": [
                    {
                        "object_id": object_ids[index],
                        "role": f"object_{'a' if index == 0 else 'b'}",
                        "rgb": list(_PAIR_COLORS[index]),
                    }
                    for index in range(2)
                ],
                "annotation_limit": (
                    "projected canonical OBB, not an occlusion-aware "
                    "segmentation contour"
                ),
            },
        ]

    def _choose_view(
        self,
        object_ids: tuple[str, str],
        *,
        focus: np.ndarray,
    ) -> tuple[_ViewProjection, dict[str, tuple[np.ndarray, np.ndarray]]]:
        ranked: list[
            tuple[
                tuple[float, float, float, float, float, float, float, str],
                _ViewProjection,
                dict[str, tuple[np.ndarray, np.ndarray]],
            ]
        ] = []
        for view in self._views:
            try:
                with Image.open(view.path) as image:
                    width, height = image.size
            except OSError as exc:
                raise CounterStrikeCollisionEvidenceError(
                    "regional_image_invalid",
                    f"could not read frozen regional image {view.path.name}",
                ) from exc
            projected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            per_object_coverage: list[float] = []
            center_visibility: list[float] = []
            for object_id in object_ids:
                obj = self._objects_by_id[object_id]
                pixels, depths = _project_world_points(
                    get_obb_corners(obj),
                    pose=view.pose,
                    width=width,
                    height=height,
                )
                projected[object_id] = pixels, depths
                in_front = depths > float(view.pose.get("near_m") or 0.02)
                in_frame = (
                    in_front
                    & (pixels[:, 0] >= 0.0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0.0)
                    & (pixels[:, 1] < height)
                )
                per_object_coverage.append(float(np.mean(in_frame)))
                center_pixel, center_depth = _project_world_points(
                    np.asarray([obj.center], dtype=float),
                    pose=view.pose,
                    width=width,
                    height=height,
                )
                center_visibility.append(
                    1.0
                    if (
                        center_depth[0] > float(view.pose.get("near_m") or 0.02)
                        and 0.0 <= center_pixel[0, 0] < width
                        and 0.0 <= center_pixel[0, 1] < height
                    )
                    else 0.0
                )
            declared_hits = len(
                set(object_ids).intersection(view.declared_target_ids)
            )
            focus_pixel, focus_depth = _project_world_points(
                np.asarray([focus], dtype=float),
                pose=view.pose,
                width=width,
                height=height,
            )
            focus_in_frame = (
                focus_depth[0] > float(view.pose.get("near_m") or 0.02)
                and 0.0 <= focus_pixel[0, 0] < width
                and 0.0 <= focus_pixel[0, 1] < height
            )
            focus_distance = float(
                np.linalg.norm(
                    focus_pixel[0]
                    - np.asarray([0.5 * width, 0.5 * height], dtype=float)
                )
            )
            focus_centrality = (
                max(
                    0.0,
                    1.0
                    - focus_distance
                    / max(math.hypot(width, height) * 0.5, 1.0),
                )
                if focus_in_frame
                else 0.0
            )
            rank = (
                1.0 if focus_in_frame else 0.0,
                1.0 if view.scope == "object_local" else 0.0,
                focus_centrality,
                float(sum(center_visibility)),
                float(declared_hits),
                min(per_object_coverage),
                float(sum(per_object_coverage)),
                view.view_id,
            )
            ranked.append((rank, view, projected))
        if not ranked:
            raise CounterStrikeCollisionEvidenceError(
                "regional_bank_empty",
                "frozen capture has no collision candidate views",
            )
        ranked.sort(
            key=lambda item: (
                -item[0][0],
                -item[0][1],
                -item[0][2],
                -item[0][3],
                -item[0][4],
                -item[0][5],
                -item[0][6],
                item[0][7],
            )
        )
        rank, view, projected = ranked[0]
        if rank[0] <= 0.0 and rank[3] <= 0.0 and rank[6] <= 0.0:
            raise CounterStrikeCollisionEvidenceError(
                "target_pair_not_visible",
                "no frozen view projects either collision target "
                "inside the image",
            )
        return view, projected

    def _render_overlay(
        self,
        view: _ViewProjection,
        *,
        object_ids: tuple[str, str],
        projections: dict[str, tuple[np.ndarray, np.ndarray]],
        focus: np.ndarray,
    ) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "version": COUNTER_STRIKE_COLLISION_EVIDENCE_VERSION,
                    "manifest": self._evidence.manifest_sha256,
                    "view": view.view_id,
                    "objects": list(object_ids),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        destination = (
            self.evidence_out_dir
            / f"collision_{digest[:20]}_{view.view_id}_obb_overlay.png"
        )
        if destination.is_file():
            return destination
        with Image.open(view.path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for index, object_id in enumerate(object_ids):
            pixels, depths = projections[object_id]
            color = _PAIR_COLORS[index]
            valid_points = pixels[
                np.isfinite(pixels).all(axis=1)
                & (depths > float(view.pose.get("near_m") or 0.02))
            ]
            if len(valid_points):
                lower = np.min(valid_points, axis=0)
                upper = np.max(valid_points, axis=0)
                draw.rectangle(
                    (
                        float(lower[0]) - 5.0,
                        float(lower[1]) - 5.0,
                        float(upper[0]) + 5.0,
                        float(upper[1]) + 5.0,
                    ),
                    outline=(*color, 115),
                    width=9,
                )
            for first, second in _OBB_EDGES:
                if (
                    depths[first] <= float(view.pose.get("near_m") or 0.02)
                    or depths[second] <= float(view.pose.get("near_m") or 0.02)
                ):
                    continue
                draw.line(
                    (
                        tuple(float(value) for value in pixels[first]),
                        tuple(float(value) for value in pixels[second]),
                    ),
                    fill=(*color, 245),
                    width=3,
                )
        focus_pixel, focus_depth = _project_world_points(
            np.asarray([focus], dtype=float),
            pose=view.pose,
            width=image.width,
            height=image.height,
        )
        if focus_depth[0] > float(view.pose.get("near_m") or 0.02):
            x, y = (float(value) for value in focus_pixel[0])
            radius = 8.0
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 210, 45, 255),
                width=4,
            )
            draw.line(
                ((x - 12.0, y), (x + 12.0, y)),
                fill=(255, 210, 45, 255),
                width=3,
            )
            draw.line(
                ((x, y - 12.0), (x, y + 12.0)),
                fill=(255, 210, 45, 255),
                width=3,
            )
        image.save(destination, format="PNG")
        return destination


def _frozen_projections(
    manifest: dict[str, Any],
    *,
    capture_dir: Path,
) -> tuple[_ViewProjection, ...]:
    controlled = manifest.get("controlled_camera")
    fallback = (
        controlled.get("style_local_fallback")
        if isinstance(controlled, dict)
        else None
    )
    regional_views = fallback.get("views") if isinstance(fallback, dict) else None
    global_views = manifest.get("views")
    if not isinstance(regional_views, list) or not isinstance(global_views, list):
        raise CounterStrikeCollisionEvidenceError(
            "regional_bank_invalid",
            "capture manifest has no frozen global/regional bank",
        )
    raw_views = list(regional_views) + list(global_views)
    result: list[_ViewProjection] = []
    for index, item in enumerate(raw_views):
        if not isinstance(item, dict):
            raise CounterStrikeCollisionEvidenceError(
                "regional_bank_invalid",
                f"regional view {index} is not a JSON object",
            )
        view_id = str(item.get("id") or "").strip()
        pose = item.get("camera_pose_canonical")
        if not view_id or not isinstance(pose, dict):
            raise CounterStrikeCollisionEvidenceError(
                "regional_bank_invalid",
                f"regional view {index} has no ID or canonical camera pose",
            )
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        try:
            path.relative_to(capture_dir)
        except ValueError as exc:
            raise CounterStrikeCollisionEvidenceError(
                "regional_path_outside_capture",
                f"regional view escaped the frozen capture: {path}",
            ) from exc
        if not path.is_file():
            raise CounterStrikeCollisionEvidenceError(
                "regional_image_missing",
                f"regional view does not exist: {path}",
            )
        _validated_perspective_pose(pose, label=view_id)
        result.append(
            _ViewProjection(
                view_id=view_id,
                path=path,
                pose=dict(pose),
                declared_target_ids=frozenset(
                    str(value) for value in item.get("target_object_ids") or []
                ),
                scope=str(item.get("scope") or ""),
            )
        )
    return tuple(result)


def _validated_perspective_pose(
    pose: dict[str, Any],
    *,
    label: str,
) -> None:
    if pose.get("camera_type") != "PERSP":
        raise CounterStrikeCollisionEvidenceError(
            "unsupported_camera_type",
            f"{label} must be a perspective camera",
        )
    for key in ("location", "target"):
        value = pose.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(float(component))
                for component in value
            )
        ):
            raise CounterStrikeCollisionEvidenceError(
                "camera_pose_invalid",
                f"{label}.{key} must be three finite numbers",
            )
    fov = pose.get("vertical_fov_degrees")
    if (
        isinstance(fov, bool)
        or not isinstance(fov, (int, float))
        or not 1.0 < float(fov) < 179.0
    ):
        raise CounterStrikeCollisionEvidenceError(
            "camera_pose_invalid",
            f"{label}.vertical_fov_degrees is invalid",
        )


def _project_world_points(
    points: np.ndarray,
    *,
    pose: dict[str, Any],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    location = np.asarray(pose["location"], dtype=float)
    target = np.asarray(pose["target"], dtype=float)
    forward = target - location
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-9:
        raise CounterStrikeCollisionEvidenceError(
            "camera_pose_invalid",
            "camera location and target coincide",
        )
    forward /= norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-9:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
    right /= right_norm
    camera_up = np.cross(right, forward)

    relative = np.asarray(points, dtype=float) - location[None, :]
    depth = relative @ forward
    vertical_tangent = math.tan(
        math.radians(float(pose["vertical_fov_degrees"])) / 2.0
    )
    horizontal_tangent = vertical_tangent * float(width) / float(height)
    safe_depth = np.where(np.abs(depth) > 1.0e-12, depth, np.nan)
    ndc_x = (relative @ right) / (safe_depth * horizontal_tangent)
    ndc_y = (relative @ camera_up) / (safe_depth * vertical_tangent)
    pixels = np.column_stack(
        (
            (ndc_x + 1.0) * 0.5 * width,
            (1.0 - ndc_y) * 0.5 * height,
        )
    )
    return pixels, depth


def _event_focus_point(
    request: dict[str, Any],
    *,
    objects: list[Any],
) -> np.ndarray:
    detector = request.get("detector_evidence")
    mesh = detector.get("mesh") if isinstance(detector, dict) else None
    focus_region = (
        mesh.get("focus_region") if isinstance(mesh, dict) else None
    )
    center = (
        focus_region.get("center")
        if isinstance(focus_region, dict)
        else None
    )
    if (
        isinstance(center, list)
        and len(center) == 3
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in center
        )
    ):
        return np.asarray(center, dtype=float)
    return np.mean(
        np.asarray([item.center for item in objects], dtype=float),
        axis=0,
    )
