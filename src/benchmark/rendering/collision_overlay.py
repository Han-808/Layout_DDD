"""Deterministic collision pair-highlighting evidence.

This module builds the backend-neutral *specification* for the collision
diagnostic overlay and ranks camera candidates from measured visibility. It
performs no rendering and imports no ``bpy``: a Blender worker (or any future
provider) consumes :func:`build_collision_overlay_spec` to draw the paired
same-pose diagnostic view, and the pure ranking helpers here decide which poses
best expose both targets.

Design constraints from the collision-evidence contract:
- object A is red, object B is cyan, other scene objects are dim but visible;
- both OBB wireframes are drawn, with a legend identifying A/B IDs and category;
- mesh closest points / definitive contact points are marked in yellow;
- no learned segmentation, SAM, or extra vision model is used - canonical object
  identity is resolved from Blender object names / custom properties;
- no new intersection-volume or concavity algorithm is introduced for drawing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.mesh_geometry import (
    geometry_entry_for_object,
    is_usable_triangle_mesh,
)
from benchmark.scene_io.object_normalization import (
    normalize_object,
    rotation_matrix_from_euler,
)


COLLISION_OVERLAY_SCHEMA_VERSION = "collision_overlay_v1"
FOCUS_OVERLAY_SCHEMA_VERSION = "focus_overlay_v1"

# Identity-mask visibility selector and paired-evidence packet versions. The v2
# selector ranks candidates from binary target-identity masks (Object Index /
# ID pass) instead of matching decorative RGB pixels, so ranking is independent
# of Blender color management, wireframes, markers, and legends.
COLLISION_VISIBILITY_SELECTOR_VERSION = "collision_visibility_mask_rank_v2"
COLLISION_EVIDENCE_PACKET_VERSION = "collision_evidence_packet_v2"

# Canonical joint-visibility outcomes recorded per collision event.
JOINT_VISIBILITY_BOTH_VISIBLE = "both_visible"
JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED = "impossible_or_occluded"
JOINT_VISIBILITY_NO_MASK_EVIDENCE = "no_mask_evidence"

# Canonical-ID custom property written on Blender roots so multi-child assets
# resolve to a single benchmark object id even after ``.001`` name mangling.
CANONICAL_ID_PROPERTY = "benchmark_object_id"

# Object-color / emission-style colors (linear RGB in [0, 1]).
COLLISION_OVERLAY_COLORS: dict[str, list[float]] = {
    "object_a": [1.0, 0.12, 0.12],
    "object_b": [0.10, 0.85, 0.92],
    "context": [0.34, 0.34, 0.38],
    "marker": [1.0, 0.94, 0.10],
    "connector": [1.0, 0.94, 0.10],
    "architecture": [0.95, 0.24, 0.92],
}

FOCUS_TARGET_COLORS: tuple[list[float], ...] = (
    [1.0, 0.12, 0.12],
    [0.10, 0.85, 0.92],
    [1.0, 0.55, 0.08],
    [0.28, 0.90, 0.30],
)

# Cube-edge index pairs: two corners share an edge iff their signed-axis codes
# differ in exactly one axis. Corner order matches ``_obb_corners`` below.
_OBB_EDGES: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1
)

_BLENDER_SUFFIX = re.compile(r"\.\d{3}$")


def build_collision_overlay_spec(
    *,
    scene: dict[str, Any],
    object_a_id: str,
    object_b_id: str,
    mesh_evidence: dict[str, Any] | None = None,
    focus_region: dict[str, Any] | None = None,
    geometry_manifest: dict[str, Any] | None = None,
    geometry_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic overlay spec for one collision pair.

    The spec is read-only data: colors, both OBB wireframes, a legend, yellow
    closest-point markers / connector, and an optional focus region. It never
    mutates the scene and never encodes a verdict.
    """

    if not isinstance(scene, dict):
        raise TypeError("collision overlay requires a scene mapping")
    objects_by_id = {
        str(item.get("id")): item
        for item in scene.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    base_dir = Path(geometry_base_dir).expanduser() if geometry_base_dir is not None else None

    target_a = _object_overlay(objects_by_id, object_a_id, "object_a", geometry_manifest, base_dir)
    target_b = _object_overlay(objects_by_id, object_b_id, "object_b", geometry_manifest, base_dir)
    target_a["required_for_visibility"] = True
    target_b["required_for_visibility"] = True

    context_objects = [
        {
            "id": object_id,
            "category": item.get("category") or item.get("retrieval_category"),
            "color": list(COLLISION_OVERLAY_COLORS["context"]),
            "role": "context",
        }
        for object_id, item in objects_by_id.items()
        if object_id not in {str(object_a_id), str(object_b_id)}
    ]

    markers, connectors = _closest_point_markers(mesh_evidence)
    resolved_focus = _focus(focus_region, mesh_evidence)
    representation_level = _representation_level(target_a["representation"], target_b["representation"])

    legend = [
        {
            "id": target_a["id"],
            "category": target_a["category"],
            "color": list(COLLISION_OVERLAY_COLORS["object_a"]),
            "role": "object_a",
            "representation": target_a["representation"],
        },
        {
            "id": target_b["id"],
            "category": target_b["category"],
            "color": list(COLLISION_OVERLAY_COLORS["object_b"]),
            "role": "object_b",
            "representation": target_b["representation"],
        },
    ]
    return {
        "schema_version": COLLISION_OVERLAY_SCHEMA_VERSION,
        "metric": "collision",
        "object_a": target_a,
        "object_b": target_b,
        "targets": [target_a, target_b],
        "context_objects": context_objects,
        "legend": legend,
        "markers": markers,
        "connectors": connectors,
        "focus": resolved_focus,
        "representation_level": representation_level,
        "colors": {key: list(value) for key, value in COLLISION_OVERLAY_COLORS.items()},
        "canonical_id_property": CANONICAL_ID_PROPERTY,
        "diagnostic_pass": "object_color_emission_read_only",
    }


def build_focus_overlay_spec(
    *,
    scene: dict[str, Any],
    metric: str,
    object_ids: list[str] | tuple[str, ...],
    detector_evidence: dict[str, Any] | None = None,
    architecture_element: str | None = None,
    geometry_manifest: dict[str, Any] | None = None,
    geometry_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a metric-neutral focus overlay for OOB/Support/local evidence.

    The first object is the primary subject. Additional objects are semantic
    context (for example candidate supports). Only the primary subject is
    required for deterministic visibility ranking; Collision uses
    :func:`build_collision_overlay_spec`, where both pair members are required.
    Exact detector evidence is copied for auditability but never changed here.
    """

    if not isinstance(scene, dict):
        raise TypeError("focus overlay requires a scene mapping")
    metric_name = str(metric).strip().lower()
    requested_ids = list(dict.fromkeys(str(value) for value in object_ids if str(value)))
    if not requested_ids:
        raise ValueError("focus overlay requires at least one target object id")
    objects_by_id = {
        str(item.get("id")): item
        for item in scene.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    missing = [object_id for object_id in requested_ids if object_id not in objects_by_id]
    if missing:
        raise ValueError(f"focus overlay targets are absent from the scene: {missing}")
    base_dir = Path(geometry_base_dir).expanduser() if geometry_base_dir is not None else None
    targets: list[dict[str, Any]] = []
    for index, object_id in enumerate(requested_ids):
        role = "primary_target" if index == 0 else f"related_target_{index}"
        target = _object_overlay(objects_by_id, object_id, role, geometry_manifest, base_dir)
        target["color"] = list(FOCUS_TARGET_COLORS[index % len(FOCUS_TARGET_COLORS)])
        target["required_for_visibility"] = index == 0
        targets.append(target)
    context_objects = [
        {
            "id": object_id,
            "category": item.get("category") or item.get("retrieval_category"),
            "color": list(COLLISION_OVERLAY_COLORS["context"]),
            "role": "context",
        }
        for object_id, item in objects_by_id.items()
        if object_id not in set(requested_ids)
    ]
    planes = _architecture_plane_overlays(scene, detector_evidence or {}) if metric_name == "oob" else []
    markers: list[dict[str, Any]] = []
    connectors: list[dict[str, Any]] = []
    focus = None
    if metric_name == "support":
        markers, connectors, focus = _support_gap_annotations(detector_evidence or {})
    legend = [
        {
            "id": target["id"],
            "category": target["category"],
            "color": list(target["color"]),
            "role": target["role"],
            "representation": target["representation"],
        }
        for target in targets
    ]
    legend.extend(
        {
            "id": plane["id"],
            "category": "room_architecture",
            "color": list(plane["color"]),
            "role": "architecture_plane",
            "representation": "analytic_plane_boundary",
        }
        for plane in planes
    )
    if connectors:
        legend.append(
            {
                "id": "support_gap",
                "category": "detector_focus",
                "color": list(COLLISION_OVERLAY_COLORS["connector"]),
                "role": "measured_support_gap",
                "representation": "analytic_vertical_segment",
            }
        )
    representations = {target["representation"] for target in targets}
    representation_level = (
        "mesh" if representations == {"mesh"} else "bbox_proxy" if representations == {"bbox_proxy"} else "mixed"
    )
    return {
        "schema_version": FOCUS_OVERLAY_SCHEMA_VERSION,
        "metric": metric_name,
        "role": "metric_focus_overlay",
        "targets": targets,
        "context_objects": context_objects,
        "legend": legend,
        "markers": markers,
        "connectors": connectors,
        "focus": focus,
        "architecture_element": architecture_element,
        "architecture_planes": planes,
        "representation_level": representation_level,
        "colors": {key: list(value) for key, value in COLLISION_OVERLAY_COLORS.items()},
        "canonical_id_property": CANONICAL_ID_PROPERTY,
        "diagnostic_pass": "object_color_emission_read_only",
    }


def _support_gap_annotations(
    detector_evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Build an auditable segment from the measured support plane to object base."""

    raw_hits = detector_evidence.get("representative_ray_hits")
    candidates: list[tuple[bool, float, list[float], dict[str, Any]]] = []
    if isinstance(raw_hits, list):
        for raw in raw_hits:
            if not isinstance(raw, dict):
                continue
            position = _point3(raw.get("position"))
            gap = raw.get("gap_m")
            if position is None or not isinstance(gap, (int, float)) or not np.isfinite(float(gap)):
                continue
            if float(gap) < 0.0:
                continue
            candidates.append((bool(raw.get("is_center")), float(gap), position, raw))
    if not candidates:
        return [], [], None

    is_center, gap, support_point, raw = min(
        candidates,
        key=lambda item: (not item[0], item[1]),
    )
    base_point = [support_point[0], support_point[1], support_point[2] + gap]
    center = [(left + right) / 2.0 for left, right in zip(support_point, base_point)]
    markers = [
        {
            "type": "support_surface_point",
            "position": support_point,
            "target": raw.get("target"),
        },
        {
            "type": "subject_base_point",
            "position": base_point,
        },
    ]
    connectors = [
        {
            "type": "measured_support_gap",
            "from": support_point,
            "to": base_point,
            "gap_m": gap,
        }
    ]
    return markers, connectors, {
        "center": center,
        "radius_m": max(0.08, min(0.5, gap * 1.5)),
        "source": "support_detector_representative_ray",
        "gap_m": gap,
        "is_center_ray": is_center,
    }


def _object_overlay(
    objects_by_id: dict[str, dict[str, Any]],
    object_id: str,
    role: str,
    geometry_manifest: dict[str, Any] | None,
    base_dir: Path | None,
) -> dict[str, Any]:
    raw = objects_by_id.get(str(object_id))
    if raw is None:
        raise ValueError(f"collision overlay target {object_id!r} is not present in the scene")
    obb = _obb_from_object(raw)
    representation = _object_representation(geometry_manifest, base_dir, str(object_id))
    return {
        "id": str(object_id),
        "category": raw.get("category") or raw.get("retrieval_category"),
        "description": raw.get("description") or raw.get("desc") or raw.get("short_desc"),
        "role": role,
        "color": list(COLLISION_OVERLAY_COLORS.get(role, COLLISION_OVERLAY_COLORS["context"])),
        "representation": representation,
        "obb": obb,
        "canonical_id_property": CANONICAL_ID_PROPERTY,
    }


def _obb_from_object(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        obj = normalize_object(raw)
        center = np.asarray(obj.center, dtype=float)
        half = np.asarray(obj.half, dtype=float)
        rotation = np.asarray(obj.rotation, dtype=float)
        matrix = np.asarray(obj.R, dtype=float)
        size = np.asarray(obj.size, dtype=float)
    except Exception:
        center = np.asarray(raw.get("center") or [0.0, 0.0, 0.0], dtype=float)
        proxy = raw.get("asset_proxy") if isinstance(raw.get("asset_proxy"), dict) else {}
        size = np.asarray(raw.get("size") or proxy.get("bbox_size") or [1.0, 1.0, 1.0], dtype=float)
        half = size / 2.0
        rotation = np.asarray(raw.get("rotation") or [0.0, 0.0, 0.0], dtype=float)
        matrix = rotation_matrix_from_euler(rotation)
    corners = _obb_corners(center, half, matrix)
    return {
        "center": [float(value) for value in center],
        "size": [float(value) for value in size],
        "rotation_degrees": [float(value) for value in rotation],
        "corners": [[float(axis) for axis in corner] for corner in corners],
        "edges": [[int(a), int(b)] for a, b in _OBB_EDGES],
    }


def _obb_corners(center: np.ndarray, half: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    local = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )
    return center + local @ matrix.T


def _object_representation(
    geometry_manifest: dict[str, Any] | None,
    base_dir: Path | None,
    object_id: str,
) -> str:
    entry = geometry_entry_for_object(geometry_manifest, object_id)
    if is_usable_triangle_mesh(entry, base_dir=base_dir):
        return "mesh"
    return "bbox_proxy"


def _representation_level(rep_a: str, rep_b: str) -> str:
    if rep_a == "mesh" and rep_b == "mesh":
        return "mesh"
    if rep_a == "mesh" or rep_b == "mesh":
        return "mixed"
    return "bbox_proxy"


def _closest_point_markers(mesh_evidence: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markers: list[dict[str, Any]] = []
    connectors: list[dict[str, Any]] = []
    if not isinstance(mesh_evidence, dict):
        return markers, connectors
    closest = mesh_evidence.get("closest_points")
    if isinstance(closest, dict):
        point_a = _point3(closest.get("object_a"))
        point_b = _point3(closest.get("object_b"))
        if point_a is not None:
            markers.append({"type": "closest_point", "role": "object_a", "position": point_a, "color": list(COLLISION_OVERLAY_COLORS["marker"])})
        if point_b is not None:
            markers.append({"type": "closest_point", "role": "object_b", "position": point_b, "color": list(COLLISION_OVERLAY_COLORS["marker"])})
        if point_a is not None and point_b is not None:
            connectors.append({"type": "closest_pair", "from": point_a, "to": point_b, "color": list(COLLISION_OVERLAY_COLORS["connector"])})
    intersection = mesh_evidence.get("intersection")
    if isinstance(intersection, dict) and intersection.get("definitive") and intersection.get("intersects"):
        for contact in intersection.get("contacts") or []:
            point = _point3(contact.get("point")) if isinstance(contact, dict) else None
            if point is not None:
                markers.append({"type": "contact", "position": point, "color": list(COLLISION_OVERLAY_COLORS["marker"])})
    return markers, connectors


def _focus(focus_region: dict[str, Any] | None, mesh_evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    source = focus_region
    if source is None and isinstance(mesh_evidence, dict):
        candidate = mesh_evidence.get("focus_region")
        source = candidate if isinstance(candidate, dict) else None
    if not isinstance(source, dict):
        return None
    center = _point3(source.get("center"))
    radius = source.get("radius_m")
    return {
        "center": center,
        "radius_m": float(radius) if isinstance(radius, (int, float)) else None,
        "source": source.get("source"),
    }


def measure_overlay_visibility(
    image_path: str | Path,
    *,
    color_a: list[float] | None = None,
    color_b: list[float] | None = None,
    tolerance: float = 0.22,
) -> dict[str, Any]:
    """Measure the visible pixel fraction of each highlighted target.

    This is a cheap deterministic target-ID measurement on the saved diagnostic
    preview: it counts pixels close to the object-A and object-B highlight colors.
    It uses no learned segmentation. Returns zero fractions if the image cannot
    be read so the caller can fall back to pose order.
    """

    color_a_arr = np.asarray(color_a if color_a is not None else COLLISION_OVERLAY_COLORS["object_a"], dtype=float)
    color_b_arr = np.asarray(color_b if color_b is not None else COLLISION_OVERLAY_COLORS["object_b"], dtype=float)
    try:
        from PIL import Image

        with Image.open(Path(image_path)) as source:
            pixels = np.asarray(source.convert("RGB"), dtype=float) / 255.0
    except Exception as exc:  # pragma: no cover - exercised via degraded fallback
        return {
            "object_a_pixel_fraction": 0.0,
            "object_b_pixel_fraction": 0.0,
            "pixel_count": 0,
            "measured": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    flat = pixels.reshape(-1, 3)
    pixel_count = int(flat.shape[0])
    if pixel_count == 0:
        return {"object_a_pixel_fraction": 0.0, "object_b_pixel_fraction": 0.0, "pixel_count": 0, "measured": True}
    # Overlay colors are authored in linear RGB, while saved PNGs are display
    # transformed and Workbench additionally changes their intensity. Compare
    # display-space chromatic direction rather than exact RGB magnitude so the
    # identity proxy remains stable across Workbench, Eevee, and Cycles.
    mask_a = _display_chromaticity_mask(flat, color_a_arr, tolerance=tolerance)
    mask_b = _display_chromaticity_mask(flat, color_b_arr, tolerance=tolerance)
    fraction_a = float(np.count_nonzero(mask_a)) / float(pixel_count)
    fraction_b = float(np.count_nonzero(mask_b)) / float(pixel_count)
    return {
        "object_a_pixel_fraction": fraction_a,
        "object_b_pixel_fraction": fraction_b,
        "pixel_count": pixel_count,
        "measured": True,
    }


def rank_collision_candidates(
    candidates: list[dict[str, Any]],
    visibility_by_id: dict[str, dict[str, Any]],
    *,
    max_views: int,
    min_visible_fraction: float = 0.001,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank collision candidates by measured two-target visibility.

    A candidate is *usable* only when both targets exceed a small nonzero pixel
    threshold, the focus region (when measured) is in frame, and neither target
    is completely occluded. Usable candidates are ordered by the weaker target's
    visibility, then combined visibility, then candidate id for deterministic
    tie-breaking. If fewer than ``max_views`` candidates are usable, the frozen
    pose order fills the remaining slots and the fallback reason is recorded.
    """

    if not candidates:
        raise ValueError("collision candidate ranking requires at least one candidate")
    budget = max(1, int(max_views))
    ordered = list(candidates)
    scored: list[dict[str, Any]] = []
    for position, candidate in enumerate(ordered):
        candidate_id = str(candidate.get("id"))
        stats = visibility_by_id.get(candidate_id, {})
        fraction_a = float(stats.get("object_a_pixel_fraction") or 0.0)
        fraction_b = float(stats.get("object_b_pixel_fraction") or 0.0)
        focus_in_frame = stats.get("focus_in_frame")
        fully_occluded = bool(stats.get("either_fully_occluded"))
        both_visible = fraction_a >= min_visible_fraction and fraction_b >= min_visible_fraction
        usable = both_visible and not fully_occluded and (focus_in_frame is not False)
        scored.append(
            {
                "id": candidate_id,
                "position": position,
                "object_a_pixel_fraction": fraction_a,
                "object_b_pixel_fraction": fraction_b,
                "min_target_fraction": min(fraction_a, fraction_b),
                "combined_fraction": fraction_a + fraction_b,
                "focus_in_frame": focus_in_frame,
                "usable": usable,
            }
        )

    usable_entries = [entry for entry in scored if entry["usable"]]
    usable_entries.sort(key=lambda entry: (-entry["min_target_fraction"], -entry["combined_fraction"], entry["id"]))
    selected_ids = [entry["id"] for entry in usable_entries[:budget]]

    fallback_reason: str | None = None
    if len(selected_ids) < budget:
        if not selected_ids:
            fallback_reason = "no_candidate_exposed_both_targets"
        else:
            fallback_reason = "insufficient_usable_candidates_filled_with_pose_order"
        for candidate in ordered:
            candidate_id = str(candidate.get("id"))
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= budget:
                break

    by_id = {str(candidate.get("id")): candidate for candidate in ordered}
    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    log = {
        "selector": "deterministic_visibility_rank_v1",
        "min_visible_fraction": float(min_visible_fraction),
        "ranked": scored,
        "selected_view_ids": [str(item.get("id")) for item in selected],
        "fallback_reason": fallback_reason,
    }
    return selected, log


def measure_target_mask_png(
    mask_path: str | Path,
    *,
    foreground_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure one binary target-identity mask (Object Index / ID pass).

    A mask pixel is foreground when its luminance is at or above the threshold.
    The result is independent of materials, lighting, exposure, Blender color
    management, wireframes, markers, and legends because the input is a binary
    identity image, not the decorative RGB render.
    """

    try:
        from PIL import Image

        with Image.open(Path(mask_path)) as source:
            pixels = np.asarray(source.convert("L"), dtype=float) / 255.0
    except Exception as exc:  # pragma: no cover - degraded renderer path
        return {"measured": False, "visible_pixels": 0, "image_pixel_count": 0, "error": f"{type(exc).__name__}: {exc}"}
    height, width = pixels.shape[:2]
    image_pixel_count = int(height * width)
    mask = pixels >= float(foreground_threshold)
    visible_pixels = int(np.count_nonzero(mask))
    if visible_pixels == 0:
        return {
            "measured": True,
            "visible_pixels": 0,
            "image_pixel_count": image_pixel_count,
            "image_size": [int(width), int(height)],
            "visible_fraction": 0.0,
            "bbox": None,
            "touches_border": False,
            "centroid_uv": None,
        }
    ys, xs = np.nonzero(mask)
    u0, u1 = int(xs.min()), int(xs.max())
    v0, v1 = int(ys.min()), int(ys.max())
    touches_border = bool(u0 == 0 or v0 == 0 or u1 == width - 1 or v1 == height - 1)
    return {
        "measured": True,
        "visible_pixels": visible_pixels,
        "image_pixel_count": image_pixel_count,
        "image_size": [int(width), int(height)],
        "visible_fraction": float(visible_pixels) / float(max(image_pixel_count, 1)),
        "bbox": [
            float(u0) / float(max(width, 1)),
            float(v0) / float(max(height, 1)),
            float(u1 + 1) / float(max(width, 1)),
            float(v1 + 1) / float(max(height, 1)),
        ],
        "touches_border": touches_border,
        "centroid_uv": [float(xs.mean()) / float(max(width, 1)), float(ys.mean()) / float(max(height, 1))],
    }


def build_candidate_mask_stats(
    view_record: dict[str, Any],
    *,
    target_ids: list[str] | tuple[str, ...],
    safe_margin_fraction: float = 0.06,
) -> dict[str, Any]:
    """Normalize one worker mask-view record into ranking statistics.

    ``view_record`` is one entry of a mask manifest and may carry, per target,
    either a ready ``visible_pixels`` count (from the Blender Object Index pass)
    or a ``mask_path`` this function measures. Focus/projected-OBB metadata are
    consumed when the worker provides them; they are optional.
    """

    status = str(view_record.get("status") or "ok")
    targets_in = view_record.get("targets") if isinstance(view_record.get("targets"), dict) else {}
    image_pixel_count = int(view_record.get("image_pixel_count") or 0)
    target_stats: dict[str, dict[str, Any]] = {}
    for target_id in target_ids:
        raw = targets_in.get(str(target_id)) if isinstance(targets_in, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        measured = raw
        if raw.get("visible_pixels") is None and raw.get("mask_path"):
            measured = measure_target_mask_png(raw["mask_path"])
            if not image_pixel_count:
                image_pixel_count = int(measured.get("image_pixel_count") or 0)
        visible_pixels = int(measured.get("visible_pixels") or 0)
        obb_area = raw.get("projected_obb_area_px")
        obb_area = float(obb_area) if isinstance(obb_area, (int, float)) and float(obb_area) > 0 else None
        pixel_count = int(measured.get("image_pixel_count") or image_pixel_count or 0)
        visible_fraction = (
            float(measured.get("visible_fraction"))
            if measured.get("visible_fraction") is not None
            else (float(visible_pixels) / float(pixel_count) if pixel_count else 0.0)
        )
        normalized = (
            float(visible_pixels) / obb_area if obb_area else visible_fraction
        )
        target_stats[str(target_id)] = {
            "visible_pixels": visible_pixels,
            "visible_fraction": visible_fraction,
            "projected_obb_area_px": obb_area,
            "normalized_visibility": float(min(1.0, normalized)) if obb_area else float(visible_fraction),
            "touches_border": bool(measured.get("touches_border")),
            "bbox": measured.get("bbox"),
            "mask_path": raw.get("mask_path"),
        }
    focus = view_record.get("focus") if isinstance(view_record.get("focus"), dict) else {}
    focus_in_frame = focus.get("in_frame")
    focus_uv = focus.get("projected_uv")
    if focus_in_frame is None and isinstance(focus_uv, (list, tuple)) and len(focus_uv) == 2:
        margin = float(safe_margin_fraction)
        focus_in_frame = bool(
            margin <= float(focus_uv[0]) <= 1.0 - margin and margin <= float(focus_uv[1]) <= 1.0 - margin
        )
    return {
        "status": status,
        "targets": target_stats,
        "focus_in_frame": focus_in_frame,
        "focus_projected_uv": list(focus_uv) if isinstance(focus_uv, (list, tuple)) else None,
        "image_pixel_count": image_pixel_count,
    }


def rank_collision_candidates_v2(
    candidates: list[dict[str, Any]],
    mask_stats_by_id: dict[str, dict[str, Any]],
    *,
    target_ids: list[str] | tuple[str, ...],
    max_views: int,
    min_visible_pixels: int = 1,
    containment_hint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank collision candidates from binary target-identity masks.

    Ranking preference (contract order):
      1. both target surfaces visible when physically possible;
      2. the weaker target's OBB-normalized visibility;
      3. the suspicious focus region in frame;
      4. useful framing without severe boundary clipping;
      5. angular diversity between selected views.

    Containment / total occlusion is explicit: if one target has no visible
    surface in any valid candidate (or a mesh containment hint is supplied), the
    joint status becomes ``impossible_or_occluded`` and only the visible outer
    target is required so a contextual + x-ray view can still be selected. A
    colored OBB line is never treated as proof a target is visible.
    """

    if not candidates:
        raise ValueError("collision candidate ranking requires at least one candidate")
    budget = max(1, int(max_views))
    ids = [str(value) for value in target_ids]
    if len(ids) < 2:
        raise ValueError("collision ranking requires two target ids")
    by_id = {str(candidate.get("id")): candidate for candidate in candidates}

    valid_ids = [
        str(candidate.get("id"))
        for candidate in candidates
        if str((mask_stats_by_id.get(str(candidate.get("id"))) or {}).get("status") or "ok") == "ok"
    ]
    dropped = [
        {
            "id": str(candidate.get("id")),
            "status": str((mask_stats_by_id.get(str(candidate.get("id"))) or {}).get("status") or "missing"),
        }
        for candidate in candidates
        if str(candidate.get("id")) not in valid_ids
    ]

    def target_visible_anywhere(target_id: str) -> bool:
        for candidate_id in valid_ids:
            stats = mask_stats_by_id.get(candidate_id) or {}
            target = (stats.get("targets") or {}).get(target_id) or {}
            if int(target.get("visible_pixels") or 0) >= min_visible_pixels:
                return True
        return False

    occluded_targets = [tid for tid in ids if not target_visible_anywhere(tid)]
    if containment_hint in ids and containment_hint not in occluded_targets:
        occluded_targets.append(containment_hint)
    if len(occluded_targets) >= len(ids):
        # No target is ever visible: mask evidence is uninformative.
        joint_visibility_status = JOINT_VISIBILITY_NO_MASK_EVIDENCE if valid_ids else JOINT_VISIBILITY_NO_MASK_EVIDENCE
        required_ids = list(ids)
    elif occluded_targets:
        joint_visibility_status = JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED
        required_ids = [tid for tid in ids if tid not in occluded_targets]
    else:
        joint_visibility_status = JOINT_VISIBILITY_BOTH_VISIBLE
        required_ids = list(ids)

    scored: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id"))
        stats = mask_stats_by_id.get(candidate_id) or {}
        status = str(stats.get("status") or "ok")
        targets = stats.get("targets") if isinstance(stats.get("targets"), dict) else {}
        per_target = {tid: (targets.get(tid) or {}) for tid in ids}
        required = [per_target[tid] for tid in required_ids]
        required_norm = [float(t.get("normalized_visibility") or 0.0) for t in required]
        required_frac = [float(t.get("visible_fraction") or 0.0) for t in required]
        weaker_norm = min(required_norm) if required_norm else 0.0
        combined_norm = sum(required_norm)
        combined_fraction = sum(required_frac)
        both_visible = bool(required) and all(
            int(t.get("visible_pixels") or 0) >= min_visible_pixels for t in required
        )
        clipping = [bool(t.get("touches_border")) for t in required]
        clip_penalty = (sum(1 for value in clipping if value) / len(clipping)) if clipping else 0.0
        union_in_frame = bool(required) and all(t.get("bbox") is not None for t in required)
        focus_in_frame = stats.get("focus_in_frame")
        focus_bonus = 1.0 if focus_in_frame is True else (0.0 if focus_in_frame is False else 0.5)
        framing = _framing_score(combined_fraction)
        usable = status == "ok" and both_visible
        base_score = (
            3.0 * weaker_norm
            + 1.0 * combined_norm
            + 0.5 * focus_bonus
            + 0.25 * framing
            + 0.10 * (1.0 if union_in_frame else 0.0)
            - 0.5 * clip_penalty
        )
        scored.append(
            {
                "id": candidate_id,
                "position": position,
                "status": status,
                "per_target_visible_pixels": {tid: int(per_target[tid].get("visible_pixels") or 0) for tid in ids},
                "per_target_visible_fraction": {tid: float(per_target[tid].get("visible_fraction") or 0.0) for tid in ids},
                "per_target_normalized_visibility": {tid: float(per_target[tid].get("normalized_visibility") or 0.0) for tid in ids},
                "per_target_touches_border": {tid: bool(per_target[tid].get("touches_border")) for tid in ids},
                "required_target_ids": list(required_ids),
                "weaker_normalized_visibility": weaker_norm,
                "combined_normalized_visibility": combined_norm,
                "combined_visible_fraction": combined_fraction,
                "union_in_frame": union_in_frame,
                "clipping_penalty": clip_penalty,
                "focus_in_frame": focus_in_frame,
                "framing_score": framing,
                "base_score": base_score,
                "usable": usable,
            }
        )

    # Sort by candidate id ascending so that, when the numeric selection key
    # ties, ``max`` (which returns the first maximal element) deterministically
    # prefers the smallest candidate id.
    usable_entries = sorted((entry for entry in scored if entry["usable"]), key=lambda entry: str(entry["id"]))
    selected_ids: list[str] = []
    while usable_entries and len(selected_ids) < budget:
        def selection_key(entry: dict[str, Any]) -> tuple[float, float, float]:
            diversity = _minimum_angular_diversity(
                by_id[entry["id"]], [by_id[view_id] for view_id in selected_ids]
            )
            # Tie-break order: score+diversity, weaker-target visibility, angular
            # diversity, then candidate id via the ascending pre-sort above.
            return (
                float(entry["base_score"]) + 0.15 * diversity,
                float(entry["weaker_normalized_visibility"]),
                diversity,
            )

        chosen = max(usable_entries, key=selection_key)
        selected_ids.append(str(chosen["id"]))
        usable_entries = [entry for entry in usable_entries if entry["id"] != chosen["id"]]

    ranked_selected = list(selected_ids)
    fallback_reason: str | None = None
    if len(selected_ids) < budget:
        fallback_reason = (
            "no_candidate_exposed_required_targets"
            if not selected_ids
            else "insufficient_usable_candidates_filled_with_pose_order"
        )
        for candidate in candidates:
            candidate_id = str(candidate.get("id"))
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= budget:
                break

    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    is_visibility_ranked = bool(ranked_selected) and fallback_reason != "no_candidate_exposed_required_targets"
    log = {
        "selector": COLLISION_VISIBILITY_SELECTOR_VERSION,
        "min_visible_pixels": int(min_visible_pixels),
        "target_ids": list(ids),
        "required_target_ids": list(required_ids),
        "joint_visibility_status": joint_visibility_status,
        "occluded_or_contained_target_ids": list(occluded_targets),
        "tie_break_order": ["base_score_plus_diversity", "weaker_target_visibility", "angular_diversity", "candidate_id"],
        "ranked": scored,
        "dropped_candidates": dropped,
        "visibility_ranked_selected_view_ids": ranked_selected,
        "selected_view_ids": [str(item.get("id")) for item in selected],
        "fallback_reason": fallback_reason,
        "is_visibility_ranked": is_visibility_ranked,
    }
    return selected, log


def measure_focus_visibility(
    image_path: str | Path,
    *,
    targets: list[dict[str, Any]],
    focus_color: list[float] | None = None,
    tolerance: float = 0.22,
) -> dict[str, Any]:
    """Measure highlighted target pixels in a generic focus-overlay image."""

    try:
        from PIL import Image

        with Image.open(Path(image_path)) as source:
            pixels = np.asarray(source.convert("RGB"), dtype=float) / 255.0
    except Exception as exc:  # pragma: no cover - degraded renderer path
        return {
            "target_pixel_fractions": {},
            "focus_pixel_fraction": 0.0,
            "pixel_count": 0,
            "measured": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    flat = pixels.reshape(-1, 3)
    pixel_count = int(flat.shape[0])
    fractions: dict[str, float] = {}
    for target in targets:
        target_id = str(target.get("id") or "")
        color = np.asarray(target.get("color") or [1.0, 0.12, 0.12], dtype=float)
        if not target_id or pixel_count == 0:
            continue
        mask = _display_chromaticity_mask(flat, color, tolerance=tolerance)
        fractions[target_id] = float(np.count_nonzero(mask)) / float(pixel_count)
    focus_fraction = 0.0
    if focus_color is not None and pixel_count:
        color = np.asarray(focus_color, dtype=float)
        mask = _display_chromaticity_mask(flat, color, tolerance=tolerance)
        focus_fraction = float(np.count_nonzero(mask)) / float(pixel_count)
    return {
        "target_pixel_fractions": fractions,
        "focus_pixel_fraction": focus_fraction,
        "pixel_count": pixel_count,
        "measured": True,
    }


def _display_chromaticity_mask(
    pixels: np.ndarray,
    linear_color: np.ndarray,
    *,
    tolerance: float,
    minimum_brightness: float = 0.04,
) -> np.ndarray:
    """Match one authored linear-RGB color after display transform/shading."""

    flat = np.asarray(pixels, dtype=float).reshape(-1, 3)
    color = np.clip(np.asarray(linear_color, dtype=float).reshape(3), 0.0, 1.0)
    display_color = np.where(
        color <= 0.0031308,
        12.92 * color,
        1.055 * np.power(color, 1.0 / 2.4) - 0.055,
    )
    display_reference = display_color / max(float(np.linalg.norm(display_color)), 1e-12)
    linear_reference = color / max(float(np.linalg.norm(color)), 1e-12)
    norms = np.linalg.norm(flat, axis=1)
    normalized = np.divide(
        flat,
        norms[:, None],
        out=np.zeros_like(flat),
        where=norms[:, None] > 1e-12,
    )
    display_distance = np.linalg.norm(normalized - display_reference[None, :], axis=1)
    linear_distance = np.linalg.norm(normalized - linear_reference[None, :], axis=1)
    chromatic_distance = np.minimum(display_distance, linear_distance)
    brightness = np.max(flat, axis=1)
    return (brightness >= float(minimum_brightness)) & (
        chromatic_distance <= float(tolerance)
    )


def rank_focus_candidates(
    candidates: list[dict[str, Any]],
    visibility_by_id: dict[str, dict[str, Any]],
    *,
    targets: list[dict[str, Any]],
    max_views: int,
    min_visible_fraction: float = 0.001,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank local views by required-target visibility, framing, and diversity.

    This is deterministic and deliberately modest: highlighted target pixels
    provide a cheap identity-aware visibility proxy, while angular diversity
    prevents two nearly identical views from consuming the evidence budget.
    It does not infer semantics or issue a metric verdict.
    """

    if not candidates:
        raise ValueError("focus candidate ranking requires at least one candidate")
    budget = max(1, int(max_views))
    required_ids = [
        str(target.get("id"))
        for target in targets
        if target.get("required_for_visibility") and target.get("id") is not None
    ]
    if not required_ids:
        required_ids = [str(targets[0].get("id"))] if targets else []
    optional_ids = [
        str(target.get("id"))
        for target in targets
        if target.get("id") is not None and str(target.get("id")) not in required_ids
    ]
    scored: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id"))
        stats = visibility_by_id.get(candidate_id, {})
        fractions = stats.get("target_pixel_fractions") if isinstance(stats, dict) else {}
        fractions = fractions if isinstance(fractions, dict) else {}
        required = [float(fractions.get(object_id) or 0.0) for object_id in required_ids]
        optional = [float(fractions.get(object_id) or 0.0) for object_id in optional_ids]
        minimum_required = min(required) if required else 0.0
        combined_required = sum(required)
        combined_optional = sum(optional)
        usable = bool(required) and all(value >= min_visible_fraction for value in required)
        framing_score = _framing_score(combined_required)
        base_score = 3.0 * minimum_required + combined_required + 0.25 * combined_optional + 0.10 * framing_score
        scored.append(
            {
                "id": candidate_id,
                "position": position,
                "required_target_fractions": dict(zip(required_ids, required)),
                "optional_target_fractions": dict(zip(optional_ids, optional)),
                "min_required_fraction": minimum_required,
                "combined_required_fraction": combined_required,
                "combined_optional_fraction": combined_optional,
                "framing_score": framing_score,
                "base_score": base_score,
                "usable": usable,
            }
        )

    by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    usable = [entry for entry in scored if entry["usable"]]
    selected_ids: list[str] = []
    while usable and len(selected_ids) < budget:
        def selection_key(entry: dict[str, Any]) -> tuple[float, float, str]:
            diversity = _minimum_angular_diversity(
                by_id[entry["id"]],
                [by_id[view_id] for view_id in selected_ids],
            )
            return (float(entry["base_score"]) + 0.15 * diversity, diversity, str(entry["id"]))

        chosen = max(usable, key=selection_key)
        selected_ids.append(str(chosen["id"]))
        usable = [entry for entry in usable if entry["id"] != chosen["id"]]

    fallback_reason: str | None = None
    if len(selected_ids) < budget:
        fallback_reason = (
            "no_candidate_exposed_required_targets"
            if not selected_ids
            else "insufficient_usable_candidates_filled_with_pose_order"
        )
        for candidate in candidates:
            candidate_id = str(candidate.get("id"))
            if candidate_id not in selected_ids:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= budget:
                break
    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    return selected, {
        "selector": "deterministic_visibility_framing_rank_v1",
        "min_visible_fraction": float(min_visible_fraction),
        "required_target_ids": required_ids,
        "optional_target_ids": optional_ids,
        "ranked": scored,
        "selected_view_ids": [str(item.get("id")) for item in selected],
        "fallback_reason": fallback_reason,
    }


def rank_support_contact_candidates(
    candidates: list[dict[str, Any]],
    visibility_by_id: dict[str, dict[str, Any]],
    *,
    targets: list[dict[str, Any]],
    max_views: int,
    min_visible_fraction: float = 0.001,
    min_focus_fraction: float = 0.00001,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank Support views by subject visibility and the exposed gap segment.

    The highlighted vertical segment is generated from detector measurements.
    Seeing both the subject and some of that segment is a deterministic proxy for
    an unobstructed contact-plane view; angular diversity prevents duplicate
    directions from consuming a two-image budget.
    """

    if not candidates:
        raise ValueError("support contact-plane ranking requires at least one candidate")
    budget = max(1, int(max_views))
    required_ids = [
        str(target.get("id"))
        for target in targets
        if target.get("required_for_visibility") and target.get("id") is not None
    ]
    if not required_ids:
        required_ids = [str(targets[0].get("id"))] if targets else []
    by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    scored: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id"))
        stats = visibility_by_id.get(candidate_id, {})
        fractions = stats.get("target_pixel_fractions") if isinstance(stats, dict) else {}
        fractions = fractions if isinstance(fractions, dict) else {}
        required = [float(fractions.get(object_id) or 0.0) for object_id in required_ids]
        minimum_required = min(required) if required else 0.0
        combined_required = sum(required)
        focus_fraction = float(stats.get("focus_pixel_fraction") or 0.0)
        target_visible = bool(required) and all(value >= min_visible_fraction for value in required)
        focus_visible = focus_fraction >= min_focus_fraction
        usable = target_visible and focus_visible
        framing_score = _framing_score(combined_required)
        # Gap visibility is deliberately dominant: a large target with a hidden
        # base is not useful Support evidence.
        base_score = 4.0 * min(focus_fraction, 0.02) / 0.02 + combined_required + 0.10 * framing_score
        scored.append(
            {
                "id": candidate_id,
                "position": position,
                "required_target_fractions": dict(zip(required_ids, required)),
                "min_required_fraction": minimum_required,
                "combined_required_fraction": combined_required,
                "focus_pixel_fraction": focus_fraction,
                "target_visible": target_visible,
                "focus_visible": focus_visible,
                "framing_score": framing_score,
                "base_score": base_score,
                "usable": usable,
            }
        )

    available = [entry for entry in scored if entry["usable"]]
    selected_ids: list[str] = []
    while available and len(selected_ids) < budget:
        def selection_key(entry: dict[str, Any]) -> tuple[float, float, str]:
            diversity = _minimum_angular_diversity(
                by_id[entry["id"]],
                [by_id[view_id] for view_id in selected_ids],
            )
            return (float(entry["base_score"]) + 0.60 * diversity, diversity, str(entry["id"]))

        chosen = max(available, key=selection_key)
        selected_ids.append(str(chosen["id"]))
        available = [entry for entry in available if entry["id"] != chosen["id"]]

    fallback_reason: str | None = None
    if len(selected_ids) < budget:
        fallback_reason = (
            "no_candidate_exposed_subject_and_gap"
            if not selected_ids
            else "insufficient_gap_visible_candidates"
        )
        remaining = sorted(
            (entry for entry in scored if entry["id"] not in selected_ids),
            key=lambda entry: (-float(entry["base_score"]), str(entry["id"])),
        )
        for entry in remaining:
            selected_ids.append(str(entry["id"]))
            if len(selected_ids) >= budget:
                break

    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    return selected, {
        "selector": "support_contact_plane_visibility_rank_v1",
        "min_visible_fraction": float(min_visible_fraction),
        "min_focus_fraction": float(min_focus_fraction),
        "required_target_ids": required_ids,
        "ranked": scored,
        "selected_view_ids": [str(item.get("id")) for item in selected],
        "fallback_reason": fallback_reason,
    }


def _framing_score(combined_fraction: float) -> float:
    if combined_fraction <= 0.0:
        return 0.0
    if combined_fraction < 0.03:
        return combined_fraction / 0.03
    if combined_fraction <= 0.65:
        return 1.0
    return max(0.0, (1.0 - combined_fraction) / 0.35)


def _minimum_angular_diversity(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    azimuth = float(candidate.get("azimuth_degrees") or 0.0)
    elevation = float(candidate.get("elevation_degrees") or 0.0)
    distances = []
    for other in selected:
        other_azimuth = float(other.get("azimuth_degrees") or 0.0)
        other_elevation = float(other.get("elevation_degrees") or 0.0)
        azimuth_delta = abs((azimuth - other_azimuth + 180.0) % 360.0 - 180.0) / 180.0
        elevation_delta = min(1.0, abs(elevation - other_elevation) / 90.0)
        distances.append(min(1.0, 0.8 * azimuth_delta + 0.2 * elevation_delta))
    return min(distances)


def _architecture_plane_overlays(scene: dict[str, Any], detector_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    flags = detector_evidence.get("plane_flags") if isinstance(detector_evidence, dict) else None
    if not isinstance(flags, dict):
        return []
    bounds = _scene_room_bounds(scene)
    if bounds is None:
        return []
    min_x, max_x, min_y, max_y, floor_z, ceiling_z = bounds
    definitions = {
        "west_oob": ("west", [[min_x, min_y, floor_z], [min_x, max_y, floor_z], [min_x, max_y, ceiling_z], [min_x, min_y, ceiling_z]], [-1.0, 0.0, 0.0]),
        "east_oob": ("east", [[max_x, min_y, floor_z], [max_x, max_y, floor_z], [max_x, max_y, ceiling_z], [max_x, min_y, ceiling_z]], [1.0, 0.0, 0.0]),
        "south_oob": ("south", [[min_x, min_y, floor_z], [max_x, min_y, floor_z], [max_x, min_y, ceiling_z], [min_x, min_y, ceiling_z]], [0.0, -1.0, 0.0]),
        "north_oob": ("north", [[min_x, max_y, floor_z], [max_x, max_y, floor_z], [max_x, max_y, ceiling_z], [min_x, max_y, ceiling_z]], [0.0, 1.0, 0.0]),
        "floor_oob": ("floor", [[min_x, min_y, floor_z], [max_x, min_y, floor_z], [max_x, max_y, floor_z], [min_x, max_y, floor_z]], [0.0, 0.0, -1.0]),
        "ceiling_oob": ("ceiling", [[min_x, min_y, ceiling_z], [max_x, min_y, ceiling_z], [max_x, max_y, ceiling_z], [min_x, max_y, ceiling_z]], [0.0, 0.0, 1.0]),
    }
    result: list[dict[str, Any]] = []
    for flag, (name, corners, normal) in definitions.items():
        if not flags.get(flag):
            continue
        center = np.mean(np.asarray(corners, dtype=float), axis=0)
        normal_end = center + np.asarray(normal, dtype=float) * 0.5
        result.append(
            {
                "id": f"room_{name}_plane",
                "flag": flag,
                "name": name,
                "corners": corners,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "normal_from": [float(value) for value in center],
                "normal_to": [float(value) for value in normal_end],
                "color": list(COLLISION_OVERLAY_COLORS["architecture"]),
            }
        )
    return result


def _scene_room_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    boundary = scene.get("boundary")
    if not isinstance(boundary, list) or len(boundary) < 3:
        return None
    try:
        points = np.asarray(boundary, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            return None
        floor_z = float(scene.get("floor_z") or 0.0)
        ceiling_z = float(scene.get("scene_height") or scene.get("height") or 2.8)
        return (
            float(np.min(points[:, 0])),
            float(np.max(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 1])),
            floor_z,
            ceiling_z,
        )
    except (TypeError, ValueError):
        return None


def resolve_canonical_object_id(
    node_chain: list[dict[str, Any]],
    *,
    known_ids: set[str] | frozenset[str] | None = None,
) -> str | None:
    """Resolve one canonical benchmark object id from a Blender node hierarchy.

    ``node_chain`` is ordered leaf-first through ancestors, each ``{"name", "canonical_id"}``.
    Resolution order matches the contract:
      1. a canonical-ID custom property, when present;
      2. an ``asset_<object_id>`` / ``proxy_<object_id>`` root name;
      3. otherwise ``None`` (caller keeps the raw name for auditing).
    Blender's ``.001`` duplicate suffixes are stripped, and ``known_ids`` (when
    supplied) disambiguates ids that themselves contain separators.
    """

    known = {str(value) for value in (known_ids or set())}
    for node in node_chain:
        canonical = node.get("canonical_id")
        if canonical is not None and str(canonical).strip():
            return str(canonical).strip()
    for node in node_chain:
        name = str(node.get("name") or "")
        for prefix in ("asset_", "proxy_"):
            if name.startswith(prefix):
                candidate = name[len(prefix):]
                if candidate in known:
                    return candidate
                stripped = _BLENDER_SUFFIX.sub("", candidate)
                if not known or stripped in known:
                    return stripped
    return None


def _point3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
