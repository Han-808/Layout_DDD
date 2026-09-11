"""Pinned renderer profile for official room-level evaluation materialization."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping


OFFICIAL_RENDER_PROFILE_ID = "room_evaluation_official_render_v1"
COLLISION_DECIMATION_POLICY_VERSION = "collision_decimate_collapse_v1"
OFFICIAL_RENDER_PROFILE: Mapping[str, Any] = MappingProxyType({
    "timeout_seconds": 1800,
    "width": 768,
    "height": 768,
    "render_engine": "BLENDER_EEVEE_NEXT",
    "cycles_device": "CPU",
    "cycles_samples": 16,
    "cycles_denoising": False,
    "require_asset_mesh": True,
    "collision_max_vertices_per_object": 400_000,
    "collision_max_faces_per_object": 550_000,
    "collision_max_total_vertices": 2_000_000,
    "collision_max_total_faces": 2_500_000,
})


def validate_official_renderer(renderer: Any) -> None:
    observed = {
        field: getattr(renderer, field, None)
        for field in OFFICIAL_RENDER_PROFILE
    }
    if observed != dict(OFFICIAL_RENDER_PROFILE):
        differences = {
            field: {
                "expected": expected,
                "observed": observed[field],
            }
            for field, expected in OFFICIAL_RENDER_PROFILE.items()
            if observed[field] != expected
        }
        raise ValueError(
            "official room evaluation renderer profile mismatch: "
            f"{differences}"
        )


def validate_official_render_manifest(
    manifest: Mapping[str, Any],
    collision_manifest: Mapping[str, Any],
) -> None:
    config = manifest.get("render_config")
    export = manifest.get("collision_geometry_export")
    coverage = manifest.get("asset_coverage")
    if not isinstance(config, Mapping):
        raise ValueError("official render manifest has no render_config")
    expected_config = {
        "width": OFFICIAL_RENDER_PROFILE["width"],
        "height": OFFICIAL_RENDER_PROFILE["height"],
        "render_engine_requested": OFFICIAL_RENDER_PROFILE["render_engine"],
        "cycles_device_requested": OFFICIAL_RENDER_PROFILE["cycles_device"],
    }
    if manifest.get("render_engine") != OFFICIAL_RENDER_PROFILE[
        "render_engine"
    ] or any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("official render manifest profile mismatch")
    if not isinstance(coverage, Mapping) or coverage.get("required") is not True:
        raise ValueError("official render manifest does not require asset meshes")
    if (
        coverage.get("bbox_proxy_count") != 0
        or coverage.get("asset_mesh_count") != coverage.get("object_count")
    ):
        raise ValueError("official render manifest has incomplete asset coverage")
    limits = (
        export.get("limits") if isinstance(export, Mapping) else None
    )
    expected_limits = {
        "max_vertices_per_object": OFFICIAL_RENDER_PROFILE[
            "collision_max_vertices_per_object"
        ],
        "max_faces_per_object": OFFICIAL_RENDER_PROFILE[
            "collision_max_faces_per_object"
        ],
        "max_total_vertices": OFFICIAL_RENDER_PROFILE[
            "collision_max_total_vertices"
        ],
        "max_total_faces": OFFICIAL_RENDER_PROFILE[
            "collision_max_total_faces"
        ],
    }
    if not isinstance(export, Mapping) or export.get("status") != "completed" or limits != expected_limits:
        raise ValueError("official collision export profile mismatch")
    objects = collision_manifest.get("objects")
    if not isinstance(objects, Mapping) or any(
        not isinstance(row, Mapping)
        or row.get("representation") != "triangle_mesh"
        or row.get("complete") is not True
        for row in objects.values()
    ):
        raise ValueError("official collision geometry coverage is incomplete")
    _validate_collision_decimation_audit(
        objects,
        collision_manifest.get("export_summary"),
    )


def _validate_collision_decimation_audit(
    objects: Mapping[str, Any],
    summary: Any,
) -> None:
    applied = []
    for row in objects.values():
        if not isinstance(row, Mapping):
            continue
        audit = row.get("decimation")
        if not isinstance(audit, Mapping) or audit.get("applied") is not True:
            continue
        applied.append((row, audit))

    if isinstance(summary, Mapping):
        reported = summary.get("decimated_mesh_count")
        if reported is not None and reported != len(applied):
            raise ValueError("official collision decimation count mismatch")
    if not applied:
        return

    for row, audit in applied:
        if (
            row.get("collision_mesh_source") != "deterministic_decimated_copy"
            or audit.get("schema_version")
            != COLLISION_DECIMATION_POLICY_VERSION
            or audit.get("algorithm") != "blender_decimate_collapse"
            or audit.get("deterministic_object_order") is not True
            or audit.get("triangulate_after_collapse") is not True
            or audit.get("visible_scene_modified") is not False
            or audit.get("saved_blend_modified") is not False
        ):
            raise ValueError("official collision decimation policy mismatch")
        counts = {
            field: audit.get(field)
            for field in (
                "original_vertex_count",
                "original_face_count",
                "final_vertex_count",
                "final_face_count",
            )
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in counts.values()
        ):
            raise ValueError("official collision decimation counts are invalid")
        if (
            counts["final_vertex_count"] != row.get("vertex_count")
            or counts["final_face_count"] != row.get("face_count")
            or (
                counts["final_vertex_count"]
                >= counts["original_vertex_count"]
                and counts["final_face_count"]
                >= counts["original_face_count"]
            )
            or counts["final_vertex_count"]
            > OFFICIAL_RENDER_PROFILE["collision_max_vertices_per_object"]
            or counts["final_face_count"]
            > OFFICIAL_RENDER_PROFILE["collision_max_faces_per_object"]
        ):
            raise ValueError("official collision decimation result is invalid")
        passes = audit.get("passes")
        if not isinstance(passes, list) or not passes:
            raise ValueError("official collision decimation pass audit is absent")
        for item in passes:
            ratio = item.get("ratio") if isinstance(item, Mapping) else None
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or not 0.0 < float(ratio) < 1.0
            ):
                raise ValueError("official collision decimation ratio is invalid")


__all__ = [
    "OFFICIAL_RENDER_PROFILE",
    "OFFICIAL_RENDER_PROFILE_ID",
    "validate_official_renderer",
    "validate_official_render_manifest",
]
