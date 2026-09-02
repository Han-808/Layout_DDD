"""Architecture evidence adapter for polygon-room Support evaluation."""

from __future__ import annotations

from typing import Any

from benchmark.non_rectangular.geometry import (
    object_footprint_polygon,
    object_vertical_interval,
    polygon_geometry_from_scene,
)


POLYGON_SUPPORT_ARCHITECTURE_VERSION = (
    "non_rectangular_polygon_support_architecture_v1"
)


def polygon_support_architecture_evidence(
    scene: dict[str, Any],
    *,
    raw_object: dict[str, Any],
    tolerance_m: float,
) -> dict[str, Any] | None:
    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        return None
    footprint = object_footprint_polygon(raw_object)
    minimum_z, _ = object_vertical_interval(raw_object)
    measurements = geometry.wall_measurements(footprint)
    nearest = geometry.nearest_wall_measurement(footprint)
    contacts = [
        {
            "plane": str(item["wall_id"]),
            "wall_id": str(item["wall_id"]),
            "signed_clearance_m": float(item["signed_clearance_m"]),
            "distance_m": float(item["distance_m"]),
            "mode": "wall_attachment",
            "inward_normal_xy": list(item["inward_normal_xy"]),
            "tangent_xy": list(item["tangent_xy"]),
        }
        for item in measurements
        if float(item["distance_m"]) <= float(tolerance_m)
    ]
    return {
        "schema_version": POLYGON_SUPPORT_ARCHITECTURE_VERSION,
        "floor_z_m": geometry.floor_z_m,
        "ceiling_in_scope": False,
        "active_physical_wall_ids": [wall.wall_id for wall in geometry.walls],
        "architecture_plane_clearances_m": {
            "floor": float(minimum_z - geometry.floor_z_m),
            **{
                str(item["wall_id"]): float(item["signed_clearance_m"])
                for item in measurements
            },
        },
        "wall_measurements": measurements,
        "architecture_contact_candidates": contacts,
        "nearest_logical_wall_measurement": nearest,
    }


__all__ = [
    "POLYGON_SUPPORT_ARCHITECTURE_VERSION",
    "polygon_support_architecture_evidence",
]
