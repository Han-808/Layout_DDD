"""Polygon OOB measurements only; common OOB owns judgement and scoring."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

from benchmark.evaluator.generic_validity.geometry import get_obb_corners


def measure_polygon_obb(obj, geometry, *, eps: float, floor_tol: float) -> dict:
    corners = get_obb_corners(obj)
    footprint = MultiPoint(corners[:, :2]).convex_hull
    outside = footprint.difference(geometry.polygon.buffer(eps, join_style="mitre"))
    horizontal = outside.area > max(eps * eps, footprint.area * 1e-10)
    minimum, maximum = np.min(corners, axis=0), np.max(corners, axis=0)
    floor = max(0.0, geometry.floor_z_m - float(minimum[2]))
    ceiling = max(0.0, float(maximum[2]) - geometry.ceiling_z_m) if geometry.ceiling_in_scope else 0.0
    edges = []
    for measurement in geometry.violated_walls(footprint) if horizontal else []:
        wall = geometry.wall_by_id(measurement["wall_id"])
        crossing = footprint.boundary.intersection(wall.line)
        wall_focus, focus = nearest_points(wall.line, footprint) if crossing.is_empty else (crossing.centroid, crossing.centroid)
        edges.append({**deepcopy(measurement), "height_m": wall.height_m,
            "focus_xy": [focus.x, focus.y], "wall_focus_xy": [wall_focus.x, wall_focus.y],
            "edge_local_frame": {"inward_normal_xy": list(wall.inward_normal_xy), "tangent_xy": list(wall.tangent_xy)}})
    flags = {"horizontal_oob": bool(horizontal), "floor_oob": floor > floor_tol + eps,
             "ceiling_oob": geometry.ceiling_in_scope and ceiling > eps}
    # Rectangle-sized AABB is a bound only, never the containment predicate.
    depth = _outside_depth(outside, geometry.polygon.boundary)
    evidence = {"room_geometry": geometry.public_dict(), "outside_area_m2": float(outside.area),
        "outside_area_ratio": float(outside.area / footprint.area) if footprint.area else 0.0,
        "maximum_horizontal_penetration_m": depth, "violated_edges": edges,
        "violated_wall_ids": [edge["wall_id"] for edge in edges],
        "footprint_xy": [list(p) for p in list(footprint.exterior.coords)[:-1]],
        # Measured along actual wall normals, compatible with the common OOB
        # full-severity ratio; no independent polygon penalty or weight.
        "boundary_penetration_ratios": [
            _edge_ratio(obj, geometry, footprint, edge) for edge in edges],
    }
    penetration = {"horizontal_oob": depth, "floor_oob": floor, "ceiling_oob": ceiling}
    return {"object_id": obj.id, "obb_intervals": {axis: [float(minimum[i]), float(maximum[i])]
            for i, axis in enumerate("xyz")}, "plane_flags": flags,
        "plane_penetration_m": penetration, "crossing_depths_m": {k: v for k, v in penetration.items() if flags[k]},
        "within_floor_contact_tolerance": eps < floor <= floor_tol + eps,
        "floor_contact_tolerance_m": floor_tol, "numerical_eps": eps,
        "floor_penetration_m": floor, "candidate_oob": any(flags.values()),
        "requires_vlm": False, "route": None, "final_verdict": None,
        "affects_oob_score": False, "judge_result": None, "adjudication_error": None,
        **evidence}


def _edge_ratio(obj, geometry, footprint, edge):
    normal = np.asarray(edge["inward_normal_xy"], dtype=float)
    extent = 2.0 * float(np.sum(np.abs(np.asarray(obj.R)[:2, :].T @ normal) * obj.half))
    # A concave wall's infinite half-plane is NOT a room boundary. Measure only
    # the footprint portion outside the polygon and local to this wall.
    wall = geometry.wall_by_id(edge["wall_id"])
    outside = footprint.difference(geometry.polygon)
    distances = []
    for component in getattr(outside, "geoms", [outside]):
        if hasattr(component, "exterior"):
            for point in component.exterior.coords:
                nearest = nearest_points(geometry.polygon.boundary, Point(point))[0]
                if wall.line.distance(nearest) <= geometry.tolerance_m * 4:
                    distances.append(max(0.0, -float(np.dot(np.asarray(point) - np.asarray(wall.start_xy), normal))))
    return max(distances, default=0.0) / max(extent, 1e-9)


def _outside_depth(outside, boundary):
    points = []
    for component in getattr(outside, "geoms", [outside]):
        if hasattr(component, "exterior"):
            points.extend(component.exterior.coords)
    return max((Point(point).distance(boundary) for point in points), default=0.0)
