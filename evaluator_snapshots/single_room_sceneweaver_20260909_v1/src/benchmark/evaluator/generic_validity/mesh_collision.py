"""Format-agnostic triangle-mesh narrow-phase collision backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.mesh_geometry import (
    TRIANGLE_MESH_REPRESENTATION,
    geometry_unavailable_reason,
    is_usable_triangle_mesh,
    load_triangle_mesh,
)


MESH_COLLISION_BACKEND_VERSION = "mesh_collision_v2"
MESH_STATES = ("separated", "near_contact", "surface_intersection", "contained", "invalid_mesh", "unknown")

# Distance backends whose reported distance may be trusted to certify separation.
# ``trimesh_fcl`` returns the exact minimum surface distance; ``numpy_aabb_gap``
# returns the gap between the world-space AABBs, which is a conservative lower
# bound on the true surface distance (mesh subset of its AABB). Vertex-sampled
# distances OVERestimate the true surface distance and must never be trusted for
# the separation fast path.
TRUSTED_DISTANCE_BACKENDS = frozenset({"trimesh_fcl", "numpy_aabb_gap"})


def evaluate_mesh_pair(
    object_a_id: str,
    object_b_id: str,
    entry_a: dict[str, Any] | None,
    entry_b: dict[str, Any] | None,
    *,
    base_dir: Path | None = None,
    separation_threshold_m: float = 0.02,
) -> dict[str, Any]:
    """Evaluate one OBB-overlapping pair when mesh evidence may be available."""

    if not is_usable_triangle_mesh(entry_a, base_dir=base_dir) or not is_usable_triangle_mesh(entry_b, base_dir=base_dir):
        return {
            "backend": MESH_COLLISION_BACKEND_VERSION,
            "status": "unavailable",
            "mesh_state": "unknown",
            "mesh_reliable_for_separation": False,
            "minimum_surface_distance_m": None,
            "surface_intersection": None,
            "containment_a_in_b": None,
            "containment_b_in_a": None,
            "reason_a": geometry_unavailable_reason(entry_a),
            "reason_b": geometry_unavailable_reason(entry_b),
            "object_a_id": object_a_id,
            "object_b_id": object_b_id,
            "evidence_level": "obb",
        }

    try:
        mesh_a = load_triangle_mesh(entry_a, base_dir=base_dir)
        mesh_b = load_triangle_mesh(entry_b, base_dir=base_dir)
    except Exception as exc:
        return {
            "backend": MESH_COLLISION_BACKEND_VERSION,
            "status": "invalid_mesh",
            "mesh_state": "invalid_mesh",
            "mesh_reliable_for_separation": False,
            "minimum_surface_distance_m": None,
            "surface_intersection": None,
            "containment_a_in_b": None,
            "containment_b_in_a": None,
            "object_a_id": object_a_id,
            "object_b_id": object_b_id,
            "evidence_level": "mesh",
            "error": f"{type(exc).__name__}: {exc}",
        }
    validity_a = _mesh_validity(mesh_a, entry_a)
    validity_b = _mesh_validity(mesh_b, entry_b)
    if not validity_a["usable"] or not validity_b["usable"]:
        return {
            "backend": MESH_COLLISION_BACKEND_VERSION,
            "status": "invalid_mesh",
            "mesh_state": "invalid_mesh",
            "mesh_reliable_for_separation": False,
            "minimum_surface_distance_m": None,
            "surface_intersection": None,
            "containment_a_in_b": None,
            "containment_b_in_a": None,
            "validity": {"object_a": validity_a, "object_b": validity_b},
            "object_a_id": object_a_id,
            "object_b_id": object_b_id,
            "evidence_level": "mesh",
        }

    distance = _minimum_surface_distance(mesh_a, mesh_b)
    intersection = _surface_intersection(mesh_a, mesh_b)
    containment_a_in_b = _containment(mesh_a["vertices"], mesh_b)
    containment_b_in_a = _containment(mesh_b["vertices"], mesh_a)
    focus = _focus_region(distance, intersection, containment_a_in_b, containment_b_in_a)

    minimum_distance = distance.get("distance_m")
    # ``None`` means the fallback broad phase found overlapping AABBs but did
    # not establish a real surface intersection.  Only literal True is an
    # observed intersection.
    intersects = intersection.get("intersects") is True
    reliable = _mesh_reliable_for_separation(
        minimum_distance=minimum_distance,
        intersects=intersects,
        intersection_definitive=bool(intersection.get("definitive")),
        containment_a_in_b=containment_a_in_b,
        containment_b_in_a=containment_b_in_a,
        distance_status=distance.get("status"),
        distance_backend=distance.get("backend"),
        separation_threshold_m=separation_threshold_m,
    )
    mesh_state = _mesh_state(
        minimum_distance=minimum_distance,
        intersects=intersects,
        containment_a_in_b=containment_a_in_b,
        containment_b_in_a=containment_b_in_a,
        intersection_definitive=bool(intersection.get("definitive")),
        separation_threshold_m=separation_threshold_m,
    )
    return {
        "backend": MESH_COLLISION_BACKEND_VERSION,
        "status": "evaluated",
        "mesh_state": mesh_state,
        "mesh_reliable_for_separation": reliable,
        "minimum_surface_distance_m": minimum_distance,
        "minimum_surface_distance_backend": distance.get("backend"),
        "minimum_surface_distance_is_lower_bound": bool(distance.get("is_lower_bound")),
        "closest_points": distance.get("closest_points"),
        "surface_intersection": intersection.get("intersects"),
        "candidate_aabb_overlap": bool(intersection.get("candidate_aabb_overlap")),
        "intersection": intersection,
        "containment_a_in_b": containment_a_in_b,
        "containment_b_in_a": containment_b_in_a,
        "validity": {"object_a": validity_a, "object_b": validity_b},
        "focus_region": focus,
        "object_a_id": object_a_id,
        "object_b_id": object_b_id,
        "evidence_level": "mesh",
        "geometry_source": {
            "object_a": entry_a.get("geometry_source"),
            "object_b": entry_b.get("geometry_source"),
        },
    }


def _mesh_validity(mesh: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(mesh["vertices"], dtype=float)
    faces = np.asarray(mesh["faces"], dtype=int)
    diagnostics: dict[str, Any] = {
        "representation": entry.get("representation", TRIANGLE_MESH_REPRESENTATION),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "finite_coordinates": bool(np.all(np.isfinite(vertices))) if len(vertices) else False,
        "valid_face_indices": bool(np.all((faces >= 0) & (faces < len(vertices)))) if len(faces) else False,
        "degenerate_face_count": 0,
        "watertight": None,
        "winding_consistent": None,
        "volume_valid": None,
    }
    if len(faces):
        degenerate = 0
        for face in faces:
            triangle = vertices[face]
            area = float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))) * 0.5
            if area <= 1.0e-12:
                degenerate += 1
        diagnostics["degenerate_face_count"] = degenerate
    usable = (
        diagnostics["vertex_count"] > 0
        and diagnostics["face_count"] > 0
        and diagnostics["finite_coordinates"]
        and diagnostics["valid_face_indices"]
        and diagnostics["degenerate_face_count"] == 0
    )
    if usable:
        try:
            import trimesh  # type: ignore

            tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            diagnostics["watertight"] = bool(tm.is_watertight)
            diagnostics["winding_consistent"] = bool(tm.is_winding_consistent)
            diagnostics["volume_valid"] = bool(tm.is_volume)
        except Exception:
            diagnostics["watertight"] = None
            diagnostics["winding_consistent"] = None
            diagnostics["volume_valid"] = None
    diagnostics["usable"] = usable
    return diagnostics


def _minimum_surface_distance(mesh_a: dict[str, Any], mesh_b: dict[str, Any]) -> dict[str, Any]:
    try:
        import trimesh  # type: ignore

        tm_a = trimesh.Trimesh(vertices=mesh_a["vertices"], faces=mesh_a["faces"], process=False)
        tm_b = trimesh.Trimesh(vertices=mesh_b["vertices"], faces=mesh_b["faces"], process=False)
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", tm_a)
        manager.add_object("b", tm_b)
        if manager.in_collision_internal():
            return {"status": "ok", "distance_m": 0.0, "closest_points": None, "backend": "trimesh_fcl", "is_lower_bound": False}
        distance = float(manager.min_distance_internal())
        return {"status": "ok", "distance_m": distance, "closest_points": None, "backend": "trimesh_fcl", "is_lower_bound": False}
    except Exception:
        return _minimum_surface_distance_aabb(mesh_a, mesh_b)


def _minimum_surface_distance_aabb(mesh_a: dict[str, Any], mesh_b: dict[str, Any]) -> dict[str, Any]:
    """Distance between world-space AABBs: a conservative lower bound on the true
    surface distance because each mesh is a subset of its own AABB."""

    vertices_a = np.asarray(mesh_a["vertices"], dtype=float)
    vertices_b = np.asarray(mesh_b["vertices"], dtype=float)
    if len(vertices_a) == 0 or len(vertices_b) == 0:
        return {"status": "failed", "distance_m": None, "closest_points": None, "backend": "numpy_aabb_gap", "error": "empty mesh"}
    min_a, max_a = vertices_a.min(axis=0), vertices_a.max(axis=0)
    min_b, max_b = vertices_b.min(axis=0), vertices_b.max(axis=0)
    per_axis_gap = np.maximum.reduce([min_a - max_b, min_b - max_a, np.zeros(3, dtype=float)])
    distance = float(np.linalg.norm(per_axis_gap))
    return {
        "status": "ok",
        "distance_m": distance,
        "closest_points": None,
        "backend": "numpy_aabb_gap",
        "is_lower_bound": True,
        "note": "axis-aligned AABB gap; conservative lower bound on true surface distance",
    }


def _surface_intersection(mesh_a: dict[str, Any], mesh_b: dict[str, Any]) -> dict[str, Any]:
    try:
        import trimesh  # type: ignore

        tm_a = trimesh.Trimesh(vertices=mesh_a["vertices"], faces=mesh_a["faces"], process=False)
        tm_b = trimesh.Trimesh(vertices=mesh_b["vertices"], faces=mesh_b["faces"], process=False)
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", tm_a)
        manager.add_object("b", tm_b)
        intersects = bool(manager.in_collision_internal())
        # CollisionManager establishes the boolean result here, but this call
        # does not return a contact position.  A centroid midpoint is not a
        # contact point and must not be presented to camera/VLM consumers as
        # one.  Keep an explicitly approximate focus hint instead.
        focus_hint = _aabb_overlap_focus_hint(mesh_a, mesh_b) if intersects else None
        return {
            "intersects": intersects,
            "definitive": True,
            "contact_count": 0,
            "contacts": [],
            "contact_geometry_available": False,
            "focus_hint": focus_hint,
            "backend": "trimesh_fcl",
        }
    except Exception:
        return _surface_intersection_aabb(mesh_a, mesh_b)


def _surface_intersection_aabb(mesh_a: dict[str, Any], mesh_b: dict[str, Any]) -> dict[str, Any]:
    """AABB overlap is a conservative over-approximation of surface intersection:
    separated AABBs prove no intersection (``definitive``); overlapping AABBs
    are candidates only and never claim a surface intersection or contact."""

    vertices_a = np.asarray(mesh_a["vertices"], dtype=float)
    vertices_b = np.asarray(mesh_b["vertices"], dtype=float)
    mins_a = vertices_a.min(axis=0)
    maxs_a = vertices_a.max(axis=0)
    mins_b = vertices_b.min(axis=0)
    maxs_b = vertices_b.max(axis=0)
    separated = bool(np.any(maxs_a < mins_b) or np.any(maxs_b < mins_a))
    candidate_overlap = not separated
    return {
        "intersects": False if separated else None,
        "definitive": separated,
        "candidate_aabb_overlap": candidate_overlap,
        "contact_count": 0,
        "contacts": [],
        "contact_geometry_available": False,
        "focus_hint": _aabb_overlap_focus_hint(mesh_a, mesh_b) if candidate_overlap else None,
        "backend": "numpy_aabb_overlap",
    }


def _aabb_overlap_focus_hint(mesh_a: dict[str, Any], mesh_b: dict[str, Any]) -> dict[str, Any] | None:
    """Return an approximate overlap-region center, explicitly not a contact."""

    vertices_a = np.asarray(mesh_a["vertices"], dtype=float)
    vertices_b = np.asarray(mesh_b["vertices"], dtype=float)
    if len(vertices_a) == 0 or len(vertices_b) == 0:
        return None
    overlap_min = np.maximum(vertices_a.min(axis=0), vertices_b.min(axis=0))
    overlap_max = np.minimum(vertices_a.max(axis=0), vertices_b.max(axis=0))
    if np.any(overlap_min > overlap_max):
        return None
    return {
        "point": ((overlap_min + overlap_max) * 0.5).tolist(),
        "source": "aabb_overlap_center",
        "is_contact": False,
    }


def _containment(inner_vertices: np.ndarray, outer_mesh: dict[str, Any]) -> str | bool:
    try:
        import trimesh  # type: ignore

        outer = trimesh.Trimesh(vertices=outer_mesh["vertices"], faces=outer_mesh["faces"], process=False)
        if not outer.is_watertight:
            return "unknown"
        # Exported closed meshes can be watertight while carrying inconsistent
        # face winding.  Repair normals on this private trimesh copy before the
        # volume/contains checks; never mutate the canonical mesh evidence.
        if not outer.is_winding_consistent or not outer.is_volume:
            outer.fix_normals(multibody=True)
        if not outer.is_winding_consistent or not outer.is_volume:
            return "unknown"
        inside = outer.contains(inner_vertices)
        if inside.size == 0:
            return "unknown"
        if bool(np.all(inside)):
            return True
        if bool(np.all(~inside)):
            return False
        return "unknown"
    except Exception:
        return _containment_numpy(inner_vertices, outer_mesh["vertices"])


def _containment_numpy(inner_vertices: np.ndarray, outer_vertices: np.ndarray) -> str | bool:
    outer_min = outer_vertices.min(axis=0)
    outer_max = outer_vertices.max(axis=0)
    inner_min = inner_vertices.min(axis=0)
    inner_max = inner_vertices.max(axis=0)
    # AABB nesting cannot prove mesh containment: the outer mesh may be hollow,
    # concave, or only a sparse shell.  Disjoint AABBs do prove non-containment;
    # all overlapping/nested cases remain unknown for conservative routing.
    if np.any(inner_max < outer_min) or np.any(inner_min > outer_max):
        return False
    return "unknown"


def _mesh_reliable_for_separation(
    *,
    minimum_distance: float | None,
    intersects: bool,
    intersection_definitive: bool,
    containment_a_in_b: str | bool,
    containment_b_in_a: str | bool,
    distance_status: str | None,
    distance_backend: str | None,
    separation_threshold_m: float,
) -> bool:
    if distance_status != "ok" or minimum_distance is None or not math.isfinite(float(minimum_distance)):
        return False
    if distance_backend not in TRUSTED_DISTANCE_BACKENDS:
        return False
    if intersects or not intersection_definitive:
        return False
    if containment_a_in_b is True or containment_b_in_a is True:
        return False
    if containment_a_in_b == "unknown" or containment_b_in_a == "unknown":
        return False
    return float(minimum_distance) > float(separation_threshold_m)


def _mesh_state(
    *,
    minimum_distance: float | None,
    intersects: bool,
    containment_a_in_b: str | bool,
    containment_b_in_a: str | bool,
    intersection_definitive: bool,
    separation_threshold_m: float,
) -> str:
    if containment_a_in_b is True or containment_b_in_a is True:
        return "contained"
    if intersects:
        return "surface_intersection"
    if not intersection_definitive:
        return "unknown"
    # Exact positive shell distance does not rule out one closed mesh being
    # wholly contained inside another.  Preserve the conservative state until
    # both directional containment checks are definitive.
    if containment_a_in_b == "unknown" or containment_b_in_a == "unknown":
        return "unknown"
    if minimum_distance is None or not math.isfinite(float(minimum_distance)):
        return "unknown"
    if float(minimum_distance) <= float(separation_threshold_m):
        return "near_contact"
    return "separated"


def _focus_region(distance: dict[str, Any], intersection: dict[str, Any], containment_a: str | bool, containment_b: str | bool) -> dict[str, Any]:
    if isinstance(intersection.get("contacts"), list) and intersection["contacts"]:
        point = intersection["contacts"][0].get("point")
        if isinstance(point, list) and len(point) == 3:
            return {"center": point, "radius_m": 0.35, "source": "intersection_contact"}
    focus_hint = intersection.get("focus_hint")
    if isinstance(focus_hint, dict):
        point = focus_hint.get("point")
        if isinstance(point, list) and len(point) == 3:
            return {
                "center": point,
                "radius_m": 0.35,
                "source": "candidate_overlap_region",
                "is_contact": False,
            }
    closest = distance.get("closest_points")
    if isinstance(closest, dict):
        point_a = closest.get("object_a")
        point_b = closest.get("object_b")
        if isinstance(point_a, list) and isinstance(point_b, list) and len(point_a) == 3 and len(point_b) == 3:
            center = [(float(point_a[index]) + float(point_b[index])) * 0.5 for index in range(3)]
            return {"center": center, "radius_m": 0.35, "source": "closest_points"}
    if containment_a is True or containment_b is True:
        return {"center": None, "radius_m": 0.5, "source": "containment"}
    return {"center": None, "radius_m": 0.5, "source": "obb_overlap_fallback"}
