"""Convert a browser game-scene probe payload into canonical benchmark artifacts.

A game level cannot be expressed losslessly as ``canonical_scene_v1``: that
contract is an oriented-bounding-box summary of a single-storey rectangular
room. The evaluator, however, consumes three separate channels, and only the
first one is lossy:

* ``canonical_scene_v1`` -- the OBB index, used for broad-phase geometry.
* ``collision_geometry_v1`` -- world-baked triangle meshes, used for the
  narrow phase. This is the geometry the metrics actually trust.
* rendered images -- real appearance, judged by the VLM.

This module builds the first two from one probe payload so that the OBB index
and the triangle meshes cannot disagree. Both are derived from the same
vertices in the same pass, which is what keeps the collision narrow phase's
mesh-enclosure guard satisfied by construction rather than by convention.

The probe reports three.js world-space data in the game's own frame. Everything
that gets applied on the way to canonical form -- unit scale, up-axis change,
and the translation that puts the level's minimum corner at the origin -- is
recorded in ``metadata.game_scene_import`` so the mapping stays invertible.

Object individualization also finishes here. The probe emits one entry per
visible mesh because that is the only level of a three.js graph that every
implementation necessarily has; deciding what constitutes *one object* is then
done from geometry, which is independent of how the game happened to organise
its graph. Two rules apply, and both are definitions this benchmark owns rather
than attempts to recover an author's intent:

* a mesh that draws only back faces is not a solid, and is dropped;
* a mesh whose fitted box lies entirely inside another's is interior detail of
  that other object, and is absorbed into it.

Every drop and every absorption is counted in
``metadata.game_scene_import.individualization`` so that a level which collapses
to a single object is visible as such instead of hiding behind a high score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.mesh_geometry import (
    BBOX_PROXY_REPRESENTATION,
    COLLISION_GEOMETRY_SCHEMA_VERSION,
    TRIANGLE_MESH_REPRESENTATION,
    write_ascii_triangle_ply,
)
from benchmark.scene_io.validate import GENERATED_MESH_GEOMETRY


GAME_SCENE_PROBE_VERSION = "game_scene_probe_v1"
SUPPORTED_UP_AXES = ("y", "z")

# Flat authored geometry (ground planes, decals) has a zero extent on one axis,
# but canonical size components must be strictly positive.
DEFAULT_MIN_EXTENT_M = 1.0e-3

# Where that fabricated extent is placed decides whether the box lies. Centring
# it puts half the padding on the visible side of the authored surface, so every
# object standing on a ground plane reports a penetration of half the minimum
# extent that exists in no game. Placing it behind the surface instead leaves the
# authored surface as a face of the box, and resting contact reads as zero
# overlap. Which side is "behind" is read from the mesh: triangle normals of a
# closed solid cancel out, those of a one-sided sheet reinforce each other, so
# the ratio of the summed normal to the summed area separates the two cases
# without having to know in advance which kind of mesh this is. Anything less
# coherent than this keeps the centred placement.
COHERENT_FACING_RATIO = 0.9

# Containment is tested with a tolerance scaled to the container so that a child
# sharing a face with its parent still counts as inside. It stays near float
# noise: the rule is "entirely inside", not "mostly inside".
CONTAINMENT_RELATIVE_EPS = 1.0e-9

# An outline shell is a slightly inflated copy of the mesh it decorates, so it
# fills most of its own volume with that mesh. A skybox or an inward-facing room
# shell contains its occupants loosely, by orders of magnitude. A back-face-only
# mesh that encloses something without tightly wrapping it is therefore reported
# as a possible level boundary rather than dropped in silence. The separation
# between the two cases is wide -- roughly 0.8 against 0.001 -- so the exact
# value here is not load-bearing.
TIGHT_WRAP_VOLUME_RATIO = 0.25

BACK_SIDE_ONLY_HINT = "back_side_only"

ALLOWED_ENTITY_KINDS = frozenset(
    {
        "static",
        "dynamic_environment",
        "dynamic_actor",
        "dynamic_player",
        "projectile",
        "effect",
        "transient_helper",
        "helper",
        "viewmodel",
    }
)
RUNTIME_EXCLUDED_ENTITY_KINDS = frozenset(
    {
        "dynamic_actor",
        "dynamic_player",
        "projectile",
        "effect",
        "transient_helper",
        "helper",
        "viewmodel",
    }
)

# Change of basis from a three.js Y-up frame to the canonical Z-up frame.
# Equivalent to a +90 degree rotation about X: (x, y, z) -> (x, -z, y).
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
_IDENTITY = np.eye(3, dtype=float)


class GameSceneExportError(ValueError):
    """Raised when a probe payload cannot produce canonical artifacts."""


def validate_probe_payload(payload: Any) -> dict[str, Any]:
    """Validate a ``game_scene_probe_v1`` payload and return it unchanged."""

    if not isinstance(payload, dict):
        raise GameSceneExportError("probe payload must be a JSON object")
    if payload.get("schema_version") != GAME_SCENE_PROBE_VERSION:
        raise GameSceneExportError(
            f"probe schema_version must be {GAME_SCENE_PROBE_VERSION!r}"
        )
    up_axis = str(payload.get("up_axis") or "").lower()
    if up_axis not in SUPPORTED_UP_AXES:
        raise GameSceneExportError(f"probe up_axis must be one of {list(SUPPORTED_UP_AXES)}")
    unit_scale = payload.get("unit_scale", 1.0)
    if isinstance(unit_scale, bool) or not isinstance(unit_scale, (int, float)):
        raise GameSceneExportError("probe unit_scale must be numeric")
    if not math.isfinite(float(unit_scale)) or float(unit_scale) <= 0.0:
        raise GameSceneExportError("probe unit_scale must be finite and positive")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise GameSceneExportError("probe objects must be a non-empty list")
    seen: set[str] = set()
    for index, entry in enumerate(objects):
        path = f"probe objects[{index}]"
        if not isinstance(entry, dict):
            raise GameSceneExportError(f"{path} must be a JSON object")
        object_id = str(entry.get("id") or "").strip()
        if not object_id:
            raise GameSceneExportError(f"{path}.id must be a non-empty string")
        if object_id in seen:
            raise GameSceneExportError(f"{path}.id duplicates {object_id!r}")
        seen.add(object_id)
        if not str(entry.get("category") or "").strip():
            raise GameSceneExportError(
                f"{path}.category must be a non-empty string; the exporter never invents labels"
            )
        _validate_quaternion(entry.get("rotation_quaternion"), path)
        has_vertices = isinstance(entry.get("vertices"), list) and entry["vertices"]
        has_bounds = isinstance(entry.get("world_bounds"), dict)
        if not has_vertices and not has_bounds:
            raise GameSceneExportError(
                f"{path} must supply either vertices or world_bounds"
            )
        if has_bounds:
            _validate_bounds(entry["world_bounds"], f"{path}.world_bounds")
        hint = entry.get("non_physical_hint")
        if hint is not None and hint != BACK_SIDE_ONLY_HINT:
            raise GameSceneExportError(
                f"{path}.non_physical_hint must be null or {BACK_SIDE_ONLY_HINT!r}"
            )
        entity_kind = str(entry.get("entity_kind") or "static").strip().lower()
        if entity_kind not in ALLOWED_ENTITY_KINDS:
            raise GameSceneExportError(
                f"{path}.entity_kind must be one of {sorted(ALLOWED_ENTITY_KINDS)}"
            )
        runtime_role = entry.get("runtime_role")
        if runtime_role is not None:
            _validate_runtime_role(runtime_role, path=f"{path}.runtime_role")
            if runtime_role["classification"] != entity_kind:
                raise GameSceneExportError(
                    f"{path}.runtime_role.classification must match entity_kind"
                )
    return payload


def build_scene_and_collision_geometry(
    payload: dict[str, Any],
    *,
    scene_id: str,
    request_id: str,
    scene_type: str,
    mesh_dir: str | Path | None = None,
    min_extent_m: float = DEFAULT_MIN_EXTENT_M,
    drop_non_physical: bool = True,
    collapse_contained: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(canonical_scene, collision_geometry_manifest)`` for one probe."""

    validate_probe_payload(payload)
    unit_scale = float(payload.get("unit_scale", 1.0))
    basis = _Y_UP_TO_Z_UP if str(payload["up_axis"]).lower() == "y" else _IDENTITY

    resolved_all: list[dict[str, Any]] = []
    for entry in payload["objects"]:
        resolved_entry = _resolve_object(entry, basis=basis, unit_scale=unit_scale, min_extent_m=min_extent_m)
        if resolved_entry is not None:
            resolved_all.append(resolved_entry)
    if not resolved_all:
        raise GameSceneExportError("probe produced no objects with usable geometry")

    resolved, runtime_filter = _filter_runtime_entities(resolved_all)
    if not resolved:
        raise GameSceneExportError(
            "runtime entity filtering removed every probed mesh; the level has "
            "no static environment geometry"
        )
    resolved, individualization = _individualize(
        resolved,
        probe_report=payload.get("individualization"),
        runtime_filter=runtime_filter,
        min_extent_m=min_extent_m,
        drop_non_physical=drop_non_physical,
        collapse_contained=collapse_contained,
    )
    if not resolved:
        raise GameSceneExportError(
            "every probed mesh was classified as non-physical; the level has no solid geometry"
        )

    corners = np.concatenate([item["corners"] for item in resolved], axis=0)
    translation = -corners.min(axis=0)
    extents = corners.max(axis=0) + translation
    scene_height = float(extents[2])
    if not math.isfinite(scene_height) or scene_height <= 0.0:
        raise GameSceneExportError("probe geometry has no positive vertical extent")
    width = float(extents[0])
    depth = float(extents[1])
    if width <= 0.0 or depth <= 0.0:
        raise GameSceneExportError("probe geometry has no positive horizontal extent")

    mesh_root = Path(mesh_dir).expanduser().resolve() if mesh_dir is not None else None
    objects: list[dict[str, Any]] = []
    geometry_entries: dict[str, dict[str, Any]] = {}
    for item in resolved:
        object_id = item["id"]
        center = (item["center"] + translation).tolist()
        objects.append(
            {
                "id": object_id,
                "category": item["category"],
                "size": item["size"].tolist(),
                "center": [float(value) for value in center],
                "rotation": [float(value) for value in item["rotation_degrees"]],
                "geometry_provenance": GENERATED_MESH_GEOMETRY,
                "entity_kind": item["entity_kind"],
                "source_names": item["source_names"],
                "metadata": {
                    "category_source": item["category_source"],
                    "absorbed_ids": item.get("absorbed_ids") or [],
                    "runtime_role": item.get("runtime_role"),
                    "material_visibility": item.get("material_visibility"),
                    "degenerate_axis_padding": item.get("degenerate_axis_padding"),
                },
            }
        )
        geometry_entries[object_id] = _collision_entry(
            item,
            translation=translation,
            mesh_root=mesh_root,
        )

    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": scene_id,
        "request_id": request_id,
        "scene_type": scene_type,
        "boundary": [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]],
        "scene_height": scene_height,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "game_scene_import": {
                "probe_schema_version": GAME_SCENE_PROBE_VERSION,
                "source_up_axis": str(payload["up_axis"]).lower(),
                "unit_scale": unit_scale,
                "translation_applied": [float(value) for value in translation],
                "captured_at_tick": payload.get("captured_at_tick"),
                "deterministic_seed": payload.get("deterministic_seed"),
                "boundary_source": "object_extent_bounding_rectangle",
                "individualization": individualization,
            },
        },
    }
    manifest = {
        "schema_version": COLLISION_GEOMETRY_SCHEMA_VERSION,
        "units": "meter",
        "up_axis": "z",
        "objects": geometry_entries,
    }
    return scene, manifest


def _resolve_object(
    entry: dict[str, Any],
    *,
    basis: np.ndarray,
    unit_scale: float,
    min_extent_m: float,
) -> dict[str, Any] | None:
    """Derive a canonical-frame OBB plus optional baked triangles for one object."""

    object_id = str(entry["id"]).strip()
    faces = entry.get("faces")
    raw_vertices = entry.get("vertices")
    complete_mesh = (
        isinstance(raw_vertices, list)
        and bool(raw_vertices)
        and isinstance(faces, list)
        and bool(faces)
        and entry.get("mesh_complete") is not False
    )

    if isinstance(raw_vertices, list) and raw_vertices:
        vertices = np.asarray(raw_vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise GameSceneExportError(f"probe object {object_id!r} vertices must be 3-vectors")
    else:
        bounds = entry["world_bounds"]
        vertices = _bounds_corners(bounds)
        complete_mesh = False
    if not np.all(np.isfinite(vertices)):
        raise GameSceneExportError(f"probe object {object_id!r} has non-finite vertices")

    canonical_vertices = (basis @ (vertices * unit_scale).T).T

    # A decimated or bounds-only object cannot justify an oriented box: use the
    # axis-aligned hull so the declared OBB still encloses the real geometry.
    rotation = (
        basis @ _quaternion_to_matrix(entry["rotation_quaternion"]) @ basis.T
        if complete_mesh
        else _IDENTITY
    )
    face_array = np.asarray(faces, dtype=int) if complete_mesh else None
    center, size, corners, padding = _fit_obb(
        canonical_vertices,
        rotation,
        min_extent_m,
        outward_normal=_outward_normal(canonical_vertices, face_array),
    )

    return {
        "id": object_id,
        "category": str(entry["category"]).strip(),
        "category_source": str(entry.get("category_source") or "unspecified"),
        "entity_kind": str(entry.get("entity_kind") or "static").lower(),
        "source_names": [str(name) for name in entry.get("source_names") or []],
        "runtime_role": (
            dict(entry["runtime_role"])
            if isinstance(entry.get("runtime_role"), dict)
            else {
                "classification": str(entry.get("entity_kind") or "static").lower(),
                "source": "legacy_probe_default",
                "declared_entity_kind": None,
                "signal_keys": [],
                "family_graph_path": None,
            }
        ),
        "material_visibility": (
            dict(entry["material_visibility"])
            if isinstance(entry.get("material_visibility"), dict)
            else None
        ),
        "graph_path": list(entry.get("graph_path") or []),
        "non_physical_hint": entry.get("non_physical_hint") or None,
        "center": center,
        "size": size,
        "rotation": rotation,
        "rotation_degrees": _matrix_to_euler_degrees(rotation),
        "corners": corners,
        "degenerate_axis_padding": padding,
        # Retained for OBB refitting after a collapse: always populated, whereas
        # the exportable mesh below exists only when the triangles are complete.
        "hull_vertices": canonical_vertices,
        "canonical_vertices": canonical_vertices if complete_mesh else None,
        "faces": face_array,
    }


def _filter_runtime_entities(
    resolved: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove actors/helpers from the static environment, with durable reasons.

    The filter consumes only the probe's safe runtime-role classification. It
    never reads names, colors, opacity, or dimensions as a standalone drop
    signal, so a wall called ``player_spawn`` and transparent static glass are
    retained. Unknown stable meshes remain in the environment and are counted.
    """

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    classification_counts: dict[str, int] = {}
    unknown_stable_ids: list[str] = []
    conflicts: list[str] = []
    for item in resolved:
        kind = str(item.get("entity_kind") or "static").lower()
        classification_counts[kind] = classification_counts.get(kind, 0) + 1
        runtime_role = (
            item.get("runtime_role")
            if isinstance(item.get("runtime_role"), dict)
            else {}
        )
        source = str(runtime_role.get("source") or "")
        if "overrode_declared_static" in source:
            conflicts.append(str(item["id"]))
        if kind in RUNTIME_EXCLUDED_ENTITY_KINDS:
            excluded.append(
                {
                    "id": str(item["id"]),
                    "entity_kind": kind,
                    "reason": "non_static_runtime_entity",
                    "classification_source": source or "probe_entity_kind",
                    "signal_keys": sorted(
                        str(value)
                        for value in runtime_role.get("signal_keys") or []
                    ),
                    "family_graph_path": runtime_role.get("family_graph_path"),
                }
            )
            continue
        if (
            kind == "static"
            and source in {"", "default_static", "legacy_probe_default"}
        ):
            unknown_stable_ids.append(str(item["id"]))
        kept.append(item)
    return kept, {
        "policy_version": "game_runtime_role_filter_v1",
        "input_mesh_count": len(resolved),
        "retained_mesh_count": len(kept),
        "excluded_mesh_count": len(excluded),
        "classification_counts": dict(sorted(classification_counts.items())),
        "excluded": excluded,
        "unknown_stable_kept_count": len(unknown_stable_ids),
        "unknown_stable_kept_ids": sorted(unknown_stable_ids),
        "declared_static_conflict_ids": sorted(conflicts),
        "static_environment_certified": (
            len(unknown_stable_ids) == 0 and len(conflicts) == 0
        ),
    }


def _outward_normal(vertices: np.ndarray | None, faces: np.ndarray | None) -> np.ndarray | None:
    """Facing direction of a one-sided sheet, or ``None`` if the mesh has none."""

    if vertices is None or faces is None or len(faces) == 0:
        return None
    triangles = vertices[faces]
    doubled = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    total_area = float(np.linalg.norm(doubled, axis=1).sum())
    if total_area <= 0.0:
        return None
    resultant = doubled.sum(axis=0)
    magnitude = float(np.linalg.norm(resultant))
    if magnitude / total_area < COHERENT_FACING_RATIO:
        return None
    return resultant / magnitude


def _fit_obb(
    vertices: np.ndarray,
    rotation: np.ndarray,
    min_extent_m: float,
    *,
    outward_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Fit a box with the given orientation around ``vertices``.

    Returns ``(center, size, corners, padding)`` in the canonical frame. The box
    is the tightest one at that orientation, so it encloses the vertices by
    construction -- which is what the collision narrow phase's mesh-enclosure
    guard requires of every object it is handed.

    ``padding`` is populated only when an axis was too thin to serve as a box
    axis, and records how much extent was fabricated and on which side it went.
    """

    local = vertices @ rotation
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    source_extent = local_max - local_min
    size = np.maximum(source_extent, float(min_extent_m))
    center_local = (local_min + local_max) / 2.0

    padding: dict[str, Any] | None = None
    thin = np.flatnonzero(size > source_extent)
    if thin.size:
        facing = outward_normal @ rotation if outward_normal is not None else np.zeros(3)
        behind = np.where(np.abs(facing) >= COHERENT_FACING_RATIO, np.sign(facing), 0.0)
        center_local = center_local - behind * (size - source_extent) / 2.0
        padding = {
            "reason": "degenerate_axis_needs_positive_extent",
            "axes": [
                {
                    "local_axis": int(axis),
                    "source_extent_m": float(source_extent[axis]),
                    "fitted_extent_m": float(size[axis]),
                    "placement": (
                        "behind_authored_surface"
                        if behind[axis]
                        else "centred_on_authored_surface"
                    ),
                }
                for axis in thin
            ],
        }

    center = rotation @ center_local
    half = size / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return center, size, center + (signs * half) @ rotation.T, padding


def _obb_contains(outer: dict[str, Any], points: np.ndarray) -> bool:
    """Whether every point lies inside ``outer``'s fitted box."""

    rotation = outer["rotation"]
    half = outer["size"] / 2.0
    eps = CONTAINMENT_RELATIVE_EPS * max(1.0, float(np.max(outer["size"])))
    local = (points - outer["center"]) @ rotation
    return bool(np.all(np.abs(local) <= half + eps))


def _individualize(
    resolved: list[dict[str, Any]],
    *,
    probe_report: Any,
    runtime_filter: dict[str, Any],
    min_extent_m: float,
    drop_non_physical: bool,
    collapse_contained: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn one entry per mesh into one entry per object, and say what it did.

    Both rules read geometry only. A rule that read the scene graph would give a
    different answer for two levels that differ solely in how the generating
    model chose to nest its groups, and could not detect which answer was wrong.
    """

    warnings: list[dict[str, Any]] = []
    kept = list(resolved)
    dropped_ids: list[str] = []

    if drop_non_physical:
        shells = [item for item in kept if item["non_physical_hint"] == BACK_SIDE_ONLY_HINT]
        survivors = [item for item in kept if item["non_physical_hint"] != BACK_SIDE_ONLY_HINT]
        for shell in shells:
            # An outline shell hugs the mesh it decorates; a room turned inside
            # out or a skybox swallows the level loosely. Both are back-face-only,
            # so how tightly the shell wraps its contents is what separates them.
            enclosed = [
                item
                for item in survivors
                if item["id"] != shell["id"] and _obb_contains(shell, item["corners"])
            ]
            if not enclosed:
                continue
            shell_volume = float(np.prod(shell["size"]))
            tightest = max(float(np.prod(item["size"])) for item in enclosed) / shell_volume
            if tightest >= TIGHT_WRAP_VOLUME_RATIO:
                continue
            warnings.append(
                {
                    "code": "enclosing_shell_dropped",
                    "object_id": shell["id"],
                    "enclosed_object_count": len(enclosed),
                    "tightest_wrap_volume_ratio": round(tightest, 6),
                    "detail": (
                        "a back-face-only mesh loosely enclosing other objects was dropped as "
                        "non-physical; confirm it was a skybox and not the level boundary"
                    ),
                }
            )
        dropped_ids = [shell["id"] for shell in shells]
        kept = survivors

    absorbed: dict[str, str] = {}
    if collapse_contained and len(kept) > 1:
        absorbed = _resolve_containers(kept)

    if not absorbed:
        report = _individualization_report(
            probe_report=probe_report,
            probed=len(resolved),
            dropped_ids=dropped_ids,
            absorbed={},
            exported=len(kept),
            warnings=warnings,
            runtime_filter=runtime_filter,
        )
        return kept, report

    merged: list[dict[str, Any]] = []
    for item in kept:
        if item["id"] in absorbed:
            continue
        children = [other for other in kept if absorbed.get(other["id"]) == item["id"]]
        merged.append(_absorb(item, children, min_extent_m=min_extent_m) if children else item)

    report = _individualization_report(
        probe_report=probe_report,
        probed=len(resolved),
        dropped_ids=dropped_ids,
        absorbed=absorbed,
        exported=len(merged),
        warnings=warnings,
        runtime_filter=runtime_filter,
    )
    return merged, report


def _resolve_containers(items: list[dict[str, Any]]) -> dict[str, str]:
    """Map each contained object to the outermost object that swallows it."""

    volumes = {item["id"]: float(np.prod(item["size"])) for item in items}
    # A strict order on (volume, id) makes "is inside" antisymmetric even for
    # coincident duplicate geometry, so the mapping below cannot cycle.
    rank = {item["id"]: (volumes[item["id"]], item["id"]) for item in items}
    candidates = _containment_candidates(items, rank)

    direct: dict[str, str] = {}
    for index, inner in enumerate(items):
        best: dict[str, Any] | None = None
        for other in candidates[index]:
            outer = items[other]
            if not _obb_contains(outer, inner["corners"]):
                continue
            # Record the tightest container, which makes the choice independent
            # of iteration order when several boxes qualify.
            if best is None or rank[outer["id"]] < rank[best["id"]]:
                best = outer
        if best is not None:
            direct[inner["id"]] = best["id"]

    # An intermediate container is itself absorbed and so cannot host anything.
    # Walking the chain sends a detail inside a part inside a body to the body,
    # which is the only object of the three that still exists afterwards.
    resolved: dict[str, str] = {}
    for inner_id in direct:
        container = direct[inner_id]
        while container in direct:
            container = direct[container]
        resolved[inner_id] = container
    return resolved


def _containment_candidates(
    items: list[dict[str, Any]],
    rank: dict[str, tuple[float, str]],
) -> list[list[int]]:
    """Pairs worth an exact test, found by axis-aligned bounds first.

    A city level runs to thousands of meshes, and the exact oriented test is far
    too costly to run on every ordered pair. An object inside another's oriented
    box is necessarily inside that object's axis-aligned box too, so the cheap
    test discards almost everything without ever discarding a real container.
    """

    count = len(items)
    lows = np.array([item["corners"].min(axis=0) for item in items], dtype=float)
    highs = np.array([item["corners"].max(axis=0) for item in items], dtype=float)
    eps = CONTAINMENT_RELATIVE_EPS * np.maximum(1.0, np.abs(highs).max(initial=1.0))

    # Built one axis at a time: the fully broadcast form would allocate an
    # (n, n, 3) array, which is the sort of thing that turns a memory budget into
    # a scaling limit on a large level.
    inside = ~np.eye(count, dtype=bool)
    for axis in range(3):
        inside &= lows[:, None, axis] >= lows[None, :, axis] - eps
        inside &= highs[:, None, axis] <= highs[None, :, axis] + eps

    order = np.array([rank[item["id"]][0] for item in items], dtype=float)
    ties = np.array([item["id"] for item in items], dtype=object)
    bigger = (order[None, :] > order[:, None]) | (
        (order[None, :] == order[:, None]) & (ties[None, :] > ties[:, None])
    )
    inside &= bigger
    return [np.flatnonzero(row).tolist() for row in inside]


def _absorb(
    container: dict[str, Any],
    children: list[dict[str, Any]],
    *,
    min_extent_m: float,
) -> dict[str, Any]:
    """Fold contained objects into their container, geometry included."""

    merged = dict(container)
    merged["absorbed_ids"] = sorted(child["id"] for child in children)

    if container["canonical_vertices"] is not None:
        vertex_blocks = [container["canonical_vertices"]]
        face_blocks = [container["faces"]]
        offset = int(container["canonical_vertices"].shape[0])
        for child in children:
            if child["canonical_vertices"] is None or child["faces"] is None:
                continue
            vertex_blocks.append(child["canonical_vertices"])
            face_blocks.append(child["faces"] + offset)
            offset += int(child["canonical_vertices"].shape[0])
        merged["canonical_vertices"] = np.concatenate(vertex_blocks, axis=0)
        merged["faces"] = np.concatenate(face_blocks, axis=0)

    # Refit at the container's own orientation. The children were inside to
    # within a float tolerance, so this grows the box by at most that tolerance
    # while making the enclosure exact rather than approximate.
    hull = np.concatenate(
        [container["hull_vertices"]] + [child["hull_vertices"] for child in children], axis=0
    )
    center, size, corners, padding = _fit_obb(
        hull,
        container["rotation"],
        min_extent_m,
        outward_normal=_outward_normal(merged["canonical_vertices"], merged["faces"]),
    )
    merged["hull_vertices"] = hull
    merged["center"] = center
    merged["size"] = size
    merged["corners"] = corners
    merged["degenerate_axis_padding"] = padding
    return merged


def _individualization_report(
    *,
    probe_report: Any,
    probed: int,
    dropped_ids: list[str],
    absorbed: dict[str, str],
    exported: int,
    warnings: list[dict[str, Any]],
    runtime_filter: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "strategy": "visible_mesh_then_geometric_collapse_v1",
        "probe_objects": int(probed),
        "dropped_non_physical": len(dropped_ids),
        "dropped_non_physical_ids": sorted(dropped_ids),
        "absorbed_into_container": len(absorbed),
        "objects_exported": int(exported),
        "warnings": warnings,
        "runtime_filter": runtime_filter,
    }
    if isinstance(probe_report, dict):
        report["probe"] = {
            "strategy": probe_report.get("strategy"),
            "top_level_child_count": probe_report.get("top_level_child_count"),
            "max_graph_depth": probe_report.get("max_graph_depth"),
            "declared_category_count": probe_report.get("declared_category_count"),
            "counts": probe_report.get("counts"),
        }
    return report


def _collision_entry(
    item: dict[str, Any],
    *,
    translation: np.ndarray,
    mesh_root: Path | None,
) -> dict[str, Any]:
    vertices = item["canonical_vertices"]
    if vertices is None or mesh_root is None:
        return {
            "representation": BBOX_PROXY_REPRESENTATION,
            "complete": False,
            "error": "triangle_mesh_not_exported",
        }
    mesh_root.mkdir(parents=True, exist_ok=True)
    geometry_path = mesh_root / f"{item['id']}.ply"
    write_ascii_triangle_ply(
        geometry_path,
        (vertices + translation).tolist(),
        item["faces"].tolist(),
    )
    return {
        "representation": TRIANGLE_MESH_REPRESENTATION,
        "geometry_path": geometry_path.as_posix(),
        "complete": True,
        "transform_baked": True,
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(item["faces"].shape[0]),
    }


def _quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 0.0:
        raise GameSceneExportError("probe rotation_quaternion must be non-degenerate")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_euler_degrees(rotation: np.ndarray) -> list[float]:
    """Invert the canonical ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` composition."""

    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1.0e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _bounds_corners(bounds: dict[str, Any]) -> np.ndarray:
    low = np.asarray(bounds["min"], dtype=float)
    high = np.asarray(bounds["max"], dtype=float)
    return np.array(
        [
            [low[0] if sx < 0 else high[0], low[1] if sy < 0 else high[1], low[2] if sz < 0 else high[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )


def _validate_quaternion(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise GameSceneExportError(f"{path}.rotation_quaternion must be a 4-vector [x, y, z, w]")
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise GameSceneExportError(f"{path}.rotation_quaternion components must be numeric")
        if not math.isfinite(float(component)):
            raise GameSceneExportError(f"{path}.rotation_quaternion components must be finite")


def _validate_bounds(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise GameSceneExportError(f"{path} must be a JSON object")
    for key in ("min", "max"):
        vector = value.get(key)
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            raise GameSceneExportError(f"{path}.{key} must be a 3-vector")
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise GameSceneExportError(f"{path}.{key} components must be numeric")
            if not math.isfinite(float(component)):
                raise GameSceneExportError(f"{path}.{key} components must be finite")


def _validate_runtime_role(value: Any, *, path: str) -> None:
    if not isinstance(value, dict):
        raise GameSceneExportError(f"{path} must be a JSON object")
    expected = {
        "classification",
        "source",
        "declared_entity_kind",
        "signal_keys",
        "family_graph_path",
    }
    if set(value) != expected:
        raise GameSceneExportError(
            f"{path} has unknown keys {sorted(set(value) - expected)} or "
            f"missing keys {sorted(expected - set(value))}"
        )
    classification = str(value["classification"] or "").strip().lower()
    if classification not in ALLOWED_ENTITY_KINDS:
        raise GameSceneExportError(
            f"{path}.classification must be one of {sorted(ALLOWED_ENTITY_KINDS)}"
        )
    if not str(value["source"] or "").strip():
        raise GameSceneExportError(f"{path}.source must be a non-empty string")
    declared = value["declared_entity_kind"]
    if declared is not None and (
        not isinstance(declared, str) or not declared.strip()
    ):
        raise GameSceneExportError(
            f"{path}.declared_entity_kind must be null or a non-empty string"
        )
    signals = value["signal_keys"]
    if not isinstance(signals, list) or any(
        not isinstance(item, str) or not item for item in signals
    ):
        raise GameSceneExportError(
            f"{path}.signal_keys must be a list of non-empty strings"
        )
    family_path = value["family_graph_path"]
    if family_path is not None and (
        not isinstance(family_path, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in family_path
        )
    ):
        raise GameSceneExportError(
            f"{path}.family_graph_path must be null or a list of non-negative integers"
        )
