"""Canonical triangle geometry artifacts for collision narrow-phase evaluation."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np


COLLISION_GEOMETRY_SCHEMA_VERSION = "collision_geometry_v1"
TRIANGLE_MESH_REPRESENTATION = "triangle_mesh"
POINT_CLOUD_REPRESENTATION = "point_cloud"
BBOX_PROXY_REPRESENTATION = "bbox_proxy"


class CollisionGeometryError(ValueError):
    """Raised when a collision geometry manifest is malformed."""


def load_collision_geometry_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise CollisionGeometryError(f"collision geometry manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollisionGeometryError(f"invalid collision geometry manifest: {manifest_path}") from exc
    return validate_collision_geometry_manifest(manifest, base_dir=manifest_path.parent)


def validate_collision_geometry_manifest(manifest: dict, *, base_dir: Path | None = None) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CollisionGeometryError("collision geometry manifest must be a JSON object")
    if manifest.get("schema_version") != COLLISION_GEOMETRY_SCHEMA_VERSION:
        raise CollisionGeometryError(
            f"collision geometry schema_version must be {COLLISION_GEOMETRY_SCHEMA_VERSION!r}"
        )
    if manifest.get("units") != "meter":
        raise CollisionGeometryError("collision geometry units must be 'meter'")
    if manifest.get("up_axis") != "z":
        raise CollisionGeometryError("collision geometry up_axis must be 'z'")
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        raise CollisionGeometryError("collision geometry objects must be a JSON object")
    for object_id, entry in objects.items():
        _validate_object_entry(str(object_id), entry, base_dir=base_dir)
    return manifest


def geometry_entry_for_object(manifest: dict | None, object_id: str) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(str(object_id))
    return entry if isinstance(entry, dict) else None


def is_usable_triangle_mesh(entry: dict[str, Any] | None, *, base_dir: Path | None = None) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("representation") != TRIANGLE_MESH_REPRESENTATION:
        return False
    if entry.get("complete") is not True:
        return False
    geometry_path = entry.get("geometry_path")
    if not isinstance(geometry_path, str) or not geometry_path.strip():
        return False
    path = Path(geometry_path)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    suffix = path.suffix.lower()
    if suffix not in {".ply", ".obj", ".glb", ".gltf"}:
        return False
    if not path.is_file():
        return False
    if _looks_like_point_cloud_only(path):
        return False
    return True


def geometry_unavailable_reason(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "geometry_entry_missing"
    representation = str(entry.get("representation") or "unknown")
    if representation == POINT_CLOUD_REPRESENTATION:
        return "point_cloud_not_triangle_mesh"
    if representation == BBOX_PROXY_REPRESENTATION:
        return "bbox_proxy_only"
    if entry.get("complete") is not True:
        return str(entry.get("error") or "geometry_incomplete")
    geometry_path = entry.get("geometry_path")
    if not isinstance(geometry_path, str) or not geometry_path.strip():
        return "geometry_path_missing"
    path = Path(geometry_path)
    if not path.is_file():
        return "geometry_file_missing"
    if _looks_like_point_cloud_only(path):
        return "point_cloud_not_triangle_mesh"
    return "geometry_not_usable"


def _validate_object_entry(object_id: str, entry: Any, *, base_dir: Path | None) -> None:
    path = f"collision_geometry.objects[{object_id!r}]"
    if not isinstance(entry, dict):
        raise CollisionGeometryError(f"{path} must be a JSON object")
    representation = entry.get("representation")
    if representation not in {TRIANGLE_MESH_REPRESENTATION, POINT_CLOUD_REPRESENTATION, BBOX_PROXY_REPRESENTATION}:
        raise CollisionGeometryError(f"{path}.representation is invalid")
    if representation == TRIANGLE_MESH_REPRESENTATION:
        geometry_path = entry.get("geometry_path")
        if not isinstance(geometry_path, str) or not geometry_path.strip():
            raise CollisionGeometryError(f"{path}.geometry_path must be a non-empty string")
        resolved = Path(geometry_path)
        if not resolved.is_absolute() and base_dir is not None:
            resolved = (base_dir / resolved).resolve()
        if entry.get("complete") is True and not resolved.is_file():
            raise CollisionGeometryError(f"{path}.geometry_path does not exist: {resolved}")
        if entry.get("transform_baked") is not True:
            raise CollisionGeometryError(f"{path}.transform_baked must be true for triangle meshes")


def _looks_like_point_cloud_only(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix != ".ply":
        return False
    try:
        stat = path.stat()
    except OSError:
        return True
    return _cached_ply_header_is_point_cloud(
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


@lru_cache(maxsize=512)
def _cached_ply_header_is_point_cloud(path: str, _mtime_ns: int, _size: int) -> bool:
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.rstrip("\r\n") for line in islice(handle, 80)]
    except OSError:
        return True
    has_face_element = any(line.strip().lower().startswith("element face") for line in lines[:80])
    if has_face_element:
        return False
    has_vertex_element = any(line.strip().lower().startswith("element vertex") for line in lines[:80])
    return has_vertex_element


def load_triangle_mesh(entry: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    """Load a canonical world-space triangle mesh from a manifest entry.

    Shared by the collision narrow-phase and the support probe so both consume
    the identical geometry artifact rather than duplicating loader logic.
    """

    geometry_path = Path(str(entry["geometry_path"]))
    if not geometry_path.is_absolute() and base_dir is not None:
        geometry_path = (base_dir / geometry_path).resolve()
    suffix = geometry_path.suffix.lower()
    if suffix == ".ply":
        return _load_ply_triangles(geometry_path)
    try:
        import trimesh  # type: ignore

        # GLB/GLTF commonly stores transforms on scene nodes rather than baking
        # them into the geometry buffers.  Loading the raw geometry dictionary
        # and concatenating its values silently drops those transforms (and any
        # repeated instances).  Force a scene for those formats and explicitly
        # bake every node transform before concatenating.
        if suffix in {".glb", ".gltf"}:
            loaded = trimesh.load(str(geometry_path), process=False, force="scene")
        else:
            loaded = trimesh.load_mesh(str(geometry_path), process=False)
        loader = "trimesh"
        if hasattr(loaded, "geometry"):
            meshes = _scene_meshes_in_world_frame(loaded)
            loaded = trimesh.util.concatenate(meshes) if meshes else None
            loader = "trimesh_scene_graph"
        if loaded is None:
            raise ValueError("trimesh returned no geometry")
        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = np.asarray(loaded.faces, dtype=int)
        return {"vertices": vertices, "faces": faces, "path": str(geometry_path), "loader": loader}
    except Exception as exc:
        raise ValueError(f"unable to load triangle mesh {geometry_path}: {exc}") from exc


def _scene_meshes_in_world_frame(scene: Any) -> list[Any]:
    """Return one transformed mesh per scene-graph geometry node.

    Iterating ``Scene.geometry.values()`` is insufficient: geometry entries are
    stored in their local frames, while transforms and instancing live on graph
    nodes.  This helper deliberately fails rather than falling back to unbaked
    local geometry, because wrong-frame vertices can create unsafe deterministic
    collision bypasses.
    """

    geometry = getattr(scene, "geometry", None)
    graph = getattr(scene, "graph", None)
    node_names = getattr(graph, "nodes_geometry", None)
    if not isinstance(geometry, dict) or graph is None or node_names is None:
        raise ValueError("trimesh scene is missing a usable geometry graph")

    meshes: list[Any] = []
    for node_name in list(node_names):
        transform, geometry_name = graph[node_name]
        source = geometry.get(geometry_name)
        if source is None:
            raise ValueError(f"scene node {node_name!r} references missing geometry {geometry_name!r}")
        mesh = source.copy()
        mesh.apply_transform(np.asarray(transform, dtype=float))
        meshes.append(mesh)
    return meshes


def _load_ply_triangles(path: Path) -> dict[str, Any]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_header = True
    section: str | None = None
    vertex_count = 0
    face_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if in_header:
            lower = stripped.lower()
            if lower == "end_header":
                in_header = False
                section = "vertex" if vertex_count else ("face" if face_count else None)
                continue
            if lower.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                continue
            if lower.startswith("element face"):
                face_count = int(stripped.split()[-1])
                continue
            continue
        if section == "vertex" and len(vertices) < vertex_count:
            parts = stripped.split()
            if len(parts) >= 3:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(vertices) >= vertex_count:
                section = "face" if face_count else None
            continue
        if section == "face" and len(faces) < face_count:
            parts = stripped.split()
            if len(parts) >= 4 and parts[0] == "3":
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    if not vertices:
        raise ValueError(f"PLY contains no vertices: {path}")
    if not faces:
        raise ValueError(f"PLY contains no faces: {path}")
    return {
        "vertices": np.asarray(vertices, dtype=float),
        "faces": np.asarray(faces, dtype=int),
        "path": str(path),
        "loader": "ascii_ply",
    }


def write_ascii_triangle_ply(path: Path, vertices: list[list[float]], faces: list[list[int]]) -> None:
    """Write a minimal ASCII PLY with triangular faces for tests and providers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_rows = []
    for vertex in vertices:
        if len(vertex) != 3:
            raise ValueError("vertices must be 3-vectors")
        for component in vertex:
            number = float(component)
            if not math.isfinite(number):
                raise ValueError("vertex coordinates must be finite")
        vertex_rows.append(f"{float(vertex[0]):.6f} {float(vertex[1]):.6f} {float(vertex[2]):.6f}")
    face_rows = []
    for face in faces:
        if len(face) != 3:
            raise ValueError("faces must be triangles")
        face_rows.append(f"3 {int(face[0])} {int(face[1])} {int(face[2])}")
    content = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertex_rows)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(face_rows)}",
            "property list uchar int vertex_indices",
            "end_header",
            *vertex_rows,
            *face_rows,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
