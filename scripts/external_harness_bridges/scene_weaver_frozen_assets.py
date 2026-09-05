"""Exact-GLB input factory component for the SceneWeaver frozen plugin.

This is generation INPUT preparation, not a converter or a placement algorithm.
It does not launch or certify the native loop. The plugin still must install the
factory in native initialization, guard all mutation paths, and observe states.
Blender and the released AssetFactory base are imported/supplied only at runtime.
"""
from __future__ import annotations

from copy import deepcopy
from array import array
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from _common import file_sha256
from scene_weaver_frozen import _anchor_basis, _orientation_basis


def _vector(value: Any, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-vector")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) and (not positive or v > 0) for v in result):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def validate_binding(slot_id: str, binding: dict, *, tolerance: float) -> dict:
    """Freeze an exact binding before bpy import; no retrieval or category fallback."""
    if not isinstance(slot_id, str) or not slot_id.strip():
        raise ValueError("frozen asset factory requires an exact slot ID")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("frozen geometry tolerance must be finite and positive")
    value = deepcopy(binding)
    if not isinstance(value.get("asset_key"), str) or not value["asset_key"]:
        raise ValueError("frozen asset factory requires asset_key")
    path = Path(str(value.get("mesh_uri") or "")).expanduser().resolve()
    if path.suffix.lower() != ".glb" or not path.is_file():
        raise ValueError("frozen asset factory requires an existing exact GLB")
    expected = value.get("mesh_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
        raise ValueError("frozen GLB hash mismatch")
    size = _vector(value.get("bbox_size_local"), "bbox_size_local", positive=True)
    scale = _vector(value.get("native_scale"), "native_scale", positive=True)
    physical = _vector(value.get("physical_dimensions"), "physical_dimensions", positive=True)
    center = _vector(value.get("bbox_center_local"), "bbox_center_local")
    if any(abs(size[i] * scale[i] - physical[i]) > tolerance for i in range(3)):
        raise ValueError("frozen local bbox, native scale and physical dimensions conflict")
    if value.get("canonical_front") is not None:
        value["canonical_front"] = _vector(value["canonical_front"], "canonical_front")
    basis = _orientation_basis(value)
    # The first track's catalog fronts are cardinal. A general baked mesh basis
    # needs a separately observed non-cardinal mesh-bbox contract, not |R|*bbox.
    if abs(basis["basis_yaw_degrees"] / 90 - round(basis["basis_yaw_degrees"] / 90)) > 1e-8:
        raise ValueError("non-cardinal frozen mesh front is not qualified for this factory")
    value.update(mesh_uri=str(path), bbox_size_local=size, bbox_center_local=center,
                 native_scale=scale, physical_dimensions=physical)
    return value


def _bounds(points):
    points = list(points)
    if not points:
        raise ValueError("exact GLB contains no mesh vertices")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("exact GLB contains non-finite mesh vertices")
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    return [high[i] - low[i] for i in range(3)], [(high[i] + low[i]) / 2 for i in range(3)]


def local_vertex_digest(obj):
    """Within-worker geometry observation, separate from the source-file hash."""
    values = array("f", (float(v) for vertex in obj.data.vertices for v in vertex.co))
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_exact_glb(slot_id: str, binding: dict, *, tolerance: float = 1e-4):
    """Return a new bottom-origin, basis-baked mesh plus observed input provenance.

    Preserve all imported mesh vertices (including loose geometry), materials,
    hierarchy transforms and frozen physical scale. Never fit/rescale to a target
    bbox. Only the declared scale/basis/origin representation transform is baked.
    """
    asset = validate_binding(slot_id, binding, tolerance=tolerance)
    import bpy
    from mathutils import Matrix, Vector

    existing = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_scene.gltf(filepath=asset["mesh_uri"])
        created = [obj for obj in bpy.data.objects if obj.name not in existing]
        meshes = sorted((obj for obj in created if obj.type == "MESH"), key=lambda obj: obj.name)
        if not meshes or any(obj.type not in {"MESH", "EMPTY"} for obj in created):
            raise ValueError("exact GLB requires static mesh/empty hierarchy only")
        if any(obj.animation_data or obj.constraints for obj in created):
            raise ValueError("animated/constrained GLB is not qualified as a frozen static asset")
        if any(obj.modifiers or obj.data.shape_keys for obj in meshes):
            raise ValueError("modified/skinned GLB is not qualified as a frozen static asset")
        matrices = {obj.name: obj.matrix_world.copy() for obj in meshes}
        size, center = _bounds(matrices[obj.name] @ vertex.co
                               for obj in meshes for vertex in obj.data.vertices)
        for observed, expected, label in [(size, asset["bbox_size_local"], "size"),
                                          (center, asset["bbox_center_local"], "center")]:
            if any(abs(observed[i] - expected[i]) > tolerance for i in range(3)):
                raise ValueError(f"observed exact GLB bbox {label} differs from frozen metadata")
        before_vertices = sum(len(obj.data.vertices) for obj in meshes)
        before_edges = sum(len(obj.data.edges) for obj in meshes)
        before_faces = sum(len(obj.data.polygons) for obj in meshes)
        basis = _orientation_basis(asset)
        anchor = _anchor_basis(asset)
        transform = (
            Matrix.Rotation(math.radians(basis["basis_yaw_degrees"]), 4, "Z")
            @ Matrix.Translation(-Vector(anchor["canonical_bottom_center_local"]))
            @ Matrix.Diagonal((*asset["native_scale"], 1.0))
        )
        for obj in meshes:
            # Make instances independent before baking each hierarchy transform.
            obj.data = obj.data.copy()
            obj.data.transform(transform @ matrices[obj.name])
            obj.parent = None
            obj.matrix_world = Matrix.Identity(4)
        for obj in created:
            if obj.type == "EMPTY":
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        obj.name = f"frozen_asset_{slot_id}"
        obj.rotation_mode = "XYZ"
        bpy.context.view_layer.update()
        if (len(obj.data.vertices), len(obj.data.edges), len(obj.data.polygons)) != (
            before_vertices, before_edges, before_faces
        ):
            raise ValueError("joining exact GLB changed mesh topology counts")
        if file_sha256(asset["mesh_uri"]) != asset["mesh_sha256"]:
            raise ValueError("exact GLB changed during import")
        obj["frozen_slot_id"] = slot_id
        obj["frozen_asset_id"] = asset["asset_key"]
        obj["frozen_mesh_sha256"] = asset["mesh_sha256"]
        obj["frozen_geometry_role"] = "asset"
        obj["frozen_vertex_sha256"] = local_vertex_digest(obj)
        audit = {
            "slot_id": slot_id, "asset_id": asset["asset_key"],
            "mesh_path": asset["mesh_uri"], "mesh_sha256": asset["mesh_sha256"],
            "source_bbox_size_local": size, "source_bbox_center_local": center,
            "canonical_local_bbox_size": [size[i] * asset["native_scale"][i] for i in range(3)],
            "native_object_dimensions": list(obj.dimensions),
            "orientation_basis": basis, "anchor_basis": anchor,
            "source_to_native_matrix": [list(row) for row in transform],
            "mesh_counts": {"vertices": before_vertices, "edges": before_edges, "faces": before_faces},
            "source_file_unchanged": True, "retrieval_calls": 0,
            "native_local_vertex_sha256": obj["frozen_vertex_sha256"],
        }
        return obj, audit
    except BaseException:
        # Only remove new in-memory objects created by this failed import. The
        # source GLB and all pre-existing scene objects remain untouched.
        for obj in list(bpy.data.objects):
            if obj.name not in existing:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise


def make_frozen_factory(base_factory, slot_id: str, binding: dict, *, tolerance: float = 1e-4):
    """Supply the release's AssetFactory base; override only its creation hooks.

    Native spawn/pose/random-seed bookkeeping stays in the base implementation.
    The caller must register native semantic usages and serialize these bindings
    for child processes; this component alone is not an executable plugin.
    """
    asset = validate_binding(slot_id, binding, tolerance=tolerance)
    identity = hashlib.sha256(json.dumps(
        {"slot_id": slot_id, "binding": asset, "tolerance_m": tolerance,
         "base_factory": f"{base_factory.__module__}.{base_factory.__qualname__}"},
        sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    class_name = f"FrozenSlot{identity[:24]}Factory"
    if class_name in globals():
        return globals()[class_name]

    def create_asset(self, **params):
        obj, audit = load_exact_glb(slot_id, asset, tolerance=tolerance)
        self.frozen_input_observations = getattr(self, "frozen_input_observations", []) + [audit]
        return obj

    def create_placeholder(self, **params):
        import bpy
        obj = self.create_asset(**params)
        corners = [tuple(corner) for corner in obj.bound_box]
        mesh = bpy.data.meshes.new(f"frozen_placeholder_{slot_id}")
        # Blender bound_box ordering: left four corners, then right four.
        mesh.from_pydata(corners, [], [(0, 1, 2, 3), (4, 7, 6, 5),
                                      (0, 4, 5, 1), (3, 2, 6, 7),
                                      (0, 3, 7, 4), (1, 5, 6, 2)])
        placeholder = bpy.data.objects.new(f"frozen_placeholder_{slot_id}", mesh)
        bpy.context.collection.objects.link(placeholder)
        for key in ("frozen_slot_id", "frozen_asset_id", "frozen_mesh_sha256"):
            placeholder[key] = obj[key]
        placeholder["frozen_geometry_role"] = "placeholder"
        bpy.data.objects.remove(obj, do_unlink=True)
        return placeholder

    factory = type(class_name, (base_factory,), {
        "__module__": __name__, "create_asset": create_asset,
        "create_placeholder": create_placeholder, "frozen_binding_sha256": identity,
        "frozen_slot_id": slot_id, "frozen_asset_id": asset["asset_key"],
    })
    globals()[class_name] = factory
    return factory
