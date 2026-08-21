from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.evaluator.generic_validity.collision import (
    CollisionEvaluationError,
    DEFAULT_COLLISION_CONFIG,
    _compliant_floor_covering_certificate,
    _shallow_surface_layer_overlap_evidence,
    _support_interface_contact_certificate,
    _tangent_plane_contact_certificate,
    check_collision,
)
from benchmark.evaluator.generic_validity.geometry import normalize_object
from benchmark.evaluator.generic_validity.mesh_geometry import (
    POINT_CLOUD_REPRESENTATION,
    TRIANGLE_MESH_REPRESENTATION,
    is_usable_triangle_mesh,
    load_triangle_mesh,
    write_ascii_triangle_ply,
)
from benchmark.evaluator.generic_validity.mesh_collision import (
    _containment_numpy,
    _surface_intersection_aabb,
    evaluate_mesh_pair,
)
from benchmark.evaluator.generic_validity.obb_sat import obb_encloses_points, obb_sat_test, obb_sat_test_parts
from benchmark.rendering.blender_worker import _uniform_contain_fit
from benchmark.visual_judge.contracts import ResponseSchemaRepairError


ROOT = Path(__file__).resolve().parents[1]


def _scene(objects: list[dict]) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "collision_scene",
        "request_id": "collision_case",
        "scene_type": "room",
        "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _obj(
    object_id: str,
    center: list[float],
    size: list[float],
    *,
    rotation: list[float] | None = None,
    category: str = "box",
    description: str = "box",
) -> dict:
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "center": center,
        "size": size,
        "rotation": rotation or [0.0, 0.0, 0.0],
        "metadata": {"interactive": False},
    }


def _box_mesh(path: Path, center: list[float], size: list[float]) -> None:
    half = [value / 2.0 for value in size]
    cx, cy, cz = center
    vertices = [
        [cx - half[0], cy - half[1], cz - half[2]],
        [cx + half[0], cy - half[1], cz - half[2]],
        [cx + half[0], cy + half[1], cz - half[2]],
        [cx - half[0], cy + half[1], cz - half[2]],
        [cx - half[0], cy - half[1], cz + half[2]],
        [cx + half[0], cy - half[1], cz + half[2]],
        [cx + half[0], cy + half[1], cz + half[2]],
        [cx - half[0], cy + half[1], cz + half[2]],
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    write_ascii_triangle_ply(path, vertices, faces)


def _geometry_manifest(tmp_path: Path, specs: dict[str, dict]) -> dict:
    entries = {}
    for object_id, spec in specs.items():
        entry = dict(spec)
        if entry.get("representation") == TRIANGLE_MESH_REPRESENTATION and entry.get("geometry_path"):
            rel = Path(entry["geometry_path"])
            if not rel.is_absolute():
                entry["geometry_path"] = str((tmp_path / rel).resolve())
        entries[object_id] = entry
    manifest = {
        "schema_version": "collision_geometry_v1",
        "units": "meter",
        "up_axis": "z",
        "objects": entries,
        "manifest_path": str((tmp_path / "collision_geometry_manifest.json").resolve()),
    }
    return manifest


class _Judge:
    vlm_control_enabled = False

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def adjudicate_p0b(self, request: dict) -> dict:
        self.calls += 1
        return {"verdict": self.verdict, "confidence": 0.9, "reason": "test"}


# 1. Full 3D OBB SAT separation and intersection
def test_obb_sat_separation_and_intersection() -> None:
    separated = obb_sat_test_parts(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.eye(3),
        np.array([3.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.eye(3),
    )
    intersecting = obb_sat_test_parts(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.eye(3),
        np.array([0.5, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.eye(3),
    )
    assert separated["intersects"] is False
    assert separated["obb_certifiably_separated"] is True
    assert separated["minimum_separation_margin_m"] > 0.0
    assert intersecting["intersects"] is True
    assert intersecting["minimum_overlap_depth_proxy_m"] > 0.0


# 2. Rotated objects and non-zero roll/pitch/yaw
def test_obb_sat_handles_full_euler_rotation() -> None:
    obj = normalize_object(_obj("rot", [1.0, 1.0, 0.5], [1.0, 0.5, 0.8], rotation=[15.0, 30.0, 45.0]))
    separated = obb_sat_test(obj, normalize_object(_obj("far", [4.0, 1.0, 0.5], [0.5, 0.5, 0.5])))
    assert separated["obb_certifiably_separated"] is True


# 3. OBB separation skips mesh and VLM
def test_obb_separation_skips_mesh_and_vlm() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]), _obj("b", [4.0, 1.0, 0.5], [0.5, 0.5, 0.5])])
    judge = _Judge("invalid")
    report = check_collision(scene, {"detector_only": False}, vlm_judge=judge)
    pair = report["pairs"][0]
    assert pair["route"] == "direct_valid_obb_separated"
    assert pair["final_verdict"] == "valid"
    assert judge.calls == 0


def test_zero_penetration_coplanar_contact_is_direct_valid_when_enabled(
    tmp_path: Path,
) -> None:
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    _box_mesh(mesh_a, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    _box_mesh(mesh_b, [1.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "a.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            },
            "b": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "b.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            },
        },
    )
    scene = _scene(
        [
            _obj("a", [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [1.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("invalid")

    report = check_collision(
        scene,
        {"zero_penetration_contact_policy": "direct_valid"},
        collision_geometry=geometry,
        vlm_judge=judge,
    )

    pair = report["pairs"][0]
    assert pair["obb_evidence"]["minimum_overlap_depth_proxy_m"] == pytest.approx(0.0)
    assert pair["route"] == "direct_valid_zero_penetration_contact"
    assert pair["final_verdict"] == "valid"
    assert report["score_mode"] == "invalid_pair_count_over_objects"
    assert report["score"] == pytest.approx(1.0)
    assert judge.calls == 0


def test_thin_plane_near_tangent_contact_has_safe_certificate() -> None:
    plane = normalize_object(
        _obj("floor_plane", [1.0, 1.0, 0.0], [3.0, 3.0, 0.001])
    )
    box = normalize_object(
        _obj("crate", [1.0, 1.0, 0.5004], [1.0, 1.0, 1.0])
    )
    obb = obb_sat_test(plane, box)

    certificate = _tangent_plane_contact_certificate(
        plane,
        box,
        obb=obb,
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        enclosure_safe=True,
        config={
            "tangent_plane_contact_policy": "direct_valid",
            "tangent_plane_max_thickness_m": 0.002,
            "tangent_contact_tolerance_m": 0.001,
        },
    )

    assert certificate is not None
    assert certificate["plane_object_id"] == "floor_plane"
    assert certificate["other_object_id"] == "crate"
    assert certificate["side"] == "positive"


def test_thin_plane_slicing_object_never_receives_tangent_certificate() -> None:
    plane = normalize_object(
        _obj("wall_plane", [1.0, 1.0, 0.5], [3.0, 3.0, 0.001])
    )
    box = normalize_object(
        _obj("crate", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    )
    obb = obb_sat_test(plane, box)

    certificate = _tangent_plane_contact_certificate(
        plane,
        box,
        obb=obb,
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        enclosure_safe=True,
        config={
            "tangent_plane_contact_policy": "direct_valid",
            "tangent_plane_max_thickness_m": 0.002,
            "tangent_contact_tolerance_m": 0.001,
        },
    )

    assert certificate is None


def test_tiny_gravity_aligned_support_interface_has_safe_certificate() -> None:
    support = normalize_object(
        _obj("counter", [1.0, 1.0, 0.45], [2.0, 1.0, 0.9])
    )
    load = normalize_object(
        _obj("mug", [1.0, 1.0, 0.9995], [0.2, 0.2, 0.2])
    )
    certificate = _support_interface_contact_certificate(
        support,
        load,
        obb=obb_sat_test(support, load),
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        enclosure_safe=True,
        config=dict(DEFAULT_COLLISION_CONFIG),
    )

    assert certificate is not None
    assert certificate["support_object_id"] == "counter"
    assert certificate["load_object_id"] == "mug"
    assert certificate["interface_penetration_m"] == pytest.approx(
        0.0005
    )


def test_lateral_or_deep_overlap_never_receives_support_certificate() -> None:
    a = normalize_object(
        _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    )
    b = normalize_object(
        _obj("b", [1.8, 1.0, 0.5], [1.0, 1.0, 1.0])
    )

    assert (
        _support_interface_contact_certificate(
            a,
            b,
            obb=obb_sat_test(a, b),
            mesh_evidence={
                "surface_intersection": True,
                "intersection": {"definitive": True},
            },
            enclosure_safe=True,
            config=dict(DEFAULT_COLLISION_CONFIG),
        )
        is None
    )


def test_shallow_floor_layer_overlap_is_context_not_an_exemption() -> None:
    layer = normalize_object(
        _obj(
            "surface_layer",
            [1.5, 1.5, 0.005],
            [2.0, 2.0, 0.01],
            category="surface covering",
        )
    )
    load = normalize_object(
        _obj(
            "load",
            [1.5, 1.5, 0.5],
            [1.0, 1.0, 1.0],
            category="furniture",
        )
    )
    evidence = _shallow_surface_layer_overlap_evidence(
        _scene([]),
        layer,
        load,
        obb=obb_sat_test(layer, load),
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        config=dict(DEFAULT_COLLISION_CONFIG),
    )

    assert evidence is not None
    assert evidence["layer_object_id"] == "surface_layer"
    assert evidence["vertical_overlap_m"] == pytest.approx(0.01)
    assert evidence["substrate_crossing_m"] == pytest.approx(0.0)
    assert evidence["decision_authority"] == "vlm_judge"
    assert evidence["carries_validity_prior"] is False
    assert evidence["automatic_exemption"] is False
    policy = check_collision(_scene([_obj("only", [1, 1, 0.5], [1, 1, 1])]))[
        "shallow_surface_layer_policy"
    ]
    assert policy["max_overlap_m"] == pytest.approx(0.0125)
    assert policy["automatic_exemption"] is False


def test_explicit_rug_semantics_upgrade_bounded_floor_overlap() -> None:
    rug = normalize_object(
        _obj(
            "rug",
            [1.5, 1.5, 0.005],
            [2.0, 2.0, 0.01],
            category="area_rug",
            description="woven area rug",
        )
    )
    chair = normalize_object(
        _obj(
            "chair",
            [1.5, 1.5, 0.5],
            [1.0, 1.0, 1.0],
            category="chair",
        )
    )
    shallow = _shallow_surface_layer_overlap_evidence(
        _scene([]),
        rug,
        chair,
        obb=obb_sat_test(rug, chair),
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        config=dict(DEFAULT_COLLISION_CONFIG),
    )

    certificate = _compliant_floor_covering_certificate(
        _scene([]),
        rug,
        chair,
        obb=obb_sat_test(rug, chair),
        mesh_evidence={
            "surface_intersection": True,
            "intersection": {"definitive": True},
        },
        config=dict(DEFAULT_COLLISION_CONFIG),
    )
    assert certificate is not None
    assert certificate["layer_object_id"] == "rug"
    assert certificate["certificate"] == (
        "semantic_compliant_floor_covering_relief"
    )


def test_explicit_rug_overlap_is_direct_valid_without_judge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import benchmark.evaluator.generic_validity.collision as collision_module

    rug_center = [1.5, 1.5, 0.005]
    rug_size = [2.0, 2.0, 0.01]
    chair_center = [1.5, 1.5, 0.5]
    chair_size = [1.0, 1.0, 1.0]
    _box_mesh(tmp_path / "rug.ply", rug_center, rug_size)
    _box_mesh(tmp_path / "chair.ply", chair_center, chair_size)
    geometry = _geometry_manifest(
        tmp_path,
        {
            object_id: {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": f"{object_id}.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            }
            for object_id in ("rug", "chair")
        },
    )
    monkeypatch.setattr(
        collision_module,
        "evaluate_mesh_pair",
        lambda *args, **kwargs: {
            "status": "evaluated",
            "mesh_state": "surface_intersection",
            "mesh_reliable_for_separation": False,
            "surface_intersection": True,
            "intersection": {"intersects": True, "definitive": True},
            "closest_points": None,
            "focus_region": None,
        },
    )
    judge = _Judge("invalid")

    report = check_collision(
        _scene(
            [
                _obj(
                    "rug",
                    rug_center,
                    rug_size,
                    category="area_rug",
                    description="woven area rug",
                ),
                _obj(
                    "chair",
                    chair_center,
                    chair_size,
                    category="chair",
                ),
            ]
        ),
        collision_geometry=geometry,
        vlm_judge=judge,
    )

    assert report["pairs"][0]["route"] == (
        "direct_valid_compliant_floor_covering"
    )
    assert report["pairs"][0]["final_verdict"] == "valid"
    assert judge.calls == 0


def test_shallow_floor_layer_context_reaches_collision_judge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import benchmark.evaluator.generic_validity.collision as collision_module

    layer_center = [1.5, 1.5, 0.005]
    layer_size = [2.0, 2.0, 0.01]
    load_center = [1.5, 1.5, 0.5]
    load_size = [1.0, 1.0, 1.0]
    _box_mesh(tmp_path / "layer.ply", layer_center, layer_size)
    _box_mesh(tmp_path / "load.ply", load_center, load_size)
    geometry = _geometry_manifest(
        tmp_path,
        {
            "layer": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "layer.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            },
            "load": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "load.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            },
        },
    )
    monkeypatch.setattr(
        collision_module,
        "evaluate_mesh_pair",
        lambda *args, **kwargs: {
            "status": "evaluated",
            "mesh_state": "surface_intersection",
            "mesh_reliable_for_separation": False,
            "surface_intersection": True,
            "intersection": {
                "intersects": True,
                "definitive": True,
            },
            "closest_points": None,
            "focus_region": None,
        },
    )

    class CapturingJudge(_Judge):
        def __init__(self) -> None:
            super().__init__("valid")
            self.requests: list[dict] = []

        def adjudicate_p0b(self, request: dict) -> dict:
            self.requests.append(request)
            return super().adjudicate_p0b(request)

    judge = CapturingJudge()
    report = check_collision(
        _scene(
            [
                _obj("layer", layer_center, layer_size),
                _obj("load", load_center, load_size),
            ]
        ),
        collision_geometry=geometry,
        vlm_judge=judge,
    )

    pair = report["pairs"][0]
    context = pair["shallow_surface_layer_overlap_evidence"]
    assert pair["route"] == "vlm_adjudicated"
    assert context["vertical_overlap_m"] == pytest.approx(0.01)
    assert (
        judge.requests[0]["detector_evidence"][
            "shallow_surface_layer_overlap"
        ]
        == context
    )


@pytest.mark.parametrize(
    ("layer_center", "layer_size", "load_center", "load_size"),
    [
        # Overlap exceeds the bounded shallow-contact depth.
        (
            [1.5, 1.5, 0.01],
            [2.0, 2.0, 0.02],
            [1.5, 1.5, 0.5],
            [1.0, 1.0, 1.0],
        ),
        # A vertical thin plane is not a horizontal support layer.
        (
            [1.5, 1.5, 0.5],
            [0.01, 2.0, 1.0],
            [1.5, 1.5, 0.5],
            [1.0, 1.0, 1.0],
        ),
        # A horizontal plane through the middle of an object is not floor contact.
        (
            [1.5, 1.5, 0.5],
            [2.0, 2.0, 0.01],
            [1.5, 1.5, 0.5],
            [1.0, 1.0, 1.0],
        ),
        # The other object crosses below the supporting substrate.
        (
            [1.5, 1.5, 0.005],
            [2.0, 2.0, 0.01],
            [1.5, 1.5, 0.495],
            [1.0, 1.0, 1.0],
        ),
    ],
)
def test_shallow_floor_layer_context_does_not_generalize_to_other_collisions(
    layer_center,
    layer_size,
    load_center,
    load_size,
) -> None:
    layer = normalize_object(
        _obj("layer", layer_center, layer_size)
    )
    load = normalize_object(
        _obj("load", load_center, load_size)
    )

    assert (
        _shallow_surface_layer_overlap_evidence(
            _scene([]),
            layer,
            load,
            obb=obb_sat_test(layer, load),
            mesh_evidence={
                "surface_intersection": True,
                "intersection": {"definitive": True},
            },
            config=dict(DEFAULT_COLLISION_CONFIG),
        )
        is None
    )


# 4. OBB overlap without mesh calls VLM exactly once
def test_obb_overlap_without_mesh_calls_vlm_once() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("valid")
    report = check_collision(
        scene,
        {"detector_only": False, "official_mode": False},
        prompt="Place two boxes.",
        vlm_judge=judge,
    )
    assert judge.calls == 1
    assert report["pairs"][0]["route"] == "vlm_adjudicated"
    assert report["pairs"][0]["final_verdict"] == "valid"


# 5. Relation claims do not auto-exempt overlap
def test_relation_claims_do_not_auto_exempt_overlap() -> None:
    scene = _scene(
        [
            _obj("table", [1.0, 1.0, 0.5], [1.0, 1.0, 0.2], category="table"),
            _obj("book", [1.0, 1.0, 0.55], [0.3, 0.3, 0.2], category="book"),
        ]
    )
    judge = _Judge("valid")
    report = check_collision(
        scene,
        {"detector_only": False},
        relationships=[{"subject": "book", "predicate": "on", "object": "table"}],
        vlm_judge=judge,
    )
    assert judge.calls == 1
    assert report["pairs"][0]["route"] == "vlm_adjudicated"


# 6. Reliable mesh separation above threshold skips VLM
def test_reliable_mesh_separation_skips_vlm(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    # OBBs overlap, but the baked meshes are separated inside those boxes.
    _box_mesh(mesh_a, [0.8, 1.0, 0.5], [0.3, 0.3, 0.3])
    _box_mesh(mesh_b, [2.2, 1.0, 0.5], [0.3, 0.3, 0.3])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "a.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "b.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [2.0, 2.0, 2.0]), _obj("b", [1.5, 1.0, 0.5], [2.0, 2.0, 2.0])])
    judge = _Judge("invalid")
    report = check_collision(scene, {"separation_threshold_m": 0.02}, collision_geometry=geometry, vlm_judge=judge)
    pair = report["pairs"][0]
    assert pair["route"] == "direct_valid_mesh_separated"
    assert pair["evidence_level"] == "mesh"
    assert judge.calls == 0


# 7. Near-contact mesh pair calls VLM
def test_near_contact_mesh_pair_calls_vlm(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    _box_mesh(mesh_a, [1.0, 1.0, 0.5], [0.4, 0.4, 0.4])
    _box_mesh(mesh_b, [1.41, 1.0, 0.5], [0.4, 0.4, 0.4])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "a.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "b.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.5, 1.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("valid")
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)
    assert judge.calls == 1
    assert report["pairs"][0]["mesh_evidence"]["mesh_state"] == "near_contact"


# 7b. Offset near-contact must not be certified separated by a vertex overestimate.
def test_offset_near_contact_is_conservative(tmp_path: Path) -> None:
    def _bounds_mesh(path: Path, lo: list[float], hi: list[float]) -> None:
        center = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
        size = [hi[i] - lo[i] for i in range(3)]
        _box_mesh(path, center, size)

    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    # True surface gap along x is 0.01 m (< threshold). Nearest vertices are far
    # apart because the meshes are offset in y, so a vertex-distance estimate
    # would wrongly report > threshold.
    _bounds_mesh(mesh_a, [0.0, -0.10, 0.0], [1.0, 0.10, 1.0])
    _bounds_mesh(mesh_b, [1.01, 0.05, 0.0], [2.0, 0.25, 1.0])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "a.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "b.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [0.5, 0.5, 0.5], [3.0, 3.0, 3.0]), _obj("b", [1.5, 0.5, 0.5], [3.0, 3.0, 3.0])])
    judge = _Judge("valid")
    report = check_collision(scene, {"separation_threshold_m": 0.02}, collision_geometry=geometry, vlm_judge=judge)
    pair = report["pairs"][0]
    assert pair["mesh_evidence"]["mesh_reliable_for_separation"] is False
    mesh_evidence = pair["mesh_evidence"]
    if mesh_evidence["minimum_surface_distance_backend"] == "trimesh_fcl":
        assert mesh_evidence["minimum_surface_distance_is_lower_bound"] is False
        assert mesh_evidence["minimum_surface_distance_m"] == pytest.approx(0.01)
    else:
        assert mesh_evidence["minimum_surface_distance_backend"] == "numpy_aabb_gap"
        assert mesh_evidence["minimum_surface_distance_is_lower_bound"] is True
    assert pair["route"] == "vlm_adjudicated"
    assert judge.calls == 1


# 8. Surface intersection calls VLM with mesh evidence
def test_surface_intersection_calls_vlm_with_mesh_evidence(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    _box_mesh(mesh_a, [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    _box_mesh(mesh_b, [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "a.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "b.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("invalid")
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)
    assert judge.calls == 1
    mesh_evidence = report["pairs"][0]["mesh_evidence"]
    intersection = mesh_evidence["intersection"]
    if intersection["definitive"]:
        assert mesh_evidence["surface_intersection"] is True
    else:
        assert mesh_evidence["surface_intersection"] is None
        assert mesh_evidence["candidate_aabb_overlap"] is True
    assert intersection["contacts"] == []
    assert intersection["contact_count"] == 0
    assert report["pairs"][0]["evidence_level"] == "mesh"


# 9. Containment calls VLM
def test_containment_calls_vlm(tmp_path: Path) -> None:
    mesh_outer = tmp_path / "outer.ply"
    mesh_inner = tmp_path / "inner.ply"
    _box_mesh(mesh_outer, [1.0, 1.0, 0.5], [2.0, 2.0, 2.0])
    _box_mesh(mesh_inner, [1.0, 1.0, 0.5], [0.4, 0.4, 0.4])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "outer": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "outer.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "inner": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "inner.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("outer", [1.0, 1.0, 0.5], [2.0, 2.0, 2.0]), _obj("inner", [1.0, 1.0, 0.5], [0.4, 0.4, 0.4])])
    judge = _Judge("valid")
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)
    assert judge.calls == 1
    mesh_evidence = report["pairs"][0]["mesh_evidence"]
    if True in {mesh_evidence["containment_a_in_b"], mesh_evidence["containment_b_in_a"]}:
        assert mesh_evidence["mesh_state"] == "contained"
    else:
        # The numpy fallback intentionally refuses to infer mesh containment
        # from AABB nesting alone.
        assert mesh_evidence["mesh_state"] == "unknown"


# 10. Invalid mesh does not direct-pass
def test_invalid_mesh_does_not_direct_pass(tmp_path: Path) -> None:
    bad_mesh = tmp_path / "bad.ply"
    bad_mesh.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n", encoding="utf-8")
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "bad.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "bad.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("valid")
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)
    assert judge.calls == 1
    assert report["pairs"][0]["route"] == "vlm_adjudicated"


# 11. Mixed mesh/proxy pair follows OBB-plus-VLM path
def test_mixed_mesh_proxy_pair_routes_to_vlm(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    _box_mesh(mesh_a, [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": "a.ply", "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    )
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("valid")
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)
    assert judge.calls == 1
    assert report["pairs"][0]["evidence_level"] == "obb"


# 12. Mesh backend failure is explicit
def test_mesh_backend_failure_is_explicit(tmp_path: Path) -> None:
    entry_a = {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(tmp_path / "missing.ply"), "transform_baked": True, "complete": True}
    entry_b = {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(tmp_path / "missing2.ply"), "transform_baked": True, "complete": True}
    result = evaluate_mesh_pair("a", "b", entry_a, entry_b, base_dir=tmp_path)
    assert result["status"] == "unavailable"
    assert result["mesh_reliable_for_separation"] is False


# 13. Missing judge in official mode fails evaluation
def test_missing_judge_in_official_mode_fails() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    with pytest.raises(CollisionEvaluationError, match="no judge"):
        check_collision(scene, {"official_mode": True}, vlm_judge=None)


# 14. Invalid VLM verdict fails parsing
def test_invalid_vlm_verdict_fails_parsing() -> None:
    class BadJudge:
        vlm_control_enabled = False

        def adjudicate_p0b(self, request: dict) -> dict:
            return {"verdict": "insufficient_evidence", "confidence": 0.5, "reason": "bad"}

    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    report = check_collision(scene, vlm_judge=BadJudge())
    assert report["pairs"][0]["adjudication_error"] is not None
    assert report["pairs"][0]["route"] == "vlm_adjudication_failed"
    assert report["coverage"]["vlm_adjudicated_pairs"] == 0


def test_binary_schema_failure_audit_is_preserved_in_event_report() -> None:
    schema_audit = {
        "policy": "single_schema_repair_retry_v1",
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": False,
        "attempts": [],
    }

    class BadJudge:
        vlm_control_enabled = False

        def adjudicate_p0b(self, request: dict) -> dict:
            raise ResponseSchemaRepairError(
                "binary response remained invalid",
                schema_audit=schema_audit,
            )

    scene = _scene([
        _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
        _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0]),
    ])

    report = check_collision(scene, vlm_judge=BadJudge())

    pair = report["pairs"][0]
    assert pair["route"] == "vlm_adjudication_failed"
    assert pair["adjudication_failure_audit"] == schema_audit
    assert report["status"] == "requires_vlm"
    assert report["score"] is None


# 15. VLM valid/invalid update pair and aggregate reports
def test_vlm_verdict_updates_pair_and_aggregate() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    valid_report = check_collision(scene, vlm_judge=_Judge("valid"))
    invalid_report = check_collision(scene, vlm_judge=_Judge("invalid"))
    assert valid_report["collision_count"] == 0
    assert valid_report["score"] == 1.0
    assert invalid_report["collision_count"] == 1
    assert invalid_report["score"] == 0.5


# 16. Point-cloud PLY is not treated as triangle mesh
def test_point_cloud_ply_is_not_triangle_mesh(tmp_path: Path) -> None:
    ply = tmp_path / "cloud.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    entry = {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(ply), "transform_baked": True, "complete": True}
    assert is_usable_triangle_mesh(entry, base_dir=tmp_path) is False


# 17. Uniformly fitted FBX geometry remains enclosed by canonical OBB
def test_uniform_fit_geometry_remains_inside_canonical_obb() -> None:
    source = [2.0, 1.0, 4.0]
    target = [4.0, 4.0, 4.0]
    fit = _uniform_contain_fit(source, target)
    scale = fit["uniform_scale"]
    rendered = fit["rendered_size"]
    center = np.array([1.0, 2.0, 0.5])
    half = np.array(target) / 2.0
    rotation = np.eye(3)
    local_corners = np.array(
        [[sx * rendered[0] / 2.0, sy * rendered[1] / 2.0, sz * rendered[2] / 2.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    world = center + local_corners
    assert obb_encloses_points(center, half, rotation, world)
    assert scale == pytest.approx(min(target[index] / source[index] for index in range(3)))


# 18. Geometry manifest preserves object IDs and world transforms
def test_geometry_manifest_preserves_object_ids_and_transforms(tmp_path: Path) -> None:
    mesh_path = tmp_path / "cabinet.ply"
    _box_mesh(mesh_path, [2.0, 2.0, 0.5], [0.8, 0.8, 1.0])
    manifest = _geometry_manifest(
        tmp_path,
        {
            "cabinet": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "cabinet.ply",
                "transform_baked": True,
                "geometry_source": "asset_fbx",
                "source_uri": "/assets/cabinet.fbx",
                "complete": True,
            }
        },
    )
    assert "cabinet" in manifest["objects"]
    assert manifest["objects"]["cabinet"]["geometry_source"] == "asset_fbx"
    assert Path(manifest["objects"]["cabinet"]["geometry_path"]).is_file()


# 19. Detector-only mode exposes requires_vlm with null score
def test_detector_only_mode_has_null_score_and_requires_vlm() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    report = check_collision(scene, {"detector_only": True})
    assert report["score"] is None
    assert report["requires_vlm_count"] == 1


def test_malformed_claimed_mesh_routes_to_vlm_instead_of_crashing(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.ply"
    malformed.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    entry = {
        "representation": TRIANGLE_MESH_REPRESENTATION,
        "geometry_path": str(malformed),
        "transform_baked": True,
        "geometry_source": "test",
        "complete": True,
    }
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    geometry = _geometry_manifest(tmp_path, {"a": entry, "b": entry})
    judge = _Judge("valid")

    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)

    mesh = report["pairs"][0]["mesh_evidence"]
    assert mesh["status"] == "invalid_mesh"
    assert mesh["mesh_reliable_for_separation"] is False
    assert mesh["error"]
    assert report["pairs"][0]["route"] == "vlm_adjudicated"
    assert judge.calls == 1


def test_aabb_fallback_reports_candidate_not_surface_contact() -> None:
    mesh_a = {
        "vertices": np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        "faces": np.asarray([[0, 1, 2], [0, 1, 3]], dtype=int),
    }
    mesh_b = {
        "vertices": np.asarray([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [0.5, 0.5, 1.0]], dtype=float),
        "faces": np.asarray([[0, 1, 2], [0, 1, 3]], dtype=int),
    }

    result = _surface_intersection_aabb(mesh_a, mesh_b)

    assert result["intersects"] is None
    assert result["definitive"] is False
    assert result["candidate_aabb_overlap"] is True
    assert result["contact_count"] == 0
    assert result["contacts"] == []
    assert result["focus_hint"]["is_contact"] is False


def test_aabb_nesting_does_not_claim_mesh_containment() -> None:
    outer = np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=float)
    nested = np.asarray([[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]], dtype=float)
    disjoint = np.asarray([[2, 2, 2], [3, 3, 3]], dtype=float)

    assert _containment_numpy(nested, outer) == "unknown"
    assert _containment_numpy(disjoint, outer) is False


def test_obb_separation_routes_when_complete_mesh_escapes_canonical_obb(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    # Canonical a is centered at x=1, but its claimed complete world mesh is at
    # x=4 beside b.  OBB separation alone is therefore not a safe certificate.
    _box_mesh(mesh_a, [4.0, 1.0, 0.5], [0.5, 0.5, 0.5])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "a.ply",
                "transform_baked": True,
                "geometry_source": "test",
                "complete": True,
            }
        },
    )
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [4.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("valid")

    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)

    pair = report["pairs"][0]
    assert pair["obb_evidence"]["obb_certifiably_separated"] is True
    assert pair["mesh_enclosure_evidence"]["object_a"]["status"] == "outside_canonical_obb"
    assert pair["route"] == "vlm_adjudicated"
    assert judge.calls == 1


def test_obb_separation_accepts_renderer_style_contained_ply(tmp_path: Path) -> None:
    mesh_a = tmp_path / "a.ply"
    _box_mesh(mesh_a, [1.0, 1.0, 0.5], [0.8, 0.6, 0.4])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "a.ply",
                "transform_baked": True,
                "geometry_source": "asset_fbx",
                "complete": True,
            }
        },
    )
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [4.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("invalid")

    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)

    pair = report["pairs"][0]
    assert pair["mesh_enclosure_evidence"]["object_a"]["status"] == "verified_inside"
    assert pair["route"] == "direct_valid_obb_separated"
    assert judge.calls == 0


def test_mesh_separation_cannot_bypass_canonical_frame_guard_when_obbs_overlap(
    tmp_path: Path,
) -> None:
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    # The canonical boxes overlap near x=1, while stale world-space meshes were
    # exported far apart. Mesh distance alone would falsely certify separation.
    _box_mesh(mesh_a, [3.0, 1.0, 0.5], [0.4, 0.4, 0.4])
    _box_mesh(mesh_b, [5.0, 1.0, 0.5], [0.4, 0.4, 0.4])
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "a.ply",
                "transform_baked": True,
                "geometry_source": "stale_export",
                "complete": True,
            },
            "b": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "b.ply",
                "transform_baked": True,
                "geometry_source": "stale_export",
                "complete": True,
            },
        },
    )
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("valid")

    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)

    pair = report["pairs"][0]
    assert pair["obb_evidence"]["obb_certifiably_separated"] is False
    assert pair["mesh_evidence"]["mesh_reliable_for_separation"] is True
    assert pair["mesh_enclosure_evidence"]["object_a"]["status"] == "outside_canonical_obb"
    assert pair["mesh_enclosure_evidence"]["object_b"]["status"] == "outside_canonical_obb"
    assert pair["route"] == "vlm_adjudicated"
    assert judge.calls == 1


def test_obb_separation_does_not_trust_unavailable_claimed_complete_mesh(tmp_path: Path) -> None:
    geometry = _geometry_manifest(
        tmp_path,
        {
            "a": {
                "representation": TRIANGLE_MESH_REPRESENTATION,
                "geometry_path": "missing.ply",
                "transform_baked": True,
                "geometry_source": "external",
                "complete": True,
            }
        },
    )
    scene = _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("b", [4.0, 1.0, 0.5], [1.0, 1.0, 1.0]),
        ]
    )
    judge = _Judge("valid")

    report = check_collision(scene, collision_geometry=geometry, vlm_judge=judge)

    pair = report["pairs"][0]
    assert pair["mesh_enclosure_evidence"]["object_a"]["status"] == "mesh_enclosure_unavailable"
    assert pair["route"] == "vlm_adjudicated"
    assert judge.calls == 1


def test_glb_scene_graph_transforms_are_baked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    glb_path = tmp_path / "instanced.glb"
    glb_path.write_bytes(b"fake glb for loader seam test")

    class FakeMesh:
        def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
            self.vertices = np.asarray(vertices, dtype=float)
            self.faces = np.asarray(faces, dtype=int)

        def copy(self):
            return deepcopy(self)

        def apply_transform(self, transform: np.ndarray) -> None:
            homogeneous = np.column_stack([self.vertices, np.ones(len(self.vertices))])
            self.vertices = (homogeneous @ np.asarray(transform, dtype=float).T)[:, :3]

    transform = np.eye(4)
    transform[:3, 3] = [3.0, 4.0, 5.0]
    source = FakeMesh(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        np.asarray([[0, 1, 2]], dtype=int),
    )

    class FakeGraph:
        nodes_geometry = ["translated_node"]

        def __getitem__(self, node_name: str):
            assert node_name == "translated_node"
            return transform, "triangle"

    scene = SimpleNamespace(geometry={"triangle": source}, graph=FakeGraph())
    calls: list[dict] = []

    def fake_load(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return scene

    fake_trimesh = SimpleNamespace(
        load=fake_load,
        util=SimpleNamespace(concatenate=lambda meshes: meshes[0]),
    )
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    mesh = load_triangle_mesh(
        {
            "representation": TRIANGLE_MESH_REPRESENTATION,
            "geometry_path": str(glb_path),
            "transform_baked": True,
            "complete": True,
        }
    )

    assert calls[0]["force"] == "scene"
    assert mesh["loader"] == "trimesh_scene_graph"
    assert mesh["vertices"].tolist() == [[3.0, 4.0, 5.0], [4.0, 4.0, 5.0], [3.0, 5.0, 5.0]]


@pytest.mark.parametrize(
    "config,match",
    [
        ({"obb_sat_eps": -1.0}, "obb_sat_eps"),
        ({"mesh_enclosure_eps_m": float("inf")}, "mesh_enclosure_eps_m"),
        ({"separation_threshold_m": float("nan")}, "separation_threshold_m"),
        (
            {
                "shallow_surface_layer_min_horizontal_alignment": 1.1
            },
            "min_horizontal_alignment",
        ),
        (
            {"shallow_surface_layer_policy": "direct_valid"},
            "shallow_surface_layer_policy",
        ),
        ({"score_mode": "unknown"}, "score_mode"),
        ({"official_mode": True, "detector_only": True}, "mutually exclusive"),
    ],
)
def test_collision_rejects_invalid_configuration(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_collision(_scene([_obj("a", [1, 1, 0.5], [1, 1, 1])]), config)
