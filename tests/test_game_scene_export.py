from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from benchmark.api.evaluation import run_evaluate
from benchmark.evaluator.generic_validity.mesh_geometry import (
    is_usable_triangle_mesh,
    load_triangle_mesh,
    validate_collision_geometry_manifest,
)
from benchmark.game_scene import (
    GameSceneExportError,
    build_scene_and_collision_geometry,
    validate_probe_payload,
)
from benchmark.scene_io.object_normalization import rotation_matrix_from_euler
from benchmark.scene_io.validate import validate_generated_scene
from benchmark.utils.io import load_yaml


_CUBE_FACES = [
    [0, 1, 3], [0, 3, 2],
    [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4],
    [1, 5, 7], [1, 7, 3],
]


def _unit_cube_vertices() -> np.ndarray:
    return np.array(
        [[sx * 0.5, sy * 0.5, sz * 0.5] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    )


def _yaw_about_y(degrees: float) -> tuple[np.ndarray, list[float]]:
    angle = math.radians(degrees)
    matrix = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=float,
    )
    quaternion = [0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)]
    return matrix, quaternion


def _probe_object(
    object_id: str,
    *,
    translation: tuple[float, float, float],
    yaw_degrees: float = 0.0,
    category: str = "crate",
    complete: bool = True,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    non_physical_hint: str | None = None,
) -> dict:
    rotation, quaternion = _yaw_about_y(yaw_degrees)
    scaled = _unit_cube_vertices() * np.asarray(scale, dtype=float)
    vertices = (rotation @ scaled.T).T + np.asarray(translation, dtype=float)
    entry = {
        "id": object_id,
        "category": category,
        "rotation_quaternion": quaternion,
        "world_bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        "mesh_complete": complete,
        "non_physical_hint": non_physical_hint,
    }
    if complete:
        entry["vertices"] = vertices.tolist()
        entry["faces"] = _CUBE_FACES
    else:
        entry["vertices"] = []
        entry["faces"] = []
    return entry


def _probe(objects: list[dict], *, up_axis: str = "y", unit_scale: float = 1.0) -> dict:
    return {
        "schema_version": "game_scene_probe_v1",
        "up_axis": up_axis,
        "unit_scale": unit_scale,
        "captured_at_tick": 60,
        "deterministic_seed": 20260727,
        "objects": objects,
    }


def _export(payload: dict, tmp_path: Path) -> tuple[dict, dict]:
    return build_scene_and_collision_geometry(
        payload,
        scene_id="game_scene",
        request_id="game_request",
        scene_type="game_level",
        mesh_dir=tmp_path / "collision_geometry",
    )


def test_export_produces_valid_canonical_scene_and_collision_manifest(tmp_path: Path) -> None:
    payload = _probe([_probe_object("cube_0000", translation=(2.0, 0.5, -3.0), yaw_degrees=45.0)])

    scene, manifest = _export(payload, tmp_path)

    validate_generated_scene(scene)
    validate_collision_geometry_manifest(manifest)
    obj = scene["objects"][0]
    assert obj["size"] == pytest.approx([1.0, 1.0, 1.0])
    assert obj["rotation"] == pytest.approx([0.0, 0.0, 45.0])
    assert obj["geometry_provenance"] == "generated_mesh"
    # A 45-degree yawed unit square has a sqrt(2) axis-aligned footprint.
    assert scene["boundary"][2] == pytest.approx([math.sqrt(2.0), math.sqrt(2.0)])
    assert scene["scene_height"] == pytest.approx(1.0)
    assert is_usable_triangle_mesh(manifest["objects"]["cube_0000"])


def test_three_js_y_up_becomes_canonical_z_up(tmp_path: Path) -> None:
    # A box lifted along the three.js up axis must land above the canonical floor.
    payload = _probe([_probe_object("cube_0000", translation=(0.0, 4.0, 0.0))])

    scene, _ = _export(payload, tmp_path)

    assert scene["scene_height"] == pytest.approx(1.0)
    assert scene["objects"][0]["center"][2] == pytest.approx(0.5)
    assert scene["metadata"]["game_scene_import"]["source_up_axis"] == "y"
    assert scene["metadata"]["game_scene_import"]["translation_applied"][2] == pytest.approx(-3.5)


def test_declared_obb_encloses_exported_mesh(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("cube_0000", translation=(2.0, 0.5, -3.0), yaw_degrees=37.0),
            _probe_object("cube_0001", translation=(-4.0, 2.0, 1.5), yaw_degrees=-12.0),
        ]
    )

    scene, manifest = _export(payload, tmp_path)

    # The collision narrow phase only trusts a mesh whose vertices sit inside
    # the canonical OBB, so the two artifacts must agree by construction.
    for obj in scene["objects"]:
        mesh = load_triangle_mesh(manifest["objects"][obj["id"]])
        rotation = rotation_matrix_from_euler(obj["rotation"])
        local = (np.asarray(mesh["vertices"], dtype=float) - np.asarray(obj["center"], dtype=float)) @ rotation
        half = np.asarray(obj["size"], dtype=float) / 2.0
        assert np.all(np.abs(local) <= half + 1.0e-6)


def test_unit_scale_converts_to_meters(tmp_path: Path) -> None:
    payload = _probe(
        [_probe_object("cube_0000", translation=(0.0, 0.5, 0.0))],
        unit_scale=0.1,
    )

    scene, _ = _export(payload, tmp_path)

    assert scene["objects"][0]["size"] == pytest.approx([0.1, 0.1, 0.1])
    assert scene["metadata"]["game_scene_import"]["unit_scale"] == pytest.approx(0.1)


def test_bounds_only_object_degrades_to_bbox_proxy(tmp_path: Path) -> None:
    payload = _probe(
        [_probe_object("cube_0000", translation=(0.0, 0.5, 0.0), yaw_degrees=30.0, complete=False)]
    )

    scene, manifest = _export(payload, tmp_path)

    entry = manifest["objects"]["cube_0000"]
    assert entry["representation"] == "bbox_proxy"
    assert is_usable_triangle_mesh(entry) is False
    # Without triangles the exporter must not claim an oriented fit.
    assert scene["objects"][0]["rotation"] == pytest.approx([0.0, 0.0, 0.0])


def test_flat_geometry_keeps_size_components_positive(tmp_path: Path) -> None:
    ground = {
        "id": "ground_0000",
        "category": "ground",
        "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
        "vertices": [[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "mesh_complete": True,
    }
    payload = _probe([ground, _probe_object("cube_0000", translation=(0.0, 1.0, 0.0))])

    scene, _ = _export(payload, tmp_path)

    validate_generated_scene(scene)
    assert min(scene["objects"][0]["size"]) > 0.0


def test_probe_validation_rejects_missing_category() -> None:
    entry = _probe_object("cube_0000", translation=(0.0, 0.5, 0.0))
    entry.pop("category")

    with pytest.raises(GameSceneExportError, match="category"):
        validate_probe_payload(_probe([entry]))


def test_probe_validation_rejects_duplicate_ids() -> None:
    payload = _probe(
        [
            _probe_object("cube_0000", translation=(0.0, 0.5, 0.0)),
            _probe_object("cube_0000", translation=(2.0, 0.5, 0.0)),
        ]
    )

    with pytest.raises(GameSceneExportError, match="duplicates"):
        validate_probe_payload(payload)


def test_probe_validation_rejects_unknown_schema_version() -> None:
    payload = _probe([_probe_object("cube_0000", translation=(0.0, 0.5, 0.0))])
    payload["schema_version"] = "game_scene_probe_v2"

    with pytest.raises(GameSceneExportError, match="schema_version"):
        validate_probe_payload(payload)


def _ground_plane(object_id: str, *, half_width: float, half_depth: float, height: float) -> dict:
    """A zero-thickness upward-facing sheet, wound the way three.js winds one.

    Vertex and index order are taken from a real ``PlaneGeometry`` carrying
    ``rotation.x = -PI/2``, which is how every ground plane in the corpus is
    authored. The winding is what tells the exporter which side is outside, so
    a hand-rolled quad would test a convention the games do not use.
    """

    vertices = [
        [-half_width, height, -half_depth],
        [half_width, height, -half_depth],
        [-half_width, height, half_depth],
        [half_width, height, half_depth],
    ]
    return {
        "id": object_id,
        "category": "floor",
        "rotation_quaternion": [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        "world_bounds": {
            "min": [-half_width, height, -half_depth],
            "max": [half_width, height, half_depth],
        },
        "mesh_complete": True,
        "non_physical_hint": None,
        "vertices": vertices,
        "faces": [[0, 2, 1], [2, 3, 1]],
    }


def test_ground_plane_thickness_is_added_below_the_authored_surface(tmp_path: Path) -> None:
    # A sheet has no thickness, but a box axis of zero length has no separating
    # axis either, so the exporter has to invent one. Centring that invention
    # would raise the walking surface by half of it and make every object in the
    # level stand half a millimetre inside the floor.
    payload = _probe(
        [
            _ground_plane("floor", half_width=6.0, half_depth=6.0, height=0.0),
            _probe_object("cube_0000", translation=(0.0, 0.5, 0.0)),
        ]
    )

    scene, _ = _export(payload, tmp_path)

    floor = next(obj for obj in scene["objects"] if obj["id"] == "floor")
    cube = next(obj for obj in scene["objects"] if obj["id"] == "cube_0000")
    padding = floor["metadata"]["degenerate_axis_padding"]
    assert padding["axes"][0]["source_extent_m"] == pytest.approx(0.0)
    assert padding["axes"][0]["placement"] == "behind_authored_surface"
    assert cube["metadata"]["degenerate_axis_padding"] is None

    rotation = rotation_matrix_from_euler(floor["rotation"])
    reach = float(np.abs(rotation[2]) @ (np.asarray(floor["size"], dtype=float) / 2.0))
    floor_top = floor["center"][2] + reach
    cube_bottom = cube["center"][2] - cube["size"][2] / 2.0
    assert floor_top == pytest.approx(cube_bottom, abs=1e-12)


def test_resting_on_a_ground_plane_is_contact_rather_than_penetration(tmp_path: Path) -> None:
    payload = _probe(
        [
            _ground_plane("floor", half_width=6.0, half_depth=6.0, height=0.0),
            _probe_object("cube_0000", translation=(-2.0, 0.5, 0.0)),
            _probe_object("cube_0001", translation=(2.0, 0.5, 0.0)),
        ]
    )
    scene, manifest = _export(payload, tmp_path)
    profile = load_yaml(
        Path("configs/evaluation/metric_profile_game_canonical_v1.yaml"), default={}
    )

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "out",
        eval_generic_validity=True,
        collision_geometry=manifest,
        evaluation_profile=profile,
    )

    pairs = report["reports"]["generic_validity"]["metrics"]["collision"]["pairs"]
    resting = [pair for pair in pairs if "floor" in (pair["object_a"], pair["object_b"])]
    assert len(resting) == 2
    # The judge is shown this number. Reporting the fabricated thickness here
    # states that the level interpenetrates when it merely touches.
    for pair in resting:
        assert pair["diagnostics"]["z_overlap"] == pytest.approx(0.0, abs=1e-12)


def test_exported_game_scene_scores_collision_under_game_profile(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("cube_0000", translation=(-2.0, 0.5, 0.0)),
            _probe_object("cube_0001", translation=(2.0, 0.5, 0.0)),
        ]
    )
    scene, manifest = _export(payload, tmp_path)
    profile = load_yaml(
        Path("configs/evaluation/metric_profile_game_canonical_v1.yaml"), default={}
    )

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "out",
        eval_generic_validity=True,
        collision_geometry=manifest,
        evaluation_profile=profile,
    )

    assert report["workflow"] == "canonical_l0_l4"
    metrics = report["reports"]["generic_validity"]["metrics"]
    assert metrics["collision"]["status"] == "checked"
    assert metrics["collision"]["collision_count"] == 0
    assert metrics["collision"]["score"] == pytest.approx(1.0)
    assert metrics["navigability"]["status"] == "checked"
    assert metrics["navigability"]["score_definition"] == (
        "largest_connected_free_area / total_free_area"
    )
    # Metrics without a valid Counter-Strike scene contract stay off.
    for disabled in ("oob", "support", "accessibility"):
        assert metrics[disabled]["status"] == "not_applicable"


def test_outline_shell_is_dropped_without_a_boundary_warning(tmp_path: Path) -> None:
    # The toon outline idiom wraps a mesh in a slightly inflated back-face copy.
    # It is scenery, and dropping it needs no operator attention.
    payload = _probe(
        [
            _probe_object("core_0000", translation=(0.0, 0.5, 0.0)),
            _probe_object(
                "shell_0001",
                translation=(0.0, 0.5, 0.0),
                scale=(1.06, 1.06, 1.06),
                non_physical_hint="back_side_only",
            ),
        ]
    )

    scene, manifest = _export(payload, tmp_path)

    validate_generated_scene(scene)
    assert [obj["id"] for obj in scene["objects"]] == ["core_0000"]
    assert "shell_0001" not in manifest["objects"]
    report = scene["metadata"]["game_scene_import"]["individualization"]
    assert report["dropped_non_physical_ids"] == ["shell_0001"]
    assert report["warnings"] == []


def test_dropping_a_loose_enclosing_shell_is_reported(tmp_path: Path) -> None:
    # A room turned inside out is also back-face-only, but it contains the level
    # instead of hugging one mesh. Losing it would lose the walls, so the drop
    # must be visible rather than silent.
    payload = _probe(
        [
            _probe_object(
                "room_0000",
                translation=(0.0, 5.0, 0.0),
                scale=(20.0, 10.0, 20.0),
                non_physical_hint="back_side_only",
            ),
            _probe_object("crate_0001", translation=(-3.0, 0.5, 0.0)),
            _probe_object("crate_0002", translation=(3.0, 0.5, 0.0)),
        ]
    )

    scene, _ = _export(payload, tmp_path)

    report = scene["metadata"]["game_scene_import"]["individualization"]
    assert [warning["code"] for warning in report["warnings"]] == ["enclosing_shell_dropped"]
    warning = report["warnings"][0]
    assert warning["object_id"] == "room_0000"
    assert warning["enclosed_object_count"] == 2
    assert warning["tightest_wrap_volume_ratio"] < 0.25


def test_contained_geometry_is_absorbed_and_stays_inside_the_container_obb(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("body_0000", translation=(0.0, 2.0, 0.0), scale=(4.0, 4.0, 4.0)),
            _probe_object("detail_0001", translation=(0.5, 2.0, 0.0)),
            _probe_object("neighbour_0002", translation=(9.0, 0.5, 0.0)),
        ]
    )

    scene, manifest = _export(payload, tmp_path)

    validate_generated_scene(scene)
    validate_collision_geometry_manifest(manifest)
    assert [obj["id"] for obj in scene["objects"]] == ["body_0000", "neighbour_0002"]
    body = scene["objects"][0]
    assert body["metadata"]["absorbed_ids"] == ["detail_0001"]
    assert body["size"] == pytest.approx([4.0, 4.0, 4.0])
    # Both meshes now live under one entry, and the enclosure guard that the
    # collision narrow phase relies on still holds for the merged geometry.
    entry = manifest["objects"]["body_0000"]
    assert entry["face_count"] == 2 * len(_CUBE_FACES)
    mesh = load_triangle_mesh(entry)
    rotation = rotation_matrix_from_euler(body["rotation"])
    local = (np.asarray(mesh["vertices"], dtype=float) - np.asarray(body["center"], dtype=float)) @ rotation
    assert np.all(np.abs(local) <= np.asarray(body["size"], dtype=float) / 2.0 + 1.0e-9)


def test_nested_containment_lands_on_the_surviving_outermost_object(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("body_0000", translation=(0.0, 4.0, 0.0), scale=(8.0, 8.0, 8.0)),
            _probe_object("part_0001", translation=(0.0, 4.0, 0.0), scale=(4.0, 4.0, 4.0)),
            _probe_object("detail_0002", translation=(0.0, 4.0, 0.0)),
        ]
    )

    scene, _ = _export(payload, tmp_path)

    assert [obj["id"] for obj in scene["objects"]] == ["body_0000"]
    assert scene["objects"][0]["metadata"]["absorbed_ids"] == ["detail_0002", "part_0001"]


def test_a_flat_level_graph_keeps_every_object(tmp_path: Path) -> None:
    # An implementation that adds each box straight to the scene must not be
    # penalised by a rule aimed at nested ones.
    payload = _probe(
        [_probe_object(f"crate_{index:04d}", translation=(index * 3.0, 0.5, 0.0)) for index in range(4)]
    )
    payload["individualization"] = {
        "strategy": "visible_mesh_v1",
        "top_level_child_count": 4,
        "max_graph_depth": 1,
        "declared_category_count": 0,
        "counts": {"meshes_visited": 4, "emitted": 4},
    }

    scene, _ = _export(payload, tmp_path)

    report = scene["metadata"]["game_scene_import"]["individualization"]
    assert report["objects_exported"] == 4
    assert report["absorbed_into_container"] == 0
    assert report["dropped_non_physical"] == 0
    # The probe's own tally travels with the artifact so that a level which
    # collapses to one object can be read off the export instead of inferred
    # from a suspiciously clean score.
    assert report["probe"]["top_level_child_count"] == 4
    assert report["probe"]["counts"]["meshes_visited"] == 4


def test_individualization_rules_can_be_disabled_for_diagnosis(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("body_0000", translation=(0.0, 2.0, 0.0), scale=(4.0, 4.0, 4.0)),
            _probe_object("detail_0001", translation=(0.5, 2.0, 0.0)),
            _probe_object(
                "shell_0002",
                translation=(0.0, 2.0, 0.0),
                scale=(4.24, 4.24, 4.24),
                non_physical_hint="back_side_only",
            ),
        ]
    )

    scene, _ = build_scene_and_collision_geometry(
        payload,
        scene_id="game_scene",
        request_id="game_request",
        scene_type="game_level",
        mesh_dir=tmp_path / "collision_geometry",
        drop_non_physical=False,
        collapse_contained=False,
    )

    assert len(scene["objects"]) == 3


def test_probe_validation_rejects_an_unknown_non_physical_hint() -> None:
    entry = _probe_object("cube_0000", translation=(0.0, 0.5, 0.0))
    entry["non_physical_hint"] = "transparent"

    with pytest.raises(GameSceneExportError, match="non_physical_hint"):
        validate_probe_payload(_probe([entry]))


def test_export_fails_when_every_mesh_is_non_physical(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object(
                "shell_0000", translation=(0.0, 0.5, 0.0), non_physical_hint="back_side_only"
            )
        ]
    )

    with pytest.raises(GameSceneExportError, match="non-physical"):
        _export(payload, tmp_path)


def test_interpenetrating_game_objects_are_not_certified_separated(tmp_path: Path) -> None:
    payload = _probe(
        [
            _probe_object("cube_0000", translation=(0.0, 0.5, 0.0)),
            _probe_object("cube_0001", translation=(0.2, 0.5, 0.0)),
        ]
    )
    scene, manifest = _export(payload, tmp_path)

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "out",
        eval_generic_validity=True,
        collision_geometry=manifest,
    )

    collision = report["reports"]["generic_validity"]["metrics"]["collision"]
    pair = collision["pairs"][0]
    assert pair["route"] != "direct_valid_obb_separated"
    assert pair["obb_evidence"]["obb_certifiably_separated"] is False


def test_legacy_game_collision_report_excludes_new_scoring_audit(
    tmp_path: Path,
) -> None:
    payload = _probe(
        [
            _probe_object("cube_0000", translation=(0.0, 0.5, 0.0)),
            _probe_object("cube_0001", translation=(0.2, 0.5, 0.0)),
        ]
    )
    scene, manifest = _export(payload, tmp_path)
    profile = load_yaml(
        Path("configs/evaluation/metric_profile_game_v1.yaml"),
        default={},
    )

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "legacy_game_out",
        eval_generic_validity=True,
        collision_geometry=manifest,
        evaluation_profile=profile,
    )

    validity = report["reports"]["generic_validity"]
    assert "scoring" not in validity
    assert all(
        "scoring_geometry" not in pair
        for pair in validity["metrics"]["collision"]["pairs"]
    )
