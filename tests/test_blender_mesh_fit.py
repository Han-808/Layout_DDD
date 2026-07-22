from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

import benchmark.rendering.blender_worker as worker
from benchmark.rendering.blender_worker import (
    FIT_MODE_BBOX_PROXY,
    FIT_MODE_UNIFORM_CONTAIN,
    VERTICAL_ANCHOR_BOTTOM,
    VERTICAL_ANCHOR_TOP,
    _anchored_root_placement,
    _build_object,
    _complexity_limit_error,
    _matvec3,
    _mesh_complexity,
    _root_location,
    _uniform_contain_fit,
    _vertical_anchor_spec,
)


def test_progress_record_allows_path_evidence(tmp_path) -> None:
    progress_path = tmp_path / "progress.jsonl"

    worker._record_progress(
        progress_path,
        "blend_saved",
        path="/tmp/scene.blend",
    )

    record = json.loads(progress_path.read_text(encoding="utf-8"))
    assert record == {"stage": "blend_saved", "path": "/tmp/scene.blend"}


def test_collision_geometry_export_reports_partial_manifest(tmp_path) -> None:
    manifest = tmp_path / "collision_geometry_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "export_summary": {
                    "complete_mesh_count": 2,
                    "incomplete_mesh_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    result = worker._collision_geometry_export_result(manifest, limits={})

    assert result["status"] == "partial"


def test_cycles_render_configuration_enables_persistent_data(monkeypatch) -> None:
    scene = SimpleNamespace(
        render=SimpleNamespace(
            engine=None,
            resolution_x=0,
            resolution_y=0,
            resolution_percentage=0,
            image_settings=SimpleNamespace(file_format=None),
            film_transparent=None,
            use_persistent_data=False,
        ),
        world=SimpleNamespace(color=None),
        cycles=SimpleNamespace(samples=0, use_denoising=False),
    )
    monkeypatch.setattr(worker, "bpy", SimpleNamespace(context=SimpleNamespace(scene=scene)))
    monkeypatch.setattr(
        worker,
        "_configure_cycles_device",
        lambda requested: {
            "cycles_device_requested": requested,
            "cycles_device_active": requested,
            "cycles_devices_enabled": [],
            "cycles_device_errors": [],
        },
    )

    config = worker._configure_render(
        512,
        384,
        "CYCLES",
        cycles_device="CUDA",
        cycles_samples=8,
        cycles_denoising=True,
    )

    assert scene.render.use_persistent_data is True
    assert config["persistent_data"] is True


def _rot_z(degrees: float) -> list[list[float]]:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


# --------------------------------------------------------------------------- #
# Uniform scaling
# --------------------------------------------------------------------------- #
def test_uniform_scale_is_min_of_fit_ratios() -> None:
    fit = _uniform_contain_fit([2.0, 1.0, 4.0], [4.0, 4.0, 4.0])
    # ratios = [2.0, 4.0, 1.0]; the mesh must fit inside the tightest axis.
    assert fit["uniform_scale"] == pytest.approx(1.0)
    assert fit["fit_mode"] == FIT_MODE_UNIFORM_CONTAIN


def test_all_three_scale_components_are_equal() -> None:
    fit = _uniform_contain_fit([3.0, 0.5, 2.0], [1.5, 1.5, 1.5])
    uniform_scale = fit["uniform_scale"]
    scale_vector = (uniform_scale, uniform_scale, uniform_scale)
    assert scale_vector[0] == scale_vector[1] == scale_vector[2]
    # rendered_size must equal source_size scaled by the single uniform factor.
    for axis in range(3):
        assert fit["rendered_size"][axis] == pytest.approx(fit["source_size"][axis] * uniform_scale)


def test_rendered_size_preserves_source_aspect_ratio() -> None:
    source = [2.0, 1.0, 4.0]
    fit = _uniform_contain_fit(source, [10.0, 3.0, 20.0])
    rendered = fit["rendered_size"]
    # Pairwise ratios between axes are identical to the source's ratios.
    assert rendered[0] / rendered[1] == pytest.approx(source[0] / source[1])
    assert rendered[0] / rendered[2] == pytest.approx(source[0] / source[2])
    assert rendered[1] / rendered[2] == pytest.approx(source[1] / source[2])


def test_rendered_size_never_exceeds_target_on_any_axis() -> None:
    cases = [
        ([2.0, 1.0, 4.0], [4.0, 4.0, 4.0]),
        ([3.0, 0.5, 2.0], [1.5, 1.5, 1.5]),
        ([0.2, 5.0, 0.7], [1.0, 2.0, 3.0]),
        ([10.0, 10.0, 10.0], [1.0, 2.0, 0.5]),
    ]
    for source, target in cases:
        fit = _uniform_contain_fit(source, target)
        for axis in range(3):
            assert fit["rendered_size"][axis] <= target[axis] + 1.0e-9
        # At least one axis is flush against the target (tightest constraint).
        assert any(
            fit["rendered_size"][axis] == pytest.approx(target[axis]) for axis in range(3)
        )


# --------------------------------------------------------------------------- #
# Validation / proxy fallback trigger
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source, target",
    [
        ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, -2.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
        ([float("inf"), 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, 1.0, float("nan")], [1.0, 1.0, 1.0]),
    ],
)
def test_invalid_dimensions_raise_value_error(source, target) -> None:
    with pytest.raises(ValueError):
        _uniform_contain_fit(source, target)


def test_blender_vector_like_values_are_valid_vec3_inputs() -> None:
    class VectorLike:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    fit = _uniform_contain_fit(VectorLike([2.0, 1.0, 4.0]), VectorLike([4.0, 4.0, 4.0]))

    assert fit["source_size"] == [2.0, 1.0, 4.0]
    assert fit["target_size"] == [4.0, 4.0, 4.0]


# --------------------------------------------------------------------------- #
# Centering
# --------------------------------------------------------------------------- #
def test_centering_maps_source_center_onto_target_center_identity_rotation() -> None:
    source_center = [1.5, -2.0, 0.75]
    target_center = [4.0, 5.0, 1.0]
    uniform_scale = 0.5
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    location = _root_location(source_center, uniform_scale, identity, target_center)
    scaled_center = [uniform_scale * value for value in source_center]
    world_center = [location[axis] + _matvec3(identity, scaled_center)[axis] for axis in range(3)]

    assert world_center == pytest.approx(target_center)


def test_centering_holds_under_rotation() -> None:
    source_center = [0.8, 0.3, 1.2]
    target_center = [2.0, -1.0, 0.5]
    uniform_scale = 1.75
    rotation = _rot_z(37.0)

    location = _root_location(source_center, uniform_scale, rotation, target_center)
    scaled_center = [uniform_scale * value for value in source_center]
    rotated = _matvec3(rotation, scaled_center)
    world_center = [location[axis] + rotated[axis] for axis in range(3)]

    assert world_center == pytest.approx(target_center)


def test_bottom_anchor_places_contain_fit_on_canonical_local_bottom() -> None:
    placement = _anchored_root_placement(
        source_center=[0.0, 0.0, 1.0],
        source_size=[2.0, 2.0, 2.0],
        uniform_scale=1.0,
        rotation_matrix=_rot_z(37.0),
        target_center=[4.0, 5.0, 2.0],
        target_size=[4.0, 4.0, 4.0],
        vertical_anchor=VERTICAL_ANCHOR_BOTTOM,
    )

    assert placement["vertical_anchor"] == VERTICAL_ANCHOR_BOTTOM
    assert placement["local_anchor_offset"] == pytest.approx([0.0, 0.0, -1.0])
    assert placement["rendered_bounds_center"] == pytest.approx([4.0, 5.0, 1.0])


def test_top_anchor_places_explicit_ceiling_attachment_on_canonical_local_top() -> None:
    placement = _anchored_root_placement(
        source_center=[0.0, 0.0, 1.0],
        source_size=[2.0, 2.0, 2.0],
        uniform_scale=1.0,
        rotation_matrix=_rot_z(0.0),
        target_center=[4.0, 5.0, 2.0],
        target_size=[4.0, 4.0, 4.0],
        vertical_anchor=VERTICAL_ANCHOR_TOP,
    )

    assert placement["vertical_anchor"] == VERTICAL_ANCHOR_TOP
    assert placement["local_anchor_offset"] == pytest.approx([0.0, 0.0, 1.0])
    assert placement["rendered_bounds_center"] == pytest.approx([4.0, 5.0, 3.0])


def test_vertical_anchor_uses_only_explicit_ceiling_attachment() -> None:
    item = {"id": "lamp", "category": "ceiling_lamp", "description": "ceiling lamp"}
    assert _vertical_anchor_spec(item, {}) == {
        "vertical_anchor": VERTICAL_ANCHOR_BOTTOM,
        "source": "default_bottom",
    }

    scene = {
        "oar_relations": [
            {
                "family": "oar",
                "subject_id": "lamp",
                "type": "hung_from_ceiling",
                "architectural_element": "ceiling",
            }
        ]
    }
    assert _vertical_anchor_spec(item, scene) == {
        "vertical_anchor": VERTICAL_ANCHOR_TOP,
        "source": "scene.oar_relations",
    }


# --------------------------------------------------------------------------- #
# Manifest entries via _build_object (bpy-dependent helpers monkeypatched)
# --------------------------------------------------------------------------- #
def test_build_object_asset_entry_reports_fit_metrics(monkeypatch) -> None:
    captured = {}

    def fake_resolve(item, asset_root):
        return "/assets/bed/bed.fbx"

    def fake_import(path, object_id, target_center, target_size, rotation_degrees, *, vertical_anchor):
        captured["import"] = (path, object_id, list(target_center), list(target_size))
        return {
            **_uniform_contain_fit([2.0, 1.0, 4.0], list(target_size)),
            "vertical_anchor": vertical_anchor,
        }

    monkeypatch.setattr(worker, "_resolve_mesh_path", fake_resolve)
    monkeypatch.setattr(worker, "_import_and_place", fake_import)

    item = {"id": "bed_1", "category": "bed", "center": [1.0, 2.0, 0.3], "size": [4.0, 4.0, 4.0], "rotation": [0, 0, 0]}
    entry = _build_object(item, asset_root=None)

    assert entry["representation"] == "asset_mesh"
    assert entry["fit_mode"] == FIT_MODE_UNIFORM_CONTAIN
    assert entry["source_size"] == [2.0, 1.0, 4.0]
    assert entry["target_size"] == [4.0, 4.0, 4.0]
    assert entry["uniform_scale"] == pytest.approx(1.0)
    assert entry["rendered_size"] == [2.0, 1.0, 4.0]
    assert entry["canonical_center"] == [1.0, 2.0, 0.3]
    assert entry["canonical_size"] == [4.0, 4.0, 4.0]
    assert entry["canonical_rotation_degrees"] == [0.0, 0.0, 0.0]
    assert entry["vertical_anchor"] == VERTICAL_ANCHOR_BOTTOM
    assert entry["vertical_anchor_source"] == "default_bottom"
    assert entry["warning"] is None


def test_build_object_proxy_entry_reports_bbox_proxy(monkeypatch) -> None:
    proxy_calls = []

    monkeypatch.setattr(worker, "_resolve_mesh_path", lambda item, asset_root: None)
    monkeypatch.setattr(worker, "_add_proxy", lambda *args, **kwargs: proxy_calls.append(args))

    item = {"id": "vase_1", "category": "vase", "center": [1.0, 1.0, 0.5], "size": [0.3, 0.3, 0.5], "rotation": [0, 0, 0]}
    entry = _build_object(item, asset_root=None)

    assert entry["representation"] == "bbox_proxy"
    assert entry["fit_mode"] == FIT_MODE_BBOX_PROXY
    assert entry["mesh_path"] is None
    assert entry["canonical_center"] == [1.0, 1.0, 0.5]
    assert entry["vertical_anchor"] == VERTICAL_ANCHOR_BOTTOM
    assert "no loadable mesh reference" in entry["warning"]
    # Proxy geometry is still built from the canonical bbox (behavior unchanged).
    assert len(proxy_calls) == 1
    assert proxy_calls[0][2] == [1.0, 1.0, 0.5]
    assert proxy_calls[0][3] == [0.3, 0.3, 0.5]


def test_build_object_falls_back_to_proxy_when_import_fails(monkeypatch) -> None:
    proxy_calls = []

    def failing_import(*args, **kwargs):
        raise RuntimeError("import produced no mesh objects")

    monkeypatch.setattr(worker, "_resolve_mesh_path", lambda item, asset_root: "/assets/x/x.fbx")
    monkeypatch.setattr(worker, "_import_and_place", failing_import)
    monkeypatch.setattr(worker, "_add_proxy", lambda *args, **kwargs: proxy_calls.append(args))

    item = {"id": "lamp_1", "category": "lamp", "center": [2.0, 2.0, 0.6], "size": [0.4, 0.4, 1.2], "rotation": [0, 0, 0]}
    entry = _build_object(item, asset_root=None)

    assert entry["representation"] == "bbox_proxy"
    assert entry["fit_mode"] == FIT_MODE_BBOX_PROXY
    assert "asset import failed; rendered proxy instead" in entry["warning"]
    assert len(proxy_calls) == 1


def test_mesh_complexity_counts_triangulated_polygons() -> None:
    mesh = SimpleNamespace(
        vertices=[object()] * 7,
        polygons=[
            SimpleNamespace(vertices=[0, 1, 2]),
            SimpleNamespace(vertices=[0, 1, 2, 3]),
        ],
    )
    child = SimpleNamespace(type="MESH", data=mesh)
    root = SimpleNamespace(type="EMPTY", data=None, children_recursive=[child])

    assert _mesh_complexity(root) == (7, 3)


def test_collision_geometry_complexity_budget_skips_only_optional_mesh_evidence() -> None:
    error = _complexity_limit_error(
        vertex_count=60,
        face_count=120,
        exported_vertices=100,
        exported_faces=200,
        max_vertices_per_object=50,
        max_faces_per_object=100,
        max_total_vertices=200,
        max_total_faces=400,
    )

    assert error is not None
    assert "vertices_per_object=60>50" in error
    assert "faces_per_object=120>100" in error
    assert _complexity_limit_error(
        vertex_count=50,
        face_count=100,
        exported_vertices=100,
        exported_faces=200,
        max_vertices_per_object=50,
        max_faces_per_object=100,
        max_total_vertices=200,
        max_total_faces=400,
    ) is None


def test_collision_geometry_frame_validation_detects_stale_origin_transform() -> None:
    result = worker._exported_bounds_frame_validation(
        {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 1.0]},
        canonical_center=[4.0, 3.0, 0.5],
        center_tolerance_m=0.05,
    )

    assert result["canonical_consistent"] is False
    assert result["world_bounds_center"] == [0.0, 0.0, 0.5]
    assert result["world_bounds_center_offset_m"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("chair.fbx", "asset_fbx"),
        ("chair.glb", "asset_glb"),
        ("chair.gltf", "asset_gltf"),
        ("chair.obj", "asset_obj"),
        (None, "asset_mesh"),
    ],
)
def test_collision_geometry_source_tracks_imported_asset_format(path, expected) -> None:
    assert worker._asset_geometry_source(path) == expected


def test_collision_geometry_frame_validation_accepts_centered_contain_fit() -> None:
    result = worker._exported_bounds_frame_validation(
        {"min": [3.5, 2.8, 0.2], "max": [4.5, 3.2, 0.8]},
        canonical_center=[4.0, 3.0, 0.5],
        center_tolerance_m=0.05,
    )

    assert result["canonical_consistent"] is True
    assert result["failure_reasons"] == []


def test_collision_geometry_frame_validation_accepts_expected_anchored_center() -> None:
    result = worker._exported_bounds_frame_validation(
        {"min": [3.5, 2.8, 0.0], "max": [4.5, 3.2, 0.6]},
        canonical_center=[4.0, 3.0, 0.5],
        expected_bounds_center=[4.0, 3.0, 0.3],
        center_tolerance_m=0.05,
    )

    assert result["canonical_consistent"] is True
    assert result["world_bounds_center_offset_m"] == pytest.approx(0.0)
    assert result["canonical_center_offset_m"] == pytest.approx(0.2)


def test_collision_geometry_export_refreshes_blender_scene_graph(monkeypatch) -> None:
    calls = []
    fake_bpy = SimpleNamespace(
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: calls.append("updated")))
    )
    monkeypatch.setattr(worker, "bpy", fake_bpy)

    worker._refresh_scene_graph()

    assert calls == ["updated"]
