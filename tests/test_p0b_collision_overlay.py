from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.rendering import BlenderRenderError, BlenderRenderer
from benchmark.rendering.collision_overlay import (
    COLLISION_EVIDENCE_PACKET_VERSION,
    COLLISION_OVERLAY_COLORS,
    COLLISION_VISIBILITY_SELECTOR_VERSION,
    JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED,
    build_candidate_mask_stats,
    build_collision_overlay_spec,
    measure_overlay_visibility,
    measure_target_mask_png,
    rank_collision_candidates,
    rank_collision_candidates_v2,
    resolve_canonical_object_id,
)
from benchmark.visual_judge import CameraEvidenceProvider, OpenAICompatibleVLMJudge
from benchmark.visual_judge.evidence_sufficiency import (
    SUFFICIENT,
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.p0b import (
    COLLISION_CANDIDATE_SELECTION_POLICY,
    P0B_METRIC_RUBRICS,
    adjudicate_p0b_event,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _scene(objects: list[dict]) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "collision_overlay_scene",
        "scene_type": "room",
        "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": objects,
    }


def _obj(object_id: str, center: list[float], size: list[float], *, category: str = "box", rotation=None) -> dict:
    return {
        "id": object_id,
        "category": category,
        "description": f"{category} {object_id}",
        "center": center,
        "size": size,
        "rotation": rotation or [0.0, 0.0, 0.0],
        "metadata": {"interactive": False},
    }


def _overlap_scene() -> dict:
    return _scene(
        [
            _obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0], category="table"),
            _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0], category="book"),
        ]
    )


def _camera_request(metric: str = "collision") -> dict:
    return {
        "metric": metric,
        "event": {"object_a": "a", "object_b": "b"},
        "scene": _overlap_scene(),
        "object_ids": ["a", "b"],
        "detector_evidence": {
            "mesh": {
                "closest_points": {"object_a": [1.4, 1.0, 0.5], "object_b": [1.45, 1.0, 0.5]},
                "focus_region": {"center": [1.42, 1.0, 0.5], "radius_m": 0.3, "source": "closest_points"},
            }
        },
        "natural_language_prompt": "Place a book on the table.",
        "extracted_relationships": [{"subject": "book", "predicate": "on", "object": "table"}],
    }


def _write_nonuniform_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), (210, 210, 210))
    image.putpixel((0, 0), (10, 20, 30))
    image.save(path)


def _color_bytes(color: list[float]) -> tuple[int, int, int]:
    return tuple(int(round(component * 255.0)) for component in color)


def _write_overlay_png(path: Path, *, both: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 8), _color_bytes(COLLISION_OVERLAY_COLORS["context"]))
    color_a = _color_bytes(COLLISION_OVERLAY_COLORS["object_a"])
    color_b = _color_bytes(COLLISION_OVERLAY_COLORS["object_b"])
    for x in range(3):
        image.putpixel((x, 0), color_a)
    if both:
        for x in range(3):
            image.putpixel((x, 7), color_b)
    image.save(path)


class _FakeRenderer:
    """Fake renderer supporting both the RGB and overlay camera passes."""

    def __init__(self, *, overlay_both: bool = True) -> None:
        self.calls: list[dict] = []
        self.overlay_both = overlay_both

    def render_camera_views(self, *, blend_file, out_dir, camera_views, preview=False):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"pass": "rgb", "preview": preview, "ids": [view["id"] for view in camera_views]})
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"rgb_{index:02d}.png"
            _write_nonuniform_png(path)
            views.append({"id": pose["id"], "path": str(path), "pose": pose})
        return {"views": views}

    def render_collision_overlay_views(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=False):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"pass": "overlay", "preview": preview, "ids": [view["id"] for view in camera_views]})
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"overlay_{index:02d}.png"
            _write_overlay_png(path, both=self.overlay_both)
            views.append({"id": pose["id"], "path": str(path), "role": "collision_pair_overlay", "pose": pose})
        return {"views": views}


class _FakeContourRenderer(_FakeRenderer):
    def render_focus_overlay_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        overlay_spec,
        preview=False,
        allow_blank_views=False,
    ):
        return self.render_collision_overlay_views(
            blend_file=blend_file,
            out_dir=out_dir,
            camera_views=camera_views,
            overlay_spec=overlay_spec,
            preview=preview,
        )

    def render_target_id_masks(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        overlay_spec,
        preview=True,
        respect_occlusion=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        target_ids = [
            str(target["id"])
            for target in overlay_spec.get("targets", [])
            if isinstance(target, dict) and target.get("id") is not None
        ]
        views = []
        for view_index, pose in enumerate(camera_views):
            targets = {}
            for target_index, target_id in enumerate(target_ids):
                path = destination / f"mask_{view_index}_{target_index}.png"
                image = Image.new("L", (8, 8), 0)
                image.putpixel((1 + target_index * 3, 2), 255)
                image.save(path)
                targets[target_id] = {"mask_path": str(path)}
            views.append({"id": pose["id"], "status": "ok", "targets": targets})
        return {"views": views}


# --------------------------------------------------------------------------- #
# 1-2. Rubric: routing carries no label prior; OBB overlap alone insufficient
# --------------------------------------------------------------------------- #
def test_collision_rubric_conveys_no_label_prior_and_obb_insufficient() -> None:
    rubric = P0B_METRIC_RUBRICS["collision"]
    lowered = rubric.lower()
    assert "high-recall" in lowered
    assert "no verdict prior" in lowered
    assert "obb overlap alone is insufficient" in lowered
    assert "actual unintended physical surface interpenetration" in lowered
    # No deterministic invalid prior: the rubric never instructs invalid on overlap.
    assert "return invalid only when" in lowered


def test_collision_request_carries_candidate_selection_policy() -> None:
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "valid", "confidence": 0.5, "reason": "no penetration shown"}

    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "a", "object_b": "b"},
        prompt="Place a book on the table.",
        relationships=[],
        scene=_overlap_scene(),
        detector_evidence={"obb": {"intersects": True}},
        judge=judge,
        object_ids=["a", "b"],
    )
    assert captured[0]["candidate_selection_policy"] == COLLISION_CANDIDATE_SELECTION_POLICY

    captured.clear()
    adjudicate_p0b_event(
        metric="oob",
        event={"object_id": "a", "plane_flags": {"west_oob": True}},
        prompt="",
        relationships=[],
        scene=_overlap_scene(),
        detector_evidence={"plane_flags": {"west_oob": True}},
        judge=judge,
        object_ids=["a"],
    )
    assert "candidate_selection_policy" not in captured[0]


def test_collision_evaluator_forwards_no_prior_policy_to_judge() -> None:
    captured: list[dict] = []

    class _Judge:
        def adjudicate_p0b(self, request: dict) -> dict:
            captured.append(request)
            return {"verdict": "valid", "confidence": 0.6, "reason": "contact only"}

    report = check_collision(_overlap_scene(), vlm_judge=_Judge())
    assert report["pairs"][0]["route"] == "vlm_adjudicated"
    assert captured[0]["candidate_selection_policy"] == COLLISION_CANDIDATE_SELECTION_POLICY
    assert captured[0]["detector_evidence"]["candidate_selection_policy"] == COLLISION_CANDIDATE_SELECTION_POLICY


def test_openai_p0b_context_includes_candidate_selection_policy(tmp_path: Path) -> None:
    image = tmp_path / "view.png"
    _write_nonuniform_png(image)

    class _FakeModel:
        model_id = "judge"
        endpoint = "http://127.0.0.1:8298/v1"
        last_request_metadata = {"image_count": 1}

        def __init__(self) -> None:
            self.messages: list = []

        def chat_messages(self, messages, **kwargs):
            self.messages = messages
            return json.dumps({"verdict": "valid", "confidence": 0.7, "reason": "ok"})

    model = _FakeModel()
    OpenAICompatibleVLMJudge(model).adjudicate_p0b(
        {
            "metric": "collision",
            "candidate_selection_policy": COLLISION_CANDIDATE_SELECTION_POLICY,
            "metric_rubric": P0B_METRIC_RUBRICS["collision"],
            "event": {"object_a": "a", "object_b": "b"},
            "detector_evidence": {"obb": {"intersects": True}},
            "render_evidence": [str(image)],
        }
    )
    content = model.messages[1]["content"][0]["text"]
    assert COLLISION_CANDIDATE_SELECTION_POLICY in content


# --------------------------------------------------------------------------- #
# 3-4. Mesh routing invariants remain (no regression in the frozen policy)
# --------------------------------------------------------------------------- #
def test_reliable_mesh_separation_stays_direct_valid(tmp_path: Path) -> None:
    from benchmark.evaluator.generic_validity.mesh_geometry import (
        TRIANGLE_MESH_REPRESENTATION,
        write_ascii_triangle_ply,
    )

    def _box(path: Path, center, size) -> None:
        half = [value / 2.0 for value in size]
        cx, cy, cz = center
        vertices = [
            [cx - half[0], cy - half[1], cz - half[2]], [cx + half[0], cy - half[1], cz - half[2]],
            [cx + half[0], cy + half[1], cz - half[2]], [cx - half[0], cy + half[1], cz - half[2]],
            [cx - half[0], cy - half[1], cz + half[2]], [cx + half[0], cy - half[1], cz + half[2]],
            [cx + half[0], cy + half[1], cz + half[2]], [cx - half[0], cy + half[1], cz + half[2]],
        ]
        faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
        write_ascii_triangle_ply(path, vertices, faces)

    _box(tmp_path / "a.ply", [0.8, 1.0, 0.5], [0.3, 0.3, 0.3])
    _box(tmp_path / "b.ply", [2.2, 1.0, 0.5], [0.3, 0.3, 0.3])
    geometry = {
        "schema_version": "collision_geometry_v1",
        "units": "meter",
        "up_axis": "z",
        "manifest_path": str((tmp_path / "collision_geometry_manifest.json").resolve()),
        "objects": {
            "a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(tmp_path / "a.ply"), "transform_baked": True, "geometry_source": "test", "complete": True},
            "b": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(tmp_path / "b.ply"), "transform_baked": True, "geometry_source": "test", "complete": True},
        },
    }

    class _Judge:
        calls = 0

        def adjudicate_p0b(self, request: dict) -> dict:
            type(self).calls += 1
            return {"verdict": "invalid", "confidence": 0.9, "reason": "x"}

    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [2.0, 2.0, 2.0]), _obj("b", [1.5, 1.0, 0.5], [2.0, 2.0, 2.0])])
    report = check_collision(scene, {"separation_threshold_m": 0.02}, collision_geometry=geometry, vlm_judge=_Judge())
    assert report["pairs"][0]["route"] == "direct_valid_mesh_separated"


def test_mixed_mesh_proxy_pair_still_routes_to_vlm(tmp_path: Path) -> None:
    from benchmark.evaluator.generic_validity.mesh_geometry import (
        TRIANGLE_MESH_REPRESENTATION,
        write_ascii_triangle_ply,
    )

    vertices = [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0], [0.5, 1.5, 0.0], [0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [1.5, 1.5, 1.0], [0.5, 1.5, 1.0]]
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    write_ascii_triangle_ply(tmp_path / "a.ply", vertices, faces)
    geometry = {
        "schema_version": "collision_geometry_v1",
        "units": "meter",
        "up_axis": "z",
        "manifest_path": str((tmp_path / "collision_geometry_manifest.json").resolve()),
        "objects": {"a": {"representation": TRIANGLE_MESH_REPRESENTATION, "geometry_path": str(tmp_path / "a.ply"), "transform_baked": True, "geometry_source": "test", "complete": True}},
    }

    calls: list[int] = []

    class _Judge:
        def adjudicate_p0b(self, request: dict) -> dict:
            calls.append(1)
            return {"verdict": "valid", "confidence": 0.5, "reason": "x"}

    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])
    report = check_collision(scene, collision_geometry=geometry, vlm_judge=_Judge())
    assert report["pairs"][0]["evidence_level"] == "obb"
    assert report["pairs"][0]["route"] == "vlm_adjudicated"
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# 5. Existing path-only local-view providers remain compatible
# --------------------------------------------------------------------------- #
def test_path_only_local_view_provider_remains_compatible(tmp_path: Path) -> None:
    view = tmp_path / "local.png"
    view.write_bytes(b"png")
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "valid", "confidence": 0.5, "reason": "ok"}

    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "a", "object_b": "b"},
        prompt="",
        relationships=[],
        scene=_overlap_scene(),
        detector_evidence={},
        judge=judge,
        object_ids=["a", "b"],
        local_view_provider=lambda request: [str(view)],
    )
    request = captured[0]
    assert request["local_render_evidence"] == [str(view)]
    assert "local_render_evidence_metadata" not in request


# --------------------------------------------------------------------------- #
# 6. Richer evidence metadata forwarded without breaking path lists
# --------------------------------------------------------------------------- #
def test_rich_local_view_metadata_forwarded_beside_path_list(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    overlay = tmp_path / "overlay.png"
    rgb.write_bytes(b"png")
    overlay.write_bytes(b"png")
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "invalid", "confidence": 0.8, "reason": "penetration visible"}

    def provider(request: dict) -> list[dict]:
        return [
            {"path": str(rgb), "role": "collision_rgb", "view_id": "v0", "object_a_id": "a", "object_b_id": "b"},
            {"path": str(overlay), "role": "collision_pair_overlay", "view_id": "v0", "representation_level": "mesh"},
        ]

    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "a", "object_b": "b"},
        prompt="",
        relationships=[],
        scene=_overlap_scene(),
        detector_evidence={},
        judge=judge,
        object_ids=["a", "b"],
        local_view_provider=provider,
    )
    request = captured[0]
    assert request["local_render_evidence"] == [str(rgb), str(overlay)]
    roles = [item["role"] for item in request["local_render_evidence_metadata"]]
    assert roles == ["collision_rgb", "collision_pair_overlay"]
    assert request["render_evidence"][0] == str(rgb)


def test_missing_path_in_rich_local_view_item_raises() -> None:
    def provider(request: dict) -> list[dict]:
        return [{"role": "collision_rgb"}]

    with pytest.raises(ValueError, match="must include a 'path'"):
        adjudicate_p0b_event(
            metric="collision",
            event={"object_a": "a", "object_b": "b"},
            prompt="",
            relationships=[],
            scene=_overlap_scene(),
            detector_evidence={},
            judge=lambda request: {"verdict": "valid", "confidence": 0.5, "reason": "x"},
            object_ids=["a", "b"],
            local_view_provider=provider,
        )


# --------------------------------------------------------------------------- #
# 7. RGB and overlay files are paired deterministically in manifests
# --------------------------------------------------------------------------- #
def test_collision_overlay_pairs_rgb_and_overlay_in_manifest(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer(overlay_both=True)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=2,
        collision_overlay=True,
    )

    items = provider(_camera_request())

    assert [item["role"] for item in items] == [
        "collision_rgb",
        "collision_pair_overlay",
        "collision_rgb",
        "collision_pair_overlay",
    ]
    assert items[0]["view_id"] == items[1]["view_id"]
    assert items[2]["view_id"] == items[3]["view_id"]
    assert items[0]["object_a_id"] == "a" and items[0]["object_b_id"] == "b"
    for item in items:
        assert Path(item["path"]).is_file()

    manifest = json.loads((tmp_path / "evidence" / next(iter((tmp_path / "evidence").iterdir())).name / "camera_evidence_manifest.json").read_text())
    assert len(manifest["pairs"]) == 2
    for pair in manifest["pairs"]:
        assert Path(pair["collision_rgb"]).is_file()
        assert Path(pair["collision_pair_overlay"]).is_file()
    assert manifest["object_a_id"] == "a"
    assert manifest["representation_level"] == "bbox_proxy"
    # Frozen bbox_track does not run visibility ranking; it renders only the
    # selected RGB and same-pose overlay.
    passes = [call["pass"] for call in renderer.calls]
    assert passes.count("overlay") == 1
    assert passes.count("rgb") == 1  # final RGB only


def test_collision_contour_pairs_same_pose_raw_and_contour(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeContourRenderer(overlay_both=True)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=1,
        collision_overlay=True,
        collision_contour=True,
    )

    items = provider(_camera_request())

    assert [item["role"] for item in items] == [
        "collision_rgb",
        "metric_local_contour",
    ]
    assert items[0]["view_id"] == items[1]["view_id"]
    assert Path(items[1]["path"]).is_file()
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads(
        (event_dir / "camera_evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pairs"][0]["contour_available"] is True
    assert manifest["contour_manifest"].endswith("segmentation_contour_manifest.json")


def test_collision_contour_measures_visible_contact_focus_in_final_image(
    tmp_path: Path,
) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _VisibleFocusContourRenderer(_FakeContourRenderer):
        def render_focus_overlay_views(
            self,
            *,
            blend_file,
            out_dir,
            camera_views,
            overlay_spec,
            preview=False,
            allow_blank_views=False,
        ):
            manifest = super().render_focus_overlay_views(
                blend_file=blend_file,
                out_dir=out_dir,
                camera_views=camera_views,
                overlay_spec=overlay_spec,
                preview=preview,
                allow_blank_views=allow_blank_views,
            )
            marker = _color_bytes(COLLISION_OVERLAY_COLORS["marker"])
            context = _color_bytes(COLLISION_OVERLAY_COLORS["context"])
            for view in manifest["views"]:
                image = Image.new("RGB", (64, 64), context)
                for y in range(58, 61):
                    for x in range(58, 61):
                        image.putpixel((x, y), marker)
                image.save(view["path"])
            return manifest

    renderer = _VisibleFocusContourRenderer(overlay_both=True)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=1,
        collision_overlay=True,
        collision_contour=True,
    )

    items = provider(_camera_request())

    raw = next(item for item in items if item["role"] == "collision_rgb")
    visibility = raw["visibility"]
    assert visibility["focus_measurement_status"] == "measured"
    assert visibility["focus_in_frame"] is True
    assert visibility["focus_pixel_fraction"] > 0.0
    assessment = assess_visual_evidence_sufficiency(
        "collision",
        items,
        request=_camera_request(),
    )
    assert assessment["status"] == SUFFICIENT


def test_collision_overlay_forwards_paired_paths_to_judge(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer(overlay_both=True)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=1,
        collision_overlay=True,
    )
    captured: list[dict] = []

    def judge(request: dict) -> dict:
        captured.append(request)
        return {"verdict": "valid", "confidence": 0.5, "reason": "no penetration"}

    adjudicate_p0b_event(
        metric="collision",
        event={"object_a": "a", "object_b": "b"},
        prompt="Place a book on the table.",
        relationships=[],
        scene=_overlap_scene(),
        detector_evidence=_camera_request()["detector_evidence"],
        judge=judge,
        object_ids=["a", "b"],
        local_view_provider=provider,
    )
    request = captured[0]
    assert len(request["local_render_evidence"]) == 2
    roles = [item["role"] for item in request["local_render_evidence_metadata"]]
    assert roles == ["collision_rgb", "collision_pair_overlay"]


# --------------------------------------------------------------------------- #
# 8. Multi-child assets resolve to one canonical object id
# --------------------------------------------------------------------------- #
def test_multi_child_asset_resolves_to_single_canonical_id() -> None:
    child_a = [{"name": "Mesh", "canonical_id": "object_3"}, {"name": "asset_object_3", "canonical_id": "object_3"}]
    child_b = [{"name": "Mesh.001", "canonical_id": "object_3"}, {"name": "asset_object_3", "canonical_id": "object_3"}]
    assert resolve_canonical_object_id(child_a) == "object_3"
    assert resolve_canonical_object_id(child_b) == "object_3"

    # Older .blend without custom properties still resolves by root name.
    legacy = [{"name": "Cube.002", "canonical_id": None}, {"name": "asset_object_3", "canonical_id": None}]
    assert resolve_canonical_object_id(legacy) == "object_3"
    assert resolve_canonical_object_id([{"name": "proxy_obj_7", "canonical_id": None}]) == "obj_7"
    assert resolve_canonical_object_id([{"name": "benchmark_floor", "canonical_id": None}]) is None


# --------------------------------------------------------------------------- #
# 9. Proxy-only highlighting works
# --------------------------------------------------------------------------- #
def test_proxy_only_overlay_spec_highlights_both_targets() -> None:
    spec = build_collision_overlay_spec(
        scene=_overlap_scene(),
        object_a_id="a",
        object_b_id="b",
    )
    assert spec["representation_level"] == "bbox_proxy"
    assert spec["object_a"]["color"] == COLLISION_OVERLAY_COLORS["object_a"]
    assert spec["object_b"]["color"] == COLLISION_OVERLAY_COLORS["object_b"]
    assert len(spec["object_a"]["obb"]["corners"]) == 8
    assert len(spec["object_a"]["obb"]["edges"]) == 12
    assert spec["object_a"]["representation"] == "bbox_proxy"
    legend_roles = {entry["role"] for entry in spec["legend"]}
    assert legend_roles == {"object_a", "object_b"}


def test_overlay_spec_draws_closest_point_markers_and_connector() -> None:
    mesh_evidence = {
        "closest_points": {"object_a": [1.4, 1.0, 0.5], "object_b": [1.5, 1.0, 0.5]},
        "focus_region": {"center": [1.45, 1.0, 0.5], "radius_m": 0.3, "source": "closest_points"},
    }
    spec = build_collision_overlay_spec(scene=_overlap_scene(), object_a_id="a", object_b_id="b", mesh_evidence=mesh_evidence)
    marker_roles = {marker.get("role") for marker in spec["markers"] if marker["type"] == "closest_point"}
    assert marker_roles == {"object_a", "object_b"}
    assert spec["connectors"] and spec["connectors"][0]["from"] == [1.4, 1.0, 0.5]
    assert all(marker["color"] == COLLISION_OVERLAY_COLORS["marker"] for marker in spec["markers"])
    assert spec["focus"]["radius_m"] == 0.3


# --------------------------------------------------------------------------- #
# 10-11. Visibility ranking, deterministic tie-break, and fallback
# --------------------------------------------------------------------------- #
def test_ranking_requires_both_targets_visible() -> None:
    candidates = [{"id": "c0"}, {"id": "c1"}, {"id": "c2"}]
    visibility = {
        "c0": {"object_a_pixel_fraction": 0.30, "object_b_pixel_fraction": 0.0},
        "c1": {"object_a_pixel_fraction": 0.05, "object_b_pixel_fraction": 0.05},
        "c2": {"object_a_pixel_fraction": 0.40, "object_b_pixel_fraction": 0.0},
    }
    selected, log = rank_collision_candidates(candidates, visibility, max_views=1)
    assert [item["id"] for item in selected] == ["c1"]
    assert log["fallback_reason"] is None


def test_ranking_tie_break_and_pose_order_fallback() -> None:
    candidates = [{"id": "c0"}, {"id": "c1"}, {"id": "c2"}]
    tie = {
        "c0": {"object_a_pixel_fraction": 0.2, "object_b_pixel_fraction": 0.2},
        "c1": {"object_a_pixel_fraction": 0.0, "object_b_pixel_fraction": 0.0},
        "c2": {"object_a_pixel_fraction": 0.2, "object_b_pixel_fraction": 0.2},
    }
    selected, _ = rank_collision_candidates(candidates, tie, max_views=1)
    assert [item["id"] for item in selected] == ["c0"]  # deterministic tie-break by id

    none_visible = {cid: {"object_a_pixel_fraction": 0.0, "object_b_pixel_fraction": 0.0} for cid in ("c0", "c1", "c2")}
    selected_fallback, log = rank_collision_candidates(candidates, none_visible, max_views=2)
    assert [item["id"] for item in selected_fallback] == ["c0", "c1"]  # frozen pose order
    assert log["fallback_reason"] == "no_candidate_exposed_both_targets"


def test_measure_overlay_visibility_counts_target_colors(tmp_path: Path) -> None:
    path = tmp_path / "overlay.png"
    _write_overlay_png(path, both=True)
    stats = measure_overlay_visibility(path)
    assert stats["object_a_pixel_fraction"] > 0.0
    assert stats["object_b_pixel_fraction"] > 0.0
    assert stats["pixel_count"] == 64

    only_a = tmp_path / "overlay_a.png"
    _write_overlay_png(only_a, both=False)
    stats_a = measure_overlay_visibility(only_a)
    assert stats_a["object_a_pixel_fraction"] > 0.0
    assert stats_a["object_b_pixel_fraction"] == 0.0


def test_measure_overlay_visibility_survives_display_transform_and_shading(tmp_path: Path) -> None:
    path = tmp_path / "shaded.png"
    image = Image.new("RGB", (20, 10), (55, 55, 58))
    for x in range(5):
        for y in range(10):
            image.putpixel((x, y), (107, 33, 30))
    for x in range(5, 10):
        for y in range(10):
            image.putpixel((x, y), (30, 105, 116))
    image.save(path)

    stats = measure_overlay_visibility(path)

    assert stats["object_a_pixel_fraction"] == pytest.approx(0.25)
    assert stats["object_b_pixel_fraction"] == pytest.approx(0.25)


def test_measure_overlay_visibility_missing_image_is_degraded() -> None:
    stats = measure_overlay_visibility("/nonexistent/overlay.png")
    assert stats["measured"] is False
    assert stats["object_a_pixel_fraction"] == 0.0


def test_collision_overlay_falls_back_to_pose_order_when_previews_fail(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _PreviewFailingRenderer(_FakeRenderer):
        def render_collision_overlay_views(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=False):
            if preview:
                raise RuntimeError("overlay preview render failed")
            return super().render_collision_overlay_views(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _PreviewFailingRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=2,
        collision_overlay=True,
    )
    items = provider(_camera_request())
    assert items  # highlighting failure must not fail the metric
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert "overlay_visibility_ranking_failed" in str(manifest["overlay_degradation_reason"])


# --------------------------------------------------------------------------- #
# 12. Source .blend is not modified by the overlay pass
# --------------------------------------------------------------------------- #
def test_overlay_renderer_is_read_only_on_source_blend(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "overlay_00_collision.png"
        _write_nonuniform_png(view_path)
        (out_dir / "collision_overlay_manifest.json").write_text(
            json.dumps(
                {
                    "backend": "blender_read_only_collision_overlay_v1",
                    "source_scene_saved": False,
                    "views": [{"id": "collision_side", "name": "collision", "path": str(view_path), "role": "collision_pair_overlay"}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="overlay rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, width=768, height=768, render_engine="CYCLES")
    manifest = renderer.render_collision_overlay_views(
        blend_file=blend_file,
        out_dir=tmp_path / "overlay",
        camera_views=[{"id": "collision_side", "name": "collision side", "location": [1.0, 2.0, 1.5], "target": [2.0, 2.0, 0.5]}],
        overlay_spec={"schema_version": "collision_overlay_v1", "object_a": {"id": "a"}, "object_b": {"id": "b"}, "legend": []},
    )

    command = captured["command"]
    assert "--disable-autoexec" in command
    assert str(blend_file.resolve()) in command
    assert "blender_collision_overlay_worker.py" in command[command.index("--python") + 1]
    assert "--overlay-spec-json" in command
    assert command[command.index("--render-engine") + 1] == "CYCLES"
    assert command[command.index("--cycles-samples") + 1] == "16"
    assert manifest["camera_evidence"]["source_blend_modified"] is False
    assert manifest["camera_evidence"]["source_blend_sha256_before"] == manifest["camera_evidence"]["source_blend_sha256_after"]
    assert (tmp_path / "overlay" / "collision_overlay_spec.json").is_file()


def test_overlay_renderer_detects_modified_blend(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "overlay_00.png"
        _write_nonuniform_png(view_path)
        (out_dir / "collision_overlay_manifest.json").write_text(
            json.dumps({"views": [{"id": "v", "name": "v", "path": str(view_path)}]}),
            encoding="utf-8",
        )
        blend_file.write_bytes(b"blend-modified-by-worker")  # simulate a save-over
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, render_engine="BLENDER_WORKBENCH")
    with pytest.raises(BlenderRenderError, match="modified the source Blender scene"):
        renderer.render_collision_overlay_views(
            blend_file=blend_file,
            out_dir=tmp_path / "overlay",
            camera_views=[{"id": "v", "location": [1.0, 2.0, 1.5], "target": [2.0, 2.0, 0.5]}],
            overlay_spec={"object_a": {"id": "a"}, "object_b": {"id": "b"}},
        )


# --------------------------------------------------------------------------- #
# v2 mask-based identity ranking (repair of the visibility_ranked path)
# --------------------------------------------------------------------------- #
def _write_binary_mask(path: Path, *, white_pixels: int, size: int = 8, touch_border: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (size, size), 0)
    placed = 0
    start = 0 if touch_border else 1
    for y in range(start, size):
        for x in range(start, size):
            if placed >= white_pixels:
                break
            image.putpixel((x, y), 255)
            placed += 1
        if placed >= white_pixels:
            break
    image.save(path)


class _MaskRenderer(_FakeRenderer):
    """Fake renderer that also produces per-target identity masks.

    ``mask_plan`` maps candidate id -> {target_id: white_pixel_count | 'fail'}.
    A candidate id mapped to the string ``"blank"`` renders both targets black.
    ``rgb_fail_ids`` forces a per-pose RGB failure to exercise backfill.
    """

    def __init__(self, *, mask_plan: dict, rgb_fail_ids: set[str] | None = None, overlay_both: bool = True) -> None:
        super().__init__(overlay_both=overlay_both)
        self.mask_plan = mask_plan
        self.rgb_fail_ids = rgb_fail_ids or set()

    def render_camera_views(self, *, blend_file, out_dir, camera_views, preview=False):
        failing = [v for v in camera_views if str(v["id"]) in self.rgb_fail_ids]
        if failing and not preview:
            raise BlenderRenderError(f"blank camera evidence for views {[v['id'] for v in failing]}")
        return super().render_camera_views(blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, preview=preview)

    def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append({"pass": "masks", "preview": preview, "ids": [v["id"] for v in camera_views]})
        target_ids = [str(t["id"]) for t in overlay_spec.get("targets", [])]
        views = []
        for index, pose in enumerate(camera_views):
            cid = str(pose["id"])
            plan = self.mask_plan.get(cid, {})
            if plan == "blank":
                plan = {tid: 0 for tid in target_ids}
            targets = {}
            status = "ok"
            for tid in target_ids:
                count = plan.get(tid, 0)
                mask_path = destination / f"mask_{index:02d}_{cid}__{tid}.png"
                _write_binary_mask(mask_path, white_pixels=int(count))
                targets[tid] = {"mask_path": str(mask_path)}
            if all(int(plan.get(tid, 0)) == 0 for tid in target_ids):
                status = "blank"
            views.append({"id": cid, "status": status, "targets": targets, "focus": {"in_frame": True}})
        return {"views": views}


def test_measure_target_mask_png_is_binary_identity(tmp_path: Path) -> None:
    mask = tmp_path / "m.png"
    _write_binary_mask(mask, white_pixels=10, size=8)
    stats = measure_target_mask_png(mask)
    assert stats["measured"] is True
    assert stats["visible_pixels"] == 10
    assert stats["image_pixel_count"] == 64
    assert stats["bbox"] is not None


def test_mask_stats_are_independent_of_color_management() -> None:
    # Two records with identical binary masks but wildly different notional RGB
    # exposure produce identical visibility, because ranking reads masks only.
    record = {
        "status": "ok",
        "targets": {
            "a": {"visible_pixels": 100, "image_pixel_count": 1000},
            "b": {"visible_pixels": 40, "image_pixel_count": 1000},
        },
        "focus": {"in_frame": True},
    }
    stats = build_candidate_mask_stats(record, target_ids=["a", "b"])
    assert stats["targets"]["a"]["visible_fraction"] == pytest.approx(0.1)
    assert stats["targets"]["b"]["visible_fraction"] == pytest.approx(0.04)


def test_v2_ranking_ignores_decorations_only_masks_prove_visibility() -> None:
    # A candidate whose overlay is full of decorative colored pixels but whose
    # target-B identity mask is empty must NOT be treated as B-visible. Ranking
    # reads binary identity masks only; a colored OBB line is never proof.
    candidates = [
        {"id": "decorated_but_occluded", "azimuth_degrees": 0.0, "elevation_degrees": 20.0},
        {"id": "plain_but_visible", "azimuth_degrees": 120.0, "elevation_degrees": 40.0},
    ]
    stats = {
        "decorated_but_occluded": {"status": "ok", "targets": {
            "a": {"visible_pixels": 500, "visible_fraction": 0.5, "bbox": [0.1, 0.1, 0.9, 0.9]},
            "b": {"visible_pixels": 0, "visible_fraction": 0.0, "bbox": None},  # B mask empty
        }, "focus_in_frame": True},
        "plain_but_visible": {"status": "ok", "targets": {
            "a": {"visible_pixels": 60, "visible_fraction": 0.06, "bbox": [0.3, 0.3, 0.45, 0.5]},
            "b": {"visible_pixels": 60, "visible_fraction": 0.06, "bbox": [0.5, 0.3, 0.65, 0.5]},
        }, "focus_in_frame": True},
    }
    selected, log = rank_collision_candidates_v2(candidates, stats, target_ids=["a", "b"], max_views=1)
    assert [item["id"] for item in selected] == ["plain_but_visible"]
    assert log["joint_visibility_status"] == "both_visible"


def test_v2_ranking_high_oblique_outranks_uninformative_side_view() -> None:
    candidates = [
        {"id": "side", "azimuth_degrees": 0.0, "elevation_degrees": 10.0},
        {"id": "oblique", "azimuth_degrees": 135.0, "elevation_degrees": 55.0},
    ]
    stats = {
        # side view: only A visible, B occluded behind it
        "side": {"status": "ok", "targets": {
            "a": {"visible_pixels": 800, "visible_fraction": 0.4, "touches_border": True, "bbox": [0.0, 0.1, 0.6, 0.9]},
            "b": {"visible_pixels": 0, "visible_fraction": 0.0, "touches_border": False, "bbox": None},
        }, "focus_in_frame": False},
        # oblique: both visible, well framed
        "oblique": {"status": "ok", "targets": {
            "a": {"visible_pixels": 200, "visible_fraction": 0.10, "touches_border": False, "bbox": [0.2, 0.2, 0.4, 0.5]},
            "b": {"visible_pixels": 180, "visible_fraction": 0.09, "touches_border": False, "bbox": [0.4, 0.2, 0.6, 0.5]},
        }, "focus_in_frame": True},
    }
    selected, log = rank_collision_candidates_v2(candidates, stats, target_ids=["a", "b"], max_views=1)
    assert [item["id"] for item in selected] == ["oblique"]
    assert log["is_visibility_ranked"] is True
    assert log["selector"] == COLLISION_VISIBILITY_SELECTOR_VERSION


def test_v2_ranking_blank_candidate_does_not_poison_others() -> None:
    candidates = [{"id": "blank"}, {"id": "good", "azimuth_degrees": 30.0, "elevation_degrees": 20.0}]
    stats = {
        "blank": {"status": "blank", "targets": {"a": {"visible_pixels": 0}, "b": {"visible_pixels": 0}}},
        "good": {"status": "ok", "targets": {
            "a": {"visible_pixels": 100, "visible_fraction": 0.1, "bbox": [0.2, 0.2, 0.4, 0.5]},
            "b": {"visible_pixels": 100, "visible_fraction": 0.1, "bbox": [0.4, 0.2, 0.6, 0.5]},
        }, "focus_in_frame": True},
    }
    selected, log = rank_collision_candidates_v2(candidates, stats, target_ids=["a", "b"], max_views=1)
    assert [item["id"] for item in selected] == ["good"]
    assert any(entry["id"] == "blank" for entry in log["dropped_candidates"])
    assert log["is_visibility_ranked"] is True


def test_v2_ranking_containment_uses_occlusion_status() -> None:
    candidates = [{"id": f"c{i}", "azimuth_degrees": i * 90.0, "elevation_degrees": 30.0} for i in range(3)]
    stats = {
        f"c{i}": {"status": "ok", "targets": {
            "a": {"visible_pixels": 300, "visible_fraction": 0.1, "bbox": [0.3, 0.3, 0.6, 0.6]},
            "b": {"visible_pixels": 0, "visible_fraction": 0.0, "bbox": None},
        }, "focus_in_frame": True}
        for i in range(3)
    }
    selected, log = rank_collision_candidates_v2(candidates, stats, target_ids=["a", "b"], max_views=1)
    assert log["joint_visibility_status"] == JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED
    assert log["required_target_ids"] == ["a"]
    assert selected  # still selects the best contextual view of the visible outer target


def test_provider_mask_ranking_produces_visibility_ranked_manifest(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    # c1 exposes both targets best; masks (not RGB colors) drive the choice.
    mask_plan = {}
    renderer = _MaskRenderer(mask_plan={})

    def plan_for(ids):
        plan = {}
        for i, cid in enumerate(ids):
            if i == 1:
                plan[cid] = {"a": 200, "b": 200}
            elif i == 0:
                plan[cid] = {"a": 300, "b": 0}  # only A
            else:
                plan[cid] = {"a": 40, "b": 40}
        return plan

    class _PlannedMaskRenderer(_MaskRenderer):
        def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
            self.mask_plan = plan_for([str(v["id"]) for v in camera_views])
            return super().render_target_id_masks(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _PlannedMaskRenderer(mask_plan={})
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    items = provider(_camera_request())
    assert items
    assert any(call["pass"] == "masks" for call in renderer.calls)
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert manifest["evidence_packet_version"] == COLLISION_EVIDENCE_PACKET_VERSION
    assert manifest["selector_version"] == COLLISION_VISIBILITY_SELECTOR_VERSION
    assert manifest["is_visibility_ranked"] is True
    assert manifest["selection"]["ranking"]["selected_view_ids"]
    # raw and highlight are paired, never raw-replaced.
    styles = {item["evidence_style"] for item in items if item.get("evidence_style")}
    assert "raw" in styles


def test_provider_raw_survives_overlay_failure(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _OverlayFinalFails(_MaskRenderer):
        def render_collision_overlay_views(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=False):
            if not preview:
                raise BlenderRenderError("final overlay render failed")
            return super().render_collision_overlay_views(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _OverlayFinalFails(mask_plan={"collision_00_separation_side": {"a": 200, "b": 200}})

    class _AllVisible(_OverlayFinalFails):
        def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
            self.mask_plan = {str(v["id"]): {"a": 200, "b": 200} for v in camera_views}
            return super().render_target_id_masks(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _AllVisible(mask_plan={})
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    items = provider(_camera_request())
    roles = [item["role"] for item in items]
    assert "collision_rgb" in roles  # raw survives
    assert "collision_pair_overlay" not in roles  # overlay failed everywhere
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert "final_overlay_failed" in str(manifest["overlay_degradation_reason"])


def test_provider_backfills_failed_final_rgb_from_next_candidate(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _BackfillRenderer(_MaskRenderer):
        def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
            ids = [str(v["id"]) for v in camera_views]
            # Best candidate is the first, but its final RGB will fail.
            self.mask_plan = {cid: {"a": 300 - i, "b": 300 - i} for i, cid in enumerate(ids)}
            self.rgb_fail_ids = {ids[0]}
            return super().render_target_id_masks(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _BackfillRenderer(mask_plan={})
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    items = provider(_camera_request())
    assert items
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert manifest["backfill"]["backfilled"] is True
    assert manifest["backfill"]["skipped_candidates"]


def test_final_rgb_backfill_preserves_active_modified_pose(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _BatchFailsRenderer(_FakeRenderer):
        def render_camera_views(
            self,
            *,
            blend_file,
            out_dir,
            camera_views,
            preview=False,
        ):
            if Path(out_dir).name == "final_rgb" and not preview:
                raise BlenderRenderError("force per-pose final backfill")
            return super().render_camera_views(
                blend_file=blend_file,
                out_dir=out_dir,
                camera_views=camera_views,
                preview=preview,
            )

    provider = CameraEvidenceProvider(
        renderer=_BatchFailsRenderer(),
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=1,
    )
    active_pose = {
        "id": "active_repair_00",
        "location": [3.0, 2.0, 1.4],
        "target": [1.4, 1.0, 0.5],
        "lens_mm": 58.0,
    }
    original_pose = {
        "id": "original_00",
        "location": [2.0, 2.0, 1.4],
        "target": [1.0, 1.0, 0.5],
        "lens_mm": 52.0,
    }

    manifest, rendered_selected, backfill = (
        provider._render_final_rgb_with_backfill(
            event_dir=tmp_path / "event",
            selected=[active_pose],
            candidates=[original_pose],
            ranking_log={"ranked": [{"id": "original_00"}]},
        )
    )

    assert backfill["backfilled"] is True
    assert backfill["rendered_view_ids"] == ["active_repair_00"]
    assert rendered_selected == [active_pose]
    assert manifest["views"][0]["id"] == "active_repair_00"
    assert manifest["views"][0]["pose"]["location"] == active_pose["location"]


def test_provider_all_candidate_rgb_failure_is_infrastructure_error(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _AllRgbFail(_MaskRenderer):
        def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
            ids = [str(v["id"]) for v in camera_views]
            self.mask_plan = {cid: {"a": 100, "b": 100} for cid in ids}
            self.rgb_fail_ids = set(ids)
            return super().render_target_id_masks(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _AllRgbFail(mask_plan={})
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    with pytest.raises(RuntimeError, match="no usable raw view"):
        provider(_camera_request())


def test_provider_containment_applies_xray_to_outer_target(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")

    class _ContainRenderer(_MaskRenderer):
        def render_target_id_masks(self, *, blend_file, out_dir, camera_views, overlay_spec, preview=True):
            # b is never visible: contained inside a.
            self.mask_plan = {str(v["id"]): {"a": 300, "b": 0} for v in camera_views}
            return super().render_target_id_masks(
                blend_file=blend_file, out_dir=out_dir, camera_views=camera_views, overlay_spec=overlay_spec, preview=preview
            )

    renderer = _ContainRenderer(mask_plan={})
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    provider(_camera_request())
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert manifest["joint_visibility_status"] == JOINT_VISIBILITY_IMPOSSIBLE_OR_OCCLUDED
    spec = json.loads((event_dir / "collision_overlay_spec.json").read_text())
    # The visible outer target 'a' is x-rayed so the contained 'b' stays inspectable.
    assert spec["object_a"]["xray"] is True
    assert spec["object_b"]["xray"] is False


def test_provider_records_fallback_not_visibility_ranked_when_masks_unavailable(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()  # no render_target_id_masks
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="visibility_ranked",
        max_views=1,
        collision_overlay=True,
    )
    provider(_camera_request())
    event_dir = next(iter((tmp_path / "evidence").iterdir()))
    manifest = json.loads((event_dir / "camera_evidence_manifest.json").read_text())
    assert manifest["is_visibility_ranked"] is False
    assert "unavailable" in str(manifest["selection"].get("mask_pass"))


def test_render_target_id_masks_is_read_only(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--out-dir") + 1])
        mask_path = out_dir / "mask_00.png"
        _write_binary_mask(mask_path, white_pixels=4)
        (out_dir / "target_id_mask_manifest.json").write_text(
            json.dumps({"views": [{"id": "c0", "status": "ok", "targets": {"a": {"mask_path": str(mask_path)}}}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="masks", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, render_engine="CYCLES")
    manifest = renderer.render_target_id_masks(
        blend_file=blend_file,
        out_dir=tmp_path / "masks",
        camera_views=[{"id": "c0", "location": [1.0, 2.0, 1.5], "target": [2.0, 2.0, 0.5]}],
        overlay_spec={"targets": [{"id": "a"}, {"id": "b"}]},
    )
    command = captured["command"]
    assert "--disable-autoexec" in command
    assert "blender_collision_mask_worker.py" in command[command.index("--python") + 1]
    assert manifest["camera_evidence"]["source_blend_modified"] is False
    assert manifest["camera_evidence"]["role"] == "target_id_masks"


# --------------------------------------------------------------------------- #
# 13. OOB and Support camera evidence remains unchanged
# --------------------------------------------------------------------------- #
def test_non_collision_metric_skips_overlay_and_returns_paths(tmp_path: Path) -> None:
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _FakeRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "evidence",
        mode="bbox_track",
        max_views=2,
        collision_overlay=True,  # enabled, but must not affect oob/support
    )

    request = {
        "metric": "oob",
        "event": {"object_id": "a", "plane_flags": {"west_oob": True}},
        "scene": _overlap_scene(),
        "object_ids": ["a"],
        "detector_evidence": {"plane_flags": {"west_oob": True}},
    }
    paths = provider(request)

    assert all(isinstance(path, Path) for path in paths)
    assert all(call["pass"] == "rgb" for call in renderer.calls)  # no overlay pass for oob
