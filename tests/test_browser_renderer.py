from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark.api.submission import _trusted_render_paths
from benchmark.rendering import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
    BrowserRenderError,
    FrozenBrowserCaptureRenderer,
    HeadlessBrowserRenderer,
)
from benchmark.rendering.browser import _game_visual_focus_bounds, _png_is_exactly_uniform
from benchmark.utils.io import read_json, write_json


_CUBE_FACES = [
    [0, 1, 3], [0, 3, 2],
    [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4],
    [1, 5, 7], [1, 7, 3],
]


def _cube_probe_object(object_id: str, translation: tuple[float, float, float]) -> dict:
    vertices = np.array(
        [[sx * 0.5, sy * 0.5, sz * 0.5] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=float,
    ) + np.asarray(translation, dtype=float)
    return {
        "id": object_id,
        "category": "crate",
        "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
        "vertices": vertices.tolist(),
        "faces": _CUBE_FACES,
        "mesh_complete": True,
        "world_bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
    }


class _FakePage:
    """Stand-in for a Playwright page that records the deterministic stepping."""

    def __init__(self, *, probe_payload: dict | None) -> None:
        self.probe_payload = probe_payload
        self.steps: list[int] = []
        self.screenshots: list[Path] = []
        self.probe_options: dict | None = None
        self.controlled_poses: list[dict] = []
        self.controlled_pose_batches: list[list[dict]] = []

    def __enter__(self) -> "_FakePage":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def step(self, frames: int) -> None:
        self.steps.append(int(frames))

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake game frame")
        self.screenshots.append(path)

    def probe(self, script: str, options: dict) -> dict | None:
        assert "game_scene_probe_v1" in script
        self.probe_options = options
        return self.probe_payload

    def render_camera_views(
        self,
        script: str,
        poses: list[dict],
        destination: Path,
    ) -> list[dict]:
        assert "original Three.js runtime" in script
        self.controlled_poses = poses
        self.controlled_pose_batches.append(poses)
        views = []
        for pose in poses:
            path = destination / f"global_{pose['id']}.png"
            path.write_bytes(b"fake controlled game frame")
            views.append(
                {
                    "id": pose["id"],
                    "name": pose["id"],
                    "path": path.as_posix(),
                    "scope": "global",
                    "presentation": "raw",
                    "backend": "threejs_original_runtime",
                    "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
                    "camera_pose_canonical": pose["canonical_pose"],
                    "camera_pose_source": pose["source_pose"],
                    "runtime_diagnostics": {
                        "registered_renderer_count": 1,
                        "direct_render_call_count": 2,
                    },
                }
            )
        return views


def _renderer(tmp_path: Path, page: _FakePage, **kwargs) -> HeadlessBrowserRenderer:
    entry = tmp_path / "index.html"
    entry.write_text("<html></html>", encoding="utf-8")
    return HeadlessBrowserRenderer(
        entry_html=entry,
        page_factory=lambda renderer: page,
        **kwargs,
    )


def _submitted_scene(tmp_path: Path) -> Path:
    return write_json(
        tmp_path / "submitted.json",
        {
            "schema_version": "canonical_scene_v1",
            "scene_id": "game_scene",
            "request_id": "game_request",
            "scene_type": "game_level",
        },
    )


def test_render_scene_fills_all_three_evidence_channels(tmp_path: Path) -> None:
    payload = {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 1.0,
        "objects": [
            _cube_probe_object("cube_0000", (-2.0, 0.5, 0.0)),
            _cube_probe_object("cube_0001", (2.0, 0.5, 0.0)),
        ],
    }
    page = _FakePage(probe_payload=payload)
    renderer = _renderer(tmp_path, page)
    out_dir = tmp_path / "renders"

    manifest = renderer.render_scene(scene_path=_submitted_scene(tmp_path), out_dir=out_dir)

    assert manifest["backend"] == BROWSER_RENDER_BACKEND
    assert [view["name"] for view in manifest["views"]] == ["gameplay", "gameplay_settled"]
    assert all(Path(view["path"]).is_file() for view in manifest["views"])
    geometry = manifest["collision_geometry"]
    assert geometry["schema_version"] == "collision_geometry_v1"
    assert set(geometry["objects"]) == {"cube_0000", "cube_0001"}
    assert all(entry["transform_baked"] is True for entry in geometry["objects"].values())
    exported = read_json(Path(manifest["exported_scene"]))
    assert {obj["geometry_provenance"] for obj in exported["objects"]} == {"generated_mesh"}


def test_controlled_global_views_use_original_runtime_and_canonical_transform(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 0.5,
        "objects": [
            _cube_probe_object("cube_0000", (-2.0, 0.5, 0.0)),
            _cube_probe_object("cube_0001", (2.0, 0.5, 0.0)),
        ],
    }
    page = _FakePage(probe_payload=payload)
    renderer = _renderer(
        tmp_path,
        page,
        controlled_camera={
            "enabled": True,
            "required": True,
            "view_family": "canonical_high_oblique_pair_v1",
            "image_budget": 2,
            "canvas_only": True,
            "include_authored_camera_diagnostics": True,
            "unsupported_render_pipeline": "fail_not_ingestable",
        },
    )

    manifest = renderer.capture_game_source(
        out_dir=tmp_path / "renders",
        scene_id="game_scene",
        request_id="game_request",
    )

    assert [view["id"] for view in manifest["views"]] == [
        "global_oblique_00",
        "global_oblique_01",
    ]
    assert all(
        view["appearance_fidelity"] == CONTROLLED_CAMERA_APPEARANCE_FIDELITY
        for view in manifest["views"]
    )
    assert manifest["controlled_camera"]["status"] == "ready"
    assert len(manifest["authored_camera_views"]) == 2
    assert len(page.controlled_poses) == 2
    # The source pose must be the exact inverse of the recorded Y-up,
    # unit-scale, and translation transform, not a second heuristic camera.
    imported = read_json(Path(manifest["exported_scene"]))["metadata"][
        "game_scene_import"
    ]
    canonical = page.controlled_poses[0]["canonical_pose"]["location"]
    source = page.controlled_poses[0]["source_pose"]["location"]
    translation = imported["translation_applied"]
    reconstructed = [
        source[0] * imported["unit_scale"] + translation[0],
        -source[2] * imported["unit_scale"] + translation[1],
        source[1] * imported["unit_scale"] + translation[2],
    ]
    assert reconstructed == pytest.approx(canonical)


def test_controlled_style_local_bank_is_frozen_and_provider_only_returns_evidence(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 1.0,
        "objects": [
            _cube_probe_object("cube_0000", (-2.0, 0.5, -2.0)),
            _cube_probe_object("cube_0001", (2.0, 0.5, 2.0)),
        ],
    }
    page = _FakePage(probe_payload=payload)
    renderer = _renderer(
        tmp_path,
        page,
        controlled_camera={
            "enabled": True,
            "required": True,
            "view_family": "canonical_high_oblique_pair_v1",
            "image_budget": 2,
            "style_local_fallback_enabled": True,
            "style_local_view_family": "canonical_style_region_quadrants_v1",
            "style_local_image_budget": 4,
            "canvas_only": True,
            "include_authored_camera_diagnostics": True,
            "unsupported_render_pipeline": "fail_not_ingestable",
        },
    )
    capture_dir = tmp_path / "renders"
    manifest = renderer.capture_game_source(
        out_dir=capture_dir,
        scene_id="game_scene",
        request_id="game_request",
    )

    fallback = manifest["controlled_camera"]["style_local_fallback"]
    assert fallback["status"] == "ready"
    assert len(fallback["views"]) == 4
    assert len(page.controlled_pose_batches) == 2
    assert all(view["scope"] == "object_local" for view in fallback["views"])

    frozen = FrozenBrowserCaptureRenderer(capture_dir=capture_dir)
    result = frozen.provide_scene_quality_evidence(
        {
            "metric": "style_consistency",
            "evidence_scope": "object_local",
            "object_ids": ["cube_0000"],
            "evidence_policy": {"image_budget": 2},
        }
    )
    assert result["status"] == "available"
    assert result["selection_role"] == (
        "visual_evidence_only_do_not_judge_metric"
    )
    assert len(result["render_evidence_items"]) == 2

    group_result = frozen.provide_scene_quality_evidence(
        {
            "metric": "style_consistency",
            "evidence_scope": "group_local",
            "object_ids": ["cube_0000", "cube_0001"],
            "evidence_policy": {"image_budget": 1},
        }
    )
    assert group_result["status"] == "available"
    assert group_result["render_evidence_items"][0]["role"] == "group_local"


def test_visual_focus_uses_rotated_world_bounds_and_ignores_flat_ground() -> None:
    scene = {
        "objects": [
            {
                "id": "ground",
                "center": [50.0, 50.0, 0.0],
                "size": [160.0, 0.001, 160.0],
                "rotation": [90.0, 0.0, 0.0],
            },
            {
                "id": "arena",
                "center": [50.0, 50.0, 2.0],
                "size": [20.0, 20.0, 4.0],
                "rotation": [0.0, 0.0, 0.0],
            },
        ]
    }

    bounds = _game_visual_focus_bounds(scene)

    assert bounds is not None
    minimum, maximum = bounds
    assert minimum == pytest.approx([40.0, 40.0, 0.0])
    assert maximum == pytest.approx([60.0, 60.0, 4.0])


def test_controlled_view_gate_only_rejects_exactly_uniform_png() -> None:
    uniform = Image.new("RGB", (2, 2), (12, 12, 12))
    uniform_bytes = BytesIO()
    uniform.save(uniform_bytes, format="PNG")
    varied = uniform.copy()
    varied.putpixel((1, 1), (13, 12, 12))
    varied_bytes = BytesIO()
    varied.save(varied_bytes, format="PNG")

    assert _png_is_exactly_uniform(uniform_bytes.getvalue()) is True
    assert _png_is_exactly_uniform(varied_bytes.getvalue()) is False


def test_render_scene_steps_deterministically_before_each_view(tmp_path: Path) -> None:
    page = _FakePage(probe_payload=None)
    renderer = _renderer(
        tmp_path,
        page,
        warmup_frames=30,
        views=({"name": "a", "step_frames": 0}, {"name": "b", "step_frames": 10}),
    )

    manifest = renderer.render_scene(scene_path=_submitted_scene(tmp_path), out_dir=tmp_path / "renders")

    assert page.steps == [30, 10]
    assert [view["tick"] for view in manifest["views"]] == [30, 40]
    assert page.probe_options["tick"] == 40
    assert page.probe_options["seed"] == renderer.seed
    assert page.probe_options["excludeCameraDescendants"] is True
    # The probe classifies but never discards; individualization is the
    # exporter's job, so no drop switch is sent into the page.
    assert "excludeBackSideOnlyMeshes" not in page.probe_options


def test_render_manifest_without_a_probe_still_provides_views(tmp_path: Path) -> None:
    page = _FakePage(probe_payload=None)
    renderer = _renderer(tmp_path, page)

    manifest = renderer.render_scene(scene_path=_submitted_scene(tmp_path), out_dir=tmp_path / "renders")

    assert manifest["probe_available"] is False
    assert "collision_geometry" not in manifest
    assert manifest["views"]


def test_game_mode_capture_requires_an_instrumented_three_scene(tmp_path: Path) -> None:
    page = _FakePage(probe_payload=None)
    renderer = _renderer(tmp_path, page)

    with pytest.raises(BrowserRenderError, match="requires an instrumented Three.js"):
        renderer.capture_game_source(
            out_dir=tmp_path / "renders",
            scene_id="game_scene",
            request_id="game_request",
            require_probe=True,
        )


def test_frozen_capture_reuses_only_matching_hashed_evidence(tmp_path: Path) -> None:
    payload = {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 1.0,
        "objects": [
            _cube_probe_object("cube_0000", (-2.0, 0.5, 0.0)),
            _cube_probe_object("cube_0001", (2.0, 0.5, 0.0)),
        ],
    }
    page = _FakePage(probe_payload=payload)
    capture_dir = tmp_path / "run" / "renders"
    manifest = _renderer(tmp_path, page).capture_game_source(
        out_dir=capture_dir,
        scene_id="game_scene",
        request_id="game_request",
    )
    frozen = FrozenBrowserCaptureRenderer(capture_dir=capture_dir)

    reused = frozen.render_scene(
        scene_path=manifest["exported_scene"],
        out_dir=capture_dir,
    )
    assert reused["capture_artifacts"] == manifest["capture_artifacts"]

    Path(manifest["views"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(BrowserRenderError, match="hash mismatch"):
        frozen.render_scene(
            scene_path=manifest["exported_scene"],
            out_dir=capture_dir,
        )


def test_render_manifest_satisfies_the_submission_render_contract(tmp_path: Path) -> None:
    page = _FakePage(probe_payload=None)
    renderer = _renderer(tmp_path, page)
    out_dir = tmp_path / "renders"

    manifest = renderer.render_scene(scene_path=_submitted_scene(tmp_path), out_dir=out_dir)

    paths = _trusted_render_paths(manifest, out_dir)
    assert len(paths) == len(manifest["views"])


def test_renderer_rejects_a_view_set_that_produces_no_images(tmp_path: Path) -> None:
    page = _FakePage(probe_payload=None)
    renderer = _renderer(tmp_path, page, views=())

    with pytest.raises(BrowserRenderError, match="no overview views"):
        renderer.render_scene(scene_path=_submitted_scene(tmp_path), out_dir=tmp_path / "renders")


def test_determinism_harness_asset_installs_before_page_scripts() -> None:
    harness = (Path("src/benchmark/game_scene/determinism.js")).read_text(encoding="utf-8")

    assert "Math.random" in harness
    assert "requestAnimationFrame" in harness
    assert "__benchmarkStep" in harness
