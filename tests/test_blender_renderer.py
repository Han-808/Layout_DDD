from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmark.rendering import BlenderRenderError, BlenderRenderer
from benchmark.grouping import (
    grouping_evidence_from_render_manifest,
    prepare_grouping_evidence,
)


def _write_nonuniform_png(path: Path) -> None:
    image = Image.new("RGB", (8, 8), (220, 220, 220))
    image.putpixel((0, 0), (10, 20, 30))
    image.save(path)


def _architecture_from_command(command: list[str]) -> dict:
    path = Path(command[command.index("--architecture-contract") + 1])
    return json.loads(path.read_text(encoding="utf-8"))


def test_blender_renderer_launches_trusted_worker_and_validates_views(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "test", "objects": []}), encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "standardized_top.png"
        _write_nonuniform_png(view_path)
        (out_dir / "render_manifest.json").write_text(
            json.dumps({
                "backend": "blender_canonical_scene_v1",
                "architecture": _architecture_from_command(command),
                "views": [{"name": "top", "path": str(view_path)}],
            }),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(
        blender_bin=blender_bin,
        timeout_seconds=42,
        width=640,
        height=480,
        render_engine="CYCLES",
        cycles_device="CUDA",
        cycles_samples=128,
        cycles_denoising=True,
    )

    manifest = renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders", asset_root=asset_root)

    command = captured["command"]
    assert command[0] == str(blender_bin)
    assert command[1:3] == ["--background", "--factory-startup"]
    assert "blender_worker.py" in command[command.index("--python") + 1]
    assert command[command.index("--asset-root") + 1] == str(asset_root.resolve())
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "480"
    assert command[command.index("--render-engine") + 1] == "CYCLES"
    assert command[command.index("--cycles-device") + 1] == "CUDA"
    assert command[command.index("--cycles-samples") + 1] == "128"
    assert "--cycles-denoising" in command
    assert captured["kwargs"]["timeout"] == 42
    assert manifest["views"][0]["name"] == "top"
    assert manifest["render_validation"]["pixel_stats_source"] == "saved_png_pillow"
    assert manifest["views"][0]["pixel_stats"]["luminance_range"] > 0
    assert (tmp_path / "renders" / "blender.stdout.log").read_text(encoding="utf-8") == "rendered"


def test_canonical_manifest_identity_pass_is_consumed_by_grouping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "scene_id": "identity",
                "objects": [{"id": "a"}, {"id": "b"}],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        del kwargs
        out_dir = Path(command[command.index("--out-dir") + 1])
        views = []
        for name, color in (
            ("top", (30, 40, 50)),
            ("perspective", (70, 80, 90)),
            ("identity_map", (120, 30, 180)),
        ):
            path = out_dir / f"standardized_{name}.png"
            image = Image.new("RGB", (8, 8), color)
            if name == "identity_map":
                image.putpixel((0, 0), (234, 66, 18))
                image.putpixel((1, 0), (18, 142, 234))
            else:
                image.putpixel((0, 0), (250, 240, 20))
            image.save(path)
            views.append({"name": name, "path": str(path)})
        (out_dir / "render_manifest.json").write_text(
            json.dumps(
                    {
                        "backend": "blender_canonical_scene_v1",
                        "architecture": _architecture_from_command(command),
                        "views": views,
                    "identity_legend": {
                        "#EA4212": "a",
                        "#128EEA": "b",
                    },
                    "identity_render": {
                        "status": "available",
                        "color_encoding": "raw_linear_rgb_8bit",
                    },
                    "objects": [
                        {"id": "a", "representation": "bbox_proxy"},
                        {"id": "b", "representation": "bbox_proxy"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0, stdout="rendered", stderr=""
        )

    monkeypatch.setattr(
        "benchmark.rendering.blender.subprocess.run", fake_run
    )
    manifest = BlenderRenderer(
        blender_bin=blender_bin
    ).render_scene(
        scene_path=scene_path,
        out_dir=tmp_path / "renders",
    )
    grouping_items = grouping_evidence_from_render_manifest(manifest)
    packet = prepare_grouping_evidence(
        grouping_items,
        expected_object_ids=("a", "b"),
    )

    assert [item["name"] for item in manifest["views"]] == [
        "top",
        "perspective",
        "identity_map",
    ]
    assert manifest["identity_legend"] == {
        "#EA4212": "a",
        "#128EEA": "b",
    }
    assert packet.input_mode == "identity_aware_perspective_top"
    assert packet.identity_legend == manifest["identity_legend"]
    assert manifest["render_validation"]["identity_map"] == {
        "status": "verified",
        "expected_object_count": 2,
        "legend_object_count": 2,
        "visible_exact_identity_count": 2,
        "color_encoding": "raw_linear_rgb_8bit",
    }


def test_blender_renderer_fails_before_subprocess_for_missing_binary(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    renderer = BlenderRenderer(blender_bin=tmp_path / "missing-blender")

    with pytest.raises(BlenderRenderError, match="does not exist"):
        renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders")


def test_blender_renderer_renders_read_only_track_camera_views(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--out-dir") + 1])
        poses = json.loads(Path(command[command.index("--camera-views-json") + 1]).read_text(encoding="utf-8"))
        view_path = out_dir / "camera_00_collision.png"
        _write_nonuniform_png(view_path)
        (out_dir / "camera_render_manifest.json").write_text(
            json.dumps(
                {
                    "backend": "blender_read_only_camera_evidence_v1",
                    "source_scene_saved": False,
                    "views": [{"id": poses[0]["id"], "name": "collision", "path": str(view_path)}],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="camera rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, width=768, height=768, render_engine="CYCLES")
    manifest = renderer.render_camera_views(
        blend_file=blend_file,
        out_dir=tmp_path / "camera",
        camera_views=[
            {
                "id": "collision_side",
                "name": "collision side",
                "location": [1.0, 2.0, 1.5],
                "target": [2.0, 2.0, 0.5],
            }
        ],
        preview=True,
    )

    command = captured["command"]
    assert "--disable-autoexec" in command
    assert str(blend_file.resolve()) in command
    assert "blender_camera_worker.py" in command[command.index("--python") + 1]
    assert command[command.index("--render-engine") + 1] == "CYCLES"
    assert command[command.index("--width") + 1] == "256"
    assert command[command.index("--cycles-samples") + 1] == "1"
    assert manifest["camera_evidence"]["render_engine"] == "CYCLES"
    assert manifest["camera_evidence"]["source_blend_modified"] is False
    assert manifest["camera_evidence"]["source_blend_sha256_before"] == manifest["camera_evidence"]["source_blend_sha256_after"]
    assert manifest["views"][0]["pixel_stats"]["luminance_range"] > 0


def test_camera_renderer_detects_same_size_same_mtime_blend_replacement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    original_stat = blend_file.stat()

    def fake_run(command, **kwargs):
        del kwargs
        blend_file.write_bytes(b"BLEND")
        os.utime(
            blend_file,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "camera.png"
        _write_nonuniform_png(view_path)
        (out_dir / "camera_render_manifest.json").write_text(
            json.dumps(
                {
                    "views": [
                        {
                            "id": "camera",
                            "name": "camera",
                            "path": str(view_path),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "benchmark.rendering.blender.subprocess.run",
        fake_run,
    )
    renderer = BlenderRenderer(blender_bin=blender_bin)

    with pytest.raises(
        BlenderRenderError,
        match="modified the source Blender scene",
    ):
        renderer.render_camera_views(
            blend_file=blend_file,
            out_dir=tmp_path / "camera",
            camera_views=[
                {
                    "id": "camera",
                    "location": [1, 1, 1],
                    "target": [0, 0, 0],
                }
            ],
        )


def test_blender_renderer_rejects_unknown_cycles_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cycles_device"):
        BlenderRenderer(blender_bin=tmp_path / "blender", cycles_device="METAL")


def test_camera_preview_rejects_near_uniform_dark_frames(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "preview.png"
        image = Image.new("RGB", (64, 64), (10, 10, 10))
        image.putpixel((0, 0), (12, 12, 12))
        image.save(view_path)
        (out_dir / "camera_render_manifest.json").write_text(
            json.dumps({"views": [{"id": "preview", "name": "preview", "path": str(view_path)}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, render_engine="CYCLES")
    with pytest.raises(BlenderRenderError, match="blank or near-uniform"):
        renderer.render_camera_views(
            blend_file=blend_file,
            out_dir=tmp_path / "preview",
            camera_views=[{"id": "preview", "location": [1, 1, 1], "target": [0, 0, 0]}],
            preview=True,
        )


def test_camera_preview_can_record_blank_candidate_without_rejecting_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "preview.png"
        Image.new("RGB", (64, 64), (10, 10, 10)).save(view_path)
        (out_dir / "camera_render_manifest.json").write_text(
            json.dumps({"views": [{"id": "preview", "name": "preview", "path": str(view_path)}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, render_engine="CYCLES")

    manifest = renderer.render_camera_views(
        blend_file=blend_file,
        out_dir=tmp_path / "preview",
        camera_views=[{"id": "preview", "location": [1, 1, 1], "target": [0, 0, 0]}],
        preview=True,
        allow_blank_views=True,
    )

    assert manifest["render_validation"]["blank_views"] == ["preview"]
    assert manifest["camera_evidence"]["blank_view_policy"] == "record"


def test_blender_renderer_bundles_final_focus_passes(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("--out-dir") + 1])
        request = json.loads(
            Path(command[command.index("--request-json") + 1]).read_text(encoding="utf-8")
        )
        rgb = out_dir / "rgb.png"
        highlight = out_dir / "highlight.png"
        global_highlight = out_dir / "global.png"
        for path in (rgb, highlight, global_highlight):
            _write_nonuniform_png(path)
        local_id = request["local_camera_views"][0]["id"]
        global_id = request["global_camera_views"][0]["id"]
        rgb_view = {"id": local_id, "path": str(rgb)}
        highlight_view = {"id": local_id, "path": str(highlight)}
        global_view = {"id": global_id, "path": str(global_highlight)}
        (out_dir / "focus_bundle_manifest.json").write_text(
            json.dumps(
                {
                    "rgb_views": [rgb_view],
                    "overlay_views": [highlight_view],
                    "global_overlay_views": [global_view],
                    "views": [rgb_view, highlight_view, global_view],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="bundled", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(
        blender_bin=blender_bin,
        render_engine="CYCLES",
        cycles_device="CUDA",
        cycles_samples=8,
    )
    manifest = renderer.render_focus_evidence_bundle(
        blend_file=blend_file,
        out_dir=tmp_path / "bundle",
        local_camera_views=[{"id": "local", "location": [1, 1, 1], "target": [0, 0, 0]}],
        global_camera_views=[{"id": "global", "location": [2, 2, 2], "target": [0, 0, 0]}],
        overlay_spec={"targets": [{"id": "object_1"}]},
    )

    command = captured["command"]
    assert "blender_focus_bundle_worker.py" in command[command.index("--python") + 1]
    assert command[command.index("--render-engine") + 1] == "CYCLES"
    assert command[command.index("--cycles-device") + 1] == "CUDA"
    assert len(manifest["views"]) == 3
    assert manifest["camera_evidence"]["bundled_process"] is True


def test_blender_renderer_rejects_blank_render_manifest(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "test", "objects": []}), encoding="utf-8")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "standardized_top.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(view_path)
        (out_dir / "render_manifest.json").write_text(
            json.dumps(
                {
                    "architecture": _architecture_from_command(command),
                    "views": [
                        {
                            "name": "top",
                            "path": str(view_path),
                            "pixel_stats": {"max_luminance": 0.0, "luminance_range": 0.0},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, render_engine="BLENDER_WORKBENCH")

    with pytest.raises(BlenderRenderError, match="blank or near-uniform"):
        renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders")


def test_blender_renderer_rejects_proxy_only_asset_run(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "test", "objects": []}), encoding="utf-8")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "standardized_top.png"
        _write_nonuniform_png(view_path)
        (out_dir / "render_manifest.json").write_text(
            json.dumps(
                {
                    "architecture": _architecture_from_command(command),
                    "views": [{"name": "top", "path": str(view_path)}],
                    "objects": [
                        {
                            "id": "bed_1",
                            "representation": "bbox_proxy",
                            "mesh_path": None,
                            "warning": "no loadable mesh reference",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(
        blender_bin=blender_bin,
        render_engine="BLENDER_WORKBENCH",
        require_asset_mesh=True,
    )

    with pytest.raises(BlenderRenderError, match="zero asset meshes"):
        renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders")

    manifest = json.loads((tmp_path / "renders" / "render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["asset_coverage"] == {
        "object_count": 1,
        "asset_mesh_count": 0,
        "bbox_proxy_count": 1,
        "asset_mesh_rate": 0.0,
        "required": True,
    }


def test_blender_renderer_preserves_completed_views_when_optional_geometry_times_out(
    monkeypatch,
    tmp_path: Path,
) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "test", "objects": []}), encoding="utf-8")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        view_path = out_dir / "standardized_top.png"
        _write_nonuniform_png(view_path)
        (out_dir / "scene.blend").write_text("blend", encoding="utf-8")
        (out_dir / "blender_worker_progress.jsonl").write_text(
            json.dumps({"stage": "base_manifest_written"}) + "\n",
            encoding="utf-8",
        )
        (out_dir / "render_manifest.json").write_text(
            json.dumps(
                {
                    "architecture": _architecture_from_command(command),
                    "views": [{"name": "top", "path": str(view_path)}],
                    "objects": [],
                    "collision_geometry_manifest": None,
                    "collision_geometry_export": {"status": "pending"},
                }
            ),
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="rendered", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, timeout_seconds=9)

    manifest = renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders")

    assert manifest["worker_completion"]["status"] == "timed_out_after_base_manifest"
    assert manifest["collision_geometry_export"]["status"] == "timed_out_after_render"
    assert manifest["views"][0]["pixel_stats"]["luminance_range"] > 0


def test_blender_renderer_reports_last_stage_when_timeout_precedes_render_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "test", "objects": []}), encoding="utf-8")

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        (out_dir / "blender_worker_progress.jsonl").write_text(
            json.dumps({"stage": "object_built"}) + "\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial", stderr="slow")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(blender_bin=blender_bin, timeout_seconds=7)

    with pytest.raises(BlenderRenderError, match="last_worker_stage='object_built'"):
        renderer.render_scene(scene_path=scene_path, out_dir=tmp_path / "renders")

    assert (tmp_path / "renders" / "blender.stdout.log").read_text(encoding="utf-8") == "partial"
    assert (tmp_path / "renders" / "blender.stderr.log").read_text(encoding="utf-8") == "slow"
