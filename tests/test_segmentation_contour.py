from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from benchmark.rendering.blender import BlenderRenderer
from benchmark.rendering.segmentation_contour import (
    compose_segmentation_contour_highlight,
    compose_segmentation_contour_manifest,
)


def test_contour_preserves_target_interior_and_unrelated_rgb(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    out_path = tmp_path / "contour.png"
    Image.new("RGB", (40, 40), (50, 80, 110)).save(rgb_path)
    mask = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(mask).rectangle((15, 15, 24, 24), fill=255)
    mask.save(mask_path)

    manifest = compose_segmentation_contour_highlight(
        rgb_path=rgb_path,
        targets=[{"id": "a", "color": [1.0, 0.0, 0.0], "mask_path": mask_path}],
        out_path=out_path,
        band_width_px=3,
        outline_width_px=1,
        band_alpha=0.5,
        outline_alpha=1.0,
    )
    with Image.open(out_path) as rendered:
        rendered = rendered.convert("RGB")
        assert rendered.getpixel((20, 20)) == (50, 80, 110)
        assert rendered.getpixel((12, 20)) != (50, 80, 110)
        assert rendered.getpixel((11, 20)) == (255, 0, 0)
        assert rendered.getpixel((0, 0)) == (50, 80, 110)
    assert manifest["target_interior_policy"] == "preserve_raw_rgb"
    assert manifest["targets"][0]["visible_pixels_at_composite_resolution"] == 100


def test_manifest_pairs_views_and_resizes_categorical_mask(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (32, 32), (40, 40, 40)).save(rgb_path)
    mask = Image.new("L", (16, 16), 0)
    ImageDraw.Draw(mask).rectangle((6, 6, 9, 9), fill=255)
    mask.save(mask_path)

    manifest = compose_segmentation_contour_manifest(
        rgb_manifest={"views": [{"id": "v0", "path": str(rgb_path), "pose": {"id": "v0"}}]},
        mask_manifest={
            "views": [
                {
                    "id": "v0",
                    "targets": {"a": {"mask_path": str(mask_path)}},
                }
            ]
        },
        overlay_spec={"targets": [{"id": "a", "color": [0.1, 0.85, 0.92]}]},
        out_dir=tmp_path / "out",
    )
    assert len(manifest["views"]) == 1
    assert manifest["views"][0]["targets"][0]["source_mask_size"] == [16, 16]
    assert manifest["views"][0]["targets"][0]["composite_mask_size"] == [32, 32]
    assert (tmp_path / "out" / "segmentation_contour_manifest.json").is_file()


def test_full_resolution_identity_mask_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    blender_bin = tmp_path / "blender"
    blender_bin.write_text("fake", encoding="utf-8")
    blender_bin.chmod(0o755)
    blend_file = tmp_path / "scene.blend"
    blend_file.write_bytes(b"blend")
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        out_dir = Path(command[command.index("--out-dir") + 1])
        mask_path = out_dir / "mask.png"
        Image.new("L", (8, 8), 255).save(mask_path)
        (out_dir / "target_id_mask_manifest.json").write_text(
            json.dumps(
                {
                    "views": [
                        {
                            "id": "v0",
                            "status": "ok",
                            "targets": {"a": {"mask_path": str(mask_path)}},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.rendering.blender.subprocess.run", fake_run)
    renderer = BlenderRenderer(
        blender_bin=blender_bin,
        width=512,
        height=384,
        preview_width=128,
        preview_height=96,
    )
    common = {
        "blend_file": blend_file,
        "camera_views": [{"id": "v0", "location": [1, 1, 1], "target": [0, 0, 0]}],
        "overlay_spec": {"targets": [{"id": "a"}]},
    }
    renderer.render_target_id_masks(out_dir=tmp_path / "preview", preview=True, **common)
    renderer.render_target_id_masks(
        out_dir=tmp_path / "final",
        preview=False,
        respect_occlusion=True,
        **common,
    )
    preview_command, final_command = captured
    assert preview_command[preview_command.index("--width") + 1] == "128"
    assert preview_command[preview_command.index("--height") + 1] == "96"
    assert final_command[final_command.index("--width") + 1] == "512"
    assert final_command[final_command.index("--height") + 1] == "384"
    assert "--respect-occlusion" in final_command
