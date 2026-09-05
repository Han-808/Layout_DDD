from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from benchmark.game_scene.counter_strike.collision_evidence import (
    CounterStrikeCollisionEvidenceError,
    CounterStrikeFrozenCaptureRenderer,
)
from benchmark.game_scene.counter_strike.loader import (
    load_counter_strike_benchmark_config,
)
from benchmark.rendering.browser import (
    BROWSER_RENDER_BACKEND,
    CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = (
    ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _view(
    *,
    view_id: str,
    path: Path,
    scope: str,
    role: str | None,
    target_object_ids: list[str],
) -> dict:
    locations = {
        "style_region_00": [2.0, -6.0, 4.0],
        "style_region_01": [12.0, 2.0, 4.0],
        "style_region_02": [5.0, -10.0, 8.0],
        "style_region_03": [-2.0, 5.0, 4.0],
        "global_oblique_00": [-4.0, -4.0, 10.0],
        "global_oblique_01": [14.0, 14.0, 10.0],
    }
    return {
        "id": view_id,
        "name": view_id,
        "path": path.as_posix(),
        "scope": scope,
        "role": role,
        "presentation": "raw",
        "backend": "threejs_original_runtime",
        "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
        "camera_pose_canonical": {
            "camera_type": "PERSP",
            "location": locations[view_id],
            "target": [5.0, 5.0, 1.0],
            "vertical_fov_degrees": 48.0,
            "near_m": 0.02,
            "far_m": 100.0,
        },
        "target_object_ids": target_object_ids,
    }


def _write_capture(
    tmp_path: Path,
    *,
    dark_regionals: bool = False,
) -> Path:
    capture = tmp_path / "capture"
    capture.mkdir()
    scene_path = capture / "probe_exported_scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "schema_version": "canonical_scene_v1",
                "scene_id": "cs_collision_evidence",
                "request_id": "cs_collision_evidence",
                "scene_type": "counter_strike_static_arena",
                "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "scene_height": 5.0,
                "objects": [
                    {
                        "id": "crate_a",
                        "category": "crate",
                        "center": [3.0, 4.0, 0.75],
                        "size": [1.5, 1.5, 1.5],
                        "rotation": [0.0, 0.0, 0.0],
                    },
                    {
                        "id": "crate_b",
                        "category": "crate",
                        "center": [4.1, 4.0, 0.75],
                        "size": [1.5, 1.5, 1.5],
                        "rotation": [0.0, 0.0, 12.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    globals_: list[dict] = []
    regionals: list[dict] = []
    artifacts = [scene_path]
    for index in range(2):
        path = capture / f"global_global_oblique_{index:02d}.png"
        Image.new("RGB", (256, 256), (210, 215, 220)).save(path)
        artifacts.append(path)
        globals_.append(
            _view(
                view_id=f"global_oblique_{index:02d}",
                path=path,
                scope="global",
                role=None,
                target_object_ids=[],
            )
        )
    for index in range(4):
        path = capture / f"local_style_region_{index:02d}.png"
        Image.new(
            "RGB",
            (256, 256),
            (
                (2 + index, 3 + index, 4 + index)
                if dark_regionals
                else (180 + 5 * index, 186, 192)
            ),
        ).save(path)
        artifacts.append(path)
        regionals.append(
            _view(
                view_id=f"style_region_{index:02d}",
                path=path,
                scope="object_local",
                role="style_local_fallback",
                target_object_ids=(
                    ["crate_a", "crate_b"] if index == 2 else ["crate_a"]
                ),
            )
        )
    manifest = {
        "backend": BROWSER_RENDER_BACKEND,
        "exported_scene": scene_path.as_posix(),
        "views": globals_,
        "controlled_camera": {
            "enabled": True,
            "status": "ready",
            "view_family": "canonical_high_oblique_pair_v1",
            "image_budget": 2,
            "appearance_fidelity": CONTROLLED_CAMERA_APPEARANCE_FIDELITY,
            "style_local_fallback": {
                "enabled": True,
                "status": "ready",
                "view_family": "canonical_style_region_quadrants_v1",
                "image_budget": 4,
                "views": regionals,
            },
        },
        "capture_artifacts": {
            path.relative_to(capture).as_posix(): _sha256(path)
            for path in artifacts
        },
    }
    (capture / "render_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return capture


def _renderer(
    tmp_path: Path,
    *,
    dark_regionals: bool = False,
) -> CounterStrikeFrozenCaptureRenderer:
    return CounterStrikeFrozenCaptureRenderer(
        capture_dir=_write_capture(
            tmp_path,
            dark_regionals=dark_regionals,
        ),
        evidence_out_dir=tmp_path / "derived",
        benchmark_config=load_counter_strike_benchmark_config(
            BENCHMARK_CONFIG
        ),
    )


def test_collision_provider_returns_two_pose_raw_and_honest_obb_overlays(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path)

    evidence = renderer(
        {
            "metric": "collision",
            "object_ids": ["crate_a", "crate_b"],
        }
    )

    assert [item["role"] for item in evidence] == [
        "collision_rgb",
        "collision_pair_overlay",
        "collision_rgb",
        "collision_pair_overlay",
    ]
    assert len({item["view_id"] for item in evidence}) == 2
    assert evidence[0]["pair_id"] == evidence[1]["pair_id"]
    assert evidence[2]["pair_id"] == evidence[3]["pair_id"]
    assert evidence[0]["pair_id"] != evidence[2]["pair_id"]
    for raw_item, overlay_item in zip(evidence[::2], evidence[1::2]):
        assert overlay_item["representation"] == (
            "same_pose_projected_canonical_obb_wireframe"
        )
        assert "segmentation" not in overlay_item["representation"]
        raw = Path(raw_item["path"])
        overlay = Path(overlay_item["path"])
        assert raw.is_file() and overlay.is_file()
        assert _sha256(raw) != _sha256(overlay)
    assert renderer.policy_config["local_view_count"] == 2
    assert renderer.policy_config["segmentation_contour_claimed"] is False


def test_collision_provider_brightness_repairs_dark_selected_angles(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path, dark_regionals=True)

    evidence = renderer(
        {
            "metric": "collision",
            "object_ids": ["crate_a", "crate_b"],
        }
    )

    raw_items = evidence[::2]
    assert len(raw_items) == 2
    assert all(item["presentation"] == "brightness_repair" for item in raw_items)
    assert all(item["luminance"]["repaired"] is True for item in raw_items)
    assert all(
        item["luminance"]["output"]["median_luminance"]
        > item["luminance"]["input"]["median_luminance"]
        for item in raw_items
    )
    assert all(item["source_view_id"] == item["view_id"] for item in raw_items)
    assert all(len(item["source_sha256"]) == 64 for item in raw_items)


def test_collision_provider_is_metric_scoped_and_fails_on_unknown_target(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path)

    assert renderer({"metric": "support", "object_ids": ["crate_a"]}) == []
    with pytest.raises(CounterStrikeCollisionEvidenceError) as caught:
        renderer(
            {
                "metric": "collision",
                "object_ids": ["crate_a", "unknown"],
            }
        )

    assert caught.value.code == "target_object_missing"
