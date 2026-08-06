from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from benchmark.game_scene.mode import (
    GAME_ACTIVE_METRICS,
    GAME_OBJECTIZATION_STRATEGY,
    GameModeConfigError,
    load_game_mode_config,
)
from benchmark.game_scene.route import run_game_mode
from benchmark.rendering import HeadlessBrowserRenderer
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parent.parent
MODE_PATH = ROOT / "configs" / "game" / "game_mode_canonical_v1.yaml"
PROFILE_PATH = (
    ROOT / "configs" / "evaluation" / "metric_profile_game_canonical_v1.yaml"
)

_CUBE_FACES = [
    [0, 1, 3], [0, 3, 2],
    [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4],
    [1, 5, 7], [1, 7, 3],
]


def _probe_object(object_id: str, translation: tuple[float, float, float]) -> dict:
    vertices = np.array(
        [
            [sx * 0.5, sy * 0.5, sz * 0.5]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    ) + np.asarray(translation, dtype=float)
    return {
        "id": object_id,
        "category": "crate",
        "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
        "vertices": vertices.tolist(),
        "faces": _CUBE_FACES,
        "mesh_complete": True,
        "world_bounds": {
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        },
    }


class _Page:
    def __init__(self) -> None:
        self.enter_count = 0
        self.probe_count = 0

    def __enter__(self) -> "_Page":
        self.enter_count += 1
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def step(self, frames: int) -> None:
        return None

    def screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (2, 2), (30, 50, 70))
        image.putpixel((0, 0), (160, 100, 40))
        image.save(path)

    def probe(self, script: str, options: dict) -> dict:
        self.probe_count += 1
        return {
            "schema_version": "game_scene_probe_v1",
            "up_axis": "y",
            "unit_scale": 1.0,
            "captured_at_tick": options["tick"],
            "deterministic_seed": options["seed"],
            "objects": [
                _probe_object("cube_0000", (-2.0, 0.5, 0.0)),
                _probe_object("cube_0001", (2.0, 0.5, 0.0)),
            ],
        }

    def render_camera_views(
        self,
        script: str,
        poses: list[dict],
        destination: Path,
    ) -> list[dict]:
        assert "original Three.js runtime" in script
        views = []
        for pose in poses:
            path = destination / f"global_{pose['id']}.png"
            image = Image.new("RGB", (2, 2), (45, 65, 85))
            image.putpixel((0, 0), (185, 125, 65))
            image.save(path)
            views.append(
                {
                    "id": pose["id"],
                    "name": pose["id"],
                    "path": path.as_posix(),
                    "scope": "global",
                    "presentation": "raw",
                    "backend": "threejs_original_runtime",
                    "appearance_fidelity": "original_runtime_direct_webgl",
                    "camera_pose_canonical": pose["canonical_pose"],
                    "camera_pose_source": pose["source_pose"],
                    "runtime_diagnostics": {
                        "registered_renderer_count": 1,
                        "direct_render_call_count": 1,
                    },
                }
            )
        return views


class _Judge:
    def chat_messages(
        self,
        messages: list[dict],
        **kwargs: object,
    ) -> str:
        return (
            '{"object_groups":[{"object_ids":["cube_0000","cube_0001"],'
            '"label":"arena ensemble","anchor_object_id":null,'
            '"reason":"Both visible objects share one local evidence scope."}],'
            '"reason":"Complete two-object partition."}'
        )

    def adjudicate_p0b(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_scene_quality(self, request: dict) -> dict:
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "no significant style defect",
            "missing_evidence": [],
            "defects": [],
        }


class _TrackingStyleJudge(_Judge):
    def __init__(self) -> None:
        self.style_requests: list[dict] = []

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.style_requests.append(request)
        return super().adjudicate_scene_quality(request)


def test_checked_in_game_mode_is_strict_and_resolves_the_canonical_profile() -> None:
    mode = load_game_mode_config(MODE_PATH)

    assert mode.raw["mode"] == "game"
    assert set(mode.raw["evaluation"]["active_metrics"]) == GAME_ACTIVE_METRICS
    assert (
        mode.raw["ingestion"]["controlled_render_contract"]
        == "threejs_direct_webgl_renderer_v1"
    )
    assert mode.renderer_kwargs["exclude_camera_descendants"] is True
    assert mode.renderer_kwargs["drop_non_physical_meshes"] is True
    assert mode.renderer_kwargs["collapse_contained_meshes"] is True
    assert mode.renderer_kwargs["controlled_camera"] == {
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
    }
    assert mode.raw["objectization"]["strategy"] == GAME_OBJECTIZATION_STRATEGY
    assert mode.evaluation_profile_path == PROFILE_PATH.resolve()
    assert mode.default_visual_style_spec_path.name == (
        "counter_strike_static_arena_style_v1.json"
    )


def test_game_mode_rejects_unknown_inert_configuration(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    (config_root / "game").mkdir(parents=True)
    (config_root / "evaluation").mkdir(parents=True)
    shutil.copy2(PROFILE_PATH, config_root / "evaluation" / PROFILE_PATH.name)
    text = MODE_PATH.read_text(encoding="utf-8")
    (config_root / "game" / MODE_PATH.name).write_text(
        f"{text}\ninert_future_option: true\n",
        encoding="utf-8",
    )

    with pytest.raises(GameModeConfigError, match="unknown keys"):
        load_game_mode_config(config_root / "game" / MODE_PATH.name)


def test_game_mode_captures_once_then_runs_the_canonical_evaluator(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "game"
    game_root.mkdir()
    entry = game_root / "index.html"
    entry.write_text(
        "<script>new THREE.Scene(); new THREE.WebGLRenderer();</script>",
        encoding="utf-8",
    )
    page = _Page()
    mode = load_game_mode_config(MODE_PATH)
    renderer = HeadlessBrowserRenderer(
        entry_html=entry,
        game_root=game_root,
        page_factory=lambda renderer: page,
        controlled_camera=mode.renderer_kwargs["controlled_camera"],
    )
    out_dir = tmp_path / "run"

    result = run_game_mode(
        game_mode_config=MODE_PATH,
        game_root=game_root,
        out_dir=out_dir,
        case_id="game_case",
        renderer=renderer,
        vlm_judge=_Judge(),
        official_mode=True,
    )

    assert page.enter_count == 1
    assert page.probe_count == 1
    assert result["evaluation_report"]["workflow"] == "canonical_l0_l4"
    assert result["evaluation_report"]["benchmark_score_status"] == "complete"
    assert result["game_mode_manifest"]["status"] == "complete"
    assert result["game_mode_manifest"]["capture"]["performed_once"] is True
    render_manifest = read_json(out_dir / "renders" / "render_manifest.json")
    assert render_manifest["controlled_camera"]["status"] == "ready"
    assert len(render_manifest["views"]) == 2
    local_fallback = render_manifest["controlled_camera"][
        "style_local_fallback"
    ]
    assert local_fallback["status"] == "ready"
    assert len(local_fallback["views"]) == 4
    assert all(
        view["appearance_fidelity"] == "original_runtime_direct_webgl"
        for view in render_manifest["views"]
    )
    assert read_json(out_dir / "case_bundle" / "asset_policy.json")[
        "appearance_owner"
    ] == "generator"
    style_spec = read_json(out_dir / "case_bundle" / "visual_style_spec.json")
    assert style_spec["scene_type"] == "counter_strike_static_arena"

    submission = read_json(out_dir / "submission_run_manifest.json")
    assert submission["rendering"]["manifest_path"] == (
        out_dir / "renders" / "render_manifest.json"
    ).as_posix()
    assert submission["vlm_evaluation_control"]["effective"][
        "camera_selector"
    ]["backend"] == "deterministic"
    assert read_json(out_dir / "render_input_scene.json") == read_json(
        out_dir / "renders" / "probe_exported_scene.json"
    )


def test_game_mode_style_mandatory_group_review_consumes_frozen_local_bank(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "game"
    game_root.mkdir()
    entry = game_root / "index.html"
    entry.write_text(
        "<script>new THREE.Scene(); new THREE.WebGLRenderer();</script>",
        encoding="utf-8",
    )
    page = _Page()
    mode = load_game_mode_config(MODE_PATH)
    renderer = HeadlessBrowserRenderer(
        entry_html=entry,
        game_root=game_root,
        page_factory=lambda renderer: page,
        controlled_camera=mode.renderer_kwargs["controlled_camera"],
    )
    judge = _TrackingStyleJudge()

    result = run_game_mode(
        game_mode_config=MODE_PATH,
        game_root=game_root,
        out_dir=tmp_path / "run",
        case_id="game_case",
        renderer=renderer,
        vlm_judge=judge,
        official_mode=True,
    )

    style = result["evaluation_report"]["layer_reports"][
        "l3_scene_quality"
    ]["metrics"]["style_consistency"]
    assert style["status"] == "evaluated"
    assert style["route"] == "global_discovery_then_forced_group_local"
    assert style["judge_call_count"] == 2
    assert style["evidence_request"]["provider_invoked"] is True
    assert len(style["local_evidence_paths"]) == 1
    assert [request["evidence_phase"] for request in judge.style_requests] == [
        "global_discovery",
        "group_local_review",
    ]
    assert page.enter_count == 1


def test_official_game_mode_requires_the_judge_before_browser_capture(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "game"
    game_root.mkdir()
    entry = game_root / "index.html"
    entry.write_text(
        "<script>new THREE.Scene(); new THREE.WebGLRenderer();</script>",
        encoding="utf-8",
    )
    page = _Page()
    renderer = HeadlessBrowserRenderer(
        entry_html=entry,
        game_root=game_root,
        page_factory=lambda renderer: page,
    )

    with pytest.raises(RuntimeError, match="requires a VLM judge"):
        run_game_mode(
            game_mode_config=MODE_PATH,
            game_root=game_root,
            out_dir=tmp_path / "run",
            case_id="missing_judge",
            renderer=renderer,
            official_mode=True,
        )
    assert page.enter_count == 0


def test_unsupported_browser_runtime_is_not_ingestable_not_a_failed_metric(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "canvas_game"
    game_root.mkdir()
    (game_root / "index.html").write_text(
        "<canvas></canvas><script>getContext('2d')</script>",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="accepts only a source"):
        run_game_mode(
            game_mode_config=MODE_PATH,
            game_root=game_root,
            out_dir=out_dir,
            case_id="canvas_case",
            vlm_judge=_Judge(),
            official_mode=True,
        )

    manifest = read_json(out_dir / "game_mode_run_manifest.json")
    assert manifest["status"] == "not_ingestable"
    assert manifest["source"]["compatibility"]["classification"] == (
        "unsupported_browser_runtime"
    )
    assert not (out_dir / "evaluation_report.json").exists()


def test_effect_composer_source_is_not_silently_downgraded_to_direct_render(
    tmp_path: Path,
) -> None:
    game_root = tmp_path / "composed_game"
    game_root.mkdir()
    (game_root / "index.html").write_text(
        """
        <script>
          const scene = new THREE.Scene();
          const renderer = new THREE.WebGLRenderer();
          const composer = new EffectComposer(renderer);
        </script>
        """,
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="accepts only a source"):
        run_game_mode(
            game_mode_config=MODE_PATH,
            game_root=game_root,
            out_dir=out_dir,
            case_id="composer_case",
            vlm_judge=_Judge(),
            official_mode=True,
        )

    manifest = read_json(out_dir / "game_mode_run_manifest.json")
    compatibility = manifest["source"]["compatibility"]
    assert manifest["status"] == "not_ingestable"
    assert compatibility["classification"] == "unsupported_threejs_render_pipeline"
    assert compatibility["has_effect_composer"] is True
