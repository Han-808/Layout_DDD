"""The game track running on the canonical L0-L4 workflow.

Covers the checked-in canonical game profile, the case bundle authored from an
exported level, and one official submission end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark.api.submission import (
    SubmissionEvaluationError,
    evaluate_submission,
    load_case_bundle,
)
from benchmark.evaluator.profile import L1, L2, L3, L4, resolve_evaluation_profile
from benchmark.game_scene import (
    GAME_ASSET_POLICY,
    GameCaseBundleError,
    build_game_case_bundle,
    build_scene_and_collision_geometry,
)
from benchmark.utils.io import load_yaml, read_json, write_json


ROOT = Path(__file__).resolve().parent.parent
GAME_PROFILE_PATH = ROOT / "configs" / "evaluation" / "metric_profile_game_canonical_v1.yaml"

_CUBE_FACES = [
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
]


def _game_profile() -> dict:
    return load_yaml(GAME_PROFILE_PATH)


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


def _exported_level(tmp_path: Path) -> tuple[dict, dict]:
    payload = {
        "schema_version": "game_scene_probe_v1",
        "up_axis": "y",
        "unit_scale": 1.0,
        "captured_at_tick": 60,
        "deterministic_seed": 20260727,
        "objects": [
            _probe_object("cube_0000", (-2.0, 0.5, 0.0)),
            _probe_object("cube_0001", (2.0, 0.5, 0.0)),
        ],
    }
    return build_scene_and_collision_geometry(
        payload,
        scene_id="game_scene",
        request_id="game_request",
        scene_type="game_level",
        mesh_dir=tmp_path / "collision_geometry",
    )


class _Renderer:
    """Stands in for HeadlessBrowserRenderer.

    The official path takes collision geometry only from the trusted renderer's
    manifest, never from the submitter, so the fake has to publish it the same
    way the browser renderer does.
    """

    def __init__(self, collision_geometry: dict) -> None:
        self._collision_geometry = collision_geometry

    def render_scene(self, *, scene_path: Path, out_dir: Path, asset_root=None) -> dict:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        views = []
        for name in ("top", "perspective"):
            frame = destination / f"game_{name}.png"
            image = Image.new("RGB", (2, 2), (35, 55, 75))
            image.putpixel((0, 0), (170, 110, 50))
            image.save(frame)
            views.append({"name": name, "path": frame.as_posix()})
        return {
            "backend": "fake_game_renderer",
            "views": views,
            "collision_geometry": self._collision_geometry,
        }


class _Judge:
    def __init__(self) -> None:
        self.scene_quality_requests: list[dict] = []

    def adjudicate_p0b(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_relation(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.scene_quality_requests.append(request)
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "no significant in-scope style defect",
            "missing_evidence": [],
            "defects": [],
        }


def _bundle(tmp_path: Path, scene: dict, **kwargs: object) -> Path:
    return build_game_case_bundle(
        scene,
        out_dir=tmp_path / "case",
        case_id="game_case",
        evaluation_profile=_game_profile(),
        **kwargs,
    )


def test_canonical_game_profile_is_frozen_and_fully_resolved() -> None:
    profile = _game_profile()

    # A trusted bundle rejects profiles that lean on code defaults, so the
    # checked-in file has to survive resolution byte-for-byte.
    assert resolve_evaluation_profile(profile) == profile
    assert profile["profile_version"] == "canonical_scene_evaluation_v1"
    assert profile["status"] == "frozen"
    assert profile["layer_weights"] == {L1: 0.40, L2: 0.0, L3: 0.60, L4: 0.0}

    live = {
        name
        for layer in (L1, L2, L3)
        for name, metric in profile[layer]["metrics"].items()
        if metric["enabled"]
    }
    assert live == {"collision", "navigability", "style_consistency"}
    assert profile[L2]["enabled"] is False
    assert profile[L1]["metric_config"]["collision"]["separation_threshold_m"] == 0.05
    assert profile[L1]["metric_config"]["navigability"][
        "boundary_source"
    ] == "non_flat_structure_envelope"


def test_bundle_derives_the_room_from_the_exported_level(tmp_path: Path) -> None:
    scene, _geometry = _exported_level(tmp_path)

    root = _bundle(tmp_path, scene)
    request = read_json(root / "scene_request.json")

    # A game level's room is its own play volume, never a generic 5x5x3 room.
    assert request["room"]["boundary"] == scene["boundary"]
    assert request["room"]["height"] == pytest.approx(scene["scene_height"])
    assert request["request_id"] == scene["request_id"]

    contract = read_json(root / "specification_contract.json")
    assert contract["frozen"] is True
    assert contract["claims"] == {"oor": [], "oar": [], "functional_semantic_fidelity": []}
    assert read_json(root / "asset_policy.json") == GAME_ASSET_POLICY

    bundle = load_case_bundle(root)
    assert bundle.asset_policy["appearance_owner"] == "generator"


def test_bundle_rejects_a_scene_without_derivable_room_bounds(tmp_path: Path) -> None:
    scene, _geometry = _exported_level(tmp_path)
    scene.pop("scene_height")

    with pytest.raises(GameCaseBundleError, match="scene_height"):
        _bundle(tmp_path, scene)


def test_official_game_submission_scores_collision_navigability_and_style(
    tmp_path: Path,
) -> None:
    scene, geometry = _exported_level(tmp_path)
    root = _bundle(tmp_path, scene)
    judge = _Judge()

    result = evaluate_submission(
        scene=scene,
        case_bundle=load_case_bundle(root),
        out_dir=tmp_path / "run",
        renderer=_Renderer(geometry),
        vlm_judge=judge,
        official_mode=True,
    )
    report = result["evaluation_report"]

    assert report["workflow"] == "canonical_l0_l4"
    assert report["protocol_scope"] == "official_submission"
    # Probe-exported geometry is generator-authored, which is what lets a game
    # case score appearance at all instead of tripping the proxy-render gate.
    assert report["evidence_provenance"]["render_input_policy"] == (
        "trusted_generator_authored_geometry"
    )
    assert report["benchmark_score"] is not None

    layers = report["layer_reports"]
    assert layers[L1]["status"] == "evaluated"
    assert layers[L1]["score"] == pytest.approx(1.0)
    assert layers[L2]["status"] == "not_applicable"
    assert layers[L3]["status"] == "evaluated"
    assert layers[L4]["status"] == "not_implemented"

    l1_metrics = report["reports"]["generic_validity"]["metrics"]
    assert l1_metrics["collision"]["status"] == "checked"
    assert l1_metrics["navigability"]["status"] == "checked"
    assert l1_metrics["navigability"]["score_definition"] == (
        "largest_connected_free_area / total_free_area"
    )
    assert l1_metrics["navigability"]["projection"] == "single_floor_2d"
    for room_shaped in ("oob", "support", "accessibility"):
        assert l1_metrics[room_shaped]["status"] == "not_applicable"

    l3_metrics = layers[L3]["metrics"]
    assert l3_metrics["style_consistency"]["status"] == "evaluated"
    assert l3_metrics["style_consistency"]["applicability"]["applicability"] == "relevant"
    for unlabelled in ("scale_consistency", "object_pairing_consistency"):
        assert l3_metrics[unlabelled]["status"] == "not_applicable"
    assert judge.scene_quality_requests


def test_report_records_the_l1_thresholds_that_produced_the_score(tmp_path: Path) -> None:
    scene, geometry = _exported_level(tmp_path)
    root = _bundle(tmp_path, scene)

    result = evaluate_submission(
        scene=scene,
        case_bundle=load_case_bundle(root),
        out_dir=tmp_path / "run",
        renderer=_Renderer(geometry),
        vlm_judge=_Judge(),
        official_mode=True,
    )

    # Threshold overrides change verdicts, so a score is not reproducible unless
    # the report states which ones were in force.
    recorded = result["evaluation_report"]["evaluation_config"]["metric_config"]
    assert recorded[L1] == _game_profile()[L1]["metric_config"]


def test_l3_stays_unresolved_when_the_case_declares_no_asset_policy(tmp_path: Path) -> None:
    scene, geometry = _exported_level(tmp_path)
    root = _bundle(tmp_path, scene)

    manifest = read_json(root / "case_bundle.json")
    del manifest["artifacts"]["asset_policy"]
    write_json(root / "case_bundle.json", manifest)

    # Without an asset policy every L3 metric is conservatively 'pending', so
    # Scene Quality cannot score and the official run refuses to certify.
    with pytest.raises(SubmissionEvaluationError, match="complete metric coverage"):
        evaluate_submission(
            scene=scene,
            case_bundle=load_case_bundle(root),
            out_dir=tmp_path / "run",
            renderer=_Renderer(geometry),
            vlm_judge=_Judge(),
            official_mode=True,
        )

    written = json.loads((tmp_path / "run" / "evaluation_report.json").read_text())
    style = written["layer_reports"][L3]["metrics"]["style_consistency"]
    assert style["status"] == "unresolved"
    assert style["reason"] == "metric_applicability_pending"
