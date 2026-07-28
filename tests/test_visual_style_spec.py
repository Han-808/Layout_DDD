from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmark.api.evaluation import run_evaluate
from benchmark.api.submission import CaseBundleError, load_case_bundle
from benchmark.evaluator.profile import resolve_evaluation_profile
from benchmark.evaluator.visual_style_spec import (
    VisualStyleSpecError,
    compile_visual_style_prompt,
    validate_visual_style_spec,
    visual_style_spec_summary,
)
from benchmark.utils.io import read_json, write_json


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(*, source: str = "benchmark_owned", frozen: bool = True) -> dict:
    return {
        "spec_version": "visual_style_spec_v1",
        "source": source,
        "frozen": frozen,
        "scene_type": "billiards",
        "directives": [
            {
                "directive_id": "style::realistic_materials",
                "statement": "Felt, rails, and balls use realistic materials with soft lighting.",
                "required": True,
            },
            {
                "directive_id": "style::unobstructed_camera",
                "statement": "The camera keeps the full table surface unobstructed.",
                "required": False,
            },
        ],
    }


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "style_scene",
        "request_id": "style_request",
        "scene_type": "billiards",
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "table",
                "category": "table",
                "size": [2.0, 1.0, 0.8],
                "center": [2.0, 2.0, 0.4],
                "rotation": [0, 0, 0],
                "metadata": {},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


class _RecordingJudge:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.requests.append(request)
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "reason": "No significant metric-scoped defect is visible.",
            "missing_evidence": [],
            "defects": [],
        }

    def adjudicate_p0b(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "ok"}

    def adjudicate_relation(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "ok"}


def _generator_asset_policy() -> dict:
    return {
        "mode": "generated_or_open_assets",
        "identity_owner": "generator",
        "category_selection_owner": "generator",
        "scale_owner": "generator",
        "appearance_owner": "generator",
        "arrangement_owner": "generator",
    }


def _style_only_config() -> dict:
    return {
        "metrics": {
            "scale_consistency": {"enabled": False},
            "object_pairing_consistency": {"enabled": False},
        }
    }


def test_compile_visual_style_prompt_is_deterministic() -> None:
    prompt = compile_visual_style_prompt(_spec())

    assert prompt == compile_visual_style_prompt(_spec())
    assert "scene_type=billiards" in prompt
    assert "1. [required] Felt, rails, and balls" in prompt
    assert "2. [optional] The camera keeps" in prompt


def test_visual_style_spec_rejects_malformed_input() -> None:
    with pytest.raises(VisualStyleSpecError, match="spec_version"):
        validate_visual_style_spec({"spec_version": "nope", "source": "manual", "frozen": True, "directives": []})

    duplicate = _spec()
    duplicate["directives"][1]["directive_id"] = duplicate["directives"][0]["directive_id"]
    with pytest.raises(VisualStyleSpecError, match="duplicates"):
        validate_visual_style_spec(duplicate)

    empty = _spec()
    empty["directives"] = []
    with pytest.raises(VisualStyleSpecError, match="non-empty list"):
        validate_visual_style_spec(empty)


def test_untrusted_source_is_rejected_only_for_official_use() -> None:
    diagnostic = _spec(source="diagnostic")

    assert validate_visual_style_spec(diagnostic) is diagnostic
    with pytest.raises(VisualStyleSpecError, match="official visual style spec source"):
        validate_visual_style_spec(diagnostic, require_trusted_source=True)


def test_summary_reports_no_spec() -> None:
    assert visual_style_spec_summary(None) == {
        "available": False,
        "spec_version": None,
        "source": None,
        "directive_count": 0,
    }


def test_run_evaluate_sends_style_spec_to_scene_quality_judge(tmp_path: Path) -> None:
    render = tmp_path / "render.png"
    render.write_bytes(b"evidence")
    judge = _RecordingJudge()

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "out",
        eval_generic_validity=True,
        render_evidence=[str(render)],
        vlm_judge=judge,
        visual_style_spec=_spec(),
        asset_policy=_generator_asset_policy(),
        scene_quality_config=_style_only_config(),
    )

    style_requests = [
        item for item in judge.requests if item["metric"] == "style_consistency"
    ]
    assert len(style_requests) == 1
    assert style_requests[0]["category"] == "l3_scene_quality"
    assert style_requests[0]["visual_style_spec"] == _spec()
    style_report = report["reports"]["scene_quality"]["metrics"]["style_consistency"]
    assert style_report["status"] == "evaluated"
    assert style_report["score"] == 1.0
    assert style_report["judgement"]["verdict"] == "valid"


def test_scene_quality_style_spec_stays_none_without_a_spec(tmp_path: Path) -> None:
    render = tmp_path / "render.png"
    render.write_bytes(b"evidence")
    judge = _RecordingJudge()

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "out",
        eval_generic_validity=True,
        render_evidence=[str(render)],
        vlm_judge=judge,
        asset_policy=_generator_asset_policy(),
        scene_quality_config=_style_only_config(),
    )

    style_request = next(
        item for item in judge.requests if item["metric"] == "style_consistency"
    )
    assert style_request["category"] == "l3_scene_quality"
    assert style_request["visual_style_spec"] is None
    assert (
        report["reports"]["scene_quality"]["metrics"]["style_consistency"]["status"]
        == "evaluated"
    )


def test_run_evaluate_accepts_a_style_spec_path(tmp_path: Path) -> None:
    spec_path = write_json(tmp_path / "style.json", _spec())
    render = tmp_path / "render.png"
    render.write_bytes(b"evidence")
    judge = _RecordingJudge()

    run_evaluate(
        scene=_scene(),
        out=tmp_path / "out",
        eval_generic_validity=True,
        render_evidence=[str(render)],
        vlm_judge=judge,
        visual_style_spec=spec_path,
        asset_policy=_generator_asset_policy(),
        scene_quality_config=_style_only_config(),
    )

    style_request = next(
        item for item in judge.requests if item["metric"] == "style_consistency"
    )
    assert style_request["category"] == "l3_scene_quality"
    assert style_request["visual_style_spec"] == _spec()


def _write_style_bundle(tmp_path: Path, spec: dict) -> Path:
    root = tmp_path / "case"
    root.mkdir(parents=True)
    profile = resolve_evaluation_profile()
    specification_contract = {
        "contract_version": "specification_contract_v1",
        "source": "trusted_case_bundle",
        "frozen": True,
        "request_id": "style_request",
        "claims": {
            "oor": [],
            "oar": [],
            "functional_semantic_fidelity": [],
        },
    }
    paths = {
        "scene_request": write_json(
            root / "scene_request.json",
            {
                "request_id": "style_request",
                "instruction": "Build a billiards table scene.",
                "scene_type": "billiards",
                "structure": False,
                "prompt_granularity": "coarse_grained",
            },
        ),
        "evaluation_profile": write_json(root / "evaluation_profile.json", profile),
        "specification_contract": write_json(
            root / "specification_contract.json", specification_contract
        ),
        "asset_policy": write_json(
            root / "asset_policy.json", _generator_asset_policy()
        ),
        "visual_style_spec": write_json(root / "visual_style_spec.json", spec),
    }
    write_json(
        root / "case_bundle.json",
        {
            "bundle_version": "benchmark_case_bundle_v1",
            "case_id": "style_case",
            "task": {"evaluator_output_type": "o1_object_state"},
            "artifacts": {
                name: {"path": path.name, "sha256": _digest(path)}
                for name, path in paths.items()
            },
            "evaluation": {
                "workflow": "canonical_l0_l4",
                "p0b_official_mode": True,
                "camera_evidence": {
                    "mode": None,
                    "metric_modes": {},
                    "max_views": 2,
                    "max_steps": 0,
                    "collision_overlay": True,
                },
            },
        },
    )
    return root


def test_case_bundle_loads_a_trusted_style_spec(tmp_path: Path) -> None:
    bundle = load_case_bundle(_write_style_bundle(tmp_path, _spec()))

    assert bundle.visual_style_spec is not None
    assert bundle.visual_style_spec["scene_type"] == "billiards"
    assert "visual_style_spec" in bundle.artifact_records


def test_case_bundle_rejects_an_untrusted_style_spec(tmp_path: Path) -> None:
    root = _write_style_bundle(tmp_path, _spec(source="diagnostic"))

    with pytest.raises(CaseBundleError, match="official visual style spec source"):
        load_case_bundle(root)


def test_case_bundle_rejects_style_spec_hash_drift(tmp_path: Path) -> None:
    root = _write_style_bundle(tmp_path, _spec())
    spec_path = root / "visual_style_spec.json"
    spec = read_json(spec_path)
    spec["directives"][0]["statement"] = "Tampered directive."
    write_json(spec_path, spec)

    with pytest.raises(CaseBundleError, match="hash mismatch"):
        load_case_bundle(root)
