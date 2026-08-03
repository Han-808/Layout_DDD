from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.evaluator import (
    resolve_asset_policy,
    scene_quality_applicability,
    validate_asset_policy,
)
from benchmark.evaluator.asset_policy import AssetPolicyError
from evaluate import run_evaluate


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "ap_scene",
        "request_id": "ap_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.9,
        "objects": [
            {
                "id": "bed",
                "category": "bed",
                "description": "bed",
                "desc": "bed",
                "size": [2.0, 1.6, 0.6],
                "center": [2.5, 2.5, 0.3],
                "rotation": [0, 0, 0],
                "asset_ref": {"source_db": "test", "asset_key": "bed_asset"},
                "asset_proxy": {"type": "obb", "bbox_center_local": [0, 0, 0], "bbox_size": [2.0, 1.6, 0.6]},
                "metadata": {"interactive": False},
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


def test_validation_defaults_owners_to_benchmark() -> None:
    policy = validate_asset_policy({"mode": "retrieval_allowed"})
    assert policy["mode"] == "retrieval_allowed"
    assert policy["identity_owner"] == "benchmark"
    assert policy["scale_owner"] == "benchmark"


def test_invalid_enum_values_fail_clearly() -> None:
    with pytest.raises(AssetPolicyError, match="mode"):
        validate_asset_policy({"mode": "teleported_assets"})
    with pytest.raises(AssetPolicyError, match="scale_owner"):
        validate_asset_policy({"mode": "benchmark_provided", "scale_owner": "vendor"})
    with pytest.raises(AssetPolicyError, match="must be a JSON object"):
        validate_asset_policy("benchmark_provided")


def test_missing_policy_is_none_backward_compatible() -> None:
    assert resolve_asset_policy(None) is None


def test_applicability_is_declarative_only() -> None:
    generated = scene_quality_applicability(
        validate_asset_policy(
            {
                "mode": "generated_or_open_assets",
                "scale_owner": "generator",
                "category_selection_owner": "generator",
                "appearance_owner": "generator",
                "arrangement_owner": "generator",
            }
        )
    )
    for metric in (
        "scale_consistency",
        "object_pairing_consistency",
        "style_consistency",
        "semantic_placement_consistency",
    ):
        assert generated[metric]["applicability"] == "relevant"
        assert generated[metric]["decision_role"] == "applicability_only"
        assert generated[metric]["workflow"] == "canonical_l0_l4"
        assert "implemented" not in generated[metric]
        assert "affects_score" not in generated[metric]

    benchmark_owned = scene_quality_applicability(validate_asset_policy({"mode": "benchmark_provided"}))
    assert benchmark_owned["scale_consistency"]["applicability"] == "not_relevant"

    # No policy declared -> pending, never a pass/fail.
    pending = scene_quality_applicability(None)
    assert pending["style_consistency"]["applicability"] == "pending"


def test_arrangement_ownership_does_not_activate_category_only_pairing() -> None:
    applicability = scene_quality_applicability(
        validate_asset_policy(
            {
                "mode": "benchmark_provided",
                "arrangement_owner": "generator",
                "category_selection_owner": "benchmark",
            }
        )
    )
    assert applicability["object_pairing_consistency"] == {
        "applicability": "not_relevant",
        "basis": [],
        "decision_role": "applicability_only",
        "workflow": "canonical_l0_l4",
    }
    assert applicability["scale_consistency"]["applicability"] == "relevant"
    assert (
        applicability["semantic_placement_consistency"]["applicability"]
        == "relevant"
    )


@pytest.mark.parametrize("granularity", ["fine_grained", "coarse_grained"])
@pytest.mark.parametrize("mode", ["benchmark_provided", "generated_or_open_assets"])
def test_all_granularity_asset_mode_combinations_are_legal(tmp_path: Path, granularity: str, mode: str) -> None:
    def judge(request: dict) -> dict:
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.8,
            "reason": "No significant defect.",
            "missing_evidence": [],
            "defects": [],
        }

    image = tmp_path / f"{granularity}_{mode}.png"
    image.write_bytes(b"test-image")

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / f"{granularity}_{mode}.json",
        eval_generic_validity=True,
        render_evidence=[str(image)],
        vlm_judge=judge,
        scene_request={
            "request_id": "ap_request",
            "instruction": "A bedroom.",
            "prompt_granularity": granularity,
            "asset_policy": {"mode": mode},
        },
    )
    assert report["evaluation_config"]["asset_policy"]["mode"] == mode
    # Prompt granularity remains metadata and is independent of asset policy.
    assert report["prompt_granularity"] == granularity


def test_absent_asset_policy_keeps_report_backward_compatible(tmp_path: Path) -> None:
    def judge(request: dict) -> dict:
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.8,
            "reason": "No significant defect.",
            "missing_evidence": [],
            "defects": [],
        }

    image = tmp_path / "global.png"
    image.write_bytes(b"test-image")

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "no_policy.json",
        eval_generic_validity=True,
        render_evidence=[str(image)],
        vlm_judge=judge,
        scene_request={"request_id": "ap_request", "instruction": "A bedroom.", "prompt_granularity": "fine_grained"},
    )
    assert report["evaluation_config"]["asset_policy"] is None
    assert all(
        entry["applicability"] == "pending"
        for entry in report["evaluation_config"]["metric_applicability"][
            "l3_scene_quality"
        ].values()
    )
