from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.evaluator.profile import build_evaluation_plan, resolve_evaluation_profile
from benchmark.utils.io import read_json
from evaluate import run_evaluate


ROOT = Path(__file__).resolve().parents[1]


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "profile_scene",
        "request_id": "profile_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.9,
        "objects": [
            {
                "id": "bed",
                "jid": "bed_asset",
                "category": "bed",
                "description": "blue velvet bed",
                "desc": "blue velvet bed",
                "size": [2.0, 1.6, 0.6],
                "center": [2.5, 2.5, 0.3],
                "rotation": [0, 0, 0],
                "asset_ref": {"source_db": "test", "asset_key": "bed_asset"},
                "asset_proxy": {
                    "type": "obb",
                    "bbox_center_local": [0, 0, 0],
                    "bbox_size": [2.0, 1.6, 0.6],
                },
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


def _spatial_ontology() -> dict:
    return {
        "bed": {
            "category": "bed",
            "count": 200,
            "dimensions": {
                "width_m": {"p5": 1.7, "median": 2.0, "p95": 2.3},
                "depth_m": {"p5": 1.3, "median": 1.6, "p95": 2.0},
                "height_m": {"p5": 0.4, "median": 0.55, "p95": 0.8},
                "n_width": 200,
                "n_depth": 200,
                "n_height": 200,
            },
            "cooccurrence": {},
        }
    }


def _reference_annotation(
    objects: list[tuple[str, str, str]],
    *,
    oar_relations: list[dict] | None = None,
) -> dict:
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": "profile_request",
        "scene_type": "bedroom",
        "inventory_policy": "open_world",
        "objects": [
            {
                "id": object_id,
                "category": category,
                "description": description,
                "count": 1,
                "claim_state": "confirmed",
            }
            for object_id, category, description in objects
        ],
        "oor_relations": [],
        "oar_relations": list(oar_relations or []),
        "room_constraints": {"claim_state": "not_mentioned"},
    }


def test_initial_profile_has_separate_mode_specific_category_2_weights() -> None:
    profile = resolve_evaluation_profile()

    assert profile["status"] == "initial_not_frozen"
    assert profile["weights"] == {
        "prompt_fidelity": 0.25,
        "spatial_fidelity": 0.25,
        "structural_validity": 0.35,
        "visual_quality": 0.40,
    }
    assert profile["structural_validity"]["backend"] == "deterministic_evidence_plus_conditional_vlm"


def test_profile_rejects_invalid_spatial_weight_and_sparse_semantics() -> None:
    invalid_weights = resolve_evaluation_profile()
    invalid_weights["spatial_fidelity"]["metric_weights"] = {
        "scale": 0.4,
        "cooccurrence_plausibility": 0.4,
        "functional_grouping": 0.0,
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        resolve_evaluation_profile(invalid_weights)

    unsafe_sparse_policy = resolve_evaluation_profile()
    unsafe_sparse_policy["spatial_fidelity"]["cooccurrence_plausibility"][
        "sparse_missing_means_unknown"
    ] = False
    with pytest.raises(ValueError, match="must remain true"):
        resolve_evaluation_profile(unsafe_sparse_policy)


def test_granularity_gate_changes_active_category_2_namespace() -> None:
    fine = build_evaluation_plan(
        prompt_granularity="fine_grained",
        has_object_plan=True,
        render_evidence_count=1,
    )
    coarse = build_evaluation_plan(
        prompt_granularity="coarse_grained",
        has_object_plan=False,
        render_evidence_count=1,
        has_spatial_fidelity_ontology=True,
    )

    assert set(fine["weights"]) == {"prompt_fidelity", "structural_validity", "visual_quality"}
    assert set(coarse["weights"]) == {"spatial_fidelity", "structural_validity", "visual_quality"}
    assert set(fine["categories"]) == set(fine["weights"])
    assert set(coarse["categories"]) == set(coarse["weights"])
    assert fine["evaluation_mode"] == "fine_grained_mode"
    assert coarse["evaluation_mode"] == "coarse_grained_mode"
    assert fine["categories"]["prompt_fidelity"]["vlm_policy"] == "fallback"
    assert fine["categories"]["prompt_fidelity"]["backend"] == "structured_claims"
    assert coarse["categories"]["spatial_fidelity"]["vlm_policy"] == "fallback"
    assert coarse["categories"]["spatial_fidelity"]["backend"] == "sceneonto_statistics_plus_conditional_vlm"


def test_public_generator_structure_cannot_change_metric_granularity(tmp_path: Path) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "public_structure_granularity.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a cozy bedroom.",
            "prompt_granularity": "coarse_grained",
        },
        object_plan={
            "request_id": "profile_request",
            "scene_type": "bedroom",
            "scene_description": "public method input",
            "prompt_granularity": "fine_grained",
            "objects": [],
            "global_constraints": [],
            "relations": [],
        },
    )

    assert report["prompt_granularity"] == "coarse_grained"
    assert report["evaluation_config"]["public_generator_structure_used_as_scoring_reference"] is False


def test_low_level_gate_records_missing_default_and_rejects_invalid_mode(tmp_path: Path) -> None:
    diagnostic = run_evaluate(
        scene=_scene(),
        out=tmp_path / "diagnostic_default.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a bedroom.",
        },
    )
    assert diagnostic["prompt_granularity"] == "fine_grained"
    assert diagnostic["evaluation_mode"] == "fine_grained_mode"
    assert diagnostic["evaluation_plan"]["gate"]["resolution_source"] == (
        "diagnostic_default"
    )

    with pytest.raises(ValueError, match="scene_request.prompt_granularity"):
        run_evaluate(
            scene=_scene(),
            out=tmp_path / "invalid_mode.json",
            scene_request={
                "request_id": "profile_request",
                "instruction": "Create a bedroom.",
                "prompt_granularity": "auto",
            },
        )


def test_coarse_spatial_fidelity_uses_statistics_and_not_prompt_fidelity_vlm(tmp_path) -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"score": 0.6}

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "evaluation_report.json",
        eval_oor=True,
        eval_oar=True,
        eval_generic_validity=True,
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a cozy bedroom.",
            "prompt_granularity": "coarse_grained",
        },
        object_plan=None,
        reference_annotation=_reference_annotation(
            [("bed", "bed", "blue velvet bed")],
            oar_relations=[
                {
                    "subject_id": "bed",
                    "type": "room_center",
                    "architectural_element": "center_region",
                    "claim_state": "confirmed",
                }
            ],
        ),
        render_evidence=["standardized_top.png", "standardized_perspective.png"],
        vlm_judge=judge,
        spatial_fidelity_ontology=_spatial_ontology(),
    )

    structural_score = report["category_reports"]["structural_validity"]["score"]
    assert "overall_score" not in report
    Draft202012Validator(read_json(ROOT / "schemas" / "evaluation_report.schema.json")).validate(report)
    assert report["evaluation_mode"] == "coarse_grained_mode"
    assert "prompt_fidelity" not in report["category_reports"]
    assert not ({"object_mapping", "object_alignment", "oor", "oar"} & set(report["reports"]))
    assert report["category_reports"]["spatial_fidelity"]["score"] == 1.0
    assert report["category_reports"]["visual_quality"]["score"] == 0.6
    assert report["benchmark_score"] == pytest.approx(0.25 * 1.0 + 0.35 * structural_score + 0.40 * 0.6)
    assert report["coverage"] == {"covered_weight": 1.0, "required_weight": 1.0, "complete": True}
    assert len(calls) == 1
    assert calls[0]["category"] == "visual_quality"
    assert calls[0]["prompt"] is None
    assert all("object_plan" not in call for call in calls)


def test_shared_metrics_are_identical_when_only_prompt_mode_changes(tmp_path: Path) -> None:
    class SharedMetricJudge:
        def __init__(self) -> None:
            self.p0b_requests: list[dict] = []
            self.visual_requests: list[dict] = []
            self.events: list[str] = []

        def adjudicate_p0b(self, request: dict) -> dict:
            self.p0b_requests.append(request)
            self.events.append("p0b")
            return {
                "verdict": "valid",
                "confidence": 1.0,
                "reason": "same shared-metric fixture",
            }

        def evaluate(self, request: dict) -> dict:
            self.visual_requests.append(request)
            self.events.append(str(request.get("category")))
            return {
                "applicable": True,
                "score": 0.7,
                "confidence": 1.0,
                "summary": "same shared-metric fixture",
            }

    scene = _scene()
    scene["objects"][0]["center"][2] = 1.0
    annotation = _reference_annotation(
        [("bed", "bed", "blue velvet bed")],
        oar_relations=[
            {
                "subject_id": "bed",
                "type": "room_center",
                "architectural_element": "center_region",
                "claim_state": "confirmed",
            }
        ],
    )
    common = {
        "scene": scene,
        "eval_oar": True,
        "eval_generic_validity": True,
        "reference_annotation": annotation,
        "render_evidence": ["standardized_top.png", "standardized_perspective.png"],
        "spatial_fidelity_ontology": _spatial_ontology(),
    }
    fine_judge = SharedMetricJudge()
    coarse_judge = SharedMetricJudge()
    fine = run_evaluate(
        **common,
        out=tmp_path / "fine_shared_metrics.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a cozy bedroom with a blue bed in the center.",
            "prompt_granularity": "fine_grained",
        },
        vlm_judge=fine_judge,
    )
    coarse = run_evaluate(
        **common,
        out=tmp_path / "coarse_shared_metrics.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a cozy bedroom with a blue bed in the center.",
            "prompt_granularity": "coarse_grained",
        },
        vlm_judge=coarse_judge,
    )

    for category in ("structural_validity", "visual_quality"):
        assert fine["category_reports"][category] == coarse["category_reports"][category]
    assert fine_judge.p0b_requests == coarse_judge.p0b_requests
    assert fine_judge.visual_requests == coarse_judge.visual_requests
    assert fine_judge.events[:2] == coarse_judge.events[:2] == [
        "p0b",
        "visual_quality",
    ]
    assert fine_judge.p0b_requests[0]["extracted_relationships"] == []
    assert set(fine_judge.visual_requests[0]["deterministic_evidence"]) == {
        "generic_validity"
    }
    assert fine["evaluation_config"]["shared_metric_invariant"] == coarse[
        "evaluation_config"
    ]["shared_metric_invariant"]
    assert set(fine["category_reports"]) == {
        "prompt_fidelity",
        "structural_validity",
        "visual_quality",
    }
    assert set(coarse["category_reports"]) == {
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    }


def test_missing_vlm_evidence_is_coverage_not_zero(tmp_path) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "evaluation_report.json",
        eval_generic_validity=True,
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a cozy bedroom.",
            "prompt_granularity": "coarse_grained",
        },
    )

    assert report["benchmark_score"] is None
    assert report["benchmark_score_status"] == "insufficient_metric_coverage"
    assert report["coverage"]["covered_weight"] == 0.35
    assert report["category_reports"]["spatial_fidelity"]["score"] is None
    assert report["evaluation_plan"]["categories"]["spatial_fidelity"]["missing_evidence"] == [
        "spatial_fidelity_ontology"
    ]
    assert report["category_reports"]["visual_quality"]["reason"] == "missing_standardized_renders"


def test_zero_weight_category_is_excluded_from_required_coverage(tmp_path) -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"score": 0.9}

    profile = resolve_evaluation_profile()
    profile["weights"] = {
        "prompt_fidelity": 0.4,
        "structural_validity": 0.6,
        "visual_quality": 0.0,
    }
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "zero_visual_weight.json",
        eval_generic_validity=True,
        scene_request={
            "request_id": "profile_request",
            "instruction": "Place one blue bed in the room.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=_reference_annotation(
            [("bed", "bed", "blue bed")],
        ),
        render_evidence=["standardized_top.png", "standardized_perspective.png"],
        vlm_judge=judge,
        evaluation_profile=profile,
    )

    assert report["category_reports"]["visual_quality"] == {
        "status": "not_applicable",
        "score": None,
        "reason": "frozen_zero_weight",
        "category": "visual_quality",
        "vlm_policy": "never",
    }
    assert report["benchmark_score_status"] == "complete"
    assert report["coverage"] == {"covered_weight": 1.0, "required_weight": 1.0, "complete": True}
    assert calls == []


def test_fine_grained_known_relation_uses_deterministic_handler_not_broad_vlm(tmp_path) -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"score": 0.6}

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "evaluation_report.json",
        eval_oar=True,
        eval_generic_validity=True,
        scene_request={
            "request_id": "profile_request",
            "instruction": "Place one blue bed in the center of the room.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=_reference_annotation(
            [("bed", "bed", "blue bed")],
            oar_relations=[
                {
                    "subject_id": "bed",
                    "type": "room_center",
                    "architectural_element": "center_region",
                    "claim_state": "confirmed",
                }
            ],
        ),
        render_evidence=["standardized_top.png", "standardized_perspective.png"],
        vlm_judge=judge,
    )

    fidelity = report["category_reports"]["prompt_fidelity"]
    assert fidelity["score"] == 1.0
    assert fidelity["vlm_policy"] == "fallback"
    assert fidelity["backend"] == "frozen_relation_registry_plus_unknown_relation_vlm"
    assert fidelity["active_relation_families"] == ["oar"]
    assert fidelity["alignment_affects_score"] is False
    assert fidelity["alignment_diagnostics"]["object_mapping"]["summary"]["resolved_match_count"] == 1
    assert fidelity["structured_diagnostics"]["oar"]["score"] == 1.0
    assert report["benchmark_score_status"] == "complete"
    assert [call["category"] for call in calls] == ["visual_quality"]
    assert "object_mapping" not in (calls[0]["deterministic_evidence"] or {})


def test_explicitly_missing_object_creates_a_fidelity_penalty_without_vlm_claim(tmp_path) -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"score": 0.75}

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "evaluation_report.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Place one chair in the room.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=_reference_annotation(
            [("chair", "chair", "wooden chair")],
        ),
        render_evidence=["standardized_top.png"],
        vlm_judge=judge,
    )

    mapping = report["reports"]["object_mapping"]
    fidelity = report["category_reports"]["prompt_fidelity"]
    assert mapping["summary"]["resolved_match_count"] == 0
    assert mapping["score"] is None
    assert mapping["summary"]["missing_reference_count"] == 1
    assert fidelity["score"] == 0.0
    assert fidelity["reason"] is None
    assert fidelity["resolved_object_claim_count"] == 1
    assert fidelity["alignment_affects_score"] is False
    assert [call["category"] for call in calls] == ["visual_quality"]
    assert "object_mapping" not in (calls[0]["deterministic_evidence"] or {})


def test_frozen_metric_applicability_controls_support_without_runtime_override(tmp_path) -> None:
    profile = resolve_evaluation_profile()
    profile["structural_validity"]["applicability"]["support"] = False

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "support_disabled_by_case.json",
        eval_generic_validity=True,
        evaluation_profile=profile,
        support_enabled=None,
    )

    generic = report["reports"]["generic_validity"]
    assert generic["metrics"]["support"]["status"] == "not_applicable"
    assert generic["metrics"]["support"]["reason"] == "disabled_by_configuration"
    assert "support" in generic["disabled_metrics"]
    assert report["protocol_scope"] == "diagnostic_evaluation_api"
    assert report["official_submission"] is False


def test_fine_grained_unknown_relation_routes_prompt_claim_and_renders_to_vlm(tmp_path) -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        if request["category"] == "relationship_fidelity_adjudication":
            return {"verdict": "valid", "confidence": 0.9, "reason": "visible in the north wall view"}
        return {"score": 0.6}

    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "unknown_relation.json",
        eval_oar=True,
        eval_generic_validity=True,
        scene_request={
            "request_id": "profile_request",
            "instruction": "Place the blue bed beneath the north window.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=_reference_annotation(
            [("bed", "bed", "blue bed")],
            oar_relations=[
                {
                    "subject_id": "bed",
                    "type": "under_window",
                    "architectural_element": "north_window",
                    "raw_relation": "beneath the north window",
                    "claim_state": "confirmed",
                }
            ],
        ),
        render_evidence=["standardized_top.png", "standardized_perspective.png"],
        vlm_judge=judge,
    )

    fidelity = report["category_reports"]["prompt_fidelity"]
    relation_check = report["reports"]["oar"]["checks"][0]
    assert fidelity["score"] == 1.0
    assert relation_check["backend"] == "vlm"
    relation_call = next(
        call for call in calls if call["category"] == "relationship_fidelity_adjudication"
    )
    assert relation_call["natural_language_prompt"] == "Place the blue bed beneath the north window."
    assert relation_call["relation"]["raw_relation"] == "beneath the north window"
    assert relation_call["render_evidence"] == [
        "standardized_top.png",
        "standardized_perspective.png",
    ]
    assert [call["category"] for call in calls] == [
        "visual_quality",
        "relationship_fidelity_adjudication",
    ]
