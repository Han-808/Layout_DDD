from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
from PIL import Image

from benchmark.evaluator.scene_quality import (
    PERCEPTUAL_VISUAL_QUALITY_METRICS,
    SCENE_QUALITY_INTERFACE_METRICS,
    SEMANTIC_PLACEMENT_METRICS,
    SEMANTIC_COHERENCE_METRICS,
    evaluate_scene_quality_interfaces,
    resolve_scene_quality_config,
    validate_authorized_deviations,
)
from benchmark.visual_judge import CameraEvidenceProvider
from benchmark.visual_judge.l3_prompts import (
    L3_METRIC_BOUNDARY_RULES,
    L3_METRIC_PHASE_PROMPTS,
    L3_METRIC_PROMPT_VERSION,
    L3_METRIC_RUBRICS,
)
from benchmark.evaluator.scene_quality.authorized_deviations import (
    AuthorizedDeviationError,
    deviation_matches,
)
from benchmark.evaluator.scene_quality.interfaces import (
    SCENE_QUALITY_INTERFACE_NAMESPACE,
    SceneQualityInterfaceConfigError,
    _apply_prompt_exemptions,
    _resolved_functional_ownership_for_placement,
)
from benchmark.evaluator.scene_quality.global_group_first import (
    _aggregate_global_and_group_results,
    _apply_functional_acquisition_budget_status,
    _evaluate_global_scope,
    _functional_probe_budget,
    _registered_placement_checks_from_controller_audit,
    _relation_episode_defect_violations,
)


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "sq_scene",
        "request_id": "sq_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.9,
        "objects": [
            {
                "id": "chair_01",
                "category": "chair",
                "description": "wooden chair",
                "size": [0.5, 0.5, 0.9],
                "center": [1.0, 1.0, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "desk_01",
                "category": "desk",
                "description": "wooden desk",
                "size": [1.2, 0.6, 0.75],
                "center": [1.8, 1.0, 0.375],
                "rotation": [0, 0, 0],
            },
        ],
    }


def _grouping_report() -> dict:
    return {
        "object_groups": [
            {
                "group_id": "group_001",
                "object_ids": ["chair_01", "desk_01"],
            }
        ]
    }


def _only(metric: str) -> dict:
    config = {
        "enabled": True,
        "metrics": {
            name: {"enabled": name == metric}
            for name in SCENE_QUALITY_INTERFACE_METRICS
        },
    }
    # Most tests below isolate the final visual-Judge contract. Keep that
    # legacy/direct route explicit; JSON-first routing has dedicated tests.
    if metric in {
        "scale_consistency",
        "object_pairing_consistency",
    }:
        config["metrics"][metric]["evidence_plan"] = {
            "evidence_strategy": "global_and_local",
            "router_options": None,
        }
    return config


def _json_first_only(metric: str) -> dict:
    config = _only(metric)
    config["metrics"][metric].pop("evidence_plan", None)
    return config


def _relevant(*metrics: str) -> dict:
    return {
        metric: {"applicability": "relevant", "basis": ["test"]}
        for metric in metrics
    }


def _images(tmp_path, *names: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in names:
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"test-image")
        paths[name] = str(path)
    return paths


def _valid(request: dict | None = None) -> dict:
    result = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.9,
        "reason": "No significant in-scope defect.",
        "missing_evidence": [],
        "defects": [],
    }
    return _with_required_placement_checks(
        request,
        _with_required_functional_checks(request, result),
    )


def _with_required_placement_checks(
    request: dict | None,
    response: dict,
) -> dict:
    result = deepcopy(response)
    checks = (
        request.get("required_placement_checks")
        if isinstance(request, dict)
        and isinstance(request.get("required_placement_checks"), list)
        else []
    )
    if not checks:
        return result
    result["placement_check_results"] = [
        {
            "check_id": str(check.get("check_id") or ""),
            "subject_id": str(check.get("subject_id") or ""),
            "context_ids": [
                str(item) for item in check.get("context_ids") or []
            ],
            "observation_status": "observed",
            "conclusion": "valid",
            "reason": (
                "The fixture explicitly acknowledges this required "
                "placement check."
            ),
        }
        for check in checks
    ]
    return result


def _with_required_functional_checks(
    request: dict | None,
    response: dict,
) -> dict:
    result = deepcopy(response)
    checks = (
        request.get("required_functional_checks")
        if isinstance(request, dict)
        and isinstance(request.get("required_functional_checks"), list)
        else []
    )
    if not checks:
        return result
    invalid_targets = [
        {
            str(item)
            for item in defect.get("target_ids") or []
            if str(item).strip()
        }
        for defect in result.get("defects") or []
        if isinstance(defect, dict)
    ]
    rows = []
    for index, check in enumerate(checks):
        target_ids = [str(item) for item in check.get("target_ids") or []]
        invalid = any(
            defect_targets
            and defect_targets <= set(target_ids)
            for defect_targets in invalid_targets
        )
        if (
            result.get("verdict") == "invalid"
            and not invalid_targets
            and index == 0
        ):
            invalid = True
        unresolved = result.get("verdict") == "ambiguous"
        rows.append(
            {
                "check_id": str(check.get("check_id") or ""),
                "target_ids": target_ids,
                "observation_status": (
                    "missing" if unresolved else "observed"
                ),
                "conclusion": (
                    "unresolved"
                    if unresolved
                    else "invalid"
                    if invalid
                    else "valid"
                ),
                "reason": (
                    "The fixture explicitly acknowledges this required check."
                ),
            }
        )
    result["functional_check_results"] = rows
    for defect in result.get("defects") or []:
        if not isinstance(defect, dict):
            continue
        defect_targets = {
            str(item)
            for item in defect.get("target_ids") or []
            if str(item).strip()
        }
        refs = [
            row["check_id"]
            for row in rows
            if row["conclusion"] == "invalid"
            and defect_targets
            and defect_targets <= set(row["target_ids"])
        ]
        if refs:
            defect["check_refs"] = refs
    return result


def _style_needs_local(target_id: str = "chair_01") -> dict:
    return {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.5,
        "reason": "The possible style outlier needs a closer local view.",
        "missing_evidence": ["group_context_visible"],
        "defects": [],
        "evidence_request": {
            "target_ids": [target_id],
            "missing_observations": ["group_context_visible"],
            "view_goal": (
                "show the possible style outlier with its local ensemble"
            ),
            "metadata": {},
        },
    }


def _style_global_then_local_config() -> dict:
    config = _only("style_consistency")
    config["metrics"]["style_consistency"].update(
        {
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_oblique",
                "selector": "deterministic",
                "image_budget": 1,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "global_screen_then_local",
                "global_policy": {
                    "view_family": "canonical_overview_perspective",
                    "image_budget": 1,
                    "top_down": False,
                    "perspective_diversity_required": False,
                },
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": "vlm_visual_evidence_scope_v2",
                    "image_budget": 1,
                    "global_context_image_budget": 1,
                    "max_packet_images": 2,
                    "image_order": [
                        "global_context",
                        "group_local",
                    ],
                    "minimum_group_members": 2,
                    "force_for_eligible_groups": False,
                },
                "router_options": None,
                "text_context": [
                    "original_prompt",
                    "authorized_deviations",
                    "asset_policy",
                ],
            },
        }
    )
    return config


def test_l3_metric_prompts_have_versioned_generic_boundaries() -> None:
    assert L3_METRIC_PROMPT_VERSION == (
        "l3_lazy_group_relation_evidence_v23"
    )
    assert set(L3_METRIC_RUBRICS) == {
        "scale_consistency",
        "style_consistency",
        "object_pairing_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    }
    assert any(
        "named observable fact is missing"
        in rule
        for rule in L3_METRIC_BOUNDARY_RULES
    )
    assert any(
        "not universal validity thresholds"
        in rule
        for rule in L3_METRIC_BOUNDARY_RULES
    )
    assert any(
        "current authored scene" in rule
        for rule in L3_METRIC_BOUNDARY_RULES
    )
    assert "relocation test" in (
        L3_METRIC_RUBRICS["object_pairing_consistency"]
    )
    assert "ordinary real-world use" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "directional_correspondence" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "relative_use_geometry" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "Resolve every listed check explicitly" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "logical room boundary" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "visible geometry and affordances" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "standalone defect threshold" in (
        L3_METRIC_RUBRICS["functional_consistency"]
    )
    assert "relocation-only test" in (
        L3_METRIC_RUBRICS["semantic_placement_consistency"]
    )


def test_functional_acquisition_budget_is_neutral_coverage_metadata() -> None:
    valid = _apply_functional_acquisition_budget_status(
        {
            "status": "evaluated",
            "score": 1.0,
            "judgement": {"verdict": "valid"},
            "functional_probe_acquisition": {
                "budget_exhausted": True
            },
        },
        metric_name="functional_consistency",
    )
    assert valid["status"] == "evaluated"
    assert valid["score"] == 1.0
    assert valid["judgement"]["verdict"] == "valid"
    assert valid["functional_acquisition_coverage_complete"] is False

    invalid = _apply_functional_acquisition_budget_status(
        {
            "status": "evaluated",
            "score": 0.0,
            "judgement": {"verdict": "invalid"},
            "functional_probe_acquisition": {
                "budget_exhausted": True
            },
        },
        metric_name="functional_consistency",
    )
    assert invalid["status"] == "evaluated"
    assert invalid["score"] == 0.0
    assert invalid["functional_acquisition_coverage_complete"] is False
    assert "Apparent image size alone" in (
        L3_METRIC_RUBRICS["scale_consistency"]
    )
    assert "multiple coherent visual languages" in (
        L3_METRIC_RUBRICS["style_consistency"]
    )
    functional_global = L3_METRIC_PHASE_PROMPTS[
        "functional_consistency"
    ]["global_discovery"]
    functional_local = L3_METRIC_PHASE_PROMPTS[
        "functional_consistency"
    ]["group_local_review"]
    functional_relation = L3_METRIC_PHASE_PROMPTS[
        "functional_consistency"
    ]["cross_group_relation_review"]
    assert "overall scene-level" in functional_global
    assert "Do not decide discovered cross-group" in functional_global
    assert "exactly the supplied cross-group" in functional_relation
    assert "non-empty subset" in functional_relation
    assert "within-group direct-use relations" in functional_local
    assert "architecture and zone context" in functional_local
    assert "dedicated relation phase owns it" in functional_local
    assert "surface hypothesis as established fact" in functional_local


def test_cross_group_relation_defect_attribution_is_object_level() -> None:
    assert _relation_episode_defect_violations(
        [
            {
                "target_ids": ["sofa"],
                "scope": "orientation_for_use",
                "relation": "faces away from television",
            }
        ],
        required_target_ids=["sofa", "television"],
    ) == []
    assert _relation_episode_defect_violations(
        [
            {
                "target_ids": ["lamp"],
                "scope": "orientation_for_use",
                "relation": "unrelated object",
            }
        ],
        required_target_ids=["sofa", "television"],
    ) == [["lamp"]]

    placement_global = L3_METRIC_PHASE_PROMPTS[
        "semantic_placement_consistency"
    ]["global_discovery"]
    placement_local = L3_METRIC_PHASE_PROMPTS[
        "semantic_placement_consistency"
    ]["group_local_review"]
    assert "scene zones, architecture" in placement_global
    assert "contextual anchors" in placement_global
    assert "only potential defect owner" in placement_global
    assert "support-surface meaning" in placement_local
    assert "Context IDs are evidence context" in placement_local
    assert "orientation or operability" in placement_local


def test_functional_probe_budget_does_not_propagate_episode_cap_metric_wide() -> None:
    class Control:
        max_total_images = 6

    class Judge:
        control = Control()

    class Provider:
        functional_probe_judge_artifacts_per_selected_view = 1
        functional_probe_full_artifacts_per_selected_view = 2

    assert _functional_probe_budget(
        {
            "prejudgement_probe_policy": {
                "max_probe_units": 8,
            }
        },
        judge=Judge(),
        global_image_count=1,
        provider=Provider(),
    ) == 8


def test_default_is_canonical_l3_and_policy_remains_overridable() -> None:
    resolved = resolve_scene_quality_config()
    assert resolved["enabled"] is True
    assert resolved["implemented"] is True
    assert all(
        resolved["metrics"][name]["implemented"] is True
        for name in SCENE_QUALITY_INTERFACE_METRICS
    )
    assert set(SEMANTIC_COHERENCE_METRICS) == {
        "scale_consistency",
        "object_pairing_consistency",
    }
    assert set(PERCEPTUAL_VISUAL_QUALITY_METRICS) == {"style_consistency"}
    functional = resolved["metrics"]["functional_consistency"]
    assert functional["enabled"] is True
    assert functional["metric_status"] == "canonical_scoring"
    assert functional["activation_policy"] == "profile_and_applicability"
    assert functional["included_in_canonical_aggregate"] is True
    placement = resolved["metrics"]["semantic_placement_consistency"]
    assert placement["enabled"] is True
    assert placement["metric_status"] == "canonical_scoring"
    assert placement["activation_policy"] == "profile_and_applicability"
    assert placement["included_in_canonical_aggregate"] is True
    assert all(
        resolved["metrics"][name]["weight"] == pytest.approx(0.2)
        for name in SCENE_QUALITY_INTERFACE_METRICS
    )
    assert set(SEMANTIC_PLACEMENT_METRICS) == {
        "semantic_placement_consistency"
    }
    serialized = json.dumps(resolved, sort_keys=True)
    assert "object_coexistence_consistency" not in serialized
    assert "sceneonto_scale_candidate_router" not in serialized
    assert "benchmark.evaluator.spatial_fidelity.scale" not in serialized
    scale = resolved["metrics"]["scale_consistency"]
    assert scale["evidence_policy"]["camera_scope"] == "group_local"
    assert scale["evidence_policy"]["image_budget"] == 3
    assert scale["evidence_policy"]["global_image_budget"] == 1
    assert scale["evidence_policy"]["scoped_image_budget"] == 1
    assert scale["evidence_plan"]["evidence_strategy"] == (
        "json_screen_then_visual"
    )
    assert scale["evidence_plan"]["local_policy"][
        "grouping_policy_id"
    ] == "vlm_visual_evidence_scope_v2"
    pairing = resolved["metrics"]["object_pairing_consistency"]
    assert pairing["evidence_plan"]["evidence_strategy"] == (
        "json_screen_then_visual"
    )
    assert pairing["evidence_policy"]["image_order"] == [
        "global_context",
        "group_local",
    ]
    style = resolved["metrics"]["style_consistency"]
    assert style["evidence_policy"]["image_budget"] == 1
    assert style["evidence_plan"]["local_policy"]["image_budget"] == 1
    assert style["evidence_plan"]["local_policy"]["image_order"] == [
        "global_context",
        "group_local",
    ]
    for name in (
        "functional_consistency",
        "semantic_placement_consistency",
    ):
        metric = resolved["metrics"][name]
        policy = metric["evidence_policy"]
        assert policy["camera_scope"] == "global"
        assert policy["camera_mode"] == "global_oblique"
        assert policy["image_budget"] == 1
        assert policy["image_order"] is None
        assert metric["evidence_plan"]["evidence_strategy"] == (
            "global_discovery_then_group_local"
        )
        assert metric["evidence_plan"]["global_policy"] == {
            "view_family": "canonical_overview_perspective",
            "image_budget": 1,
            "top_down": False,
            "perspective_diversity_required": False,
        }
        assert metric["evidence_plan"]["local_policy"][
            "minimum_group_members"
        ] == 2
        assert metric["evidence_plan"]["local_policy"][
            "force_for_eligible_groups"
        ] is True
        assert metric["evidence_plan"]["local_policy"][
            "global_context_image_budget"
        ] == 1
        assert metric["evidence_plan"]["local_policy"][
            "max_packet_images"
        ] == 2
        assert metric["evidence_plan"]["local_policy"]["image_order"] == [
            "global_context",
            "group_local",
        ]
    assert style["evidence_plan"]["evidence_strategy"] == (
        "global_screen_then_local"
    )
    assert style["evidence_plan"]["local_policy"][
        "force_for_eligible_groups"
    ] is False
    probe_policy = functional["evidence_plan"][
        "prejudgement_probe_policy"
    ]
    assert probe_policy["enabled"] is True
    assert probe_policy["max_probe_units"] == 6
    assert probe_policy["candidate_count_by_probe_kind"] == {
        "functional_frontage": 4,
        "functional_correspondence": 4,
        "approach_clearance": 4,
    }
    assert probe_policy["preferred_lens_mm"] == 32.0
    assert probe_policy["judge_presentation"] == "raw_rgb_only"
    assert probe_policy["usable_surface"]["decode_scope"] == (
        "directed_or_uncertain_clearance_targets_before_probe_budget"
    )
    assert "decode_only_scheduled_targets" not in (
        probe_policy["usable_surface"]
    )

    overridden = resolve_scene_quality_config(
        {
            "metrics": {
                "style_consistency": {
                    "evidence_policy": {
                        "camera_mode": "metric_local",
                        "image_budget": 3,
                        "presentation": "highlight",
                        "include_global_context": False,
                        "image_order": ["global_context", "metric_local"],
                    }
                }
            }
        }
    )
    policy = overridden["metrics"]["style_consistency"]["evidence_policy"]
    assert policy["camera_mode"] == "metric_local"
    assert policy["image_budget"] == 3
    assert policy["presentation"] == "highlight"

    historical = resolve_scene_quality_config(
        profile={
            "profile_version": "canonical_scene_evaluation_v1",
            "l3_scene_quality": {
                "metrics": {
                    "style_consistency": {
                        "enabled": True,
                        "weight": 1.0 / 3.0,
                    },
                    "scale_consistency": {
                        "enabled": True,
                        "weight": 1.0 / 3.0,
                    },
                    "object_pairing_consistency": {
                        "enabled": True,
                        "weight": 1.0 / 3.0,
                    },
                }
            },
        }
    )
    for name in (
        "functional_consistency",
        "semantic_placement_consistency",
    ):
        assert historical["metrics"][name]["enabled"] is False
        assert (
            historical["metrics"][name]["metric_status"]
            == "historical_profile_excluded"
        )
        assert (
            historical["metrics"][name][
                "included_in_canonical_aggregate"
            ]
            is False
        )


def test_style_global_clear_skips_group_local_review(tmp_path) -> None:
    images = _images(tmp_path, "global_a", "global_b", "local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": images["local"], "role": "group_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global_a"], images["global_b"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["score"] == 1.0
    assert metric["route"] == "global_clear"
    assert metric["judge_call_count"] == 1
    assert metric["local_review"]["requested"] is False
    assert metric["budget_exhaustion_forced_choice"] == {
        "applied": False
    }
    assert len(provider_calls) == 0
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_screen",
    ]
    assert judge_calls[0]["render_evidence"] == [images["global_a"]]


def test_metric_report_exposes_forced_choice_without_reason_parsing(
    tmp_path,
) -> None:
    global_image = _images(tmp_path, "global")["global"]
    forced = {
        "applied": True,
        "trigger": "max_total_images_exhausted",
        "ambiguity_before_forcing": True,
        "pre_force_judge_status": "need_more_evidence",
        "pre_force_evidence_request": {
            "target_ids": ["chair_01"],
        },
        "pre_force_reason": "need local detail",
        "available_image_count": 1,
        "final_verdict": "valid",
        "final_confidence": 0.7,
        "evidence_artifacts": [global_image],
    }
    response = {
        **_valid(),
        "budget_exhaustion_forced_choice": forced,
    }

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [global_image]},
        vlm_judge=lambda request: response,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["budget_exhaustion_forced_choice"]["applied"] is True
    assert metric["budget_exhaustion_forced_choice"]["trigger"] == (
        "max_total_images_exhausted"
    )
    assert metric["budget_exhaustion_forced_choice"][
        "occurrence_count"
    ] == 1


def test_evaluation_schema_accepts_forced_choice_metric_audit() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "evaluation_report.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    metric_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/l3MetricAudit",
    }
    validator = Draft202012Validator(metric_schema)

    validator.validate(
        {
            "budget_exhaustion_forced_choice": {
                "applied": False
            }
        }
    )
    validator.validate(
        {
            "budget_exhaustion_forced_choice": {
                "applied": True,
                "trigger": "max_total_images_exhausted",
                "ambiguity_before_forcing": True,
                "pre_force_judge_status": "need_more_evidence",
                "pre_force_evidence_request": {
                    "target_ids": ["chair_01"]
                },
                "pre_force_reason": "need another view",
                "available_image_count": 2,
                "final_verdict": "valid",
                "final_confidence": 0.8,
                "evidence_artifacts": [
                    "global.png",
                    "local.png",
                ],
            }
        }
    )


def test_style_global_suspicion_requires_local_confirmation(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    judge_calls: list[dict] = []
    provider_calls: list[dict] = []
    confirmed = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.9,
        "reason": "The chair is a conclusive visible style outlier.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "significant_visible_style_incompatibility",
                "target_ids": ["chair_01"],
                "relation": "rendering_style_outlier",
                "reason": "The global view conclusively shows the mismatch.",
            }
        ],
    }

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": images["local"], "role": "group_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return confirmed if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert metric["route"] == "global_screen_then_group_local"
    assert metric["judge_call_count"] == 2
    assert metric["global_screen"]["final_metric_verdict"] is False
    assert len(metric["global_screen_candidate_claims"]) == 1
    assert metric["final_defect_claims"] == []
    assert metric["final_object_findings"] == []
    assert metric["judgement"]["object_penalty_count"] == 0
    assert len(provider_calls) == 1
    assert len(judge_calls) == 2


def test_style_localized_suspicion_reviews_only_implicated_group(
    tmp_path,
) -> None:
    scene = _scene()
    scene["objects"].extend(
        [
            {
                "id": "sofa_01",
                "category": "sofa",
                "size": [2.0, 0.8, 0.9],
                "center": [3.5, 3.5, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "table_01",
                "category": "coffee_table",
                "size": [1.0, 0.6, 0.4],
                "center": [3.5, 2.7, 0.2],
                "rotation": [0, 0, 0],
            },
        ]
    )
    images = _images(tmp_path, "global", "work_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []
    suspicion = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair may be a style outlier.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "significant_visible_style_incompatibility",
                "target_ids": ["chair_01"],
                "relation": "possible_style_outlier",
                "reason": "Confirm locally.",
            }
        ],
    }

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": images["work_local"], "role": "group_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return suspicion if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config=_style_global_then_local_config(),
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "work",
                    "object_ids": ["chair_01", "desk_01"],
                },
                {
                    "group_id": "lounge",
                    "object_ids": ["sofa_01", "table_01"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["score"] == 1.0
    assert metric["selected_group_ids"] == ["work"]
    assert len(provider_calls) == 1
    assert provider_calls[0]["group_scope"]["group_id"] == "work"
    assert len(judge_calls) == 2


def test_style_required_local_confirmation_without_grouping_is_unresolved(
    tmp_path,
) -> None:
    global_image = _images(tmp_path, "global")["global"]
    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _style_needs_local()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        render_evidence={"global": [global_image]},
        camera_evidence_provider=lambda request: pytest.fail(
            "local provider must not run without trusted grouping"
        ),
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "unresolved"
    assert metric["reason"] == (
        "object_grouping_unavailable_for_style_confirmation"
    )
    assert metric["local_review"]["requested"] is True
    assert len(judge_calls) == 1


def test_default_style_flow_uses_one_global_then_global_plus_local(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global_a", "global_b", "local")
    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return (
            _style_needs_local()
            if len(judge_calls) == 1
            else _valid()
        )

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global_a"], images["global_b"]]},
        camera_evidence_provider=lambda request: [
            {"path": images["local"], "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert len(judge_calls) == 2
    assert judge_calls[0]["render_evidence"] == [
        images["global_a"]
    ]
    assert judge_calls[1]["render_evidence"] == [
        images["global_a"],
        images["local"],
    ]
    assert judge_calls[0]["scene_summary"]["objects"]


@pytest.mark.parametrize(
    "metric_name",
    [
        "scale_consistency",
        "object_pairing_consistency",
    ],
)
def test_json_screen_clear_skips_camera_and_visual_confirmation(
    tmp_path,
    metric_name,
) -> None:
    global_image = _images(tmp_path, "global")["global"]
    judge_calls: list[dict] = []

    class Provider:
        def __call__(self, request: dict) -> list[dict]:
            raise AssertionError(
                "clear JSON screen must not request camera evidence"
            )

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_json_first_only(metric_name),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [global_image]},
        camera_evidence_provider=Provider(),
        vlm_judge=judge,
        metric_applicability=_relevant(metric_name),
    )["metrics"][metric_name]

    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert metric["route"] == "json_screen_resolved"
    assert metric["judge_call_count"] == 1
    assert metric["renderer_invoked"] is False
    assert len(judge_calls) == 1
    assert judge_calls[0]["evidence_phase"] == "json_screen"
    assert judge_calls[0]["decision_mode"] == "screen"
    assert judge_calls[0]["render_evidence"] == []
    assert metric["dependencies"]["evidence_source"] == (
        "structured_scene_json"
    )
    assert metric["dependencies"]["evidence_scope_satisfied"] is True
    assert metric["json_screen"]["screen_state"] == "clear"


@pytest.mark.parametrize(
    ("screen_response", "expected_router_state"),
    [
        (
            {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": 0.7,
                "reason": "The chair dimensions look suspicious.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": (
                            "significant_visible_category_relative_"
                            "scale_incoherence"
                        ),
                        "target_ids": ["chair_01"],
                        "relation": "scale_outlier_candidate",
                        "reason": "The JSON dimensions merit confirmation.",
                    }
                ],
            },
            "suspicious",
        ),
        (
            {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "confidence": 0.2,
                "reason": "Relative scale needs visual context.",
                "missing_evidence": ["group_context_visible"],
                "defects": [],
                "evidence_request": {
                    "target_ids": ["chair_01"],
                    "missing_observations": [
                        "group_context_visible"
                    ],
                    "view_goal": (
                        "show the chair with a nearby reference object"
                    ),
                    "metadata": {},
                },
            },
            "insufficient_evidence",
        ),
    ],
)
def test_json_screen_routes_only_suspicious_or_insufficient_to_visual(
    tmp_path,
    screen_response,
    expected_router_state,
) -> None:
    images = _images(
        tmp_path,
        "global",
        "local_a",
        "local_over_budget",
    )
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {
                "path": images["local_a"],
                "role": "group_local",
            },
            {
                "path": images["local_over_budget"],
                "role": "group_local",
            },
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return screen_response if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_json_first_only("scale_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert metric["route"] == "json_screen_then_group_visual"
    assert metric["router_state"] == expected_router_state
    assert metric["judge_call_count"] == 2
    assert len(provider_calls) == 1
    assert [call["evidence_phase"] for call in judge_calls] == [
        "json_screen",
        "visual_confirmation",
    ]
    assert judge_calls[0]["render_evidence"] == []
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["local_a"],
    ]
    assert metric["evidence_request"]["global_image_budget"] == 1
    assert metric["evidence_request"]["scoped_image_budget"] == 1
    assert metric["dependencies"]["evidence_source"] == (
        "per_group_camera_evidence"
    )
    assert metric["dependencies"]["evidence_scope_satisfied"] is True
    if expected_router_state == "suspicious":
        assert metric["json_screen"]["screen_state"] == (
            "material_candidate"
        )
        assert metric["json_screen"]["verdict"] == "candidate"
        assert metric["json_screen"]["defects"] == []
        assert len(metric["json_screen"]["candidate_defects"]) == 1
        assert metric["json_screen"]["final_metric_verdict"] is False
        assert len(metric["routed_candidate_claims"]) == 1
        assert judge_calls[1]["routed_screen_claims"] == (
            metric["routed_candidate_claims"]
        )
        assert metric["group_results"][0][
            "claim_correspondence"
        ][0]["relationship"] == "routed_candidate_not_confirmed"
    else:
        assert metric["json_screen"]["screen_state"] == (
            "review_required"
        )


def test_object_pairing_suspicious_json_screen_routes_to_visual(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    judge_calls: list[dict] = []
    suspicious = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.7,
        "reason": "One group member may not belong in this scene.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "group_member_category_compatibility",
                "target_ids": ["chair_01"],
                "relation": "category_compatibility_candidate",
                "reason": "The JSON category merits visual confirmation.",
            }
        ],
    }

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return suspicious if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_json_first_only("object_pairing_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: [
            {"path": images["local"], "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant(
            "object_pairing_consistency"
        ),
    )["metrics"]["object_pairing_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["router_state"] == "suspicious"
    assert metric["route"] == "json_screen_then_group_visual"
    assert [call["evidence_phase"] for call in judge_calls] == [
        "json_screen",
        "visual_confirmation",
    ]
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["local"],
    ]


def test_json_candidate_confirmation_is_one_final_defect(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    candidate = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.7,
        "reason": "The dimensions identify a candidate.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": (
                    "significant_visible_category_relative_"
                    "scale_incoherence"
                ),
                "target_ids": ["chair_01", "desk_01"],
                "relation": "chair_too_large_for_desk",
                "reason": "The JSON dimensions merit visual confirmation.",
            }
        ],
    }
    confirmed = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.9,
        "reason": "The visual evidence confirms the routed candidate.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": (
                    "significant_visible_category_relative_"
                    "scale_incoherence"
                ),
                "target_ids": ["desk_01", "chair_01"],
                "relation": "chair_too_large_for_desk",
                "reason": "The local view confirms the scale defect.",
            }
        ],
    }
    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return candidate if len(judge_calls) == 1 else confirmed

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_json_first_only("scale_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: [
            {"path": images["local"], "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]

    assert metric["score"] == 0.0
    assert metric["json_screen"]["verdict"] == "candidate"
    assert metric["json_screen"]["defects"] == []
    assert len(metric["judgement"]["defects"]) == 1
    assert len(metric["final_defect_claims"]) == 1
    correspondence = metric["group_results"][0][
        "claim_correspondence"
    ]
    assert correspondence[0]["relationship"] == (
        "confirmed_routed_candidate"
    )
    assert correspondence[0]["routed_candidate_id"] == (
        metric["routed_candidate_claims"][0]["claim_id"]
    )


@pytest.mark.parametrize(
    "metric_name",
    [
        "functional_consistency",
        "semantic_placement_consistency",
    ],
)
def test_function_and_placement_force_every_eligible_group_review(
    tmp_path,
    metric_name,
) -> None:
    images = _images(
        tmp_path,
        "global_perspective",
        "global_top",
        "local",
        "local_over_budget",
    )
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []
    config = {
        "enabled": True,
        "metrics": {
            name: {"enabled": False}
            for name in SCENE_QUALITY_INTERFACE_METRICS
        },
    }
    config["metrics"][metric_name] = {"enabled": True}

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {"path": images["local"], "role": "group_local"},
            {
                "path": images["local_over_budget"],
                "role": "group_local",
            },
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [
                images["global_top"],
                images["global_perspective"],
            ]
        },
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant(metric_name),
    )["metrics"][metric_name]

    assert metric["status"] == "evaluated"
    assert len(provider_calls) == 1
    assert len(judge_calls) == 2
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_discovery",
        "group_local_review",
    ]
    assert judge_calls[0]["render_evidence"] == [
        images["global_perspective"],
    ]
    assert judge_calls[1]["render_evidence"] == [
        images["global_perspective"],
        images["local"],
    ]
    assert metric["route"] == (
        "global_then_cross_group_relations_then_group_local"
        if metric_name == "functional_consistency"
        else "global_discovery_then_forced_group_local"
    )
    assert metric["global_discovery"]["final_metric_verdict"] is True
    assert metric["group_phase"]["status"] == "complete"
    assert metric["judge_call_count"] == 2
    assert metric["combined_evidence_budget"][
        "budget_enforcement_scope"
    ] == "judge_episode"
    assert metric["combined_evidence_budget"][
        "episode_seed_counting"
    ] == "judge_facing_evidence"
    assert metric["combined_evidence_budget"][
        "judge_triggered_render_counting"
    ] == "physical_artifacts"
    assert metric["combined_evidence_budget"][
        "metric_aggregate_counting"
    ] == "physical_artifacts"
    assert metric["combined_evidence_budget"][
        "metric_aggregate_is_budget_authority"
    ] is False
    assert metric["global_evidence_paths"] == [
        images["global_perspective"],
    ]
    assert provider_calls[0]["evidence_policy"][
        "scoped_image_budget"
    ] == 1
    assert provider_calls[0]["evidence_policy"][
        "global_image_budget"
    ] == 1
    assert provider_calls[0]["evidence_policy"]["image_budget"] == 2
    assert provider_calls[0]["existing_global_evidence"] == [
        images["global_perspective"]
    ]


def test_functional_global_uses_prejudgement_raw_probe_packet(
    tmp_path,
) -> None:
    images = _images(
        tmp_path,
        "global_perspective",
        "probe_frontage",
        "probe_relation",
        "group_local",
    )
    order: list[str] = []
    planner_calls: list[dict] = []
    probe_calls: list[dict] = []
    local_calls: list[dict] = []
    judge_calls: list[dict] = []

    class Planner:
        def plan_functional_evidence(self, request: dict) -> dict:
            order.append("planner")
            planner_calls.append(request)
            return {
                "schema_version": "functional_probe_plan_v2",
                "probe_units": [
                    {
                        "probe_id": "functional_probe_01",
                        "kind": "functional_frontage",
                        "target_ids": ["chair_01"],
                        "related_target_ids": [],
                        "required_observations": [
                            "target_visible",
                            "interaction_side_visible",
                            "front_back_disambiguated",
                            "approach_zone_visible",
                            "limited_local_context",
                        ],
                        "priority": 1,
                        "reason": "decode the usable side",
                    },
                    {
                        "probe_id": "functional_probe_02",
                        "kind": "functional_correspondence",
                        "target_ids": ["chair_01"],
                        "related_target_ids": ["desk_01"],
                        "required_observations": [
                            "target_visible",
                            "joint_visibility",
                            "interaction_side_visible",
                            "front_back_disambiguated",
                            "approach_zone_visible",
                            "group_context_visible",
                            "limited_local_context",
                        ],
                        "priority": 2,
                        "reason": "show the interaction orientation",
                    },
                ],
                "reason": "bounded visual coverage",
                "request_metadata": {"model": "planner"},
            }

    def probe_provider(request: dict) -> list[dict]:
        order.append("probe")
        probe_calls.append(request)
        path = (
            images["probe_frontage"]
            if len(probe_calls) == 1
            else images["probe_relation"]
        )
        return [
            {
                "path": path,
                "role": "functional_probe_rgb",
                "evidence_style": "raw",
                "image_transform": "none",
            },
            {
                "path": images["global_perspective"],
                "role": "metric_highlighted_global",
                "evidence_style": "raw_highlight",
            },
        ]

    def local_provider(request: dict) -> list[dict]:
        local_calls.append(request)
        return [
            {
                "path": images["group_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        order.append("judge")
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {
                    "enabled": False,
                },
            }
        },
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "group_001",
                    "object_ids": ["chair_01", "desk_01"],
                    "reason": "must not reach functional Judge",
                }
            ]
        },
        render_evidence={
            "global": [images["global_perspective"]]
        },
        camera_evidence_provider=local_provider,
        functional_evidence_planner=Planner(),
        functional_probe_evidence_provider=probe_provider,
        vlm_judge=judge,
        metric_applicability=_relevant(
            "functional_consistency"
        ),
    )["metrics"]["functional_consistency"]

    assert metric["status"] == "evaluated"
    assert order[:4] == ["planner", "probe", "probe", "judge"]
    assert len(planner_calls) == 1
    assert planner_calls[0]["objects"] == [
        {"id": "chair_01", "category": "chair"},
        {"id": "desk_01", "category": "desk"},
    ]
    assert planner_calls[0]["architecture_context"] == {
        "source": "scene_boundary_adapter",
        "logical_boundary_enabled": True,
        "logical_boundary_xy": [
            [0, 0],
            [5, 0],
            [5, 5],
            [0, 5],
        ],
        "physical_walls_rendered": False,
        "physical_wall_ids": [],
    }
    assert metric["functional_probe_acquisition"]["planner_input"][
        "architecture_context"
    ] == planner_calls[0]["architecture_context"]
    assert "center" not in planner_calls[0]["objects"][0]
    # The configured limit is four; the planner emitted two concrete needs,
    # so acquisition must not synthesize two unnecessary calls.
    assert len(probe_calls) == 2
    assert all(
        request["evidence_policy"]["camera_pose_mode"]
        == "query_cov"
        and request["evidence_policy"]["presentation"] == "raw"
        and request["presentation_invariant"][
            "overlay_allowed_in_judge_packet"
        ]
        is False
        for request in probe_calls
    )
    assert judge_calls[0]["render_evidence"] == [
        images["global_perspective"],
        images["probe_frontage"],
        images["probe_relation"],
    ]
    assert judge_calls[0]["camera_acquisition_ledger"][
        "total_images_acquired"
    ] == 3
    assert judge_calls[1]["render_evidence"] == [
        images["global_perspective"],
        images["group_local"],
    ]
    assert judge_calls[1]["camera_acquisition_ledger"][
        "total_images_acquired"
    ] == 2
    assert judge_calls[1]["scene_summary"]["group_scope"] == {
        "group_id": "group_001",
        "member_ids": ["chair_01", "desk_01"],
    }
    assert "target_bounds" not in judge_calls[1]["scene_summary"][
        "group_scope"
    ]
    assert judge_calls[0]["scene_summary"]["objects"] == [
        {"id": "chair_01", "category": "chair"},
        {"id": "desk_01", "category": "desk"},
    ]
    assert judge_calls[0]["object_groups"] == [
        {
            "group_id": "group_001",
            "object_ids": ["chair_01", "desk_01"],
        }
    ]
    packet = judge_calls[0]["functional_probe_evidence"]
    assert [item["role"] for item in packet["image_order"]] == [
        "scene_global",
        "functional_probe",
        "functional_probe",
    ]
    assert packet["probe_inclusion_is_invalidity_prior"] is False
    assert metric["functional_probe_acquisition"]["status"] == (
        "complete"
    )
    assert metric["functional_probe_evidence_paths"] == [
        images["probe_frontage"],
        images["probe_relation"],
    ]
    assert metric["global_context_evidence_paths"] == [
        images["global_perspective"]
    ]
    assert len(local_calls) == 1


def test_disabled_functional_prejudgement_skips_only_proactive_stage(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "group_local")
    local_calls: list[dict] = []
    judge_calls: list[dict] = []

    class Planner:
        def __getattr__(self, name: str):
            raise AssertionError(
                f"disabled proactive stage called planner.{name}"
            )

    def proactive_provider(request: dict) -> list[dict]:
        raise AssertionError(
            f"disabled proactive stage called provider: {request}"
        )

    def local_provider(request: dict) -> list[dict]:
        local_calls.append(request)
        return [
            {
                "path": images["group_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "functional_prejudgement_evidence": {
                "mode": "disabled"
            },
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {
                    "enabled": False,
                },
            },
        },
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=local_provider,
        functional_evidence_planner=Planner(),
        functional_probe_evidence_provider=proactive_provider,
        vlm_judge=judge,
        metric_applicability=_relevant(
            "functional_consistency"
        ),
    )["metrics"]["functional_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["functional_prejudgement_evidence_mode"] == (
        "disabled"
    )
    assert metric["functional_prejudgement_evidence"]["status"] == (
        "disabled"
    )
    assert metric["prejudgement_functional_stage"] == {
        "planner_calls": 0,
        "usable_surface_detector_calls": 0,
        "selector_calls": 0,
        "preview_render_count": 0,
        "full_render_count": 0,
        "judge_facing_image_count": 0,
        "cache_hits": 0,
    }
    assert len(local_calls) == 1
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_discovery",
        "group_local_review",
    ]
    assert judge_calls[0]["functional_probe_evidence"] is None


def test_functional_discovery_routes_cross_group_and_unusual_to_group(
    tmp_path,
) -> None:
    scene = {
        **_scene(),
        "scene_type": "living_room",
        "objects": [
            {
                "id": "sofa",
                "category": "sofa",
                "size": [2.0, 0.9, 0.9],
                "center": [1.2, 2.5, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "television",
                "category": "television",
                "size": [1.2, 0.2, 0.8],
                "center": [4.0, 2.5, 1.2],
                "rotation": [0, 0, 0],
            },
            {
                "id": "cabinet",
                "category": "cabinet",
                "size": [0.8, 0.5, 1.5],
                "center": [4.1, 3.4, 0.75],
                "rotation": [0, 0, 0],
            },
        ],
    }
    images = _images(
        tmp_path,
        "global",
        "cross_group",
        "baseline_group_local",
        "group_confirmation",
    )
    probe_calls: list[dict] = []
    local_calls: list[dict] = []
    judge_calls: list[dict] = []

    class Discovery:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v1",
                "inspected_object_ids": [
                    "sofa",
                    "television",
                    "cabinet",
                ],
                "object_coverage": [
                    {"object_id": object_id, "inspected": True}
                    for object_id in (
                        "sofa",
                        "television",
                        "cabinet",
                    )
                ],
                "directed_surface_targets": [
                    {
                        "discovery_id": "directed_surface_01",
                        "target_id": "sofa",
                        "surface_roles": ["seating_side"],
                        "need_clearance": False,
                        "observation_goal": "show the seating side",
                        "owning_group_id": "group_001",
                    },
                    {
                        "discovery_id": "directed_surface_02",
                        "target_id": "television",
                        "surface_roles": ["display_side"],
                        "need_clearance": False,
                        "observation_goal": "show the display side",
                        "owning_group_id": "group_002",
                    },
                    {
                        "discovery_id": "directed_surface_03",
                        "target_id": "cabinet",
                        "surface_roles": ["opening_side"],
                        "need_clearance": False,
                        "observation_goal": "show the opening side",
                        "owning_group_id": "group_002",
                    },
                ],
                "within_group_correspondences": [],
                    "cross_group_correspondences": [
                        {
                            "discovery_id": "functional_direction_01",
                            "target_ids": ["sofa", "television"],
                            "group_ids": ["group_001", "group_002"],
                            "scope": "cross_group",
                            "predicate": "directional_correspondence",
                            "observation_goal": (
                                "show both usable sides and mutual orientation"
                            ),
                        },
                        {
                            "discovery_id": "functional_geometry_01",
                            "target_ids": ["sofa", "television"],
                            "group_ids": ["group_001", "group_002"],
                            "scope": "cross_group",
                            "predicate": "relative_use_geometry",
                            "observation_goal": (
                                "show their relative layout for ordinary use"
                            ),
                        }
                    ],
                "approach_clearance_targets": [],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [
                    {
                        "discovery_id": "unusual_unconfirmed_01",
                        "target_ids": ["cabinet"],
                        "owning_group_id": "group_002",
                        "observation_goal": (
                            "show cabinet access within its group context"
                        ),
                        "audit_reason": (
                            "the global view leaves the access side unclear"
                        ),
                        "confirmation_scope": "group_local",
                        "decision_authority": "none",
                    }
                ],
                "reason": "complete",
                "provenance": {"request_metadata": {"model": "discovery"}},
            }

    def probe_provider(request: dict) -> list[dict]:
        probe_calls.append(request)
        path = (
            images["cross_group"]
            if request["functional_acquisition_route"]["route_scope"]
            == "cross_group"
            else images["group_confirmation"]
        )
        return [
            {
                "path": path,
                "role": "functional_probe_rgb",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    def local_provider(request: dict) -> list[dict]:
        local_calls.append(request)
        return [
            {
                "path": images["baseline_group_local"],
                "role": "group_local",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report={
            "object_groups": [
                {"group_id": "group_001", "object_ids": ["sofa"]},
                {
                    "group_id": "group_002",
                    "object_ids": ["television", "cabinet"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=local_provider,
        functional_evidence_planner=Discovery(),
        functional_probe_evidence_provider=probe_provider,
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert metric["status"] == "evaluated"
    # Exact-target reuse leaves four unique probe identities even though
    # the default bounded capacity is six.
    assert len(probe_calls) == 4
    assert len(local_calls) == 2
    cross_request = next(
        request
        for request in probe_calls
        if request["functional_acquisition_route"]["route_scope"]
        == "cross_group"
    )
    group_request = next(
        request
        for request in probe_calls
        if request["functional_acquisition_route"]["route_scope"]
        == "group_local"
    )
    assert cross_request["group_ids"] == []
    assert cross_request["object_groups"] == []
    assert group_request["group_ids"] == ["group_002"]
    assert group_request["object_groups"] == [
        {
            "group_id": "group_002",
            "object_ids": ["television", "cabinet"],
        }
    ]
    assert group_request["functional_probe"]["group_member_ids"] == [
        "television",
        "cabinet",
    ]
    assert group_request["functional_probe"][
        "camera_scope_composition"
    ] == "specific_target_focus_plus_owning_group_context"
    assert {
        item["id"] for item in group_request["scene_summary"]["objects"]
    } == {"television", "cabinet"}
    assert judge_calls[0]["render_evidence"] == [images["global"]]
    assert judge_calls[0]["functional_probe_evidence"] is None
    assert judge_calls[1]["evidence_phase"] == (
        "cross_group_relation_review"
    )
    assert judge_calls[1]["target_object_ids"] == [
        "sofa",
        "television",
    ]
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["cross_group"],
    ]
    assert judge_calls[1]["functional_probe_evidence"][
        "episode_scope"
    ] == "single_cross_group_relation_target_set"
    assert judge_calls[1]["functional_probe_evidence"][
        "required_check_count"
    ] == 2
    assert {
        item["predicate"]
        for item in judge_calls[1]["required_functional_checks"]
    } == {
        "directional_correspondence",
        "relative_use_geometry",
    }
    assert sum(
        request["evidence_phase"] == "cross_group_relation_review"
        for request in judge_calls
    ) == 1
    group_002_judge = next(
        request
        for request in judge_calls[2:]
        if request["scene_summary"]["group_scope"]["group_id"]
        == "group_002"
    )
    assert group_002_judge["render_evidence"] == [
        images["global"],
        images["baseline_group_local"],
        images["group_confirmation"],
    ]
    observation_goals = [
        item["neutral_observation_goal"]
        for item in group_002_judge["functional_probe_evidence"][
            "observation_requests"
        ]
    ]
    assert any(
        "show cabinet access within its group context" in goal
        for goal in observation_goals
    )
    assert "global view leaves" not in json.dumps(
        judge_calls[1]["functional_probe_evidence"]
    )
    assert metric["functional_probe_acquisition"][
        "cross_group_evidence_paths"
    ] == [images["cross_group"]]
    assert metric["functional_probe_acquisition"][
        "group_evidence_paths"
    ] == {
        "group_001": [images["group_confirmation"]],
        "group_002": [images["group_confirmation"]],
    }
    group_002_result = next(
        item
        for item in metric["group_results"]
        if item["group_id"] == "group_002"
    )
    assert group_002_result["evidence_resolution"][
        "functional_probe_reuse"
    ]["baseline_group_local_preserved"] is True
    assert group_002_result["evidence_resolution"][
        "functional_probe_reuse"
    ]["appended_probe_paths"] == [images["group_confirmation"]]
    assert metric["cross_group_relation_phase"] == {
        "required": True,
        "scheduled_relation_count": 1,
        "judge_eligible_relation_count": 1,
        "skipped_missing_pair_evidence_count": 0,
        "resolved_relation_count": 1,
        "status": "complete",
        "max_probe_units": 6,
    }
    assert metric["judge_call_count"] == 4


def test_each_acquired_cross_group_relation_gets_one_judge_episode(
    tmp_path,
) -> None:
    images = _images(
        tmp_path,
        "global",
        "sofa_tv",
        "piano_bench",
    )
    scene = {
        **_scene(),
        "scene_type": "living_room",
        "objects": [
            {"id": "sofa", "category": "sofa"},
            {"id": "television", "category": "television"},
            {"id": "piano", "category": "piano"},
            {"id": "bench", "category": "piano_bench"},
        ],
    }

    class Discovery:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v3",
                "inspected_object_ids": [
                    "sofa",
                    "television",
                    "piano",
                    "bench",
                ],
                "directed_surface_targets": [],
                "within_group_correspondences": [],
                "cross_group_correspondences": [
                    {
                        "discovery_id": "sofa_tv_relation",
                        "target_ids": ["sofa", "television"],
                        "group_ids": ["sofa_group", "tv_group"],
                        "scope": "cross_group",
                        "observation_kinds": ["mutual_orientation"],
                        "observation_goal": (
                            "show sofa and television facing compatibility"
                        ),
                    },
                    {
                        "discovery_id": "piano_bench_relation",
                        "target_ids": ["piano", "bench"],
                        "group_ids": ["piano_group", "bench_group"],
                        "scope": "cross_group",
                        "observation_kinds": ["cooperative_operation"],
                        "observation_goal": (
                            "show piano and bench joint-use alignment"
                        ),
                    },
                ],
                "approach_clearance_targets": [],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [],
                "reason": "two cross-group relations",
                "provenance": {},
            }

    def probe_provider(request: dict) -> list[dict]:
        targets = set(request["object_ids"])
        path = (
            images["sofa_tv"]
            if targets == {"sofa", "television"}
            else images["piano_bench"]
        )
        return [
            {
                "path": path,
                "role": "functional_probe_rgb",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        phase = request.get("evidence_phase")
        relation_targets = set(
            request.get("target_object_ids") or []
        )
        if phase == "global_discovery" or (
            phase == "cross_group_relation_review"
            and relation_targets == {"sofa", "television"}
        ):
            return _with_required_functional_checks(request, {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": 0.9,
                "reason": "The two usable sides do not correspond.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": "orientation_for_use",
                        "target_ids": ["sofa", "television"],
                        "relation": "incompatible_facing_direction",
                        "reason": (
                            "The seating and display sides face away from "
                            "their intended joint-use direction."
                        ),
                    }
                ],
            })
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report={
            "object_groups": [
                {"group_id": "sofa_group", "object_ids": ["sofa"]},
                {
                    "group_id": "tv_group",
                    "object_ids": ["television"],
                },
                {"group_id": "piano_group", "object_ids": ["piano"]},
                {"group_id": "bench_group", "object_ids": ["bench"]},
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: pytest.fail(
            "singleton groups must not enter group-local review"
        ),
        functional_evidence_planner=Discovery(),
        functional_probe_evidence_provider=probe_provider,
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_discovery",
        "cross_group_relation_review",
        "cross_group_relation_review",
    ]
    relation_calls = judge_calls[1:]
    assert [
        set(call["target_object_ids"]) for call in relation_calls
    ] == [
        {"piano", "bench"},
        {"sofa", "television"},
    ]
    assert all(
        len(call["functional_probe_evidence"][
            "relation_observation_requests"
        ])
        == 1
        for call in relation_calls
    )
    assert metric["cross_group_relation_phase"][
        "scheduled_relation_count"
    ] == 2
    assert metric["cross_group_relation_phase"]["status"] == "complete"
    assert metric["group_phase"]["status"] == (
        "not_required_singleton_only"
    )
    assert metric["judge_call_count"] == 3
    assert metric["status"] == "evaluated"
    assert metric["score"] == 0.0
    assert metric["global_discovery"]["final_metric_verdict"] is False
    assert "owned by the cross-group relation stage" in (
        metric["global_discovery"]["error"]
    )
    assert metric["global_scene_claims"] == []
    assert len(metric["cross_group_relation_claims"]) == 1
    assert metric["cross_group_relation_claims"][0][
        "source_phase"
    ] == "cross_group_relation_review:sofa_tv_relation"


def test_failed_cross_group_acquisition_does_not_start_judge_episode(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global")
    scene = {
        **_scene(),
        "scene_type": "living_room",
        "objects": [
            {"id": "sofa", "category": "sofa"},
            {"id": "television", "category": "television"},
        ],
    }

    class Discovery:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v3",
                "inspected_object_ids": ["sofa", "television"],
                "directed_surface_targets": [],
                "within_group_correspondences": [],
                "cross_group_correspondences": [
                    {
                        "discovery_id": "sofa_tv_relation",
                        "target_ids": ["sofa", "television"],
                        "group_ids": ["sofa_group", "tv_group"],
                        "scope": "cross_group",
                        "observation_kinds": ["mutual_orientation"],
                        "observation_goal": (
                            "show sofa and television facing compatibility"
                        ),
                    }
                ],
                "approach_clearance_targets": [],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [],
                "reason": "one cross-group relation",
                "provenance": {},
            }

    def failed_probe_provider(request: dict) -> list[dict]:
        raise RuntimeError(
            "no_feasible_candidate: functional candidate bank is empty"
        )

    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report={
            "object_groups": [
                {"group_id": "sofa_group", "object_ids": ["sofa"]},
                {
                    "group_id": "tv_group",
                    "object_ids": ["television"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: pytest.fail(
            "singleton groups must not enter group-local review"
        ),
        functional_evidence_planner=Discovery(),
        functional_probe_evidence_provider=failed_probe_provider,
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_discovery",
    ]
    assert metric["cross_group_relation_phase"] == {
        "required": True,
        "scheduled_relation_count": 1,
        "judge_eligible_relation_count": 0,
        "skipped_missing_pair_evidence_count": 1,
        "resolved_relation_count": 0,
        "status": "unresolved",
        "max_probe_units": 6,
    }
    scheduled = metric["functional_cross_group_relation_schedule"][0]
    assert scheduled["pair_specific_evidence_available"] is False
    assert scheduled["judge_episode"] == (
        "not_started_pair_specific_evidence_unavailable"
    )
    relation_result = metric["cross_group_relation_results"][0]
    assert relation_result["vlm_invoked"] is False
    assert relation_result["reason"] == "pair_specific_evidence_unavailable"
    assert relation_result["available_global_context_evidence_paths"] == [
        images["global"]
    ]
    assert metric["status"] == "unresolved"


def test_functional_discovery_forces_singleton_unusual_confirmation(
    tmp_path,
) -> None:
    scene = {
        **_scene(),
        "objects": [
            {
                "id": "cabinet",
                "category": "cabinet",
                "size": [0.8, 0.5, 1.5],
                "center": [4.1, 3.4, 0.75],
                "rotation": [0, 0, 0],
            }
        ],
    }
    images = _images(
        tmp_path,
        "global",
        "baseline_group_local",
        "confirmation",
    )
    judge_calls: list[dict] = []

    class Discovery:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v1",
                "inspected_object_ids": ["cabinet"],
                "object_coverage": [
                    {"object_id": "cabinet", "inspected": True}
                ],
                "directed_surface_targets": [],
                "within_group_correspondences": [],
                "cross_group_correspondences": [],
                "approach_clearance_targets": [],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [
                    {
                        "discovery_id": "unusual_unconfirmed_01",
                        "target_ids": ["cabinet"],
                        "owning_group_id": "group_001",
                        "observation_goal": (
                            "show the usable side and limited local context"
                        ),
                        "audit_reason": "global cue remains unresolved",
                        "confirmation_scope": "group_local",
                        "decision_authority": "none",
                    }
                ],
                "reason": "complete",
                "provenance": {"request_metadata": {}},
            }

    def probe_provider(request: dict) -> list[dict]:
        return [
            {
                "path": images["confirmation"],
                "role": "functional_probe_rgb",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    def local_provider(request: dict) -> list[dict]:
        return [
            {
                "path": images["baseline_group_local"],
                "role": "group_local",
                "evidence_style": "raw",
                "image_transform": "none",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report={
            "object_groups": [
                {"group_id": "group_001", "object_ids": ["cabinet"]}
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=local_provider,
        functional_evidence_planner=Discovery(),
        functional_probe_evidence_provider=probe_provider,
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["group_filter"][
        "functional_confirmation_forced_group_ids"
    ] == ["group_001"]
    assert metric["group_filter"]["eligible_group_ids"] == ["group_001"]
    assert len(judge_calls) == 2
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["baseline_group_local"],
        images["confirmation"],
    ]


def test_functional_boundary_facts_reach_judge_with_zero_probe_budget(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "group_local")
    judge_calls: list[dict] = []
    boundary_calls: list[dict] = []

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "functional_discovery_v3",
                "inspected_object_ids": ["chair_01", "desk_01"],
                "directed_surface_targets": [
                    {
                        "discovery_id": "directed_surface_01",
                        "target_id": "chair_01",
                        "directionality": "directed",
                        "surface_roles": ["seating_side"],
                        "need_clearance": True,
                        "boundary_review_state": "routine",
                        "owning_group_id": "group_001",
                        "observation_goal": (
                            "identify seating-side approach space"
                        ),
                    }
                ],
                "within_group_correspondences": [],
                "cross_group_correspondences": [],
                "approach_clearance_targets": [
                    {
                        "discovery_id": "approach_clearance_01",
                        "target_id": "chair_01",
                        "need_clearance": True,
                        "owning_group_id": "group_001",
                        "observation_goal": (
                            "show ordinary seating approach"
                        ),
                    }
                ],
                "boundary_sensitive_targets": [],
                "unusual_unconfirmed": [],
                "reason": "complete",
                "provenance": {},
            }

    class BoundaryProvider:
        def provide_functional_boundary_evidence(
            self,
            request: dict,
        ) -> dict:
            boundary_calls.append(request)
            return {
                "status": "complete",
                "decision_authority": "none",
                "scene_access": "read_only",
                "surface_targets": request["surface_targets"],
                "usable_surface_hypotheses": [
                    {
                        "target_id": "chair_01",
                        "status": "identified",
                        "surfaces": [
                            {
                                "surface_role": "seating_side",
                                "side_id": "local_pos_x",
                                "visual_cues": ["seat and back geometry"],
                                "confidence": 0.8,
                            }
                        ],
                        "reason": "seating side",
                    }
                ],
                "functional_geometry": {
                    "schema_version": "functional_geometry_v1",
                    "decision_authority": "none",
                    "scene_access": "read_only",
                    "logical_boundary_available": True,
                    "surface_observations": [
                        {
                            "target_id": "chair_01",
                            "nearest_boundary_distance_m": 0.2,
                            "outward_ray_boundary_distance_m": 0.2,
                            "approach_samples": [],
                        }
                    ],
                    "observation_status": "available",
                },
                "decoder_audit": {
                    "decoder_calls": 1,
                    "cache_hits": 0,
                    "preview_render_count": 8,
                },
                "provenance": {"geometry_source": "deterministic"},
            }

    def local_provider(request: dict) -> list[dict]:
        return [
            {
                "path": images["group_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    config = {
        "metrics": {
            "style_consistency": {"enabled": False},
            "scale_consistency": {"enabled": False},
            "object_pairing_consistency": {"enabled": False},
            "functional_consistency": {
                "enabled": True,
                "weight": 1.0,
                "evidence_plan": {
                    "prejudgement_probe_policy": {
                        "enabled": True,
                        "max_probe_units": 0,
                    }
                },
            },
            "semantic_placement_consistency": {"enabled": False},
        }
    }
    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=local_provider,
        functional_evidence_planner=Planner(),
        functional_probe_evidence_provider=BoundaryProvider(),
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert metric["status"] == "evaluated"
    assert len(boundary_calls) == 1
    assert metric["functional_probe_evidence_paths"] == []
    assert judge_calls[0]["render_evidence"] == [images["global"]]
    assert judge_calls[0]["functional_probe_evidence"] is None
    structured = judge_calls[1]["functional_probe_evidence"][
        "boundary_clearance_evidence"
    ]
    assert structured["decision_authority"] == "none"
    assert structured["functional_geometry"][
        "surface_observations"
    ][0]["outward_ray_boundary_distance_m"] == 0.2
    assert judge_calls[0]["camera_acquisition_ledger"][
        "total_images_acquired"
    ] == 1


@pytest.mark.parametrize(
    "metric_name",
    [
        "style_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    ],
)
def test_visual_global_local_metrics_skip_singleton_group_review(
    tmp_path,
    metric_name,
) -> None:
    scene = _scene()
    scene["objects"].append(
        {
            "id": "lamp_01",
            "category": "floor_lamp",
            "size": [0.3, 0.3, 1.6],
            "center": [3.5, 3.5, 0.8],
            "rotation": [0, 0, 0],
        }
    )
    images = _images(tmp_path, "global", "work_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {"path": images["work_local"], "role": "group_local"}
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        if (
            metric_name == "style_consistency"
            and len(judge_calls) == 1
        ):
            return _style_needs_local("chair_01")
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                metric_name: {"enabled": True, "weight": 1.0},
            }
        },
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "work",
                    "object_ids": ["chair_01", "desk_01"],
                },
                {
                    "group_id": "lamp",
                    "object_ids": ["lamp_01"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant(metric_name),
    )["metrics"][metric_name]

    assert metric["status"] == "evaluated"
    assert len(judge_calls) == 2
    assert len(provider_calls) == 1
    assert provider_calls[0]["group_scope"]["group_id"] == "work"
    assert metric["group_filter"]["eligible_group_ids"] == ["work"]
    assert metric["group_filter"]["skipped_groups"] == [
        {
            "group_id": "lamp",
            "member_ids": ["lamp_01"],
            "member_count": 1,
            "reason": "singleton_group",
        }
    ]


def test_localized_style_candidate_forces_its_singleton_group(
    tmp_path,
) -> None:
    scene = _scene()
    scene["objects"].append(
        {
            "id": "lamp_01",
            "category": "floor_lamp",
            "size": [0.3, 0.3, 1.6],
            "center": [3.5, 3.5, 0.8],
            "rotation": [0, 0, 0],
        }
    )
    images = _images(tmp_path, "global", "singleton_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {"path": images["singleton_local"], "role": "group_local"}
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        if len(judge_calls) == 1:
            return _style_needs_local("lamp_01")
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config=_only("style_consistency"),
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "work",
                    "object_ids": ["chair_01", "desk_01"],
                },
                {
                    "group_id": "lamp",
                    "object_ids": ["lamp_01"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert len(judge_calls) == 2
    assert provider_calls[0]["group_scope"]["group_id"] == "lamp"
    assert metric["selected_group_ids"] == ["lamp"]
    assert metric["group_filter"]["explicit_singleton_group_ids"] == [
        "lamp"
    ]


def test_supported_invalid_is_final_even_when_coverage_is_incomplete() -> None:
    base = {
        "functional_check_coverage": {
            "unresolved_check_ids": ["functional_check_pending"],
        }
    }
    result = _aggregate_global_and_group_results(
        base,
        metric_name="functional_consistency",
        global_record={
            "verdict": "valid",
            "confidence": 0.9,
            "defects": [],
        },
        global_outcome={"status": "evaluated", "score": 1.0},
        scene_claims=[],
        relation_claims=[],
        relation_results=[],
        relation_phase_complete=True,
        group_results=[
            {
                "group_id": "work",
                "status": "evaluated",
                "score": 0.0,
                "judgement": {
                    "verdict": "invalid",
                    "confidence": 0.8,
                    "defects": [
                        {
                            "scope": "clearance",
                            "target_ids": ["chair_01"],
                            "relation": "clearance",
                            "reason": "Ordinary use is materially blocked.",
                        }
                    ],
                },
            },
            {
                "group_id": "other",
                "status": "unresolved",
                "score": None,
            },
        ],
        group_phase_required=True,
        group_phase_complete=False,
        functional_check_phase_complete=False,
        placement_check_phase_complete=True,
    )

    assert result["status"] == "evaluated"
    assert result["score"] == 0.0
    assert result["coverage"]["complete"] is False
    assert result["judgement"]["verdict"] == "invalid"
    assert result["judgement"]["unresolved_scopes"] == [
        "group_local_judgement:other",
        "functional_check:functional_check_pending",
    ]


def test_functional_global_pass_preserves_cross_group_defect(
    tmp_path,
) -> None:
    scene = {
        **_scene(),
        "objects": [
            {
                "id": "sofa_01",
                "category": "sofa",
                "size": [2.0, 0.8, 0.9],
                "center": [1.5, 2.0, 0.45],
                "rotation": [0, 0, 180],
            },
            {
                "id": "coffee_table_01",
                "category": "coffee_table",
                "size": [1.0, 0.6, 0.4],
                "center": [2.4, 2.0, 0.2],
                "rotation": [0, 0, 0],
            },
            {
                "id": "television_01",
                "category": "television",
                "size": [1.2, 0.2, 0.8],
                "center": [4.2, 2.0, 1.2],
                "rotation": [0, 0, 180],
            },
            {
                "id": "tv_cabinet_01",
                "category": "tv_cabinet",
                "size": [1.5, 0.5, 0.6],
                "center": [4.2, 2.0, 0.3],
                "rotation": [0, 0, 0],
            },
        ],
    }
    images = _images(
        tmp_path,
        "global",
        "seating_local",
        "media_local",
    )
    judge_calls: list[dict] = []

    global_invalid = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.92,
        "reason": "The sofa faces away from the television.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "facing_and_interaction_direction",
                "target_ids": ["sofa_01", "television_01"],
                "relation": "sofa_faces_away_from_television",
                "reason": (
                    "The seating-to-display interaction direction is unusable."
                ),
            }
        ],
    }

    def provider(request: dict) -> list[dict]:
        group_id = request["group_scope"]["group_id"]
        return [
            {
                "path": images[f"{group_id}_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return global_invalid if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
            }
        },
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "seating",
                    "object_ids": ["sofa_01", "coffee_table_01"],
                },
                {
                    "group_id": "media",
                    "object_ids": [
                        "television_01",
                        "tv_cabinet_01",
                    ],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert len(judge_calls) == 3
    assert metric["status"] == "evaluated"
    assert metric["score"] == 0.0
    assert metric["coverage"]["complete"] is True
    assert metric["judgement"]["defects"] == global_invalid["defects"]
    assert len(metric["global_scene_claims"]) == 1
    assert len(metric["final_defect_claims"]) == 1


@pytest.mark.parametrize(
    ("metric_name", "scope"),
    [
        (
            "style_consistency",
            "significant_visible_style_incompatibility",
        ),
        (
            "semantic_placement_consistency",
            "implausible_local_context",
        ),
    ],
)
def test_global_local_duplicate_claim_is_one_metric_object_penalty(
    tmp_path,
    metric_name,
    scope,
) -> None:
    images = _images(tmp_path, "global", "local")
    defect = {
        "scope": scope,
        "target_ids": ["chair_01", "desk_01"],
        "relation": "same_object_level_defect",
        "reason": "The same object-level defect is visible in both phases.",
    }
    if metric_name == "semantic_placement_consistency":
        defect.update(
            scope="semantically_inappropriate_scene_zone",
            target_ids=["chair_01"],
            relation="scene_zone",
            check_id="placement_scene_zone_proposal",
        )
        defect["severity"] = "material_contextual_mismatch"
    invalid = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.9,
        "reason": defect["reason"],
        "missing_evidence": [],
        "defects": [defect],
    }
    if metric_name == "semantic_placement_consistency":
        invalid["judge_originated_placement_results"] = [
            {
                "proposal_id": "placement_scene_zone_proposal",
                "subject_id": "chair_01",
                "context_ids": [],
                "check_type": "scene_zone",
                "observation_goal": (
                    "Inspect the chair's room zone in global context."
                ),
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": defect["reason"],
                "severity": "material_contextual_mismatch",
            }
        ]

    judge_call_count = 0

    def judge(request: dict) -> dict:
        nonlocal judge_call_count
        judge_call_count += 1
        if (
            metric_name == "semantic_placement_consistency"
            and judge_call_count > 1
        ):
            return _valid(request)
        return deepcopy(invalid)

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                metric_name: {
                    "enabled": True,
                    "weight": 1.0,
                },
            }
        },
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: [
            {"path": images["local"], "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant(metric_name),
    )["metrics"][metric_name]

    assert metric["score"] == 0.0
    assert len(metric["judgement"]["defects"]) == 1
    assert len(metric["final_defect_claims"]) == 1
    if metric_name == "style_consistency":
        correspondence = metric["group_results"][0][
            "claim_correspondence"
        ]
        assert correspondence[0]["relationship"] == (
            "confirmed_routed_candidate"
        )
        assert correspondence[0]["routed_candidate_id"] == (
            metric["global_screen_candidate_claims"][0]["claim_id"]
        )
        assert metric["object_level_attribution"] == {
            "enabled": True,
            "unit": "object",
            "deduplication_key": ["metric", "object_id"],
            "cross_phase_deduplication": True,
            "cross_metric_deduplication": False,
            "raw_defect_observation_count": 2,
            "unique_object_count": 2,
            "merged_duplicate_observation_count": 0,
            "penalty_unit_count": 2,
        }
    else:
        assert metric["placement_check_coverage"]["complete"] is True
        assert len(metric["placement_check_ledger"]["checks"]) == 1
        assert metric["object_level_attribution"] == {
            "enabled": True,
            "unit": "object",
            "deduplication_key": ["metric", "object_id"],
            "cross_phase_deduplication": True,
            "cross_metric_deduplication": False,
            "raw_defect_observation_count": 1,
            "unique_object_count": 1,
            "merged_duplicate_observation_count": 0,
            "penalty_unit_count": 1,
        }
        assert metric["placement_severity"][
            "highest_severity"
        ] == "material_contextual_mismatch"
        assert metric["placement_severity"][
            "strict_failure_present"
        ] is False
        assert metric["placement_severity"][
            "extended_issue_present"
        ] is True
    expected_findings = {
        "chair_01": 1,
    } if metric_name == "semantic_placement_consistency" else {
        "chair_01": 1,
        "desk_01": 1,
    }
    assert {
        finding["object_id"]: finding["observation_count"]
        for finding in metric["final_object_findings"]
    } == expected_findings
    assert all(
        finding["observed_in_global_and_local"]
        is False
        for finding in metric["final_object_findings"]
    )


def test_same_metric_object_is_one_penalty_across_distinct_phase_claims(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    global_invalid = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.9,
        "reason": "The chair is unusably far from the desk.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "reachability",
                "target_ids": ["chair_01"],
                "relation": "chair_cannot_reach_desk",
                "reason": "The chair cannot serve the desk.",
            }
        ],
    }
    local_invalid = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.85,
        "reason": "The chair's interaction side is unusable.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["chair_01"],
                "relation": "chair_interaction_side_blocked",
                "reason": "The same chair cannot be used from this side.",
            }
        ],
    }
    judge_calls: list[dict] = []

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return global_invalid if len(judge_calls) == 1 else local_invalid

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "metrics": {
                "style_consistency": {"enabled": False},
                "scale_consistency": {"enabled": False},
                "object_pairing_consistency": {"enabled": False},
                "functional_consistency": {
                    "enabled": True,
                    "weight": 1.0,
                },
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=lambda request: [
            {"path": images["local"], "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant("functional_consistency"),
    )["metrics"]["functional_consistency"]

    assert metric["score"] == 0.0
    assert len(metric["final_defect_claims"]) == 2
    assert len(metric["final_object_findings"]) == 1
    finding = metric["final_object_findings"][0]
    assert finding["metric"] == "functional_consistency"
    assert finding["object_id"] == "chair_01"
    assert finding["observation_count"] == 2
    assert finding["merged_duplicate_observation_count"] == 1
    assert finding["observed_in_global_and_local"] is True
    assert metric["judgement"]["object_penalty_count"] == 1
    correspondence = metric["group_results"][0][
        "scene_claim_correspondence"
    ][0]
    assert correspondence["relationship"] == (
        "same_metric_object_already_flagged_global"
    )
    assert correspondence["object_level_deduplication"][
        "duplicate_penalty_suppressed"
    ] is True
    assert correspondence["object_level_deduplication"][
        "cross_metric_deduplication"
    ] is False


def test_visual_confirmation_accepts_fewer_images_than_budget_when_sufficient(
    tmp_path,
) -> None:
    local_image = _images(tmp_path, "local")["local"]
    judge_calls: list[dict] = []
    suspicious = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.7,
        "reason": "The chair dimensions look suspicious.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": (
                    "significant_visible_category_relative_"
                    "scale_incoherence"
                ),
                "target_ids": ["chair_01"],
                "relation": "scale_outlier_candidate",
                "reason": "The JSON dimensions merit confirmation.",
            }
        ],
    }

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return suspicious if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_json_first_only("scale_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence=None,
        camera_evidence_provider=lambda request: [
            {"path": local_image, "role": "group_local"}
        ],
        vlm_judge=judge,
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]

    assert metric["status"] == "evaluated"
    assert judge_calls[1]["render_evidence"] == [local_image]
    assert metric["evidence_request"]["image_budget"] == 3


def test_style_global_screen_plan_clear_skips_local(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global_a", "global_b", "local_a", "local_b")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> dict:
        provider_calls.append(request)
        return {
            "status": "available",
            "render_evidence_items": [
                {"path": images["local_a"], "role": "object_local"},
                {"path": images["local_b"], "role": "object_local"},
            ],
        }

    config = _style_global_then_local_config()
    config["metrics"]["style_consistency"]["evidence_plan"][
        "evidence_strategy"
    ] = "global_screen_then_local"

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global_a"], images["global_b"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["score"] == 1.0
    assert metric["route"] == "global_clear"
    assert metric["judge_call_count"] == 1
    assert metric["local_evidence_paths"] == []
    assert provider_calls == []
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_screen",
    ]


class _StyleLocalRenderer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_camera_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        preview=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "pass": "rgb",
                "preview": preview,
                "camera_views": camera_views,
            }
        )
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"rgb_{index:02d}.png"
            image = Image.new("RGB", (32, 32), (80, 110, 140))
            image.putpixel((0, 0), (160, 70, 40))
            image.save(path)
            views.append(
                {"id": pose["id"], "path": str(path), "pose": pose}
            )
        return {"views": views}

    def render_focus_overlay_views(
        self,
        *,
        blend_file,
        out_dir,
        camera_views,
        overlay_spec,
        preview=False,
    ):
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.calls.append(
            {
                "pass": "focus",
                "preview": preview,
                "camera_views": camera_views,
            }
        )
        color = tuple(
            round(channel * 255)
            for channel in overlay_spec["targets"][0]["color"]
        )
        views = []
        for index, pose in enumerate(camera_views):
            path = destination / f"focus_{index:02d}.png"
            image = Image.new("RGB", (32, 32), (40, 40, 40))
            for x in range(7, 25):
                for y in range(7, 25):
                    image.putpixel((x, y), color)
            image.save(path)
            views.append(
                {"id": pose["id"], "path": str(path), "pose": pose}
            )
        return {"views": views}


def test_style_routed_local_uses_real_provider_group_local_override(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global_a", "global_b")
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"blend")
    renderer = _StyleLocalRenderer()
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend,
        out_dir=tmp_path / "style_camera",
        mode="auto",
        max_views=1,
    )
    judge_calls: list[dict] = []
    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return (
            _style_needs_local()
            if len(judge_calls) == 1
            else _valid()
        )

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        object_grouping_report=_grouping_report(),
        render_evidence=[images["global_a"], images["global_b"]],
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["route"] == "global_screen_then_group_local"
    assert metric["judgement"]["verdict"] == "valid"
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_screen",
        "local_confirmation",
    ]
    assert any(call["pass"] == "focus" for call in renderer.calls)
    manifests = list(
        (tmp_path / "style_camera").glob(
            "*/camera_evidence_manifest.json"
        )
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["resolved_mode"] == "visibility_ranked"
    assert manifest["metric"] == "style_consistency"
    assert metric["local_evidence_paths"]


def test_canonical_profile_l3_section_is_the_profile_override() -> None:
    profile = {
        "l3_scene_quality": {
            "enabled": True,
            "metrics": {
                "style_consistency": {"enabled": False, "weight": 0.5},
                "scale_consistency": {"enabled": True, "weight": 0.25},
                "object_pairing_consistency": {"enabled": True, "weight": 0.25},
            },
        }
    }
    resolved = resolve_scene_quality_config(profile=profile)
    assert resolved["metrics"]["style_consistency"]["enabled"] is False
    assert resolved["metrics"]["style_consistency"]["weight"] == 0.5


def test_three_metric_subset_evaluates_and_aggregates_when_complete(tmp_path) -> None:
    calls: list[dict] = []
    images = _images(
        tmp_path,
        "global",
        "style_local",
        "scale",
        "pair_1",
        "pair_2",
    )

    def judge(request: dict) -> dict:
        calls.append(request)
        return _valid()

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "metrics": {
                "functional_consistency": {"enabled": False},
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "style_consistency": {
                "group_001": [images["style_local"]],
            },
            "scale_consistency": [images["scale"]],
            "object_pairing_consistency": [
                images["pair_1"],
                images["pair_2"],
            ],
        },
        vlm_judge=judge,
        prompt="Create a coherent bedroom work corner.",
        metric_applicability=_relevant(
            "style_consistency",
            "scale_consistency",
            "object_pairing_consistency",
        ),
    )
    assert report["category"] == SCENE_QUALITY_INTERFACE_NAMESPACE == "l3_scene_quality"
    assert report["implemented"] is True
    assert report["status"] == "evaluated"
    assert report["score"] == 1.0
    assert report["coverage"] == {
        "eligible_count": 3,
        "resolved_count": 3,
        "fraction": 1.0,
        "complete": True,
    }
    assert report["renderer_invoked"] is False
    assert report["vlm_invoked"] is True
    assert {request["metric"] for request in calls} == {
        "style_consistency",
        "scale_consistency",
        "object_pairing_consistency",
    }
    assert all(request["vlm_role"] == "judge" for request in calls)
    assert all(
        request["decision_contract"] == "canonical_metric_v1"
        for request in calls
    )
    assert all(
        request["judge_method"] == "adjudicate_scene_quality"
        for request in calls
    )
    assert all(
        request["metric_prompt_version"] == L3_METRIC_PROMPT_VERSION
        and request["metric_boundary_rules"] == list(
            L3_METRIC_BOUNDARY_RULES
        )
        for request in calls
    )
    pairing_request = next(
        request
        for request in calls
        if request["metric"] == "object_pairing_consistency"
    )
    assert pairing_request["object_groups"][0]["group_id"] == "group_001"
    assert "orientation" in pairing_request["judgment_scope"]["excluded"]
    assert "scene_member_category_compatibility" in (
        pairing_request["judgment_scope"]["included"]
    )
    assert "evidence scope, not compatibility ground truth" in (
        pairing_request["metric_rubric"]
    )
    assert pairing_request["natural_language_prompt"].startswith("Create")
    assert pairing_request["render_evidence"] == []
    assert pairing_request["evidence_phase"] == "json_screen"
    assert pairing_request["decision_mode"] == "screen"
    assert report["active_metric_signature"] == (
        "style_consistency+scale_consistency+object_pairing_consistency"
    )
    serialized = json.dumps(report, sort_keys=True)
    assert "metric_aliases" not in report
    assert all("aliases" not in metric for metric in report["metrics"].values())
    assert "object_coexistence_consistency" not in serialized
    assert "benchmark.evaluator.spatial_fidelity.scale" not in serialized


@pytest.mark.parametrize(
    ("response", "status", "score", "reason"),
    [
        (
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "reason": "No significant defect.",
                "missing_evidence": [],
                "defects": [],
            },
            "evaluated",
            1.0,
            None,
        ),
        (
            {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "reason": "The chair is visibly many times too large.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": "significant_visible_category_relative_scale_incoherence",
                        "target_ids": ["chair_01", "desk_01"],
                        "relation": "chair_too_large_for_desk",
                        "reason": "The chair is visibly many times too large.",
                    }
                ],
            },
            "evaluated",
            0.0,
            None,
        ),
        (
            {
                "evidence_status": "sufficient",
                "verdict": "ambiguous",
                "reason": "A genuine semantic borderline.",
                "missing_evidence": [],
                "defects": [],
            },
            "unresolved",
            None,
            "ambiguous_scene_quality_judgement",
        ),
        (
            {
                "evidence_status": "insufficient",
                "verdict": "ambiguous",
                "reason": "The complete silhouettes are not visible.",
                    "missing_evidence": ["complete target silhouettes"],
                    "defects": [],
            },
            "unresolved",
            None,
            "insufficient_visual_evidence",
        ),
    ],
)
def test_canonical_judgement_states(
    tmp_path,
    response: dict,
    status: str,
    score: float | None,
    reason: str | None,
) -> None:
    image = _images(tmp_path, "scale")["scale"]
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        render_evidence={"scale_consistency": [image]},
        vlm_judge=lambda request: response,
        metric_applicability=_relevant("scale_consistency"),
    )
    metric = report["metrics"]["scale_consistency"]
    assert metric["status"] == status
    assert metric["score"] == score
    assert metric["reason"] == reason
    if status == "unresolved":
        assert report["score"] is None


def test_missing_inputs_are_unresolved_never_valid() -> None:
    report = evaluate_scene_quality_interfaces(
        _scene(),
        object_grouping_report=None,
        render_evidence=None,
        vlm_judge=None,
        metric_applicability=_relevant(*SCENE_QUALITY_INTERFACE_METRICS),
    )
    assert report["score"] is None
    assert report["status"] == "unresolved"
    assert report["metrics"]["style_consistency"]["reason"] == "vlm_judge_not_configured"
    assert report["metrics"]["scale_consistency"]["reason"] == "vlm_judge_not_configured"
    pairing = report["metrics"]["object_pairing_consistency"]
    assert pairing["status"] == "unresolved"
    assert pairing["reason"] == "object_grouping_unavailable"

    no_judge = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        render_evidence=["global.png"],
        metric_applicability=_relevant("style_consistency"),
    )
    assert no_judge["metrics"]["style_consistency"]["reason"] == (
        "vlm_judge_not_configured"
    )


def test_pairing_requires_grouping_and_ignores_singleton_groups(tmp_path) -> None:
    group_image = _images(tmp_path, "group")["group"]
    missing = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("object_pairing_consistency"),
        render_evidence={"object_pairing_consistency": [group_image]},
        vlm_judge=lambda request: _valid(),
        metric_applicability=_relevant("object_pairing_consistency"),
    )["metrics"]["object_pairing_consistency"]
    assert missing["status"] == "unresolved"
    assert missing["reason"] == "object_grouping_unavailable"

    singleton = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("object_pairing_consistency"),
        object_grouping_report={
            "object_groups": [
                {"group_id": "g1", "object_ids": ["chair_01"]},
                {"group_id": "g2", "object_ids": ["desk_01"]},
            ]
        },
        render_evidence={"object_pairing_consistency": [group_image]},
        vlm_judge=lambda request: pytest.fail("singleton groups must not be judged"),
        metric_applicability=_relevant("object_pairing_consistency"),
    )["metrics"]["object_pairing_consistency"]
    assert singleton["status"] == "not_applicable"
    assert singleton["reason"] == "no_eligible_targets"
    assert singleton["affects_score"] is False


def test_pairing_mechanically_drops_structured_out_of_scope_defects(tmp_path) -> None:
    group_image = _images(tmp_path, "group")["group"]
    def judge(request: dict) -> dict:
        return {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "reason": "The chair faces away from the desk.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "orientation",
                    "target_ids": ["chair_01", "desk_01"],
                    "relation": "chair_faces_away_from_desk",
                    "reason": "direction",
                }
            ],
        }

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("object_pairing_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"object_pairing_consistency": [group_image]},
        vlm_judge=judge,
        metric_applicability=_relevant("object_pairing_consistency"),
    )["metrics"]["object_pairing_consistency"]
    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert metric["judgement"]["original_verdict"] == "invalid"
    assert metric["judgement"]["out_of_scope_defects"][0]["scope"] == "orientation"


def test_pairing_emits_one_camera_and_judge_request_per_group(
    tmp_path,
) -> None:
    scene = _scene()
    scene["objects"].extend(
        [
            {
                "id": "sofa_01",
                "category": "sofa",
                "description": "small sofa",
                "size": [2.0, 0.8, 0.9],
                "center": [3.5, 3.5, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "table_01",
                "category": "coffee_table",
                "description": "coffee table",
                "size": [1.0, 0.6, 0.4],
                "center": [3.5, 2.7, 0.2],
                "rotation": [0, 0, 0],
            },
        ]
    )
    grouping = {
        "grouping_backend": "vlm",
        "grouping_policy_id": "vlm_visual_evidence_scope_v2",
        "object_groups": [
            {
                "group_id": "work",
                "object_ids": ["chair_01", "desk_01"],
            },
            {
                "group_id": "lounge",
                "object_ids": ["sofa_01", "table_01"],
            },
        ],
    }
    images = _images(
        tmp_path,
        "global",
        "work_local",
        "lounge_local",
    )
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        group_id = request["group_scope"]["group_id"]
        return [
            {
                "path": images["global"],
                "role": "global_context",
            },
            {
                "path": images[f"{group_id}_local"],
                "role": "group_local",
            },
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config=_only("object_pairing_consistency"),
        object_grouping_report=grouping,
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant(
            "object_pairing_consistency"
        ),
    )["metrics"]["object_pairing_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert len(provider_calls) == len(judge_calls) == 2
    assert [
        call["group_scope"]["group_id"] for call in provider_calls
    ] == ["work", "lounge"]
    assert [
        call["object_ids"] for call in provider_calls
    ] == [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
    assert all(len(call["group_ids"]) == 1 for call in provider_calls)
    assert [
        call["target_object_ids"] for call in judge_calls
    ] == [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
    work_scope = provider_calls[0]["group_scope"]
    assert work_scope["member_ids"] == ["chair_01", "desk_01"]
    assert work_scope["target_bounds"] == {
        "min": [0.75, 0.7, 0.0],
        "max": [2.4, 1.3, 0.9],
    }
    assert work_scope["focus_center"] == pytest.approx(
        [1.575, 1.0, 0.45]
    )
    assert work_scope["extent"] == pytest.approx(
        [1.65, 0.6, 0.9]
    )
    assert all(
        call["grouping_role"]
        == "primary_visual_evidence_decomposition"
        for call in provider_calls
    )


def test_group_local_judge_cannot_report_a_defect_from_another_group(
    tmp_path,
) -> None:
    scene = _scene()
    scene["objects"].extend(
        [
            {
                "id": "sofa_01",
                "category": "sofa",
                "size": [2.0, 0.8, 0.9],
                "center": [3.5, 3.5, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "table_01",
                "category": "coffee_table",
                "size": [1.0, 0.6, 0.4],
                "center": [3.5, 2.7, 0.2],
                "rotation": [0, 0, 0],
            },
        ]
    )
    grouping = {
        "object_groups": [
            {
                "group_id": "work",
                "object_ids": ["chair_01", "desk_01"],
            },
            {
                "group_id": "lounge",
                "object_ids": ["sofa_01", "table_01"],
            },
        ],
    }
    images = _images(tmp_path, "work", "lounge")

    def judge(request: dict) -> dict:
        if request["target_group_ids"] == ["work"]:
            return {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": 0.9,
                "reason": "Out-of-group target must be rejected.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": "category_and_role_compatibility",
                        "target_ids": ["sofa_01"],
                        "relation": "out_of_group_claim",
                        "reason": "wrong evidence scope",
                    }
                ],
            }
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        scene,
        config=_only("object_pairing_consistency"),
        object_grouping_report=grouping,
        render_evidence={
            "object_pairing_consistency": {
                "work": [images["work"]],
                "lounge": [images["lounge"]],
            }
        },
        vlm_judge=judge,
        metric_applicability=_relevant(
            "object_pairing_consistency"
        ),
    )["metrics"]["object_pairing_consistency"]

    assert metric["status"] == "unresolved"
    assert metric["score"] is None
    assert metric["group_results"][0]["reason"] == "vlm_judge_failed"
    assert metric["group_results"][1]["status"] == "evaluated"


def test_group_scope_bounds_include_object_rotation(tmp_path) -> None:
    scene = _scene()
    scene["objects"][0].update(
        center=[2.0, 2.0, 0.5],
        size=[2.0, 1.0, 1.0],
        rotation=[0.0, 0.0, 90.0],
    )
    scene["objects"] = [scene["objects"][0]]
    image = _images(tmp_path, "local")["local"]
    provider_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": image, "role": "group_local"}]

    metric = evaluate_scene_quality_interfaces(
        scene,
        config=_only("scale_consistency"),
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "rotated",
                    "object_ids": ["chair_01"],
                }
            ]
        },
        camera_evidence_provider=provider,
        vlm_judge=lambda request: _valid(),
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]

    assert metric["status"] == "evaluated"
    assert provider_calls[0]["target_bounds"]["min"] == pytest.approx(
        [1.5, 1.0, 0.0]
    )
    assert provider_calls[0]["target_bounds"]["max"] == pytest.approx(
        [2.5, 3.0, 1.0]
    )
    assert provider_calls[0]["target_extent"] == pytest.approx(
        [1.0, 2.0, 1.0]
    )


@pytest.mark.parametrize(
    "metric_name",
    [
        "scale_consistency",
        "style_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    ],
)
def test_other_group_metrics_keep_camera_requests_per_group(
    tmp_path,
    metric_name: str,
) -> None:
    scene = _scene()
    scene["objects"].extend(
        [
            {
                "id": "sofa_01",
                "category": "sofa",
                "size": [2.0, 0.8, 0.9],
                "center": [3.5, 3.5, 0.45],
                "rotation": [0, 0, 0],
            },
            {
                "id": "table_01",
                "category": "coffee_table",
                "size": [1.0, 0.6, 0.4],
                "center": [3.5, 2.7, 0.2],
                "rotation": [0, 0, 0],
            },
        ]
    )
    grouping = {
        "grouping_backend": "vlm",
        "grouping_policy_id": "vlm_visual_evidence_scope_v2",
        "object_groups": [
            {
                "group_id": "work",
                "object_ids": ["chair_01", "desk_01"],
            },
            {
                "group_id": "lounge",
                "object_ids": ["sofa_01", "table_01"],
            },
        ],
    }
    images = _images(
        tmp_path,
        "global",
        "work_local",
        "lounge_local",
    )
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        group_id = request["group_scope"]["group_id"]
        return [
            {
                "path": images[f"{group_id}_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        if (
            metric_name == "style_consistency"
            and len(judge_calls) == 1
        ):
            return _style_needs_local("scene")
        return _valid()

    config = _only(metric_name)
    if metric_name in {
        "style_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    }:
        config["metrics"][metric_name] = {
            "enabled": True,
            "weight": 1.0,
        }
    metric = evaluate_scene_quality_interfaces(
        scene,
        config=config,
        object_grouping_report=grouping,
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant(metric_name),
    )["metrics"][metric_name]

    assert metric["status"] == "evaluated"
    assert metric["affects_score"] is True
    assert [request["object_ids"] for request in provider_calls] == [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
    expected_local_targets = [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
    if metric_name == "style_consistency":
        assert judge_calls[0]["target_object_ids"] == []
        assert [
            request["target_object_ids"]
            for request in judge_calls[1:]
        ] == expected_local_targets
        assert [
            request["evidence_phase"] for request in judge_calls
        ] == [
            "global_screen",
            "local_confirmation",
            "local_confirmation",
        ]
    elif metric_name in {
        "functional_consistency",
        "semantic_placement_consistency",
    }:
        assert judge_calls[0]["target_object_ids"] == []
        assert [
            request["target_object_ids"]
            for request in judge_calls[1:]
        ] == expected_local_targets
        assert [
            request["evidence_phase"] for request in judge_calls
        ] == [
            "global_discovery",
            "group_local_review",
            "group_local_review",
        ]
    else:
        assert [
            request["target_object_ids"] for request in judge_calls
        ] == expected_local_targets
    assert all(
        request["group_scope"]["member_ids"] == request["object_ids"]
        for request in provider_calls
    )
    assert all(
        request["existing_global_evidence"] == [images["global"]]
        and request["global_context_mode"] == "reuse_existing"
        for request in provider_calls
    )
    assert all(
        images["global"] in request["render_evidence"]
        for request in judge_calls
    )
    group_judge_calls = [
        request
        for request in judge_calls
        if isinstance(request.get("group_scope"), dict)
    ]
    assert len(group_judge_calls) == 2
    assert [
        request["camera_acquisition_ledger"][
            "total_images_acquired"
        ]
        for request in group_judge_calls
    ] == [2, 2]
    if metric_name == "semantic_placement_consistency":
        assert all(
                "collision" in request["judgment_scope"]["excluded"]
                and "physical_support" in request["judgment_scope"]["excluded"]
                and "Evaluate the current support surface"
                in request["metric_rubric"]
                for request in judge_calls
            )


def test_placement_discovery_context_frames_camera_but_not_defect_scope(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "placement_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    class Planner:
        def discover_functional_evidence(self, request: dict) -> dict:
            raise AssertionError(
                "placement must not invoke functional discovery"
            )

        def discover_placement_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "placement_discovery_v1",
                "considered_object_ids": ["chair_01", "desk_01"],
                "candidates": [
                    {
                        "subject_id": "chair_01",
                        "context_ids": ["desk_01"],
                        "observation_kind": "adjacency_context",
                        "observation_goal": (
                            "show the subject relative to its context"
                        ),
                    }
                ],
                "reason": "complete",
                "decision_authority": "none",
            }

    class Provider:
        def provide_functional_boundary_evidence(
            self,
            request: dict,
        ) -> dict:
            raise AssertionError(
                "placement must not request usable-side boundary evidence"
            )

        def __call__(self, request: dict) -> list[dict]:
            provider_calls.append(request)
            return [
                {
                    "path": images["placement_local"],
                    "role": "group_local",
                }
            ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    config = _only("semantic_placement_consistency")
    config["metrics"]["semantic_placement_consistency"] = {
        "enabled": True,
        "weight": 1.0,
    }
    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report={
            "object_groups": [
                {
                    "group_id": "chair_group",
                    "object_ids": ["chair_01"],
                },
                {
                    "group_id": "desk_group",
                    "object_ids": ["desk_01"],
                },
            ]
        },
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=Provider(),
        functional_evidence_planner=Planner(),
        vlm_judge=judge,
        metric_applicability=_relevant(
            "semantic_placement_consistency"
        ),
    )["metrics"]["semantic_placement_consistency"]

    assert metric["status"] == "evaluated"
    assert provider_calls == []
    assert len(judge_calls) == 1
    global_request = judge_calls[0]
    assert global_request["target_object_ids"] == []
    assert global_request["response_contract"][
        "allowed_target_ids"
    ] == ["chair_01", "desk_01"]
    required = global_request["required_placement_checks"]
    assert len(required) == 1
    assert required[0]["check_type"] == "contextual_anchor"
    assert required[0]["subject_id"] == "chair_01"
    assert required[0]["context_ids"] == ["desk_01"]
    assert required[0]["owner_stage"] == "scene_global"
    assert global_request["response_contract"]["defects"]["fields"] == [
        "scope",
        "target_ids",
        "relation",
        "reason",
        "check_id",
        "placement_check_type",
        "severity",
    ]
    assert global_request["response_contract"]["defects"][
        "allowed_field_values"
    ] == {
        "severity": [
            "clear_semantic_misplacement",
            "material_contextual_mismatch",
        ]
    }


def test_style_unlocalized_screen_reviews_every_eligible_group(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "work_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []
    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {
                "path": images["work_local"],
                "role": "group_local",
            }
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return (
            _style_needs_local("scene")
            if len(judge_calls) == 1
            else _valid()
        )

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["route"] == "global_screen_then_group_local"
    assert metric["local_review"]["routing_fallback"] == (
        "all_eligible_non_singleton_groups"
    )
    assert len(provider_calls) == 1
    assert provider_calls[0]["group_scope"]["group_id"] == "group_001"
    assert provider_calls[0]["object_ids"] == ["chair_01", "desk_01"]
    assert judge_calls[1]["target_object_ids"] == [
        "chair_01",
        "desk_01",
    ]


def test_style_group_review_reuses_presupplied_group_evidence(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    judge_calls: list[dict] = []
    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return (
            _style_needs_local()
            if len(judge_calls) == 1
            else _valid()
        )

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "style_consistency": {
                "group_001": [images["local"]],
            },
        },
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert len(judge_calls) == 2
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["local"],
    ]
    assert metric["evidence_request"]["provider_invoked"] is False


def test_style_local_quota_does_not_truncate_global_anchor(
    tmp_path,
) -> None:
    images = _images(
        tmp_path,
        "global",
        "local_a",
        "local_b",
        "local_over_budget",
    )
    config = _style_global_then_local_config()
    config["metrics"]["style_consistency"]["evidence_plan"][
        "local_policy"
    ]["image_budget"] = 2
    judge_calls: list[dict] = []
    def provider(request: dict) -> list[dict]:
        return [
            {"path": images["local_a"], "role": "group_local"},
            {"path": images["local_b"], "role": "group_local"},
            {
                "path": images["local_over_budget"],
                "role": "group_local",
            },
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return (
            _style_needs_local()
            if len(judge_calls) == 1
            else _valid()
        )

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [images["global"]]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["status"] == "evaluated"
    assert judge_calls[1]["render_evidence"] == [
        images["global"],
        images["local_a"],
        images["local_b"],
    ]


def test_global_context_cannot_replace_group_scoped_evidence(
    tmp_path,
) -> None:
    global_image = _images(tmp_path, "global")["global"]
    judge_calls: list[dict] = []
    provider_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {
                "path": global_image,
                "role": "global_context",
            }
        ]

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("object_pairing_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={"global": [global_image]},
        camera_evidence_provider=provider,
        vlm_judge=lambda request: judge_calls.append(request),
        metric_applicability=_relevant(
            "object_pairing_consistency"
        ),
    )["metrics"]["object_pairing_consistency"]

    assert len(provider_calls) == 1
    assert judge_calls == []
    assert metric["status"] == "unresolved"
    assert metric["score"] is None
    assert metric["group_results"][0]["evidence_resolution"][
        "scope_satisfied"
    ] is False
    assert metric["group_results"][0]["evidence_paths"] == [
        global_image
    ]
    assert metric["evidence_request"]["group_requests"][0][
        "group_scope"
    ]["require_global_anchor"] is True


def test_group_scoped_evidence_can_omit_optional_global_context(
    tmp_path,
) -> None:
    local_image = _images(tmp_path, "group_local")["group_local"]
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []
    config = _only("object_pairing_consistency")
    config["metrics"]["object_pairing_consistency"][
        "evidence_policy"
    ] = {"include_global_context": False}

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": local_image, "role": "group_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report=_grouping_report(),
        render_evidence=None,
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant(
            "object_pairing_consistency"
        ),
    )["metrics"]["object_pairing_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert len(provider_calls) == len(judge_calls) == 1
    assert (
        provider_calls[0]["group_scope"]["require_global_anchor"]
        is False
    )
    assert judge_calls[0]["render_evidence"] == [local_image]


def test_exact_prompt_exemption_removes_only_the_matching_defect(tmp_path) -> None:
    scale_image = _images(tmp_path, "scale")["scale"]
    deviation = {
        "metric": "scale_consistency",
        "target_ids": ["chair_01", "desk_01"],
        "relation": "chair_intentionally_larger_than_desk",
        "source": "explicit_prompt_requirement",
        "prompt_span": "make the chair intentionally much larger",
    }

    def judge(request: dict) -> dict:
        assert request["authorized_deviations"] == [
            validate_authorized_deviations(
                [deviation],
                metric_normalizer=str,
                allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
            )[0]
        ]
        return {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "reason": "The chair is much larger than the desk.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "significant_visible_category_relative_scale_incoherence",
                    "target_ids": ["chair_01", "desk_01"],
                    "relation": "chair_intentionally_larger_than_desk",
                    "reason": "scale ratio",
                }
            ],
        }

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        render_evidence={"scale_consistency": [scale_image]},
        vlm_judge=judge,
        authorized_deviations=[deviation],
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]
    assert metric["score"] == 1.0
    assert metric["judgement"]["original_verdict"] == "invalid"
    assert len(metric["judgement"]["prompt_authorized_defects"]) == 1

    normalized = validate_authorized_deviations(
        [deviation],
        metric_normalizer=str,
        allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
    )
    assert json.loads(json.dumps(normalized)) == normalized
    assert deviation_matches(
        normalized[0],
        metric="scale_consistency",
        target_ids=["chair_01", "desk_01"],
        relation="chair_intentionally_larger_than_desk",
    )
    assert not deviation_matches(
        normalized[0],
        metric="scale_consistency",
        target_ids=["chair_01"],
        relation="different_relation",
    )


def test_functional_exemption_reconciles_only_its_explicit_check_ref() -> None:
    deviation = {
        "metric": "functional_consistency",
        "target_ids": ["chair_01"],
        "relation": "authorized_direction",
        "source": "explicit_prompt_requirement",
        "prompt_span": "author the chair with this exact direction",
    }
    judgement = {
        "verdict": "invalid",
        "reason": "Two independent checks fail.",
        "defects": [
            {
                "scope": "orientation_for_use",
                "target_ids": ["chair_01"],
                "relation": "authorized_direction",
                "reason": "The authored direction is explicitly required.",
                "check_refs": ["check_direction"],
            },
            {
                "scope": "functional_relation",
                "target_ids": ["chair_01"],
                "relation": "relative_use_geometry",
                "reason": "The chair remains too far from the table.",
                "check_refs": ["check_geometry"],
            },
        ],
        "functional_check_results": [
            {
                "check_id": "check_direction",
                "target_ids": ["chair_01", "desk_01"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The direction check fails.",
            },
            {
                "check_id": "check_geometry",
                "target_ids": ["chair_01", "desk_01"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The use-geometry check fails.",
            },
        ],
    }

    adjusted = _apply_prompt_exemptions(
        judgement,
        metric_name="functional_consistency",
        authorized_deviations=[deviation],
    )

    assert adjusted["verdict"] == "invalid"
    assert [
        row["conclusion"]
        for row in adjusted["functional_check_results"]
    ] == ["valid", "invalid"]
    assert adjusted["defects"][0]["check_refs"] == ["check_geometry"]


def test_asset_applicability_controls_denominator_without_defaulting_valid() -> None:
    calls: list[dict] = []

    not_relevant = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        render_evidence=["global.png"],
        vlm_judge=lambda request: calls.append(request),
        metric_applicability={
            "style_consistency": {"applicability": "not_relevant"}
        },
    )
    metric = not_relevant["metrics"]["style_consistency"]
    assert metric["status"] == "not_applicable"
    assert metric["affects_score"] is False
    assert calls == []

    pending = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        render_evidence=["global.png"],
        vlm_judge=lambda request: calls.append(request),
        metric_applicability={
            "style_consistency": {"applicability": "pending"}
        },
    )
    metric = pending["metrics"]["style_consistency"]
    assert metric["status"] == "unresolved"
    assert metric["reason"] == "metric_applicability_pending"
    assert metric["score"] is None
    assert calls == []


def test_metric_specific_evidence_budget_and_style_context(tmp_path) -> None:
    calls: list[dict] = []
    images = _images(tmp_path, "g1", "g2", "s1")
    visual_style_spec = {
        "spec_version": "visual_style_spec_v1",
        "source": "benchmark_owned",
        "frozen": True,
        "directives": [
            {
                "directive_id": "style_1",
                "statement": "Use a consistent low-poly style.",
                "required": True,
            }
        ],
    }

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        render_evidence={
            "style_consistency": [images["g1"], images["g2"]],
            "scale_consistency": [images["s1"]],
        },
        vlm_judge=lambda request: (
            calls.append(request)
            or {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "reason": "consistent",
                "missing_evidence": [],
                "defects": [],
            }
        ),
        prompt="A consistent low-poly room.",
        visual_style_spec=visual_style_spec,
        metric_applicability=_relevant("style_consistency"),
    )
    style = report["metrics"]["style_consistency"]
    assert style["evidence_paths"] == [images["g1"]]  # default budget is one
    assert calls[0]["visual_style_spec"] == visual_style_spec
    assert calls[0]["render_evidence"] == [images["g1"]]


def test_camera_provider_not_invoked_when_scope_correct_evidence_exists(tmp_path) -> None:
    class Provider:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"camera provider must not be invoked: {name}")

    images = _images(tmp_path, "global", "local")
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "style_consistency": {
                "group_001": [images["local"]],
            },
        },
        camera_evidence_provider=Provider(),
        vlm_judge=lambda request: {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "reason": "consistent",
            "missing_evidence": [],
            "defects": [],
        },
        metric_applicability=_relevant("style_consistency"),
    )
    assert report["renderer_invoked"] is False
    assert report["metrics"]["style_consistency"]["status"] == "evaluated"


def test_flat_overview_can_resolve_a_clear_global_style_screen(
    tmp_path,
) -> None:
    global_image = _images(tmp_path, "global")["global"]
    calls: list[str] = []

    def judge(request: dict) -> dict:
        calls.append(request["metric"])
        return _valid()

    report = evaluate_scene_quality_interfaces(
        _scene(),
        object_grouping_report=_grouping_report(),
        render_evidence=[global_image],
        vlm_judge=judge,
        metric_applicability=_relevant(*SCENE_QUALITY_INTERFACE_METRICS),
    )
    assert calls == [
        "style_consistency",
        "scale_consistency",
        "object_pairing_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    ]
    assert report["metrics"]["style_consistency"]["status"] == "evaluated"
    assert report["metrics"]["style_consistency"]["route"] == (
        "global_clear"
    )
    assert report["metrics"]["scale_consistency"]["route"] == (
        "json_screen_resolved"
    )
    assert report["metrics"]["object_pairing_consistency"][
        "route"
    ] == "json_screen_resolved"
    # The global image resolves the three global/screen-first metrics, but it
    # cannot replace the required local scopes of functional and placement.
    assert report["status"] == "partial"
    assert report["score"] is None
    assert report["resolved_score"] == 1.0


def test_local_provider_is_selection_only_and_supplies_missing_scope(tmp_path) -> None:
    images = _images(tmp_path, "provider_global", "scale_local")
    local_image = images["scale_local"]
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [
            {
                "path": images["provider_global"],
                "role": "metric_highlighted_global",
            },
            {"path": local_image, "role": "metric_local_rgb"},
        ]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        render_evidence=[],
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("scale_consistency"),
    )
    metric = report["metrics"]["scale_consistency"]
    assert metric["status"] == "evaluated"
    assert metric["evidence_paths"] == [local_image]
    assert report["camera_evidence_provider_invoked"] is True
    assert report["renderer_invoked"] is False
    assert len(provider_calls) == len(judge_calls) == 1
    request = provider_calls[0]
    assert request["metric"] == "scale_consistency"
    assert request["evidence_scope"] == "object_local"
    assert request["selection_role"] == "visual_evidence_only_do_not_judge_metric"
    assert set(request["object_ids"]) == {"chair_01", "desk_01"}


@pytest.mark.parametrize(
    "applicability",
    [
        {"applicability": "not_relevant"},
        {"applicability": "pending"},
    ],
)
def test_provider_not_invoked_before_applicability_gate(
    applicability: dict,
) -> None:
    calls: list[dict] = []

    def provider(request: dict) -> list[str]:
        calls.append(request)
        return ["must_not_be_used.png"]

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        camera_evidence_provider=provider,
        vlm_judge=lambda request: pytest.fail("judge must not run"),
        metric_applicability={"scale_consistency": applicability},
    )
    assert calls == []
    assert report["metrics"]["scale_consistency"]["score"] is None


def test_provider_not_invoked_without_final_judge() -> None:
    calls: list[dict] = []

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        camera_evidence_provider=lambda request: calls.append(request),
        metric_applicability=_relevant("scale_consistency"),
    )
    assert calls == []
    assert report["metrics"]["scale_consistency"]["reason"] == (
        "vlm_judge_not_configured"
    )


@pytest.mark.parametrize(
    "provider_result",
    [
        [],
        {"status": "insufficient", "reason": "target_occluded"},
        {"status": "failed", "error": "renderer unavailable"},
        {"unexpected": "shape"},
    ],
)
def test_provider_failure_never_calls_final_judge(
    provider_result: object,
) -> None:
    judge_calls: list[dict] = []
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        camera_evidence_provider=lambda request: provider_result,
        vlm_judge=lambda request: judge_calls.append(request),
        metric_applicability=_relevant("scale_consistency"),
    )
    metric = report["metrics"]["scale_consistency"]
    assert metric["status"] == "unresolved"
    assert metric["score"] is None
    assert judge_calls == []


def test_nonexistent_scoped_evidence_never_reaches_custom_judge() -> None:
    calls: list[dict] = []
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        render_evidence={"scale_consistency": ["/definitely/missing/image.png"]},
        vlm_judge=lambda request: calls.append(request),
        metric_applicability=_relevant("scale_consistency"),
    )
    metric = report["metrics"]["scale_consistency"]
    assert metric["status"] == "unresolved"
    assert metric["reason"] == "render_evidence_path_missing"
    assert metric["evidence_request"]["missing_paths"] == [
        "/definitely/missing/image.png"
    ]
    assert calls == []


@pytest.mark.parametrize(
    "response",
    [
        {"score": 1.0},
        {"evidence_status": "sufficient", "verdict": "valid"},
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "missing_evidence": [],
            "defects": [],
        },
        {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "missing_evidence": ["still missing"],
            "defects": [],
        },
        {
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "missing_evidence": [],
            "defects": [],
        },
        {
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "missing_evidence": ["target"],
            "defects": [
                {
                    "scope": "significant_visible_category_relative_scale_incoherence",
                    "target_ids": ["chair_01"],
                    "relation": "scale",
                    "reason": "asserted despite insufficient evidence",
                }
            ],
        },
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "reason": "wrong scope",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "orientation",
                    "target_ids": ["chair_01"],
                    "relation": "faces",
                    "reason": "wrong metric",
                }
            ],
        },
    ],
)
def test_malformed_judgement_is_unresolved(
    tmp_path,
    response: dict,
) -> None:
    scale_image = _images(tmp_path, "scale")["scale"]
    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("scale_consistency"),
        render_evidence={"scale_consistency": [scale_image]},
        vlm_judge=lambda request: response,
        metric_applicability=_relevant("scale_consistency"),
    )["metrics"]["scale_consistency"]
    assert metric["status"] == "unresolved"
    assert metric["reason"] == "vlm_judge_failed"
    assert metric["score"] is None


def test_declared_applicability_map_is_closed() -> None:
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        metric_applicability={"scale_consistency": {"applicability": "relevant"}},
    )
    assert report["metrics"]["style_consistency"]["reason"] == (
        "metric_applicability_pending"
    )
    with pytest.raises(ValueError, match="unknown metrics"):
        evaluate_scene_quality_interfaces(
            _scene(),
            metric_applicability={"visual_vibes": True},
        )
    with pytest.raises(TypeError, match="JSON object"):
        evaluate_scene_quality_interfaces(
            _scene(),
            metric_applicability=["style_consistency"],
        )


@pytest.mark.parametrize(
    "grouping",
    [
        {"groups": []},
        {
            "object_groups": [
                {"group_id": "g", "object_ids": ["chair_01", "chair_01"]},
                {"group_id": "g2", "object_ids": ["desk_01"]},
            ]
        },
        {
            "object_groups": [
                {"group_id": "g", "object_ids": ["chair_01", "ghost"]},
                {"group_id": "g2", "object_ids": ["desk_01"]},
            ]
        },
        {
            "object_groups": [
                {"group_id": "g", "object_ids": ["chair_01"]},
            ]
        },
    ],
)
def test_malformed_grouping_is_rejected(grouping: dict) -> None:
    with pytest.raises(ValueError, match="object_grouping_report"):
        evaluate_scene_quality_interfaces(
            _scene(),
            object_grouping_report=grouping,
        )


def test_complete_l3_subset_aggregation_uses_enabled_metric_scores(tmp_path) -> None:
    images = _images(tmp_path, "global", "style_local", "scale", "pair")

    def judge(request: dict) -> dict:
        if request["metric"] != "scale_consistency":
            return _valid()
        return {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "The chair is visibly too large for the desk.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "significant_visible_category_relative_scale_incoherence",
                    "target_ids": ["chair_01", "desk_01"],
                    "relation": "chair_too_large_for_desk",
                    "reason": "The relative scale is significantly incoherent.",
                }
            ],
        }

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config={
            "metrics": {
                "functional_consistency": {"enabled": False},
                "semantic_placement_consistency": {"enabled": False},
            }
        },
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "style_consistency": {
                "group_001": [images["style_local"]],
            },
            "scale_consistency": [images["scale"]],
            "object_pairing_consistency": [images["pair"]],
        },
        vlm_judge=judge,
        metric_applicability=_relevant(
            "style_consistency",
            "scale_consistency",
            "object_pairing_consistency",
        ),
    )
    assert report["status"] == "evaluated"
    assert report["score"] == pytest.approx(2.0 / 3.0)
    assert report["resolved_score"] == pytest.approx(2.0 / 3.0)
    assert report["resolved_metrics"] == [
        "style_consistency",
        "scale_consistency",
        "object_pairing_consistency",
    ]


def test_retired_namespaces_and_metric_aliases_are_rejected() -> None:
    for retired_namespace in (
        "scene_quality_interfaces",
        "visual_quality_interfaces",
    ):
        with pytest.raises(SceneQualityInterfaceConfigError, match="retired"):
            resolve_scene_quality_config(
                profile={retired_namespace: {"enabled": True}}
            )
        with pytest.raises(SceneQualityInterfaceConfigError, match="retired"):
            resolve_scene_quality_config(
                {retired_namespace: {"enabled": True}}
            )

    with pytest.raises(SceneQualityInterfaceConfigError, match="retired"):
        resolve_scene_quality_config(
            {
                "metrics": {
                    "object_coexistence_consistency": {
                        "evidence_policy": {"image_budget": 2}
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="unknown metrics"):
        evaluate_scene_quality_interfaces(
            _scene(),
            metric_applicability={
                "object_coexistence_consistency": {
                    "applicability": "relevant",
                }
            },
        )
    with pytest.raises(ValueError, match="retired L3 metric keys"):
        evaluate_scene_quality_interfaces(
            _scene(),
            render_evidence={"object_coexistence_consistency": ["/tmp/old.png"]},
        )


def test_invalid_canonical_configs_are_explicit() -> None:
    with pytest.raises(SceneQualityInterfaceConfigError, match="camera_scope"):
        resolve_scene_quality_config(
            {
                "metrics": {
                    "style_consistency": {
                        "evidence_policy": {"camera_scope": "orbit"}
                    }
                }
            }
        )
    with pytest.raises(SceneQualityInterfaceConfigError, match="must remain true"):
        resolve_scene_quality_config({"implemented": False})
    with pytest.raises(AuthorizedDeviationError, match="target-specific"):
        validate_authorized_deviations(
            [
                {
                    "metric": "style_consistency",
                    "target_ids": [],
                    "relation": "contrast",
                    "source": "explicit_prompt_requirement",
                }
            ],
            allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
        )


def test_functional_ownership_is_exposed_only_after_metric_resolution() -> None:
    ledger = {
        "schema_version": "functional_ownership_ledger_v1",
        "source_metric": "functional_consistency",
        "events": [
            {
                "event_id": "functional_event:chair-blocker",
                "affected_object_ids": ["desk_01"],
                "cause_kind": "external_object",
                "causal_object_ids": ["chair_01"],
                "scoring_target_ids": ["chair_01"],
                "decision_ref": "judge:function:1",
                "lifecycle_status": "final",
                "decision_authority": "none",
            }
        ],
        "event_count": 1,
        "decision_authority": "none",
        "projection_mode": "posthoc_read_only",
    }
    unresolved = {
        "functional_consistency": {
            "status": "unresolved",
            "functional_ownership_ledger": deepcopy(ledger),
        }
    }
    evaluated = {
        "functional_consistency": {
            "status": "evaluated",
            "functional_ownership_ledger": deepcopy(ledger),
        }
    }

    assert _resolved_functional_ownership_for_placement(
        unresolved,
        object_ids=["chair_01", "desk_01"],
    ) is None
    assert _resolved_functional_ownership_for_placement(
        evaluated,
        object_ids=["chair_01", "desk_01"],
    ) == ledger
    malformed = deepcopy(evaluated)
    malformed["functional_consistency"]["functional_ownership_ledger"][
        "events"
    ][0]["lifecycle_status"] = "pending"
    with pytest.raises(ValueError, match="must be final"):
        _resolved_functional_ownership_for_placement(
            malformed,
            object_ids=["chair_01", "desk_01"],
        )


def test_controller_audit_projects_deferred_local_placement_check() -> None:
    check = {
        "check_id": "placement_check:deferred",
        "check_type": "support_and_height",
        "subject_id": "chair_01",
        "context_ids": [],
        "owner_stage": "group_local",
    }
    audit_records = [
        {
            "audit": {
                "judge_request": {
                    "context": {
                        "required_placement_checks": [],
                        "deferred_placement_checks": [check],
                    }
                }
            }
        }
    ]

    assert _registered_placement_checks_from_controller_audit(
        audit_records,
        audit_start=0,
    ) == [check]


def test_global_scope_does_not_require_deferred_local_check_result() -> None:
    deferred = {
        "check_id": "placement_check:deferred-support",
        "check_type": "support_and_height",
        "subject_id": "chair_01",
        "context_ids": [],
        "target_ids": ["chair_01"],
        "owner_stage": "group_local",
        "owning_group_id": "group_001",
        "group_ids": ["group_001"],
        "required_observations": [
            "target_visible",
            "contact_surface_visible",
            "group_context_visible",
        ],
        "observation_goals": ["Inspect support and height."],
        "source_observation_kinds": ["support_and_height"],
        "source_discovery_refs": ["global-miss"],
        "origin": "judge_originated_evidence_request",
        "lifecycle_status": "evidence_requested",
        "acquisition_status": "pending",
        "observation_complete": False,
        "judge_status": "pending",
        "judge_result_ref": None,
        "decision_authority": "none",
        "handoff_status": "deferred_to_group_local",
        "handoff_from_stage": "scene_global",
    }

    class Judge:
        audit_records: list[dict] = []

    judge = Judge()

    def call_judge(_judge, _request):
        judge.audit_records.append(
            {
                "audit": {
                    "judge_request": {
                        "context": {
                            "required_placement_checks": [],
                            "deferred_placement_checks": [deferred],
                        }
                    }
                }
            }
        )
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "No scene-global placement defect.",
            "missing_evidence": [],
            "defects": [],
            "placement_check_results": [],
        }

    base = {
        "evidence_request": {},
        "placement_check_ledger": {
            "schema_version": "placement_check_ledger_v1",
            "checks": [],
            "accepted_check_count": 0,
            "decision_authority": "none",
        },
    }
    record, outcome, _ = _evaluate_global_scope(
        base=base,
        metric_name="semantic_placement_consistency",
        scene=_scene(),
        object_ids=["chair_01", "desk_01"],
        groups=_grouping_report()["object_groups"],
        global_evidence=["global.png"],
        functional_probe_packet=None,
        vlm_judge=judge,
        prompt=None,
        visual_style_spec=None,
        authorized_deviations=[],
        build_judge_request=lambda **kwargs: deepcopy(kwargs),
        call_judge=call_judge,
        apply_prompt_exemptions=lambda value, **_: deepcopy(value),
        normalize_judgement=lambda value, **_: {
            "status": "evaluated",
            "score": 1.0,
            "reason": None,
        },
        camera_acquisition_ledger={},
        forbidden_cross_group_target_sets=[],
        required_placement_checks=[],
        functional_ownership_ledger=None,
    )

    assert outcome["status"] == "evaluated"
    assert record["global_status"] == "clear"
    check = base["placement_check_ledger"]["checks"][0]
    assert check["check_id"] == deferred["check_id"]
    assert check["judge_status"] == "pending"
