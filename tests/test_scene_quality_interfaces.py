from __future__ import annotations

import json

import pytest

from benchmark.evaluator.scene_quality import (
    PERCEPTUAL_VISUAL_QUALITY_METRICS,
    SCENE_QUALITY_INTERFACE_METRICS,
    SEMANTIC_COHERENCE_METRICS,
    evaluate_scene_quality_interfaces,
    resolve_scene_quality_config,
    validate_authorized_deviations,
)
from benchmark.evaluator.scene_quality.authorized_deviations import (
    AuthorizedDeviationError,
    deviation_matches,
)
from benchmark.evaluator.scene_quality.interfaces import (
    SCENE_QUALITY_INTERFACE_NAMESPACE,
    SceneQualityInterfaceConfigError,
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
    return {
        "enabled": True,
        "metrics": {
            name: {"enabled": name == metric}
            for name in SCENE_QUALITY_INTERFACE_METRICS
        },
    }


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


def _valid() -> dict:
    return {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.9,
        "reason": "No significant in-scope defect.",
        "missing_evidence": [],
        "defects": [],
    }


def _style_global_then_local_config() -> dict:
    config = _only("style_consistency")
    config["metrics"]["style_consistency"].update(
        {
            "evidence_policy": {
                "camera_scope": "global",
                "camera_mode": "global_top",
                "selector": "deterministic",
                "image_budget": 2,
                "presentation": "raw",
                "image_order": None,
                "include_global_context": True,
                "camera_pose_mode": None,
            },
            "evidence_plan": {
                "evidence_strategy": "global_screen_then_local",
                "global_policy": {
                    "view_family": "canonical_high_oblique_pair_v1",
                    "image_budget": 2,
                    "top_down": False,
                },
                "local_policy": {
                    "camera_scope": "object_local",
                    "grouping_policy_id": "vlm_visual_evidence_scope_v2",
                    "image_budget": 4,
                    "trigger_states": [
                        "suspicious",
                        "insufficient_evidence",
                    ],
                },
                "router_options": {
                    "global_screen_then_local": {
                        "router": "vlm_global_screen",
                        "trigger_states": [
                            "suspicious",
                            "insufficient_evidence",
                        ],
                        "executes_router": False,
                    }
                },
                "text_context": [
                    "original_prompt",
                    "authorized_deviations",
                    "asset_policy",
                ],
            },
        }
    )
    return config


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
    assert (
        resolved["metrics"]["functional_consistency"]["enabled"]
        is False
    )
    serialized = json.dumps(resolved, sort_keys=True)
    assert "object_coexistence_consistency" not in serialized
    assert "sceneonto_scale_candidate_router" not in serialized
    assert "benchmark.evaluator.spatial_fidelity.scale" not in serialized
    scale = resolved["metrics"]["scale_consistency"]
    assert scale["evidence_policy"]["camera_scope"] == "group_local"
    assert scale["evidence_plan"]["local_policy"][
        "grouping_policy_id"
    ] == "vlm_visual_evidence_scope_v2"

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


def test_style_global_screen_does_not_request_local_when_clear(tmp_path) -> None:
    images = _images(tmp_path, "global_a", "global_b")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> dict:
        provider_calls.append(request)
        return {"status": "available", "paths": []}

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        render_evidence=[images["global_a"], images["global_b"]],
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["score"] == 1.0
    assert metric["route"] == "global_screen_resolved"
    assert metric["router_state"] == "not_suspicious"
    assert metric["judge_call_count"] == 1
    assert provider_calls == []
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_screen"
    ]


def test_style_suspicion_requests_local_then_rejudges(tmp_path) -> None:
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

    suspicious = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair appears to use a conflicting rendering style.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "significant_visible_style_incompatibility",
                "target_ids": ["chair_01"],
                "relation": "rendering_style_outlier",
                "reason": "The chair appears to use a conflicting rendering style.",
            }
        ],
    }

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return suspicious if len(judge_calls) == 1 else _valid()

    metric = evaluate_scene_quality_interfaces(
        _scene(),
        config=_style_global_then_local_config(),
        render_evidence=[images["global_a"], images["global_b"]],
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability=_relevant("style_consistency"),
    )["metrics"]["style_consistency"]

    assert metric["score"] == 1.0
    assert metric["route"] == "global_screen_then_local"
    assert metric["router_state"] == "suspicious"
    assert metric["judge_call_count"] == 2
    assert metric["local_evidence_paths"] == [
        images["local_a"],
        images["local_b"],
    ]
    assert provider_calls[0]["object_ids"] == ["chair_01"]
    assert [call["evidence_phase"] for call in judge_calls] == [
        "global_screen",
        "local_confirmation",
    ]
    assert judge_calls[1]["render_evidence"] == [
        images["local_a"],
        images["local_b"],
        images["global_a"],
        images["global_b"],
    ]


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


def test_all_three_metrics_evaluate_and_aggregate_when_complete(tmp_path) -> None:
    calls: list[dict] = []
    images = _images(tmp_path, "global", "scale", "pair_1", "pair_2")

    def judge(request: dict) -> dict:
        calls.append(request)
        return _valid()

    report = evaluate_scene_quality_interfaces(
        _scene(),
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "scale_consistency": [images["scale"]],
            "object_pairing_consistency": [
                images["pair_1"],
                images["pair_2"],
            ],
        },
        vlm_judge=judge,
        prompt="Create a coherent bedroom work corner.",
        metric_applicability=_relevant(*SCENE_QUALITY_INTERFACE_METRICS),
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
    assert {request["metric"] for request in calls} == set(
        SCENE_QUALITY_INTERFACE_METRICS
    )
    assert all(request["vlm_role"] == "judge" for request in calls)
    assert all(
        request["decision_contract"] == "canonical_metric_v1"
        for request in calls
    )
    assert all(
        request["judge_method"] == "adjudicate_scene_quality"
        for request in calls
    )
    pairing_request = next(
        request
        for request in calls
        if request["metric"] == "object_pairing_consistency"
    )
    assert pairing_request["object_groups"][0]["group_id"] == "group_001"
    assert "orientation" in pairing_request["judgment_scope"]["excluded"]
    assert pairing_request["natural_language_prompt"].startswith("Create")
    assert pairing_request["render_evidence"] == [
        images["global"],
        images["pair_1"],
        images["pair_2"],
    ]
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
    ["scale_consistency", "functional_consistency"],
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
        return _valid()

    config = _only(metric_name)
    if metric_name == "functional_consistency":
        config["metrics"]["functional_consistency"] = {
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
    assert metric["affects_score"] is (
        metric_name == "scale_consistency"
    )
    assert [request["object_ids"] for request in provider_calls] == [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
    assert [request["target_object_ids"] for request in judge_calls] == [
        ["chair_01", "desk_01"],
        ["sofa_01", "table_01"],
    ]
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


def test_style_suspicion_drills_into_the_implicated_group_only(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "work_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []
    suspicious = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair may use an inconsistent style.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "significant_visible_style_incompatibility",
                "target_ids": ["chair_01"],
                "relation": "rendering_style_outlier",
                "reason": "The chair is the suspected outlier.",
            }
        ],
    }

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
        return suspicious if len(judge_calls) == 1 else _valid()

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
    assert len(provider_calls) == 1
    assert provider_calls[0]["group_scope"]["group_id"] == "group_001"
    assert provider_calls[0]["object_ids"] == ["chair_01", "desk_01"]
    assert judge_calls[1]["target_object_ids"] == [
        "chair_01",
        "desk_01",
    ]


def test_style_drilldown_reuses_presupplied_group_evidence(
    tmp_path,
) -> None:
    images = _images(tmp_path, "global", "local")
    judge_calls: list[dict] = []
    suspicious = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The chair may be a style outlier.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "significant_visible_style_incompatibility",
                "target_ids": ["chair_01"],
                "relation": "rendering_style_outlier",
                "reason": "suspected outlier",
            }
        ],
    }

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return suspicious if len(judge_calls) == 1 else _valid()

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
        images["local"],
        images["global"],
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
    suspicious = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.4,
        "reason": "Local confirmation is needed.",
        "missing_evidence": ["local group style"],
        "defects": [],
        "evidence_request": {
            "target_ids": ["chair_01"],
            "missing_observations": ["group_context_visible"],
            "view_goal": "inspect the suspected group style",
        },
    }

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
        return suspicious if len(judge_calls) == 1 else _valid()

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
        images["local_a"],
        images["local_b"],
        images["global"],
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

    global_image = _images(tmp_path, "global")["global"]
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_only("style_consistency"),
        render_evidence=[global_image],
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


def test_flat_overview_satisfies_only_global_style_scope(tmp_path) -> None:
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
    assert calls == ["style_consistency"]
    assert report["metrics"]["style_consistency"]["status"] == "evaluated"
    assert report["metrics"]["scale_consistency"]["reason"] == (
        "group_local_render_evidence_unavailable"
    )
    assert report["metrics"]["object_pairing_consistency"]["reason"] == (
        "group_local_render_evidence_unavailable"
    )
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


def test_complete_l3_aggregation_uses_only_canonical_metric_scores(tmp_path) -> None:
    images = _images(tmp_path, "global", "scale", "pair")

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
        object_grouping_report=_grouping_report(),
        render_evidence={
            "global": [images["global"]],
            "scale_consistency": [images["scale"]],
            "object_pairing_consistency": [images["pair"]],
        },
        vlm_judge=judge,
        metric_applicability=_relevant(*SCENE_QUALITY_INTERFACE_METRICS),
    )
    assert report["status"] == "evaluated"
    assert report["score"] == pytest.approx(2.0 / 3.0)
    assert report["resolved_score"] == pytest.approx(2.0 / 3.0)
    assert report["resolved_metrics"] == list(SCENE_QUALITY_INTERFACE_METRICS)


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
