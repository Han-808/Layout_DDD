from __future__ import annotations

import json

import pytest
from PIL import Image

import benchmark.evaluator as evaluator_package
import benchmark.evaluator.specification_fidelity as specification_package
from benchmark.evaluator.specification_fidelity.coarse_interfaces import (
    DEFAULT_COARSE_SPECIFICATION_CONFIG,
    DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG,
    FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE,
    FUNCTIONAL_SEMANTIC_INTERFACE_VERSION,
    FUNCTIONAL_SEMANTIC_METRICS,
    CoarseSpecificationConfigError,
    evaluate_coarse_specification_interfaces,
    evaluate_functional_semantic_fidelity,
    normalize_coarse_metric_name,
    normalize_functional_semantic_metric_name,
    resolve_functional_semantic_config,
    resolve_coarse_specification_config,
)
from benchmark.visual_judge import OpenAICompatibleVLMJudge


def _scene() -> dict:
    return {
        "scene_id": "s",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "desk",
                "category": "desk",
                "center": [1.5, 1.5, 0.4],
                "size": [1.2, 0.7, 0.8],
                "rotation": [0, 0, 0],
            },
            {
                "id": "chair",
                "category": "chair",
                "center": [1.5, 2.3, 0.5],
                "size": [0.5, 0.5, 1.0],
                "rotation": [0, 0, 180],
            },
            {
                "id": "unrelated_lamp",
                "category": "lamp",
                "center": [3.5, 3.5, 0.8],
                "size": [0.4, 0.4, 1.6],
                "rotation": [0, 0, 0],
            },
        ],
    }


def _contract(*claims: dict) -> dict:
    return {
        "contract_version": "specification_contract_v1",
        "source": "benchmark_owned",
        "frozen": True,
        "claims": {"functional_semantic_fidelity": list(claims)},
    }


def _claim(claim_id: str, component: str, **extra) -> dict:
    return {
        "claim_id": claim_id,
        "claim_family": "functional_semantic_fidelity",
        "component": component,
        **extra,
    }


class RecordingProvider:
    def __init__(self, paths: list[str] | None = None) -> None:
        self.paths = paths or ["local.png"]
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> list[str]:
        self.requests.append(request)
        return list(self.paths)


class RecordingJudge:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def adjudicate_functional_semantic(self, request: dict) -> dict:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeCanonicalModel:
    model_id = "fake-canonical-model"
    endpoint = "http://127.0.0.1/fake"
    response_format_json = True
    last_request_metadata: dict = {}

    def __init__(self) -> None:
        self.messages: list = []

    def chat_messages(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "the requested room type is visible",
                "defects": [],
                "missing_evidence": [],
            }
        )


def test_disabled_runtime_is_non_invoking() -> None:
    provider = RecordingProvider()
    judge = RecordingJudge([{"verdict": "valid"}])
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": False},
        specification_contract=_contract(
            _claim("room", "room_scene_type", expected={"scene_type": "bedroom"})
        ),
        render_evidence=["global.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
    )
    assert report["implemented"] is True
    assert report["status"] == "not_applicable"
    assert report["score"] is None
    assert report["renderer_invoked"] is False
    assert report["vlm_invoked"] is False
    assert "metric_aliases" not in report
    assert "aliases" not in report["metrics"]["functional_semantic_fidelity"]
    assert provider.requests == []
    assert judge.requests == []


def test_legacy_function_names_are_aliases_of_canonical_runtime() -> None:
    assert evaluate_coarse_specification_interfaces is evaluate_functional_semantic_fidelity
    assert resolve_coarse_specification_config is resolve_functional_semantic_config


def test_canonical_names_are_primary_and_deprecated_helper_is_isolated() -> None:
    assert FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE == "functional_semantic_fidelity"
    assert (
        FUNCTIONAL_SEMANTIC_INTERFACE_VERSION
        == "functional_semantic_fidelity_runtime_v1"
    )
    assert FUNCTIONAL_SEMANTIC_METRICS == ("functional_semantic_fidelity",)
    assert DEFAULT_COARSE_SPECIFICATION_CONFIG is DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG
    assert (
        normalize_functional_semantic_metric_name("room_scene_type")
        == "functional_semantic_fidelity"
    )
    assert normalize_coarse_metric_name is normalize_functional_semantic_metric_name
    assert not hasattr(
        specification_package, "normalize_functional_semantic_metric_name"
    )
    assert not hasattr(specification_package, "FUNCTIONAL_SEMANTIC_ALIASES")
    assert not hasattr(evaluator_package, "FUNCTIONAL_SEMANTIC_ALIASES")


def test_canonical_l2_profile_controls_only_functional_metric() -> None:
    resolved = resolve_coarse_specification_config(
        profile={
            "l2_specification_fidelity": {
                "enabled": True,
                "metrics": {
                    "oor": {"enabled": False, "weight": 1.0 / 3.0},
                    "oar": {"enabled": False, "weight": 1.0 / 3.0},
                    "functional_semantic_fidelity": {
                        "enabled": False,
                        "weight": 1.0 / 3.0,
                    },
                },
            }
        }
    )
    assert resolved["enabled"] is True
    assert resolved["metrics"]["functional_semantic_fidelity"]["enabled"] is False
    assert "oor" not in resolved["metrics"]
    assert "oar" not in resolved["metrics"]


def test_global_room_claim_uses_prompt_and_global_evidence_only() -> None:
    provider = RecordingProvider()
    judge = RecordingJudge(
        [{"verdict": "valid", "confidence": 0.9, "reason": "bedroom visible"}]
    )
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Create a bedroom.",
        specification_contract=_contract(
            _claim("room", "room_scene_type", expected={"scene_type": "bedroom"})
        ),
        render_evidence=["global_a.png", "global_b.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
    )
    metric = report["metrics"]["functional_semantic_fidelity"]
    assert report["category"] == "functional_semantic_fidelity"
    assert report["level"] == "l2_specification_fidelity"
    assert report["affects_aggregation"] is True
    assert metric["namespace"] == "functional_semantic_fidelity"
    assert metric["status"] == "evaluated"
    assert metric["score"] == 1.0
    assert metric["coverage"]["complete"] is True
    assert provider.requests == []
    assert len(judge.requests) == 1
    request = judge.requests[0]
    assert request["phase"] == "global"
    assert request["natural_language_prompt"] == "Create a bedroom."
    assert request["claims"][0]["claim_id"] == "room"
    assert request["components"] == ["room_scene_type"]
    assert request["evidence_phase"] == "global"
    assert request["judgment_scope"]["excluded"] == [
        "generic_object_pairing",
        "generic_scale_coherence",
        "visual_style",
        "oor",
        "oar",
        "unprompted_local_functionality",
    ]
    assert request["relevant_global_visual_evidence"] == [
        "global_a.png",
        "global_b.png",
    ]
    assert request["relevant_local_visual_evidence"] == []
    assert request["generic_pairing_scan"] is False


def test_prompt_explicit_local_functionality_is_claim_target_scoped() -> None:
    provider = RecordingProvider(["desk_chair_local.png"])
    judge = RecordingJudge([{"verdict": "valid", "confidence": 0.8}])
    grouping = {
        "groups": [
            {
                "group_id": "generic_group",
                "object_ids": ["desk", "chair", "unrelated_lamp"],
            }
        ]
    }
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="The chair should serve the desk.",
        specification_contract=_contract(
            _claim(
                "local",
                "local_functionality",
                target_ids=["desk", "chair"],
                expected={"function": "seated desk use"},
            )
        ),
        render_evidence=["global.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
        object_grouping_report=grouping,
    )
    assert report["score"] == 1.0
    assert report["object_grouping_report_supplied"] is True
    assert report["object_grouping_report_consumed"] is False
    assert len(provider.requests) == 1
    evidence_request = provider.requests[0]
    assert evidence_request["scene"]["scene_id"] == "s"
    assert evidence_request["object_ids"] == ["desk", "chair"]
    assert "unrelated_lamp" not in evidence_request["object_ids"]
    assert evidence_request["trigger"] == "prompt_specified_local_functionality"
    assert evidence_request["generic_pairing_scan"] is False
    assert judge.requests[0]["phase"] == "prompt_scoped_local"
    assert judge.requests[0]["relevant_local_visual_evidence"] == [
        "desk_chair_local.png"
    ]
    assert "object_grouping_report" not in judge.requests[0]
    assert judge.requests[0]["render_evidence"] == [
        "desk_chair_local.png",
        "global.png",
    ]
    assert judge.requests[0]["image_order"] == "local_first_then_global"


def test_required_area_suspicious_global_screen_triggers_local_fallback() -> None:
    provider = RecordingProvider(["work_area_local.png"])
    judge = RecordingJudge(
        [
            {
                "router_state": "suspicious",
                "confidence": 0.7,
                "reason": "area partly occluded",
            },
            {
                "verdict": "valid",
                "confidence": 0.9,
                "reason": "local view resolves it",
            },
        ]
    )
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Include a usable work area.",
        specification_contract=_contract(
            _claim(
                "area",
                "required_functional_areas",
                target_ids=["desk", "chair"],
                expected={"area": "work"},
            )
        ),
        render_evidence=["global.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["status"] == "checked"
    assert check["score"] == 1.0
    assert check["router_state"] == "suspicious"
    assert len(provider.requests) == 1
    assert provider.requests[0]["trigger"] == "suspicious"
    assert [request["phase"] for request in judge.requests] == [
        "required_area_global_screen",
        "required_area_local_fallback",
    ]


def test_required_area_not_suspicious_resolves_without_local_call() -> None:
    provider = RecordingProvider()
    # A strict screen response is preferred. A legacy binary valid response is
    # also normalized to not_suspicious.
    judge = RecordingJudge([{"verdict": "valid", "confidence": 0.95}])
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Include a work area.",
        specification_contract=_contract(
            _claim("area", "required_functional_areas", expected={"area": "work"})
        ),
        render_evidence=["global.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["route"] == "global_screen_resolved"
    assert check["score"] == 1.0
    assert provider.requests == []


def test_required_area_insufficient_without_provider_stays_unresolved() -> None:
    judge = RecordingJudge(
        [{"router_state": "insufficient_evidence", "reason": "occluded"}]
    )
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Include a work area.",
        specification_contract=_contract(
            _claim("area", "required_functional_areas", expected={"area": "work"})
        ),
        render_evidence=["global.png"],
        vlm_judge=judge,
    )
    metric = report["metrics"]["functional_semantic_fidelity"]
    assert metric["status"] == "incomplete"
    assert metric["score"] is None
    assert metric["partial_score"] is None
    assert metric["checks"][0]["reason"] == "camera_evidence_provider_not_configured"


def test_required_area_without_claim_targets_cannot_request_generic_local() -> None:
    provider = RecordingProvider()
    judge = RecordingJudge([{"router_state": "suspicious"}])
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Include a work area.",
        specification_contract=_contract(
            _claim("area", "required_functional_areas", expected={"area": "work"})
        ),
        render_evidence=["global.png"],
        camera_evidence_provider=provider,
        vlm_judge=judge,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["status"] == "requires_vlm"
    assert check["score"] is None
    assert check["reason"] == "claim_scoped_required_area_targets_missing"
    assert provider.requests == []


def test_local_provider_failure_is_failed_not_valid_or_zero() -> None:
    class Boom:
        def __call__(self, request: dict) -> list[str]:
            raise RuntimeError("render failed")

    judge = RecordingJudge([{"verdict": "valid"}])
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="The chair should serve the desk.",
        specification_contract=_contract(
            _claim(
                "local",
                "local_functionality",
                target_ids=["desk", "chair"],
            )
        ),
        camera_evidence_provider=Boom(),
        vlm_judge=judge,
    )
    metric = report["metrics"]["functional_semantic_fidelity"]
    assert metric["status"] == "incomplete"
    assert metric["score"] is None
    assert metric["failed_claim_count"] == 1
    assert metric["checks"][0]["status"] == "evidence_provider_failed"
    assert judge.requests == []


def test_provider_reported_failure_is_failed_not_missing_evidence() -> None:
    provider = RecordingProvider()

    def reported_failure(request: dict) -> dict:
        provider.requests.append(request)
        return {"status": "failed", "error": "renderer died"}

    judge = RecordingJudge([{"verdict": "valid"}])
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="The chair should serve the desk.",
        specification_contract=_contract(
            _claim(
                "local",
                "local_functionality",
                target_ids=["desk", "chair"],
            )
        ),
        camera_evidence_provider=reported_failure,
        vlm_judge=judge,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["status"] == "evidence_provider_failed"
    assert check["reason"] == "camera_evidence_provider_reported_failure"
    assert check["score"] is None


def test_insufficient_or_malformed_judge_output_never_becomes_valid_or_zero() -> None:
    insufficient = RecordingJudge(
        [
            {
                "evidence_status": "insufficient",
                "verdict": "valid",
                "confidence": 0.9,
            }
        ]
    )
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Create a bedroom.",
        specification_contract=_contract(
            _claim("room", "room_scene_type")
        ),
        render_evidence=["global.png"],
        vlm_judge=insufficient,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["status"] == "requires_vlm"
    assert check["score"] is None
    assert check["judge_result"]["verdict"] == "insufficient_evidence"

    invalid_without_defect = RecordingJudge(
        [{"evidence_status": "sufficient", "verdict": "invalid"}]
    )
    malformed = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Create a bedroom.",
        specification_contract=_contract(
            _claim("room", "room_scene_type")
        ),
        render_evidence=["global.png"],
        vlm_judge=invalid_without_defect,
    )
    malformed_check = malformed["metrics"]["functional_semantic_fidelity"][
        "checks"
    ][0]
    assert malformed_check["status"] == "vlm_adjudication_failed"
    assert malformed_check["score"] is None


def test_explicit_invalid_defect_can_score_zero() -> None:
    judge = RecordingJudge(
        [
            {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "defects": [
                    {
                        "claim_id": "room",
                        "description": "the render is a kitchen, not a bedroom",
                    }
                ],
            }
        ]
    )
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Create a bedroom.",
        specification_contract=_contract(
            _claim("room", "room_scene_type")
        ),
        render_evidence=["global.png"],
        vlm_judge=judge,
    )
    check = report["metrics"]["functional_semantic_fidelity"]["checks"][0]
    assert check["status"] == "checked"
    assert check["verdict"] == "invalid"
    assert check["score"] == 0.0


def test_actual_openai_adapter_receives_canonical_claim_context(tmp_path) -> None:
    image_path = tmp_path / "global.png"
    Image.new("RGB", (8, 8), (120, 120, 120)).save(image_path)
    model = FakeCanonicalModel()
    judge = OpenAICompatibleVLMJudge(model, max_images=2)
    report = evaluate_coarse_specification_interfaces(
        _scene(),
        config={"enabled": True},
        prompt="Create a bedroom.",
        specification_contract=_contract(
            _claim(
                "room",
                "room_scene_type",
                expected={"scene_type": "bedroom"},
            )
        ),
        render_evidence=[str(image_path)],
        vlm_judge=judge,
    )
    assert report["score"] == 1.0
    outbound = str(model.messages)
    assert "room_scene_type" in outbound
    assert "room" in outbound
    assert "global" in outbound


def test_component_names_are_rejected_as_metric_config_keys() -> None:
    for alias in (
        "room_scene_type",
        "broad_semantic_intent",
        "visual_functional_intent",
        "required_functional_areas",
        "required_zones",
        "local_functionality",
    ):
        assert normalize_coarse_metric_name(alias) == "functional_semantic_fidelity"
        with pytest.raises(CoarseSpecificationConfigError, match="claim components"):
            resolve_coarse_specification_config(
                {"metrics": {alias: {"enabled": False}}}
            )
    for retired_namespace in (
        "coarse_specification_interfaces",
        "functional_semantic_interfaces",
    ):
        with pytest.raises(CoarseSpecificationConfigError, match="retired"):
            resolve_coarse_specification_config(
                {retired_namespace: {"enabled": True}}
            )
        with pytest.raises(CoarseSpecificationConfigError, match="retired"):
            resolve_coarse_specification_config(
                profile={retired_namespace: {"enabled": True}}
            )
    with pytest.raises(CoarseSpecificationConfigError, match="claim components"):
        resolve_coarse_specification_config(
            profile={
                "l2_specification_fidelity": {
                    "metrics": {"room_scene_type": {"enabled": True}}
                }
            }
        )


def test_invalid_flag_fails_clearly() -> None:
    with pytest.raises(CoarseSpecificationConfigError, match="implemented=false"):
        resolve_coarse_specification_config({"implemented": False})
    with pytest.raises(CoarseSpecificationConfigError, match="enabled must be boolean"):
        resolve_coarse_specification_config({"enabled": "yes"})
    with pytest.raises(CoarseSpecificationConfigError, match="unknown metrics"):
        resolve_coarse_specification_config(
            {"metrics": {"oor": {"enabled": True}}}
        )


def test_functional_semantic_default_policy_is_global_then_claim_local() -> None:
    resolved = resolve_coarse_specification_config()
    plan = resolved["metrics"]["functional_semantic_fidelity"]["evidence_plan"]
    assert plan["evidence_strategy"] == "global_screen_then_local"
    assert plan["global_policy"]["top_down"] is False
    assert plan["global_policy"]["image_budget"] == 2
    assert (
        plan["global_policy"]["view_family"]
        == "wall_occlusion_aware_room_perspective"
    )
    local = plan["local_policy"]
    assert local["activation_condition"] == "prompt_specified_local_functionality"
    assert local["grouping_policy_id"] == "benchmark_owned_claim_targets"
    assert local["target_source"] == "benchmark_owned_claim_only"
    assert local["generic_pairing_scan"] is False
    assert set(local["trigger_states"]) == {
        "prompt_specified_local_functionality",
        "suspicious",
        "insufficient_evidence",
    }
    assert "original_prompt" in plan["text_context"]


def test_invalid_evidence_plan_fails_clearly() -> None:
    with pytest.raises(CoarseSpecificationConfigError, match="image_budget"):
        resolve_coarse_specification_config(
            {
                "metrics": {
                    "functional_semantic_fidelity": {
                        "evidence_plan": {
                            "global_policy": {
                                "view_family": "x",
                                "image_budget": 0,
                                "top_down": False,
                            },
                            "text_context": ["original_prompt"],
                        }
                    }
                }
            }
        )
