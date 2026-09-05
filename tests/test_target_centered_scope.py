from __future__ import annotations

from pathlib import Path

from PIL import Image

from benchmark.evaluator.scene_quality.interfaces import (
    _resolve_metric_evidence,
    evaluate_scene_quality_interfaces,
)
from benchmark.evaluator.scene_quality.target_scoped import (
    _retained_global_fallback_allowed,
    evaluate_target_scoped_judgements,
    resolve_target_evidence_packets,
)
from benchmark.evaluator.scene_quality.target_scope import (
    build_target_camera_scope,
)


def _image(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.png"
    path.write_bytes(b"not-decoded-by-unit-provider")
    return str(path)


def _decodable_image(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.png"
    image = Image.new("RGB", (3, 2), (20, 30, 40))
    image.putpixel((2, 1), (220, 210, 200))
    image.save(path)
    return str(path)


def _scene() -> dict:
    return {
        "scene_id": "target_scope_scene",
        "scene_type": "office",
        "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "center": [2.0, 2.0, 0.5],
                "size": [0.6, 0.6, 1.0],
                "rotation": [0, 0, 0],
            },
            {
                "id": "desk",
                "category": "desk",
                "center": [2.8, 2.0, 0.4],
                "size": [1.2, 0.7, 0.8],
                "rotation": [0, 0, 0],
            },
            {
                "id": "lamp",
                "category": "floor_lamp",
                "center": [4.8, 4.0, 0.8],
                "size": [0.3, 0.3, 1.6],
                "rotation": [0, 0, 0],
            },
        ],
    }


def _config(metric: str) -> dict:
    metrics = {
        name: {"enabled": False}
        for name in (
            "style_consistency",
            "scale_consistency",
            "object_pairing_consistency",
            "functional_consistency",
            "semantic_placement_consistency",
        )
    }
    metrics[metric] = {"enabled": True, "weight": 1.0}
    return {"metrics": metrics}


def _valid(request: dict | None = None) -> dict:
    result = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.9,
        "reason": "No in-scope defect is established.",
        "missing_evidence": [],
        "defects": [],
    }
    checks = (
        request.get("required_placement_checks")
        if isinstance(request, dict)
        else None
    )
    if checks:
        result["placement_check_results"] = [
            {
                "check_id": check["check_id"],
                "subject_id": check["subject_id"],
                "context_ids": check.get("context_ids") or [],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "The target-local evidence resolves the check.",
            }
            for check in checks
        ]
    return result


def test_object_pairing_singleton_candidate_gets_non_group_target_scope(
    tmp_path: Path,
) -> None:
    global_image = _image(tmp_path, "global")
    local_image = _image(tmp_path, "chair_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": local_image, "role": "object_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        if request["evidence_phase"] == "json_screen":
            return {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": 0.8,
                "reason": "The chair category requires visual confirmation.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": "group_member_category_compatibility",
                        "target_ids": ["chair"],
                        "relation": "category_compatibility_candidate",
                        "reason": "Confirm the isolated target visually.",
                    }
                ],
            }
        return _valid(request)

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_config("object_pairing_consistency"),
        object_grouping_report={
            "object_groups": [
                {"group_id": "chair_group", "object_ids": ["chair"]},
                {"group_id": "desk_group", "object_ids": ["desk"]},
                {"group_id": "lamp_group", "object_ids": ["lamp"]},
            ]
        },
        render_evidence={"global": [global_image]},
        camera_evidence_provider=provider,
        vlm_judge=judge,
        metric_applicability={
            "object_pairing_consistency": {"applicability": "relevant"}
        },
    )
    metric = report["metrics"]["object_pairing_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["route"] == "json_screen_then_target_visual"
    assert metric["selected_group_ids"] == []
    assert len(metric["target_scope_results"]) == 1
    assert metric["target_scope_policy"]["creates_group"] is False
    assert len(provider_calls) == 1
    provider_request = provider_calls[0]
    assert "group_scope" not in provider_request
    assert provider_request["object_groups"] == []
    assert provider_request["target_scope"]["target_id"] == "chair"
    assert provider_request["target_scope"]["group_identity"] is None
    target_judge = judge_calls[1]
    assert target_judge["target_object_ids"] == ["chair"]
    assert target_judge["framing_object_ids"][0] == "chair"
    assert target_judge["response_contract"]["allowed_target_ids"] == [
        "chair"
    ]
    assert target_judge["object_groups"] == []
    assert target_judge["render_evidence"] == [global_image, local_image]


def test_placement_without_group_owner_routes_check_to_target_scope(
    tmp_path: Path,
) -> None:
    global_image = _image(tmp_path, "global")
    local_image = _image(tmp_path, "placement_local")
    provider_calls: list[dict] = []
    judge_calls: list[dict] = []

    class Planner:
        def discover_placement_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "placement_discovery_v2",
                "considered_object_ids": ["chair", "desk", "lamp"],
                "candidates": [
                    {
                        "candidate_id": "candidate_chair_desk",
                        "subject_id": "chair",
                        "context_ids": ["desk"],
                        "check_type": "contextual_anchor",
                        "observation_goal": (
                            "show the chair relative to its intended desk context"
                        ),
                    }
                ],
                "reason": "complete",
                "decision_authority": "none",
            }

    def provider(request: dict) -> list[dict]:
        provider_calls.append(request)
        return [{"path": local_image, "role": "object_local"}]

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_config("semantic_placement_consistency"),
        render_evidence={"global": [global_image]},
        camera_evidence_provider=provider,
        functional_evidence_planner=Planner(),
        vlm_judge=judge,
        metric_applicability={
            "semantic_placement_consistency": {"applicability": "relevant"}
        },
    )
    metric = report["metrics"]["semantic_placement_consistency"]

    assert metric["status"] == "evaluated"
    assert metric["route"] == "global_discovery_then_target_local"
    assert metric["group_results"] == []
    assert len(metric["target_scope_results"]) == 1
    assert metric["target_scope_phase"]["status"] == "complete"
    check = metric["placement_check_ledger"]["checks"][0]
    assert check["owner_stage"] == "target_local"
    assert check["owning_group_id"] is None
    assert check["judge_status"] == "resolved"
    assert len(provider_calls) == 1
    assert provider_calls[0]["target_scope"]["target_id"] == "chair"
    target_judge = judge_calls[1]
    assert target_judge["evidence_phase"] == "target_local_confirmation"
    assert target_judge["target_object_ids"] == ["chair"]
    assert target_judge["response_contract"]["allowed_target_ids"] == [
        "chair"
    ]
    assert target_judge["required_placement_checks"][0]["check_id"] == (
        check["check_id"]
    )


def test_target_scope_requires_real_global_anchor(tmp_path: Path) -> None:
    local_image = _image(tmp_path, "local_without_global")

    packets = resolve_target_evidence_packets(
        {},
        metric_name="semantic_placement_consistency",
        policy={
            "camera_scope": "object_local",
            "image_budget": 2,
            "scoped_image_budget": 1,
            "global_image_budget": 1,
            "include_global_context": True,
            "image_order": ["global_context", "object_local"],
        },
        scene=_scene(),
        prompt=None,
        targets=[{"target_id": "chair", "context_ids": ["desk"]}],
        camera_evidence_provider=lambda request: [
            {"path": local_image, "role": "object_local"}
        ],
        resolve_metric_evidence=_resolve_metric_evidence,
    )

    resolution = packets[0]["resolution"]
    assert resolution["local_scope_satisfied"] is True
    assert resolution["global_anchor_required"] is True
    assert resolution["global_anchor_satisfied"] is False
    assert resolution["scope_satisfied"] is False
    assert resolution["provider_reason"] == (
        "global_anchor_render_evidence_unavailable"
    )


def test_target_local_acquisition_failure_uses_retained_global_forced_final(
    tmp_path: Path,
) -> None:
    global_image = _decodable_image(tmp_path, "retained_global")
    judge_calls: list[dict] = []

    class Planner:
        def discover_placement_evidence(self, request: dict) -> dict:
            return {
                "schema_version": "placement_discovery_v2",
                "considered_object_ids": ["chair", "desk", "lamp"],
                "candidates": [
                    {
                        "candidate_id": "candidate_chair_desk",
                        "subject_id": "chair",
                        "context_ids": ["desk"],
                        "check_type": "contextual_anchor",
                        "observation_goal": "show chair relative to desk",
                    }
                ],
                "reason": "complete",
                "decision_authority": "none",
            }

    def judge(request: dict) -> dict:
        judge_calls.append(request)
        return _valid(request)

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_config("semantic_placement_consistency"),
        render_evidence={"global": [global_image]},
        camera_evidence_provider=lambda request: {
            "status": "insufficient",
            "reason": "no_feasible_candidate",
        },
        functional_evidence_planner=Planner(),
        vlm_judge=judge,
        metric_applicability={
            "semantic_placement_consistency": {"applicability": "relevant"}
        },
    )
    metric = report["metrics"]["semantic_placement_consistency"]
    target = metric["target_scope_results"][0]

    assert target["status"] == "evaluated"
    assert target["terminal_state"] == "evaluated_degraded"
    assert target["retained_global_forced_final"] is True
    assert target["evidence_paths"] == [global_image]
    assert target["evidence_resolution"]["scope_satisfied"] is False
    assert target["evidence_coverage"]["grounded"] is False
    assert target["judgement"]["evidence_ambiguous"] is True
    assert target["judgement"]["forced_binary"] is True
    assert target["placement_check_resolution"]["rows"][0][
        "observation_status"
    ] == "inferred_under_budget"
    assert judge_calls[-1]["render_evidence"] == [global_image]
    assert judge_calls[-1]["budget_exhaustion_finalization"]["required"] is True
    assert metric["coverage"]["score_grounding"]["fraction"] < 1.0


def test_corrupt_global_anchor_cannot_enable_target_fallback(
    tmp_path: Path,
) -> None:
    corrupt_global = _image(tmp_path, "corrupt_retained_global")
    packets = resolve_target_evidence_packets(
        {"global": [corrupt_global]},
        metric_name="semantic_placement_consistency",
        policy={
            "camera_scope": "object_local",
            "image_budget": 2,
            "scoped_image_budget": 1,
            "global_image_budget": 1,
            "include_global_context": True,
            "image_order": ["global_context", "object_local"],
        },
        scene=_scene(),
        prompt=None,
        targets=[{"target_id": "chair", "context_ids": ["desk"]}],
        camera_evidence_provider=lambda request: {
            "status": "insufficient",
            "reason": "no_feasible_candidate",
        },
        resolve_metric_evidence=_resolve_metric_evidence,
    )

    allowed, integrity = _retained_global_fallback_allowed(
        packets[0],
        metric_name="semantic_placement_consistency",
    )
    assert allowed is False
    assert integrity is not None
    assert integrity["ready"] is False
    assert "undecodable_render" in integrity["reason_codes"]


def test_target_context_object_cannot_become_defect_owner(
    tmp_path: Path,
) -> None:
    global_image = _image(tmp_path, "global_context_owner")
    local_image = _image(tmp_path, "target_context_owner")

    def judge(request: dict) -> dict:
        if request["evidence_phase"] != "target_local_confirmation":
            return _valid(request)
        value = _valid(request)
        value.update(
            verdict="invalid",
            reason="The context desk is invalid.",
            defects=[
                {
                    "scope": "object_environment_fit",
                    "target_ids": ["desk"],
                    "relation": "context_object_wrongly_attributed",
                    "reason": "Context objects cannot own this episode.",
                }
            ],
        )
        return value

    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=_config("object_pairing_consistency"),
        object_grouping_report={
            "object_groups": [
                {"group_id": "chair_group", "object_ids": ["chair"]},
                {"group_id": "desk_group", "object_ids": ["desk"]},
                {"group_id": "lamp_group", "object_ids": ["lamp"]},
            ]
        },
        render_evidence={"global": [global_image]},
        camera_evidence_provider=lambda request: [
            {"path": local_image, "role": "object_local"}
        ],
        vlm_judge=lambda request: (
            {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": 0.8,
                "reason": "Confirm chair locally.",
                "missing_evidence": [],
                "defects": [
                    {
                        "scope": "group_member_category_compatibility",
                        "target_ids": ["chair"],
                        "relation": "candidate",
                        "reason": "Confirm visually.",
                    }
                ],
            }
            if request["evidence_phase"] == "json_screen"
            else judge(request)
        ),
        metric_applicability={
            "object_pairing_consistency": {"applicability": "relevant"}
        },
    )
    target = report["metrics"]["object_pairing_consistency"][
        "target_scope_results"
    ][0]
    assert target["status"] == "failed"
    assert target["terminal_state"] == "infrastructure_failure"
    assert target["reason"] == "vlm_judge_failed"
    assert target["judgement"]["error_type"] == "ValueError"


def test_target_local_placement_can_deduplicate_exact_function_event(
    tmp_path: Path,
) -> None:
    local_image = _image(tmp_path, "target_local_exact_event")
    scope = build_target_camera_scope(
        _scene(),
        target_id="chair",
        metric="semantic_placement_consistency",
        explicit_context_ids=["desk"],
    )
    required_check = {
        "check_id": "placement_check_exact_event",
        "subject_id": "chair",
        "context_ids": ["desk"],
        "check_type": "contextual_anchor",
    }
    ownership = {
        "events": [
            {
                "event_id": "functional_event_exact",
                "affected_object_ids": ["chair"],
                "causal_object_ids": ["desk"],
                "scoring_target_ids": ["chair"],
                "counterpart_object_ids": ["desk"],
            }
        ]
    }
    delivered_ledgers: list[dict] = []

    def build_request(**kwargs: object) -> dict:
        delivered_ledgers.append(
            kwargs["functional_ownership_ledger"]  # type: ignore[arg-type]
        )
        return dict(kwargs)

    def judge(_judge: object, request: dict) -> dict:
        assert request["functional_ownership_ledger"] == ownership
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.8,
            "reason": "The same physical event is already Function-owned.",
            "missing_evidence": [],
            "defects": [],
            "placement_check_results": [
                {
                    "check_id": required_check["check_id"],
                    "subject_id": "chair",
                    "context_ids": ["desk"],
                    "observation_status": "observed",
                    "conclusion": "excluded_function_owned",
                    "reason": "Exact duplicate of the cited Function event.",
                    "function_event_ref": "functional_event_exact",
                    "same_physical_event": True,
                }
            ],
        }

    records = evaluate_target_scoped_judgements(
        metric_name="semantic_placement_consistency",
        scene=_scene(),
        prompt=None,
        packets=[
            {
                "target_id": "chair",
                "context_ids": ["desk"],
                "framing_ids": list(scope.framing_ids),
                "target_scope": scope,
                "paths": [local_image],
                "resolution": {
                    "scope_satisfied": True,
                    "global_anchor_satisfied": True,
                    "local_scope_satisfied": True,
                },
                "required_placement_checks": [required_check],
            }
        ],
        vlm_judge=object(),
        authorized_deviations=[],
        visual_style_spec=None,
        build_judge_request=build_request,
        call_judge=judge,
        apply_prompt_exemptions=lambda value, **_: value,
        normalize_judgement=lambda value, **_: {
            "status": "evaluated",
            "score": 1.0,
            "reason": value["reason"],
        },
        functional_ownership_ledger=ownership,
    )

    assert delivered_ledgers == [ownership]
    assert records[0]["status"] == "evaluated"
    assert records[0]["placement_check_resolution"][
        "excluded_function_owned_check_ids"
    ] == [required_check["check_id"]]
