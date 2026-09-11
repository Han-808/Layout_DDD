"""Canonical runtime for L2 Functional Semantic Fidelity.

L2 evaluates only whether the generated scene satisfies requirements expressed
by the prompt. Room/scene type, broad visual-functional intent, required
functional areas, and explicitly requested local functionality are components of
one canonical metric, ``functional_semantic_fidelity``. They are not separately
weighted metrics:

- global evidence judges room/scene type and broad visual-functional intent;
- required functional areas are screened globally and receive claim-scoped local
  evidence only when the global screen is suspicious or insufficient;
- local evidence is requested directly only for functionality explicitly
  specified by the benchmark-owned prompt contract.

``room_scene_type``, ``visual_functional_intent``,
``required_functional_areas``, and ``local_functionality`` are claim
components only. They are never metric names or configuration aliases.

No generic object-pair scan is performed. For prompt-owned group-local claims,
the supplied grouping partition is consumed only to decompose camera evidence
and Judge calls into bounded local scopes; grouping never creates a new claim.

Ownership note: generic scale appropriateness and generic object
co-occurrence/pairing coherence are **not** owned by high-level L2 in the
canonical hierarchy; they belong to L3 Scene Quality (Semantic Coherence).
Prompt-specific scale or pairing *requirements* may still be represented under
L2 as explicit prompt requirements ("did the scene follow the prompt?"), which
is a different judgment from L3 generic coherence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.architecture_policy import architecture_contract_from_scene
from benchmark.evaluator.evidence_contract import (
    FINAL_VLM_CONTEXT_CONTRACT,
    FUNCTIONAL_SEMANTIC_COMPONENTS,
    GROUPING_POLICY_ID,
    canonical_hierarchy,
    validate_evidence_plan,
)
from benchmark.evaluator.specification_fidelity.contract import (
    canonical_specification_claims,
)
from benchmark.evaluator.specification_fidelity.group_scoped import (
    aggregate_group_judgements as _aggregate_group_judgements,
    normalize_grouping_partition as _normalize_functional_grouping,
    request_group_local_evidence as _request_group_local_evidence,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.group_scope import (
    GroupCameraScope,
    group_scope_evidence_goal,
)


FUNCTIONAL_SEMANTIC_INTERFACE_VERSION = "functional_semantic_fidelity_runtime_v2"
FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE = "functional_semantic_fidelity"
CANONICAL_L2_PROFILE_KEY = "l2_specification_fidelity"

FUNCTIONAL_SEMANTIC_FIDELITY = "functional_semantic_fidelity"
SPECIFICATION_CLAIM_TARGET_POLICY_ID = GROUPING_POLICY_ID
FUNCTIONAL_SEMANTIC_METRICS = (FUNCTIONAL_SEMANTIC_FIDELITY,)
_COMPONENT_ONLY_NAMES = tuple(FUNCTIONAL_SEMANTIC_COMPONENTS)
_RETIRED_METRIC_NAMES = (
    *_COMPONENT_ONLY_NAMES,
    "broad_semantic_intent",
    "required_zones",
)
_RETIRED_CONFIG_NAMESPACES = (
    "coarse_specification_interfaces",
    "functional_semantic_interfaces",
)

# Camera selection remains provider-owned. This module only issues metric- and
# claim-scoped evidence requests. ``global_top`` is deliberately not the default
# global room view.
_WALL_OCCLUSION_AWARE_GLOBAL = {
    "view_family": "wall_occlusion_aware_room_perspective",
    "image_budget": 2,
    "top_down": False,
    "perspective_diversity_required": True,
    "occluder_policy": "suppress_occluding_architecture",
    "preserve_boundary_cues": True,
}

DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG: dict[str, Any] = {
    "enabled": True,
    "implemented": True,
    "version": FUNCTIONAL_SEMANTIC_INTERFACE_VERSION,
    "metrics": {
        FUNCTIONAL_SEMANTIC_FIDELITY: {
            "enabled": True,
            "implemented": True,
            "evidence_plan": {
                "evidence_strategy": "global_screen_then_local",
                "global_policy": deepcopy(_WALL_OCCLUSION_AWARE_GLOBAL),
                "local_policy": {
                    "camera_scope": "group_local",
                    "grouping_policy_id": SPECIFICATION_CLAIM_TARGET_POLICY_ID,
                    "include_global_context": True,
                    "activation_condition": "prompt_specified_local_functionality",
                    "trigger_states": [
                        "prompt_specified_local_functionality",
                        "suspicious",
                        "insufficient_evidence",
                    ],
                    "target_source": "benchmark_owned_claim_only",
                    "generic_pairing_scan": False,
                },
                "router_options": None,
                "text_context": [
                    "original_prompt",
                    "parsed_functional_semantic_requirements",
                    "parsed_room_scene_type",
                    "parsed_broad_intent",
                    "parsed_required_functional_areas",
                    "parsed_local_functionality_requirements",
                    "authorized_deviations",
                    "asset_policy",
                    "scene_metadata",
                ],
            },
        },
    },
}

class FunctionalSemanticConfigError(ValueError):
    """Raised when the functional-semantic runtime config is malformed."""


def resolve_functional_semantic_config(
    config: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the single canonical Functional Semantic configuration."""

    layers: list[dict[str, Any]] = []
    if isinstance(profile, dict):
        retired_namespaces = sorted(
            set(profile) & set(_RETIRED_CONFIG_NAMESPACES)
        )
        if retired_namespaces:
            raise FunctionalSemanticConfigError(
                "retired L2 profile namespaces are not accepted by the "
                f"canonical resolver: {retired_namespaces}; use "
                f"{CANONICAL_L2_PROFILE_KEY!r}"
            )
        canonical_section = profile.get(CANONICAL_L2_PROFILE_KEY)
        if canonical_section is not None:
            layers.append(
                _canonical_profile_layer(
                    canonical_section,
                    "canonical L2 profile",
                )
            )
    if config is not None:
        layers.append(_normalize_layer(config, "config override"))
    if run_overrides is not None:
        layers.append(_normalize_layer(run_overrides, "run override"))

    resolved = deepcopy(DEFAULT_FUNCTIONAL_SEMANTIC_CONFIG)
    for layer in layers:
        resolved = _deep_merge(resolved, layer)

    _validate_flags(resolved, FUNCTIONAL_SEMANTIC_INTERFACE_NAMESPACE)
    metrics = resolved.get("metrics")
    if not isinstance(metrics, dict):
        raise FunctionalSemanticConfigError(
            "functional_semantic_fidelity.metrics must be a JSON object"
        )
    unknown_metrics = set(metrics) - set(FUNCTIONAL_SEMANTIC_METRICS)
    if unknown_metrics:
        raise FunctionalSemanticConfigError(
            "functional-semantic config contains unknown metrics "
            f"{sorted(unknown_metrics)}; OOR/OAR are configured by canonical L2, "
            "not by this runtime"
        )
    for metric_name in FUNCTIONAL_SEMANTIC_METRICS:
        metric_config = metrics.get(metric_name)
        if not isinstance(metric_config, dict):
            raise FunctionalSemanticConfigError(
                f"functional_semantic_fidelity.metrics.{metric_name} must be a JSON object"
            )
        _validate_flags(
            metric_config, f"functional_semantic_fidelity.metrics.{metric_name}"
        )
        plan = metric_config.get("evidence_plan")
        if plan is not None:
            try:
                validate_evidence_plan(
                    plan, where=f"functional_semantic_fidelity.metrics.{metric_name}.evidence_plan"
                )
            except Exception as exc:  # normalize to this module's error type
                raise FunctionalSemanticConfigError(str(exc)) from exc
    return resolved


def evaluate_functional_semantic_fidelity(
    scene: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | None = None,
    prompt: str = "",
    specification_contract: dict[str, Any] | None = None,
    render_evidence: list[Any] | None = None,
    camera_evidence_provider: Any = None,
    vlm_judge: Any = None,
    object_grouping_report: Any = None,
    authorized_deviations: Any = None,
    asset_policy: Any = None,
) -> dict[str, Any]:
    """Evaluate prompt-conditioned functional semantics.

    ``render_evidence`` is the global evidence packet. The camera provider is
    invoked only for one benchmark-owned claim at a time:

    - immediately for an explicit ``local_functionality`` claim;
    - after a ``required_functional_areas`` global screen returns suspicious or
      insufficient.

    Room type and broad intent are global-only. No generic object-pairing or
    grouping scan is legal in this evaluator.
    """

    if not isinstance(scene, dict):
        raise TypeError("functional semantic fidelity scene must be a JSON object")
    resolved = resolve_functional_semantic_config(
        config, profile=profile, run_overrides=run_overrides
    )

    top_enabled = bool(resolved["enabled"])
    metric_config = resolved["metrics"][FUNCTIONAL_SEMANTIC_FIDELITY]
    metric_enabled = bool(metric_config["enabled"])
    evidence_plan = metric_config.get("evidence_plan")
    local_policy = (
        evidence_plan.get("local_policy")
        if isinstance(evidence_plan, dict)
        and isinstance(evidence_plan.get("local_policy"), dict)
        else {}
    )
    include_group_global_context = bool(
        local_policy.get("include_global_context", True)
    )
    global_paths = _normalize_evidence_paths(render_evidence)
    grouping_groups = _normalize_functional_grouping(
        object_grouping_report,
        scene=scene,
    )

    if not top_enabled or not metric_enabled:
        checks: list[dict[str, Any]] = []
        metric_report = _metric_report(
            metric_config=metric_config,
            status="not_applicable",
            reason=(
                "disabled_by_configuration"
                if not top_enabled
                else "metric_disabled_by_configuration"
            ),
            checks=checks,
            provider_calls=0,
            judge_calls=0,
        )
    elif specification_contract is None:
        metric_report = _metric_report(
            metric_config=metric_config,
            status="not_evaluable",
            reason="missing_specification_contract",
            checks=[],
            provider_calls=0,
            judge_calls=0,
        )
    else:
        canonical = canonical_specification_claims(specification_contract)
        claims = [
            claim
            for claim in (
                canonical.get(FUNCTIONAL_SEMANTIC_FIDELITY) or []
            )
            if claim.get("required") is not False
        ]
        checks = []
        counters = {"provider": 0, "judge": 0}
        for claim in claims:
            checks.append(
                _evaluate_functional_claim(
                    claim=claim,
                    scene=scene,
                    prompt=str(prompt or ""),
                    global_paths=global_paths,
                    camera_evidence_provider=camera_evidence_provider,
                    vlm_judge=vlm_judge,
                    authorized_deviations=authorized_deviations,
                    asset_policy=asset_policy,
                    counters=counters,
                    grouping_groups=grouping_groups,
                    grouping_report=(
                        object_grouping_report
                        if isinstance(object_grouping_report, dict)
                        else None
                    ),
                    include_global_context=include_group_global_context,
                )
            )
        status, reason = _metric_status(checks)
        metric_report = _metric_report(
            metric_config=metric_config,
            status=status,
            reason=reason,
            checks=checks,
            provider_calls=counters["provider"],
            judge_calls=counters["judge"],
        )

    grouping_consumed = any(
        isinstance(check, dict)
        and (
            "group_results" in check
            or check.get("reason")
            in {
                "object_grouping_unavailable",
                "claim_targets_not_mapped_to_group",
            }
        )
        for check in metric_report.get("checks") or []
    )
    metric_report["object_grouping_report_consumed"] = grouping_consumed
    metric_report["grouping_policy"] = (
        {
            "policy_id": object_grouping_report.get(
                "grouping_policy_id"
            ),
            "backend": object_grouping_report.get(
                "grouping_backend"
            ),
        }
        if grouping_consumed
        and isinstance(object_grouping_report, dict)
        else None
    )
    metric_reports = {FUNCTIONAL_SEMANTIC_FIDELITY: metric_report}
    provider_invoked = bool(metric_report["camera_evidence_provider_invoked"])
    judge_invoked = bool(metric_report["vlm_invoked"])
    top_status = metric_report["status"]
    top_reason = metric_report["reason"]
    active_claims = bool(metric_report["eligible_claim_count"] > 0)
    affects_aggregation = bool(
        top_enabled and metric_enabled and active_claims
    )
    return {
        "category": FUNCTIONAL_SEMANTIC_FIDELITY,
        "level": CANONICAL_L2_PROFILE_KEY,
        "interface_version": str(
            resolved.get("version") or FUNCTIONAL_SEMANTIC_INTERFACE_VERSION
        ),
        "implemented": True,
        "enabled": top_enabled,
        "status": top_status,
        "reason": top_reason,
        "score": metric_report["score"],
        "partial_score": metric_report["partial_score"],
        "affects_score": affects_aggregation,
        "affects_aggregation": affects_aggregation,
        "renderer_invoked": provider_invoked,
        "camera_evidence_provider_invoked": provider_invoked,
        "vlm_invoked": judge_invoked,
        "metrics": metric_reports,
        "grouping_policy": (
            {
                "policy_id": object_grouping_report.get(
                    "grouping_policy_id"
                ),
                "backend": object_grouping_report.get(
                    "grouping_backend"
                ),
            }
            if grouping_groups is not None
            and isinstance(object_grouping_report, dict)
            else None
        ),
        "object_grouping_report_consumed": grouping_consumed,
        "object_grouping_report_supplied": object_grouping_report is not None,
        "final_vlm_context_contract": list(FINAL_VLM_CONTEXT_CONTRACT),
        "hierarchy": canonical_hierarchy(),
        "l2_l3_boundary": {
            "l2_question": "Did the scene follow the prompt?",
            "l3_question": "Is the scene coherent, except for prompt-authorized deviations?",
            "generic_scale_pairing_owner": "L3 scene_quality (semantic_coherence)",
            "oor_oar_owner": "L2 explicit-relation Fidelity (not L1 Physical Plausibility)",
            "functional_semantic_owner": (
                "L2 prompt-conditioned room, intent, area, and explicitly "
                "requested local-function requirements"
            ),
            "prompt_owned_group_precedence": (
                "Prompt-owned group/function requirements are resolved here; "
                "L3 pairing must not score the same requirement again."
            ),
        },
        "notes": [
            "Room type, broad intent, required areas, and prompt-specified local functionality are components of one family, not separate weighted metrics.",
            "Room type and broad intent use global evidence.",
            "Required areas use global screening; suspicious or insufficient cases alone receive claim-scoped local fallback.",
            "Local functionality receives local evidence only when explicitly present in the benchmark-owned prompt contract.",
            "No generic object-pairing or object-grouping scan is performed.",
            "Generic scale/pairing coherence is owned by L3 Scene Quality, not coarse L2.",
            "Prompt-specific scale/pairing requirements may still appear here as explicit prompt requirements.",
            "The camera provider owns pose generation; this evaluator issues only claim-scoped evidence requests.",
            "Insufficient evidence remains unresolved and never defaults to valid or zero.",
        ],
    }


def _evaluate_functional_claim(
    *,
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    global_paths: list[str],
    camera_evidence_provider: Any,
    vlm_judge: Any,
    authorized_deviations: Any,
    asset_policy: Any,
    counters: dict[str, int],
    grouping_groups: list[dict[str, Any]] | None,
    grouping_report: dict[str, Any] | None,
    include_global_context: bool,
) -> dict[str, Any]:
    claim_id = str(claim.get("claim_id") or "")
    component = str(claim.get("component") or "")
    base = {
        "claim_id": claim_id,
        "family": FUNCTIONAL_SEMANTIC_FIDELITY,
        "component": component,
        "required": claim.get("required", True) is not False,
        "target_ids": _claim_target_ids(claim),
        "score": None,
        "verdict": None,
        "global_evidence_paths": list(global_paths),
        "local_evidence_paths": [],
        "generic_pairing_scan": False,
    }
    if vlm_judge is None:
        return {
            **base,
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "vlm_judge_not_configured",
        }

    if component == "local_functionality":
        local_result = _request_group_local_evidence(
            claim=claim,
            scene=scene,
            prompt=prompt,
            trigger="prompt_specified_local_functionality",
            camera_evidence_provider=camera_evidence_provider,
            counters=counters,
            grouping_groups=grouping_groups,
            grouping_report=grouping_report,
            include_global_context=include_global_context,
            metric=FUNCTIONAL_SEMANTIC_FIDELITY,
            claim_target_ids=_claim_target_ids,
            request_local_evidence=_request_local_evidence,
        )
        if local_result["status"] != "available":
            return {**base, **local_result}
        return _final_group_judgements(
            base={
                **base,
                "local_evidence_paths": local_result["paths"],
            },
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase="prompt_scoped_local",
            global_paths=(
                global_paths if include_global_context else []
            ),
            packets=local_result["group_packets"],
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
        )

    if component == "required_functional_areas":
        screen = _required_area_global_screen(
            base=base,
            claim=claim,
            scene=scene,
            prompt=prompt,
            global_paths=global_paths,
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
        )
        if screen.get("status") in {"checked", "vlm_adjudication_failed"}:
            return screen
        trigger = str(screen.get("router_state") or "insufficient_evidence")
        local_result = _request_group_local_evidence(
            claim=claim,
            scene=scene,
            prompt=prompt,
            trigger=trigger,
            camera_evidence_provider=camera_evidence_provider,
            counters=counters,
            grouping_groups=grouping_groups,
            grouping_report=grouping_report,
            include_global_context=include_global_context,
            metric=FUNCTIONAL_SEMANTIC_FIDELITY,
            claim_target_ids=_claim_target_ids,
            request_local_evidence=_request_local_evidence,
        )
        if local_result["status"] != "available":
            return {
                **base,
                **local_result,
                "global_screen": screen.get("global_screen"),
                "router_state": trigger,
            }
        return _final_group_judgements(
            base={
                **base,
                "local_evidence_paths": local_result["paths"],
                "global_screen": screen.get("global_screen"),
                "router_state": trigger,
            },
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase="required_area_local_fallback",
            global_paths=(
                global_paths if include_global_context else []
            ),
            packets=local_result["group_packets"],
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
        )

    # room_scene_type and visual_functional_intent are deliberately global-only.
    if component in {"room_scene_type", "visual_functional_intent"}:
        if not global_paths:
            return {
                **base,
                "status": "requires_vlm",
                "route": "unresolved",
                "reason": "global_render_evidence_not_available",
            }
        return _final_judgement(
            base=base,
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase="global",
            global_paths=global_paths,
            local_paths=[],
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
        )

    return {
        **base,
        "status": "requires_vlm",
        "route": "unresolved",
        "reason": "unsupported_functional_semantic_component",
    }


def _required_area_global_screen(
    *,
    base: dict[str, Any],
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    global_paths: list[str],
    vlm_judge: Any,
    authorized_deviations: Any,
    asset_policy: Any,
    counters: dict[str, int],
) -> dict[str, Any]:
    if not global_paths:
        return {
            **base,
            "status": "requires_local_evidence",
            "route": "global_screen_then_local",
            "reason": "global_render_evidence_not_available",
            "router_state": "insufficient_evidence",
            "global_screen": {
                "router_state": "insufficient_evidence",
                "reason": "global_render_evidence_not_available",
            },
        }
    try:
        raw = _invoke_judge(
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase="required_area_global_screen",
            decision_mode="screen",
            global_paths=global_paths,
            local_paths=[],
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
        )
        normalized = _normalize_judge_response(raw, decision_mode="screen")
    except Exception as exc:
        return {
            **base,
            "status": "vlm_adjudication_failed",
            "route": "vlm_adjudication_failed",
            "reason": "required_area_global_screen_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    router_state = normalized["router_state"]
    if router_state == "not_suspicious":
        return {
            **base,
            "status": "checked",
            "route": "global_screen_resolved",
            "reason": None,
            "verdict": "valid",
            "score": 1.0,
            "confidence": normalized.get("confidence"),
            "global_screen": normalized,
        }
    return {
        **base,
        "status": "requires_local_evidence",
        "route": "global_screen_then_local",
        "reason": f"required_area_global_screen_{router_state}",
        "router_state": router_state,
        "global_screen": normalized,
    }


def _final_judgement(
    *,
    base: dict[str, Any],
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    phase: str,
    global_paths: list[str],
    local_paths: list[str],
    vlm_judge: Any,
    authorized_deviations: Any,
    asset_policy: Any,
    counters: dict[str, int],
    group_scope: GroupCameraScope | None = None,
) -> dict[str, Any]:
    if not global_paths and not local_paths:
        return {
            **base,
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "render_evidence_not_available",
        }
    try:
        raw = _invoke_judge(
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase=phase,
            decision_mode="final",
            global_paths=global_paths,
            local_paths=local_paths,
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
            group_scope=group_scope,
        )
        normalized = _normalize_judge_response(raw, decision_mode="final")
    except Exception as exc:
        return {
            **base,
            "status": "vlm_adjudication_failed",
            "route": "vlm_adjudication_failed",
            "reason": "functional_semantic_vlm_adjudication_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    verdict = normalized["verdict"]
    if verdict == "insufficient_evidence":
        return {
            **base,
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "vlm_reported_insufficient_evidence",
            "judge_result": normalized,
        }
    return {
        **base,
        "status": "checked",
        "route": "vlm_adjudicated",
        "reason": normalized.get("reason"),
        "verdict": verdict,
        "score": 1.0 if verdict == "valid" else 0.0,
        "confidence": normalized.get("confidence"),
        "judge_result": normalized,
    }


def _final_group_judgements(
    *,
    base: dict[str, Any],
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    phase: str,
    global_paths: list[str],
    packets: list[dict[str, Any]],
    vlm_judge: Any,
    authorized_deviations: Any,
    asset_policy: Any,
    counters: dict[str, int],
) -> dict[str, Any]:
    def judge_group(
        packet: dict[str, Any],
        scope: GroupCameraScope,
    ) -> dict[str, Any]:
        return _final_judgement(
            base={
                "group_id": scope.group_id,
                "member_ids": list(scope.member_ids),
                "group_scope": scope.to_dict(),
                "local_evidence_paths": list(packet["paths"]),
            },
            claim=claim,
            scene=scene,
            prompt=prompt,
            phase=phase,
            global_paths=global_paths,
            local_paths=packet["paths"],
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            asset_policy=asset_policy,
            counters=counters,
            group_scope=scope,
        )

    return _aggregate_group_judgements(
        base=base,
        packets=packets,
        judge_group=judge_group,
    )


def _invoke_judge(
    *,
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    phase: str,
    decision_mode: str,
    global_paths: list[str],
    local_paths: list[str],
    vlm_judge: Any,
    authorized_deviations: Any,
    asset_policy: Any,
    counters: dict[str, int],
    group_scope: GroupCameraScope | None = None,
) -> dict[str, Any]:
    request = _judge_request(
        claim=claim,
        scene=scene,
        prompt=prompt,
        phase=phase,
        decision_mode=decision_mode,
        global_paths=global_paths,
        local_paths=local_paths,
        authorized_deviations=authorized_deviations,
        asset_policy=asset_policy,
        group_scope=group_scope,
    )
    call = getattr(vlm_judge, "adjudicate_functional_semantic", None)
    if not callable(call):
        call = getattr(vlm_judge, "adjudicate_functional_semantics", None)
    if not callable(call):
        call = getattr(vlm_judge, "adjudicate_specification_fidelity", None)
    if not callable(call) and callable(vlm_judge):
        call = vlm_judge
    if not callable(call):
        raise TypeError(
            "functional-semantic VLM judge must be callable or expose "
            "adjudicate_functional_semantic(request)"
        )
    counters["judge"] += 1
    raw = call(request)
    if not isinstance(raw, dict):
        raise ValueError("functional-semantic VLM response must be a JSON object")
    return raw


def _judge_request(
    *,
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    phase: str,
    decision_mode: str,
    global_paths: list[str],
    local_paths: list[str],
    authorized_deviations: Any,
    asset_policy: Any,
    group_scope: GroupCameraScope | None = None,
) -> dict[str, Any]:
    target_ids = _claim_target_ids(claim)
    required_response = (
        {
            "router_state": (
                "not_suspicious|suspicious|insufficient_evidence"
            )
        }
        if decision_mode == "screen"
        else {"verdict": "valid|invalid|insufficient_evidence"}
    )
    request = {
        "category": "functional_semantic_fidelity_adjudication",
        "metric": FUNCTIONAL_SEMANTIC_FIDELITY,
        "phase": phase,
        "decision_mode": decision_mode,
        "component": claim.get("component"),
        "components": [claim.get("component")],
        "claim_id": claim.get("claim_id"),
        "claim": deepcopy(claim),
        "claims": [deepcopy(claim)],
        "evidence_phase": phase,
        "judgment_scope": {
            "included": [
                "this_benchmark_owned_prompt_claim_only",
                str(claim.get("component") or ""),
            ],
            "excluded": [
                "generic_object_pairing",
                "generic_scale_coherence",
                "visual_style",
                "oor",
                "oar",
                "unprompted_local_functionality",
            ],
        },
        "event": {
            "type": str(claim.get("component") or "functional_semantic_claim"),
            "claim_id": claim.get("claim_id"),
            "object_ids": target_ids,
        },
        "detector_evidence": {
            "source": "benchmark_owned_specification_contract",
            "claim": deepcopy(claim),
        },
        "deterministic_evidence": {
            "source": "benchmark_owned_specification_contract",
            "claim_id": claim.get("claim_id"),
            "component": claim.get("component"),
            "phase": phase,
        },
        "natural_language_prompt": prompt,
        "prompt": prompt,
        "camera_scene_context": deepcopy(scene),
        "scene_summary": _compact_scene(scene),
        "target_ids": target_ids,
        "involved_objects": _target_objects(scene, target_ids),
        # Local fallback evidence is ordered first so a downstream max-image
        # budget cannot silently truncate the evidence that justified the
        # fallback. Dedicated global/local fields preserve provenance.
        "render_evidence": list(local_paths) + list(global_paths),
        "image_order": (
            "local_first_then_global"
            if local_paths
            else "global_only"
        ),
        "relevant_global_visual_evidence": list(global_paths),
        "relevant_local_visual_evidence": list(local_paths),
        "authorized_deviations": deepcopy(authorized_deviations),
        "asset_policy": deepcopy(asset_policy),
        "required_response": required_response,
        "rubric": (
            "Judge only whether this explicit prompt-scoped functional-semantic "
            "claim is satisfied. Do not judge generic object pairing, generic "
            "scale coherence, visual style, OOR, or OAR. An invalid final verdict "
            "requires a significant explicitly identified visible failure. If "
            "the supplied views cannot establish the claim, return "
            "insufficient_evidence rather than guessing."
        ),
        "generic_pairing_scan": False,
        "object_grouping_report_consumed": group_scope is not None,
        **vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method="adjudicate_functional_semantic",
        ),
    }
    if group_scope is not None:
        scope_value = group_scope.to_dict()
        group_member_ids = list(group_scope.member_ids)
        group_member_set = set(group_member_ids)
        group_claim_target_ids = [
            target_id
            for target_id in target_ids
            if target_id in group_member_set
        ]
        group_objects = _target_objects(scene, group_member_ids)
        request["scene_summary"]["objects"] = deepcopy(group_objects)
        request["scene_summary"]["object_count"] = len(group_objects)
        request["scene_summary"]["group_scope"] = deepcopy(
            scope_value
        )
        request.update(
            {
                "group_scope": scope_value,
                "group_id": group_scope.group_id,
                "member_ids": group_member_ids,
                "group_claim_target_ids": group_claim_target_ids,
                "target_ids": group_claim_target_ids,
                "involved_objects": deepcopy(group_objects),
                "target_bounds": deepcopy(
                    scope_value["target_bounds"]
                ),
                "focus_center": list(group_scope.focus_center),
                "target_extent": list(group_scope.extent),
                "evidence_goal": group_scope_evidence_goal(
                    group_scope
                ),
                "grouping_role": (
                    "primary_visual_evidence_decomposition"
                ),
            }
        )
        request["event"]["group_id"] = group_scope.group_id
        request["event"]["object_ids"] = group_member_ids
        request["event"]["claim_target_ids"] = (
            group_claim_target_ids
        )
        request["event"]["focus_region"] = deepcopy(
            scope_value["target_bounds"]
        )
    return request


def _normalize_judge_response(
    raw: dict[str, Any],
    *,
    decision_mode: str,
) -> dict[str, Any]:
    evidence_status = raw.get("evidence_status")
    if evidence_status is not None and evidence_status not in {
        "sufficient",
        "insufficient",
    }:
        raise ValueError(
            "functional-semantic VLM evidence_status must be exactly "
            "'sufficient' or 'insufficient'"
        )
    confidence = raw.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("functional-semantic VLM confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "functional-semantic VLM confidence must be between 0 and 1"
            )
    reason = raw.get("reason")
    if decision_mode == "screen":
        router_state = raw.get("router_state")
        verdict = raw.get("verdict")
        if evidence_status == "insufficient":
            router_state = "insufficient_evidence"
        if router_state is None:
            router_state = {
                "valid": "not_suspicious",
                "invalid": "suspicious",
                "insufficient_evidence": "insufficient_evidence",
            }.get(verdict)
        if router_state not in {
            "not_suspicious",
            "suspicious",
            "insufficient_evidence",
        }:
            raise ValueError(
                "required-area global screen must return router_state exactly "
                "'not_suspicious', 'suspicious', or 'insufficient_evidence'"
            )
        expected_from_verdict = {
            "valid": "not_suspicious",
            "invalid": "suspicious",
            "insufficient_evidence": "insufficient_evidence",
        }.get(verdict)
        if (
            evidence_status != "insufficient"
            and expected_from_verdict is not None
            and router_state != expected_from_verdict
        ):
            raise ValueError(
                "required-area global screen returned contradictory verdict "
                "and router_state"
            )
        return {
            "router_state": router_state,
            "evidence_status": evidence_status,
            "confidence": confidence,
            "reason": reason,
            "judgement": deepcopy(raw),
        }

    verdict = raw.get("verdict")
    if evidence_status == "insufficient":
        verdict = "insufficient_evidence"
    if verdict not in {"valid", "invalid", "insufficient_evidence"}:
        raise ValueError(
            "functional-semantic final verdict must be exactly 'valid', "
            "'invalid', or 'insufficient_evidence'"
        )
    defects = raw.get("defects")
    if verdict == "invalid" and (
        not isinstance(defects, list) or not defects
    ):
        raise ValueError(
            "functional-semantic invalid verdict requires one or more "
            "explicitly identified visible prompt-scoped defects"
        )
    return {
        "verdict": verdict,
        "evidence_status": evidence_status,
        "confidence": confidence,
        "reason": reason,
        "defects": deepcopy(defects) if isinstance(defects, list) else [],
        "judgement": deepcopy(raw),
    }


def _request_local_evidence(
    *,
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    trigger: str,
    camera_evidence_provider: Any,
    counters: dict[str, int],
    group_scope: GroupCameraScope | None = None,
) -> dict[str, Any]:
    if camera_evidence_provider is None:
        return {
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "camera_evidence_provider_not_configured",
            "paths": [],
        }
    target_ids = _claim_target_ids(claim)
    if (
        claim.get("component") == "local_functionality"
        and not target_ids
    ):
        return {
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "prompt_scoped_local_functionality_targets_missing",
            "paths": [],
        }
    if (
        claim.get("component") == "required_functional_areas"
        and not target_ids
    ):
        return {
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "claim_scoped_required_area_targets_missing",
            "paths": [],
        }
    scoped_target_ids = (
        list(group_scope.member_ids)
        if group_scope is not None
        else target_ids
    )
    request = {
        "category": "functional_semantic_evidence_request",
        "metric": FUNCTIONAL_SEMANTIC_FIDELITY,
        "component": claim.get("component"),
        "claim_id": claim.get("claim_id"),
        "claim": deepcopy(claim),
        "scene": deepcopy(scene),
        "event": {
            "type": str(claim.get("component") or "functional_semantic_claim"),
            "claim_id": claim.get("claim_id"),
            "object_ids": scoped_target_ids,
            "area_id": claim.get("area_id"),
        },
        "object_ids": scoped_target_ids,
        "claim_target_ids": target_ids,
        "natural_language_prompt": prompt,
        "scene_summary": _compact_scene(scene),
        "evidence_scope": "benchmark_claim_target_local",
        "trigger": trigger,
        "selection_role": "visual_evidence_only_do_not_judge_metric",
        "generic_pairing_scan": False,
        "object_grouping_report_consumed": group_scope is not None,
    }
    if group_scope is not None:
        scope_value = group_scope.to_dict()
        request.update(
            {
                "group_scope": scope_value,
                "group_id": group_scope.group_id,
                "member_ids": list(group_scope.member_ids),
                "target_bounds": deepcopy(
                    scope_value["target_bounds"]
                ),
                "focus_center": list(group_scope.focus_center),
                "target_extent": list(group_scope.extent),
                "evidence_goal": group_scope_evidence_goal(
                    group_scope
                ),
                "grouping_role": (
                    "primary_visual_evidence_decomposition"
                ),
            }
        )
        request["event"]["group_id"] = group_scope.group_id
        request["event"]["focus_region"] = deepcopy(
            scope_value["target_bounds"]
        )
    call = getattr(
        camera_evidence_provider,
        "provide_functional_semantic_evidence",
        None,
    )
    if not callable(call) and callable(camera_evidence_provider):
        call = camera_evidence_provider
    if not callable(call):
        return {
            "status": "evidence_provider_failed",
            "route": "evidence_provider_failed",
            "reason": "camera_evidence_provider_not_callable",
            "paths": [],
        }
    try:
        counters["provider"] += 1
        provider_result = call(request)
    except Exception as exc:
        return {
            "status": "evidence_provider_failed",
            "route": "evidence_provider_failed",
            "reason": "camera_evidence_provider_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "paths": [],
        }
    if isinstance(provider_result, dict):
        provider_status = str(provider_result.get("status") or "").strip().lower()
        provider_error = provider_result.get("error")
        if provider_status in {"failed", "error"} or provider_error:
            return {
                "status": "evidence_provider_failed",
                "route": "evidence_provider_failed",
                "reason": "camera_evidence_provider_reported_failure",
                "error": str(provider_error or provider_status),
                "paths": [],
            }
        if provider_status in {
            "insufficient",
            "unavailable",
            "not_available",
        }:
            return {
                "status": "requires_vlm",
                "route": "unresolved",
                "reason": (
                    str(provider_result.get("reason") or "")
                    or "local_render_evidence_not_available"
                ),
                "paths": [],
            }
    elif not isinstance(provider_result, (list, tuple, str, Path)):
        return {
            "status": "evidence_provider_failed",
            "route": "evidence_provider_failed",
            "reason": "camera_evidence_provider_returned_malformed_result",
            "paths": [],
        }
    paths = _normalize_local_evidence_paths(provider_result)
    if not paths:
        return {
            "status": "requires_vlm",
            "route": "unresolved",
            "reason": "local_render_evidence_not_available",
            "paths": [],
        }
    return {
        "status": "available",
        "route": "claim_scoped_local_evidence",
        "reason": None,
        "paths": paths,
    }


def _metric_status(checks: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not checks:
        return "not_applicable", "no_functional_semantic_claims"
    failed = any(
        check.get("status")
        in {
            "vlm_adjudication_failed",
            "camera_evidence_failed",
            "evidence_provider_failed",
        }
        for check in checks
    )
    unresolved = any(check.get("status") != "checked" for check in checks)
    if failed:
        return "incomplete", "module_failures_present"
    if unresolved:
        return "incomplete", "claims_unresolved"
    return "evaluated", None


def _metric_report(
    *,
    metric_config: dict[str, Any],
    status: str,
    reason: str | None,
    checks: list[dict[str, Any]],
    provider_calls: int,
    judge_calls: int,
) -> dict[str, Any]:
    resolved_scores = [
        float(check["score"])
        for check in checks
        if check.get("status") == "checked" and _is_score(check.get("score"))
    ]
    failed_count = sum(
        check.get("status")
        in {
            "vlm_adjudication_failed",
            "camera_evidence_failed",
            "evidence_provider_failed",
        }
        for check in checks
    )
    unresolved_count = len(checks) - len(resolved_scores) - failed_count
    complete = bool(
        checks
        and len(resolved_scores) == len(checks)
        and failed_count == 0
    )
    partial_score = (
        sum(resolved_scores) / len(resolved_scores)
        if resolved_scores
        else None
    )
    return {
        "metric": FUNCTIONAL_SEMANTIC_FIDELITY,
        "namespace": FUNCTIONAL_SEMANTIC_FIDELITY,
        "components": list(FUNCTIONAL_SEMANTIC_COMPONENTS),
        "component_scoring": "single_family_no_separate_component_denominators",
        "interface_version": FUNCTIONAL_SEMANTIC_INTERFACE_VERSION,
        "implemented": True,
        "enabled": bool(metric_config["enabled"]),
        "status": status,
        "reason": reason,
        "score": partial_score if complete else None,
        "partial_score": partial_score,
        "affects_score": bool(checks),
        "renderer_invoked": provider_calls > 0,
        "camera_evidence_provider_invoked": provider_calls > 0,
        "camera_evidence_provider_call_count": provider_calls,
        "vlm_invoked": judge_calls > 0,
        "vlm_call_count": judge_calls,
        "evidence_plan": deepcopy(metric_config.get("evidence_plan")),
        "grouping_policy": None,
        "object_grouping_report_consumed": False,
        "local_functionality_activation": {
            "condition": "prompt_specified_local_functionality",
            "generic_or_unprompted_local_function_check": False,
        },
        "required_area_local_fallback": {
            "condition": "global_screen_suspicious_or_insufficient_evidence",
            "generic_pairing_scan": False,
        },
        "judgment_contract": {
            "evidence_first": True,
            "insufficient_evidence_result": "unresolved",
            "invalid_requires": (
                "one_or_more_significant_explicitly_identified_visible_prompt_scoped_failures"
            ),
            "otherwise_when_evidence_sufficient": "valid",
        },
        "eligible_claim_count": len(checks),
        "resolved_claim_count": len(resolved_scores),
        "unresolved_claim_count": max(0, unresolved_count),
        "failed_claim_count": failed_count,
        "coverage": {
            "eligible_count": len(checks),
            "resolved_count": len(resolved_scores),
            "failed_count": failed_count,
            "fraction": (
                len(resolved_scores) / len(checks)
                if checks
                else None
            ),
            "complete": complete,
        },
        "checks": checks,
    }


def _claim_target_ids(claim: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("target_ids", "object_ids", "member_ids"):
        if isinstance(claim.get(key), list):
            values.extend(claim[key])
    for key in ("target_id", "object_id", "subject_id"):
        if claim.get(key) is not None:
            values.append(claim[key])
    return list(
        dict.fromkeys(
            str(value)
            for value in values
            if str(value).strip()
        )
    )


def _target_objects(
    scene: dict[str, Any],
    target_ids: list[str],
) -> list[dict[str, Any]]:
    wanted = set(target_ids)
    result = []
    for obj in scene.get("objects", []) if isinstance(scene.get("objects"), list) else []:
        if not isinstance(obj, dict) or str(obj.get("id")) not in wanted:
            continue
        result.append(
            {
                "id": obj.get("id"),
                "category": obj.get("category"),
                "description": (
                    obj.get("description")
                    or obj.get("desc")
                    or obj.get("short_desc")
                ),
                "center": deepcopy(obj.get("center")),
                "size": deepcopy(obj.get("size")),
                "rotation_degrees": deepcopy(obj.get("rotation")),
            }
        )
    return result


def _compact_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical judge summary without a legacy L2 dependency."""

    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    return {
        "scene_id": scene.get("scene_id"),
        "request_id": scene.get("request_id"),
        "scene_type": scene.get("scene_type"),
        "boundary": deepcopy(scene.get("boundary")),
        "scene_height": scene.get("scene_height"),
        "architecture": architecture_contract_from_scene(scene),
        "objects": [
            {
                "id": obj.get("id"),
                "category": obj.get("category"),
                "description": (
                    obj.get("description")
                    or obj.get("desc")
                    or obj.get("short_desc")
                ),
                "center": deepcopy(obj.get("center")),
                "size": deepcopy(obj.get("size")),
                "rotation_degrees": deepcopy(obj.get("rotation")),
            }
            for obj in objects
            if isinstance(obj, dict)
        ],
    }


def _normalize_evidence_paths(evidence: Any) -> list[str]:
    paths: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (str, Path)):
            text = str(Path(value).expanduser())
            if text.strip():
                paths.append(text)
            return
        if isinstance(value, list) or isinstance(value, tuple):
            for item in value:
                collect(item)
            return
        if not isinstance(value, dict):
            return
        for key in ("path", "image_path"):
            if value.get(key) is not None:
                collect(value[key])
                return
        for key in (
            "render_evidence",
            "render_evidence_items",
            "evidence_paths",
            "paths",
            "views",
            "images",
        ):
            if value.get(key) is not None:
                collect(value[key])

    collect(evidence)
    return list(dict.fromkeys(paths))


def _normalize_local_evidence_paths(evidence: Any) -> list[str]:
    if isinstance(evidence, (list, tuple)) and all(
        isinstance(item, dict) for item in evidence
    ):
        ordered = sorted(
            evidence,
            key=lambda item: (
                0
                if "local" in str(item.get("role") or "")
                else 2
                if "global" in str(item.get("role") or "")
                else 1
            ),
        )
        return _normalize_evidence_paths(ordered)
    return _normalize_evidence_paths(evidence)


def _is_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _validate_flags(config: dict[str, Any], label: str) -> None:
    for flag in ("enabled", "implemented"):
        if not isinstance(config.get(flag), bool):
            raise FunctionalSemanticConfigError(f"{label}.{flag} must be boolean")
    if config.get("implemented") is not True:
        raise FunctionalSemanticConfigError(
            f"{label} is an implemented canonical runtime; implemented=false is unsupported"
        )


def _canonical_profile_layer(section: Any, label: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise FunctionalSemanticConfigError(
            f"{label} must be a JSON object"
        )
    result: dict[str, Any] = {}
    if "enabled" in section:
        result["enabled"] = deepcopy(section["enabled"])
    metrics = section.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise FunctionalSemanticConfigError(f"{label}.metrics must be a JSON object")
    if isinstance(metrics, dict):
        retired_metrics = sorted(set(metrics) & set(_RETIRED_METRIC_NAMES))
        if retired_metrics:
            raise FunctionalSemanticConfigError(
                f"{label}.metrics uses claim components as metric keys "
                f"{retired_metrics}; configure only "
                f"{FUNCTIONAL_SEMANTIC_FIDELITY!r}"
            )
        functional = metrics.get(FUNCTIONAL_SEMANTIC_FIDELITY)
        if functional is not None:
            if not isinstance(functional, dict):
                raise FunctionalSemanticConfigError(
                    f"{label}.metrics.{FUNCTIONAL_SEMANTIC_FIDELITY} "
                    "must be a JSON object"
                )
            # OOR/OAR are intentionally not copied: this resolver owns only the
            # functional-semantic runtime within canonical L2.
            result["metrics"] = {
                FUNCTIONAL_SEMANTIC_FIDELITY: deepcopy(functional)
            }
    return result


def _normalize_layer(layer: Any, label: str) -> dict[str, Any]:
    if not isinstance(layer, dict):
        raise FunctionalSemanticConfigError(
            f"functional_semantic_fidelity {label} must be a JSON object"
        )
    retired_namespaces = sorted(
        set(layer) & set(_RETIRED_CONFIG_NAMESPACES)
    )
    if retired_namespaces:
        raise FunctionalSemanticConfigError(
            "retired L2 config namespaces are not accepted: "
            f"{retired_namespaces}; pass the canonical config directly"
        )
    retired_top_level = sorted(set(layer) & set(_RETIRED_METRIC_NAMES))
    if retired_top_level:
        raise FunctionalSemanticConfigError(
            "functional-semantic claim components are not top-level config keys: "
            f"{retired_top_level}"
        )
    result = deepcopy(layer)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        retired_metrics = sorted(set(metrics) & set(_RETIRED_METRIC_NAMES))
        if retired_metrics:
            raise FunctionalSemanticConfigError(
                "functional-semantic claim components are not metric config keys: "
                f"{retired_metrics}; configure only "
                f"{FUNCTIONAL_SEMANTIC_FIDELITY!r}"
            )
    return result


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise FunctionalSemanticConfigError(
            "functional_semantic_fidelity config patch must be a JSON object"
        )
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
