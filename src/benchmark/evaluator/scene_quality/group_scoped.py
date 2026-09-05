"""Per-group evidence routing and scene-quality result aggregation.

This module keeps group-loop orchestration out of the metric contract module.
Metric rubrics, response validation, and provider adapters remain injected by
the caller so this layer cannot redefine benchmark semantics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.evaluator.scene_quality.claim_identity import (
    claim_records,
    deduplicate_defects,
    match_final_defects_to_routed_claims,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    canonicalize_clearance_causal_attribution,
    canonicalize_functional_defect_check_linkage,
    canonicalize_typed_invalid_envelope,
    validate_functional_check_results,
)
from benchmark.evaluator.scene_quality.functional_group_evidence import (
    FunctionalGroupEvidenceBank,
)
from benchmark.evaluator.scene_quality.functional_measurements import (
    compact_functional_measurements_for_checks,
)
from benchmark.evaluator.scene_quality.placement_checks import (
    canonicalize_placement_defect_linkage,
    merge_placement_checks,
    normalize_judge_originated_placement_results,
    validate_placement_check_results,
)
from benchmark.evaluator.scene_quality.placement_severity import (
    placement_severity_summary,
)
from benchmark.evaluator.scene_quality.terminal import (
    infrastructure_failure_from_scope,
    scope_was_defaulted,
    terminalize_required_scope,
)
from benchmark.visual_judge.group_scope import (
    GroupCameraScope,
    build_group_camera_scope,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)
from benchmark.visual_judge.orchestration.budget import (
    extend_acquisition_ledger,
    merge_acquisition_ledger_delta,
)
from benchmark.visual_judge.orchestration.evidence_window import (
    SHARED_GROUP_BANK_POLICY,
)


FUNCTIONAL_GROUP_LOCAL_EVIDENCE_POLICIES = {
    "isolated_episode",
    SHARED_GROUP_BANK_POLICY,
}


def resolve_group_evidence_packets(
    value: list[str] | dict[str, Any] | None,
    *,
    metric_name: str,
    policy: dict[str, Any],
    scene: dict[str, Any],
    prompt: str | None,
    groups: list[dict[str, Any]],
    grouping_report: dict[str, Any] | None,
    camera_evidence_provider: Any,
    resolve_metric_evidence: Callable[..., tuple[list[str], dict[str, Any]]],
    initial_acquisition_ledger: dict[str, Any] | None = None,
    max_total_images: int | None = None,
    camera_target_ids_by_group: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    metric_ledger = deepcopy(initial_acquisition_ledger)
    for group in groups:
        group_id = str(group["group_id"])
        try:
            scope = build_group_camera_scope(
                scene,
                group,
                metric=metric_name,
                include_global_context=bool(
                    policy.get("include_global_context")
                ),
                grouping_report=grouping_report,
            )
            camera_target_ids = list(
                dict.fromkeys(
                    str(item)
                    for item in (
                        (camera_target_ids_by_group or {}).get(group_id)
                        or scope.member_ids
                    )
                    if str(item).strip()
                )
            )
            camera_scope = scope
            if tuple(camera_target_ids) != scope.member_ids:
                camera_scope = build_group_camera_scope(
                    scene,
                    {
                        "group_id": group_id,
                        "object_ids": camera_target_ids,
                    },
                    metric=metric_name,
                    include_global_context=bool(
                        policy.get("include_global_context")
                    ),
                    grouping_report=grouping_report,
                )
        except Exception as exc:
            packets.append(
                {
                    "group": deepcopy(group),
                    "group_scope": _unavailable_group_scope(group),
                    "paths": [],
                    "resolution": {
                        "scope_satisfied": False,
                        "source": "group_scope_failure",
                        "provider_invoked": False,
                        "provider_status": "failed",
                        "provider_reason": "group_camera_scope_invalid",
                        "missing_paths": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                }
            )
            continue
        group_value = _evidence_for_group(
            value,
            metric_name=metric_name,
            scope=str(policy["camera_scope"]),
            group_id=group_id,
            single_group=len(groups) == 1,
        )
        packet_policy = deepcopy(policy)
        paths, resolution = resolve_metric_evidence(
            group_value,
            metric_name=metric_name,
            policy=packet_policy,
            scene=scene,
            prompt=prompt,
            selected_object_ids=camera_target_ids,
            selected_group_ids=[group_id],
            selected_groups=[group],
            camera_evidence_provider=camera_evidence_provider,
            group_scope=camera_scope,
        )
        artifact_paths = _resolution_artifact_paths(
            paths,
            resolution,
        )
        episode_ledger_before = extend_acquisition_ledger(
            None,
            artifact_ids=[],
        )
        episode_ledger_after = extend_acquisition_ledger(
            episode_ledger_before,
            artifact_ids=evidence_artifact_refs(paths),
        )
        metric_ledger_before = deepcopy(metric_ledger)
        metric_ledger = extend_acquisition_ledger(
            metric_ledger,
            artifact_ids=evidence_artifact_refs(artifact_paths),
        )
        over_budget = bool(
            max_total_images is not None
            and int(
                episode_ledger_after.get("total_images_acquired") or 0
            )
            > max_total_images
        )
        resolution["acquisition_budget"] = {
            "scope": "group_judge_episode",
            "counting": (
                "judge_packet_seed_then_controller_acquired_artifacts"
            ),
            "max_total_images": max_total_images,
            "initial_judge_evidence_count": int(
                episode_ledger_after.get("total_images_acquired") or 0
            ),
            "acquired_artifact_count": len(
                evidence_artifact_refs(artifact_paths)
            ),
            "metric_artifact_count_after": int(
                metric_ledger.get("total_images_acquired") or 0
            ),
        }
        if over_budget:
            resolution.update(
                scope_satisfied=False,
                provider_reason=(
                    "group_judge_episode_evidence_budget_exceeded"
                ),
                acquired_artifact_paths=artifact_paths,
            )
        packets.append(
            {
                "group": deepcopy(group),
                "group_scope": scope,
                "camera_target_scope": camera_scope,
                "camera_target_ids": list(camera_target_ids),
                "paths": paths,
                "resolution": resolution,
                "budget_scope": "group_judge_episode",
                "camera_acquisition_ledger_before": (
                    episode_ledger_before
                ),
                "camera_acquisition_ledger_after": deepcopy(
                    episode_ledger_after
                ),
                "metric_camera_acquisition_ledger_before": (
                    metric_ledger_before
                ),
                "metric_camera_acquisition_ledger_after": deepcopy(
                    metric_ledger
                ),
            }
        )
    return packets


def _resolution_artifact_paths(
    paths: list[str],
    resolution: dict[str, Any],
) -> list[str]:
    usage = resolution.get("provider_usage")
    acquired = (
        usage.get("acquired_artifact_paths")
        if isinstance(usage, dict)
        else None
    )
    values = [
        *paths,
        *(acquired if isinstance(acquired, list) else []),
    ]
    return list(
        dict.fromkeys(
            str(item)
            for item in values
            if isinstance(item, (str, bytes))
            and str(item).strip()
        )
    )


def evaluate_group_scoped_judgements(
    *,
    base: dict[str, Any],
    metric_name: str,
    scene: dict[str, Any],
    prompt: str | None,
    packets: list[dict[str, Any]],
    vlm_judge: Any,
    authorized_deviations: list[dict[str, Any]],
    visual_style_spec: dict[str, Any] | None,
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
    evidence_phase: str = "final",
    decision_mode: str = "final",
    group_local_check_granularity: str = "batched",
    group_local_evidence_policy: str = "isolated_episode",
    group_local_active_window_max_images: int = 6,
) -> dict[str, Any]:
    """Evaluate group-local scopes in batched or atomic-check mode.

    ``per_check`` is deliberately limited to Functional required checks.  In
    ``isolated_episode`` mode every atomic episode receives the same immutable
    seed packet and owns its follow-up evidence.  In ``shared_group_bank`` mode
    the episodes remain semantically isolated while relevant visual artifacts
    can flow forward through a bounded active window.  Both policies fold the
    atomic results back into one group result before metric aggregation.
    """

    if group_local_check_granularity not in {"batched", "per_check"}:
        raise ValueError(
            "group_local_check_granularity must be exactly 'batched' or "
            "'per_check'"
        )
    if group_local_evidence_policy not in (
        FUNCTIONAL_GROUP_LOCAL_EVIDENCE_POLICIES
    ):
        raise ValueError(
            "group_local_evidence_policy must be exactly "
            "'isolated_episode' or 'shared_group_bank'"
        )
    if (
        group_local_evidence_policy == SHARED_GROUP_BANK_POLICY
        and group_local_check_granularity != "per_check"
    ):
        raise ValueError(
            "shared_group_bank requires per_check Functional group-local "
            "granularity"
        )
    if (
        isinstance(group_local_active_window_max_images, bool)
        or not isinstance(group_local_active_window_max_images, int)
        or group_local_active_window_max_images < 2
    ):
        raise ValueError(
            "group_local_active_window_max_images must be an integer >= 2"
        )
    base["group_local_check_granularity"] = (
        group_local_check_granularity
    )
    base["group_local_evidence_policy"] = group_local_evidence_policy
    base["group_local_active_window_max_images"] = (
        group_local_active_window_max_images
    )
    if (
        metric_name != "functional_consistency"
        or group_local_check_granularity == "batched"
    ):
        return _evaluate_group_scoped_judgements_batched(
            base=base,
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            packets=packets,
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            build_judge_request=build_judge_request,
            call_judge=call_judge,
            apply_prompt_exemptions=apply_prompt_exemptions,
            normalize_judgement=normalize_judgement,
            evidence_phase=evidence_phase,
            decision_mode=decision_mode,
        )

    if group_local_evidence_policy == SHARED_GROUP_BANK_POLICY:
        return _evaluate_functional_checks_with_shared_bank(
            base=base,
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            packets=packets,
            vlm_judge=vlm_judge,
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            build_judge_request=build_judge_request,
            call_judge=call_judge,
            apply_prompt_exemptions=apply_prompt_exemptions,
            normalize_judgement=normalize_judgement,
            evidence_phase=evidence_phase,
            decision_mode=decision_mode,
            max_active_images=group_local_active_window_max_images,
        )

    expanded_packets = _expand_functional_check_packets(packets)
    expanded = _evaluate_group_scoped_judgements_batched(
        base=deepcopy(base),
        metric_name=metric_name,
        scene=scene,
        prompt=prompt,
        packets=expanded_packets,
        vlm_judge=vlm_judge,
        authorized_deviations=authorized_deviations,
        visual_style_spec=visual_style_spec,
        build_judge_request=build_judge_request,
        call_judge=call_judge,
        apply_prompt_exemptions=apply_prompt_exemptions,
        normalize_judgement=normalize_judgement,
        evidence_phase=evidence_phase,
        decision_mode=decision_mode,
    )
    combined = _combine_functional_check_episodes(
        packets=packets,
        episode_results=list(expanded.get("group_results") or []),
    )
    expanded.pop("infrastructure_failures", None)
    expanded["group_local_check_granularity"] = "per_check"
    expanded["group_local_evidence_policy"] = "isolated_episode"
    return _aggregate_group_results(
        expanded,
        combined,
        metric_name=metric_name,
    )


def _evaluate_functional_checks_with_shared_bank(
    *,
    base: dict[str, Any],
    metric_name: str,
    scene: dict[str, Any],
    prompt: str | None,
    packets: list[dict[str, Any]],
    vlm_judge: Any,
    authorized_deviations: list[dict[str, Any]],
    visual_style_spec: dict[str, Any] | None,
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
    evidence_phase: str,
    decision_mode: str,
    max_active_images: int,
) -> dict[str, Any]:
    """Evaluate atomic checks sequentially while sharing visual artifacts."""

    episode_results: list[dict[str, Any]] = []
    bank_records: dict[str, dict[str, Any]] = {}
    metric_ledger = deepcopy(base.get("camera_acquisition_ledger"))

    for packet in packets:
        group_id = str((packet.get("group") or {}).get("group_id") or "")
        bank: FunctionalGroupEvidenceBank | None = None
        bank_error: str | None = None
        if packet.get("paths") and (
            packet.get("resolution") or {}
        ).get("scope_satisfied"):
            try:
                bank = FunctionalGroupEvidenceBank.from_packet(
                    packet,
                    max_active_images=max_active_images,
                )
            except (TypeError, ValueError) as exc:
                bank_error = f"{type(exc).__name__}: {exc}"

        expanded_packets = _expand_functional_check_packets([packet])
        for expanded_packet in expanded_packets:
            episode = deepcopy(expanded_packet)
            functional = episode.get("functional_probe_evidence")
            required_checks = (
                functional.get("required_checks")
                if isinstance(functional, dict)
                and isinstance(functional.get("required_checks"), list)
                else []
            )
            if required_checks:
                check = deepcopy(required_checks[0])
                include_reusable = True
            else:
                check = {
                    "check_id": f"group_baseline:{group_id}",
                    "target_ids": list(
                        (episode.get("group") or {}).get("object_ids") or []
                    ),
                    "required_observations": [],
                }
                include_reusable = False

            if bank is not None:
                active_evidence, initial_window = bank.initial_window(
                    check,
                    include_reusable=include_reusable,
                )
                episode["paths"] = list(active_evidence)
                episode["functional_group_evidence_initial_window"] = (
                    initial_window
                )
                episode["functional_group_evidence_window"] = (
                    bank.window_context(check=check)
                )
                resolution = deepcopy(episode.get("resolution") or {})
                resolution["functional_group_evidence"] = {
                    "policy": SHARED_GROUP_BANK_POLICY,
                    "max_active_images": max_active_images,
                    "fixed_artifact_ids": list(bank.fixed_artifact_ids),
                    "initial_artifact_ids": list(
                        initial_window["selected_artifact_ids"]
                    ),
                    "bank_reuse_precedes_camera_selection": True,
                }
                acquisition_budget = resolution.get("acquisition_budget")
                if isinstance(acquisition_budget, dict):
                    acquisition_budget["initial_judge_evidence_count"] = len(
                        evidence_artifact_refs(active_evidence)
                    )
                    acquisition_budget["active_window_max_images"] = (
                        max_active_images
                    )
                episode["resolution"] = resolution
                episode["camera_acquisition_ledger_after"] = (
                    extend_acquisition_ledger(
                        None,
                        artifact_ids=evidence_artifact_refs(active_evidence),
                    )
                )
            elif bank_error is not None:
                # A malformed shared-bank contract must never silently run as
                # the isolated policy.  Route the atomic scope through the
                # existing required-scope terminalizer without invoking the
                # Judge so the failure remains explicit and auditable.
                episode["paths"] = []
                resolution = deepcopy(episode.get("resolution") or {})
                resolution.update(
                    scope_satisfied=False,
                    provider_status="failed",
                    provider_reason=(
                        "group_evidence_bank_validation_failed"
                    ),
                    group_evidence_bank_error=bank_error,
                )
                episode["resolution"] = resolution

            episode_base = deepcopy(base)
            if isinstance(metric_ledger, dict):
                episode_base["camera_acquisition_ledger"] = deepcopy(
                    metric_ledger
                )
            evaluated = _evaluate_group_scoped_judgements_batched(
                base=episode_base,
                metric_name=metric_name,
                scene=scene,
                prompt=prompt,
                packets=[episode],
                vlm_judge=vlm_judge,
                authorized_deviations=authorized_deviations,
                visual_style_spec=visual_style_spec,
                build_judge_request=build_judge_request,
                call_judge=call_judge,
                apply_prompt_exemptions=apply_prompt_exemptions,
                normalize_judgement=normalize_judgement,
                evidence_phase=evidence_phase,
                decision_mode=decision_mode,
            )
            results = list(evaluated.get("group_results") or [])
            if len(results) != 1:
                raise ValueError(
                    "shared Functional evidence evaluation must produce "
                    "exactly one atomic episode result"
                )
            record = deepcopy(results[0])
            if bank is not None:
                evidence_window = bank.absorb_controller_audit(
                    record.get("camera_control_audit"),
                    check=check,
                    initial_window=initial_window,
                )
                record["functional_group_evidence_window_audit"] = (
                    evidence_window
                )
                reused = list(
                    dict.fromkeys(
                        str(artifact_id)
                        for event in evidence_window.get("events") or []
                        if isinstance(event, dict)
                        for artifact_id in event.get(
                            "reused_artifact_ids"
                        )
                        or []
                    )
                )
                record["shared_dynamic_evidence_reused"] = bool(reused)
                record["shared_reused_artifact_ids"] = reused
                record["camera_selector_avoided_by_bank_reuse"] = bool(
                    reused
                )
            elif bank_error is not None:
                record["functional_group_evidence_window_audit"] = {
                    "policy": SHARED_GROUP_BANK_POLICY,
                    "status": "failed_closed",
                    "reason": "group_evidence_bank_validation_failed",
                    "error": bank_error,
                }
            episode_results.append(record)
            next_ledger = evaluated.get("camera_acquisition_ledger")
            if isinstance(next_ledger, dict):
                metric_ledger = deepcopy(next_ledger)

        if bank is not None:
            bank_records[group_id] = bank.to_dict()
        else:
            bank_records[group_id] = {
                "schema_version": "functional_group_evidence_bank_v1",
                "policy": SHARED_GROUP_BANK_POLICY,
                "decision_authority": "none",
                "group_id": group_id,
                "status": "unavailable",
                "reason": (
                    "group_evidence_bank_validation_failed"
                    if bank_error is not None
                    else "group_seed_evidence_unavailable"
                ),
                **({"error": bank_error} if bank_error is not None else {}),
            }

    combined = _combine_functional_check_episodes(
        packets=packets,
        episode_results=episode_results,
    )
    result = deepcopy(base)
    result.pop("infrastructure_failures", None)
    if isinstance(metric_ledger, dict):
        result["camera_acquisition_ledger"] = deepcopy(metric_ledger)
    result["group_local_check_granularity"] = "per_check"
    result["group_local_evidence_policy"] = SHARED_GROUP_BANK_POLICY
    result["group_local_active_window_max_images"] = max_active_images
    result["functional_group_evidence_bank"] = {
        "schema_version": "functional_group_evidence_bank_collection_v1",
        "policy": SHARED_GROUP_BANK_POLICY,
        "decision_authority": "none",
        "groups": bank_records,
    }
    return _aggregate_group_results(
        result,
        combined,
        metric_name=metric_name,
    )


def _evaluate_group_scoped_judgements_batched(
    *,
    base: dict[str, Any],
    metric_name: str,
    scene: dict[str, Any],
    prompt: str | None,
    packets: list[dict[str, Any]],
    vlm_judge: Any,
    authorized_deviations: list[dict[str, Any]],
    visual_style_spec: dict[str, Any] | None,
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
    evidence_phase: str = "final",
    decision_mode: str = "final",
) -> dict[str, Any]:
    """Judge each supplied packet once and aggregate without score averaging."""

    group_results: list[dict[str, Any]] = []
    metric_ledger = deepcopy(
        base.get("camera_acquisition_ledger")
        if isinstance(base.get("camera_acquisition_ledger"), dict)
        else None
    )
    for packet in packets:
        group = packet["group"]
        group_id = str(group["group_id"])
        members = [str(item) for item in group["object_ids"]]
        resolution = packet["resolution"]
        episode_ledger_before_judge = _packet_episode_ledger(packet)
        packet_metric_ledger = packet.get(
            "metric_camera_acquisition_ledger_after"
        )
        if isinstance(packet_metric_ledger, dict):
            metric_ledger = extend_acquisition_ledger(
                metric_ledger,
                artifact_ids=[
                    str(item)
                    for item in (
                        packet_metric_ledger.get("artifact_ids") or []
                    )
                    if str(item).strip()
                ],
            )
        metric_ledger = extend_acquisition_ledger(
            metric_ledger,
            artifact_ids=list(
                episode_ledger_before_judge.get("artifact_ids") or []
            ),
        )
        record: dict[str, Any] = {
            "group_id": group_id,
            "member_ids": members,
            "group_scope": packet["group_scope"].to_dict(),
            "camera_target_scope": (
                packet.get("camera_target_scope").to_dict()
                if isinstance(
                    packet.get("camera_target_scope"),
                    GroupCameraScope,
                )
                else packet["group_scope"].to_dict()
            ),
            "camera_target_ids": list(
                packet.get("camera_target_ids") or members
            ),
            "evidence_paths": list(packet["paths"]),
            "evidence_resolution": deepcopy(resolution),
            "status": "unresolved",
            "terminal_state": "pending",
            "score": None,
            "reason": resolution.get("provider_reason"),
            "vlm_invoked": False,
            "judge_episode_count": 0,
            "functional_check_episode_id": packet.get(
                "functional_check_episode_id"
            ),
            "functional_check_granularity": packet.get(
                "functional_check_granularity", "batched"
            ),
            "functional_group_evidence_window": deepcopy(
                packet.get("functional_group_evidence_window")
            ),
            "functional_group_evidence_initial_window": deepcopy(
                packet.get("functional_group_evidence_initial_window")
            ),
            "judgement": None,
            "routed_candidate_claims": deepcopy(
                packet.get("routed_candidate_claims") or []
            ),
            "functional_probe_evidence": deepcopy(
                packet.get("functional_probe_evidence")
            ),
            "functional_check_resolution": None,
            "placement_discovery": deepcopy(
                packet.get("placement_discovery")
            ),
            "required_placement_checks": deepcopy(
                packet.get("required_placement_checks") or []
            ),
            "placement_check_resolution": None,
            "functional_ownership_ledger": deepcopy(
                packet.get("functional_ownership_ledger")
            ),
            "claim_correspondence": [],
            "camera_acquisition_episode": {
                "scope": "group_judge_episode",
                "ledger_before_judge": deepcopy(
                    episode_ledger_before_judge
                ),
                "ledger_after_judge": deepcopy(
                    episode_ledger_before_judge
                ),
            },
        }
        if not resolution.get("scope_satisfied") or not packet["paths"]:
            record["reason"] = (
                resolution.get("provider_reason")
                or "group_local_render_evidence_unavailable"
            )
            terminal = terminalize_required_scope(
                record,
                phase=_group_episode_phase(packet, group_id),
            )
            group_results.append(
                _attach_default_typed_rows(
                    terminal,
                    metric_name=metric_name,
                )
            )
            continue

        judge_request_kwargs = {
            "metric_name": metric_name,
            "scene": scene,
            "prompt": prompt,
            "render_evidence": packet["paths"],
            "selected_object_ids": members,
            "selected_group_ids": [group_id],
            "groups": [group],
            "authorized_deviations": authorized_deviations,
            "visual_style_spec": visual_style_spec,
            "group_scope": packet["group_scope"],
            "evidence_phase": evidence_phase,
            "decision_mode": decision_mode,
            "routed_screen_claims": record[
                "routed_candidate_claims"
            ],
        }
        if record["functional_probe_evidence"] is not None:
            judge_request_kwargs["functional_probe_evidence"] = record[
                "functional_probe_evidence"
            ]
        if record["functional_group_evidence_window"] is not None:
            judge_request_kwargs["functional_group_evidence_window"] = (
                deepcopy(record["functional_group_evidence_window"])
            )
        if record["placement_discovery"] is not None:
            judge_request_kwargs["placement_discovery"] = record[
                "placement_discovery"
            ]
        if record["required_placement_checks"]:
            judge_request_kwargs["required_placement_checks"] = deepcopy(
                record["required_placement_checks"]
            )
        if record["functional_ownership_ledger"] is not None:
            judge_request_kwargs["functional_ownership_ledger"] = deepcopy(
                record["functional_ownership_ledger"]
            )
        request = build_judge_request(
            **judge_request_kwargs,
        )
        functional_preflight = _functional_visual_preflight(
            packet,
            required_checks=required_functional_checks_from_packet(packet),
        )
        if functional_preflight is not None:
            request["functional_evidence_preflight"] = functional_preflight
        request["camera_acquisition_ledger"] = deepcopy(
            episode_ledger_before_judge
        )
        record["vlm_invoked"] = True
        record["judge_episode_count"] = 1
        audit_records = getattr(vlm_judge, "audit_records", None)
        audit_start = (
            len(audit_records)
            if isinstance(audit_records, list)
            else None
        )
        try:
            raw = call_judge(vlm_judge, request)
            adjusted = apply_prompt_exemptions(
                raw,
                metric_name=metric_name,
                authorized_deviations=authorized_deviations,
            )
            if (
                record.get("required_placement_checks")
                or (
                    record.get("functional_probe_evidence") or {}
                ).get("required_checks")
            ):
                adjusted = canonicalize_typed_invalid_envelope(adjusted)
            placement_resolution = None
            if metric_name == "semantic_placement_consistency":
                registered_checks = (
                    _registered_placement_checks_from_controller_audit(
                        audit_records,
                        audit_start=audit_start,
                    )
                )
                if registered_checks:
                    base["placement_check_ledger"] = (
                        merge_placement_checks(
                            base["placement_check_ledger"],
                            registered_checks,
                        )
                    )
                internal_registrations = adjusted.get(
                    "judge_originated_placement_check_registrations"
                )
                if isinstance(internal_registrations, list):
                    judge_originated_checks = [
                        deepcopy(item)
                        for item in internal_registrations
                        if isinstance(item, dict)
                    ]
                else:
                    adjusted, judge_originated_checks = (
                        normalize_judge_originated_placement_results(
                            adjusted,
                            known_ids=set(members),
                            groups=[group],
                            existing_checks=list(
                                (
                                    base.get("placement_check_ledger") or {}
                                ).get("checks")
                                or []
                            ),
                            expected_owner_stage="group_local",
                        )
                    )
                if judge_originated_checks:
                    base["placement_check_ledger"] = (
                        merge_placement_checks(
                            base["placement_check_ledger"],
                            judge_originated_checks,
                        )
                    )
                phase_ids = {
                    str(item.get("check_id") or "")
                    for item in [
                        *record["required_placement_checks"],
                        *registered_checks,
                        *judge_originated_checks,
                    ]
                    if isinstance(item, dict) and item.get("check_id")
                }
                phase_checks = [
                    deepcopy(check)
                    for check in (
                        base.get("placement_check_ledger") or {}
                    ).get("checks")
                    or []
                    if str(check.get("check_id") or "") in phase_ids
                ]
                adjusted = canonicalize_placement_defect_linkage(
                    adjusted,
                    required_checks=phase_checks,
                )
                placement_resolution = (
                    validate_placement_check_results(
                        adjusted,
                        required_checks=phase_checks,
                        function_events=list(
                            (
                                record.get(
                                    "functional_ownership_ledger"
                                )
                                or {}
                            ).get("events")
                            or []
                        ),
                    )
                )
            required_functional_checks = (
                (
                    record.get("functional_probe_evidence")
                    or {}
                ).get("required_checks")
                or []
            )
            if (
                metric_name == "functional_consistency"
                and required_functional_checks
            ):
                adjusted = canonicalize_clearance_causal_attribution(
                    adjusted,
                    required_checks=[
                        deepcopy(item)
                        for item in required_functional_checks
                        if isinstance(item, dict)
                    ],
                )
                adjusted = canonicalize_functional_defect_check_linkage(
                    adjusted,
                    required_checks=[
                        deepcopy(item)
                        for item in required_functional_checks
                        if isinstance(item, dict)
                    ],
                )
            check_resolution = (
                validate_functional_check_results(
                    adjusted,
                    required_checks=[
                        deepcopy(item)
                        for item in required_functional_checks
                        if isinstance(item, dict)
                    ],
                )
                if metric_name == "functional_consistency"
                and required_functional_checks
                else None
            )
            outcome = normalize_judgement(
                adjusted,
                metric_name=metric_name,
                # A group-local Judge may only report defects on members of
                # this immutable evidence scope.  Scene-wide validation would
                # allow a defect from another group to leak into this result.
                valid_object_ids=(
                    {
                        str(item.get("id"))
                        for item in scene.get("objects") or []
                        if isinstance(item, dict) and item.get("id")
                    }
                    if metric_name == "functional_consistency"
                    and any(
                        isinstance(item, dict)
                        and item.get("check_type") == "clearance"
                        for item in required_functional_checks
                    )
                    else set(members)
                ),
            )
            record.update(
                status=outcome["status"],
                score=outcome["score"],
                reason=outcome["reason"],
                judgement=adjusted,
                functional_check_resolution=check_resolution,
                placement_check_resolution=placement_resolution,
                claim_correspondence=(
                    match_final_defects_to_routed_claims(
                        metric_name,
                        adjusted.get("defects") or [],
                        record["routed_candidate_claims"],
                    )
                ),
            )
        except Exception as exc:
            schema_audit = response_schema_audit_from_exception(exc)
            record.update(
                status="failed",
                reason="vlm_judge_failed",
                judgement={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **(
                        {"response_schema_audit": schema_audit}
                        if schema_audit is not None
                        else {}
                    ),
                },
            )
        if (
            audit_start is not None
            and isinstance(audit_records, list)
            and len(audit_records) > audit_start
        ):
            record["camera_control_audit"] = deepcopy(
                audit_records[-1]
            )
            next_ledger = _camera_acquisition_ledger_from_audit(
                audit_records[-1]
            )
            if next_ledger is not None:
                record["camera_acquisition_episode"][
                    "ledger_after_judge"
                ] = deepcopy(next_ledger)
                metric_ledger = merge_acquisition_ledger_delta(
                    metric_ledger,
                    episode_before=episode_ledger_before_judge,
                    episode_after=next_ledger,
                )
        terminal = terminalize_required_scope(
            record,
            phase=_group_episode_phase(packet, group_id),
        )
        group_results.append(
            _attach_default_typed_rows(
                terminal,
                metric_name=metric_name,
            )
        )

    if isinstance(metric_ledger, dict):
        base["camera_acquisition_ledger"] = deepcopy(metric_ledger)
    return _aggregate_group_results(
        base,
        group_results,
        metric_name=metric_name,
    )


def _attach_default_typed_rows(
    record: dict[str, Any],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Complete only the typed rows owned by one defaulted Judge episode."""

    if not scope_was_defaulted(record):
        return record
    judgement = (
        record.get("judgement")
        if isinstance(record.get("judgement"), dict)
        else {}
    )
    if metric_name == "functional_consistency":
        required = list(
            (
                record.get("functional_probe_evidence")
                if isinstance(
                    record.get("functional_probe_evidence"), dict
                )
                else {}
            ).get("required_checks")
            or []
        )
        rows = [
            {
                "check_id": str(check["check_id"]),
                "target_ids": [
                    str(item) for item in check.get("target_ids") or []
                ],
                "observation_status": "inferred_under_budget",
                "conclusion": "valid",
                "reason": (
                    "The atomic check defaulted valid after a non-hard "
                    "episode failure; ambiguity is retained in audit."
                ),
            }
            for check in required
            if isinstance(check, dict) and check.get("check_id")
        ]
        judgement["functional_check_results"] = rows
        if required:
            record["functional_check_resolution"] = (
                validate_functional_check_results(
                    judgement,
                    required_checks=required,
                )
            )
    elif metric_name == "semantic_placement_consistency":
        required = [
            deepcopy(check)
            for check in record.get("required_placement_checks") or []
            if isinstance(check, dict) and check.get("check_id")
        ]
        rows = [
            {
                "check_id": str(check["check_id"]),
                "subject_id": str(check["subject_id"]),
                "context_ids": sorted(
                    str(item) for item in check.get("context_ids") or []
                ),
                "observation_status": "inferred_under_budget",
                "conclusion": "valid",
                "reason": (
                    "The atomic check defaulted valid after a non-hard "
                    "episode failure; ambiguity is retained in audit."
                ),
            }
            for check in required
        ]
        judgement["placement_check_results"] = rows
        record["placement_check_resolution"] = (
            validate_placement_check_results(
                judgement,
                required_checks=required,
                function_events=list(
                    (
                        record.get("functional_ownership_ledger")
                        if isinstance(
                            record.get("functional_ownership_ledger"),
                            dict,
                        )
                        else {}
                    ).get("events")
                    or []
                ),
            )
        )
    record["judgement"] = judgement
    record["defaulted_scoring_unit_count"] = max(
        1,
        len(
            judgement.get("functional_check_results")
            or judgement.get("placement_check_results")
            or []
        ),
    )
    return record


def _group_episode_phase(packet: dict[str, Any], group_id: str) -> str:
    check_id = str(packet.get("functional_check_episode_id") or "").strip()
    if check_id:
        return f"group_local:{group_id}:check:{check_id}"
    return f"group_local:{group_id}"


def _expand_functional_check_packets(
    packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for packet in packets:
        functional = packet.get("functional_probe_evidence")
        required = (
            functional.get("required_checks")
            if isinstance(functional, dict)
            else None
        )
        required_checks = [
            deepcopy(item)
            for item in required or []
            if isinstance(item, dict) and item.get("check_id")
        ]
        if not required_checks:
            baseline = deepcopy(packet)
            baseline["functional_check_granularity"] = "per_check"
            baseline["functional_check_episode_id"] = None
            expanded.append(baseline)
            continue
        for check in required_checks:
            check_id = str(check["check_id"])
            episode = deepcopy(packet)
            episode["functional_check_granularity"] = "per_check"
            episode["functional_check_episode_id"] = check_id
            episode["shared_seed_evidence_reused"] = True
            episode_functional = deepcopy(functional)
            episode_functional["required_checks"] = [deepcopy(check)]
            episode_functional["required_check_ids"] = [check_id]
            episode_functional["required_check_count"] = 1
            episode_functional["episode_scope"] = (
                "single_group_local_functional_check"
            )
            if isinstance(
                episode_functional.get("functional_measurements"), dict
            ):
                episode_functional["functional_measurements"] = (
                    compact_functional_measurements_for_checks(
                        episode_functional["functional_measurements"],
                        [check_id],
                    )
                )
            for key in (
                "observation_requests",
                "relation_observation_requests",
                "image_order",
            ):
                values = episode_functional.get(key)
                if isinstance(values, list):
                    episode_functional[key] = [
                        deepcopy(item)
                        for item in values
                        if not isinstance(item, dict)
                        or not item.get("check_ids")
                        or check_id
                        in {
                            str(value)
                            for value in item.get("check_ids") or []
                        }
                    ]
            episode["functional_probe_evidence"] = episode_functional
            episode = _scope_functional_episode_evidence(
                episode,
                check=check,
            )
            expanded.append(episode)
    return expanded


def required_functional_checks_from_packet(
    packet: dict[str, Any],
) -> list[dict[str, Any]]:
    functional = packet.get("functional_probe_evidence")
    return [
        deepcopy(item)
        for item in (
            functional.get("required_checks")
            if isinstance(functional, dict)
            else []
        )
        or []
        if isinstance(item, dict) and item.get("check_id")
    ]


def _scope_functional_episode_evidence(
    packet: dict[str, Any],
    *,
    check: dict[str, Any],
) -> dict[str, Any]:
    """Keep the fixed group packet plus evidence relevant to one check.

    The probe acquisition stage is group-owned, so its original packet may
    contain probes for several objects.  Per-check judging is meaningful only
    when those dynamic paths are filtered with the same check identity as the
    structured obligations.  Fixed angled-global and group-local evidence is
    always retained.
    """

    result = deepcopy(packet)
    paths = list(
        dict.fromkeys(
            str(path)
            for path in result.get("paths") or []
            if str(path).strip()
        )
    )
    resolution = (
        deepcopy(result.get("resolution"))
        if isinstance(result.get("resolution"), dict)
        else {}
    )
    reuse = (
        resolution.get("functional_probe_reuse")
        if isinstance(
            resolution.get("functional_probe_reuse"), dict
        )
        else {}
    )
    baseline = [
        str(path)
        for path in reuse.get("baseline_packet_paths") or []
        if str(path).strip() and str(path) in paths
    ]
    requested = {
        str(path)
        for path in reuse.get("requested_probe_paths") or []
        if str(path).strip()
    }
    if not baseline:
        # Packets without supplementary probe metadata are already scoped and
        # must remain unchanged.  With probe metadata, the first two entries
        # are the frozen global/local seed contract.
        baseline = paths if not requested else paths[:2]

    functional = result.get("functional_probe_evidence")
    image_order = (
        functional.get("image_order")
        if isinstance(functional, dict)
        and isinstance(functional.get("image_order"), list)
        else []
    )
    check_id = str(check.get("check_id") or "")
    targets = {
        str(item) for item in check.get("target_ids") or [] if str(item)
    }
    relevant: set[str] = {
        str(path)
        for path in check.get("evidence_refs") or []
        if str(path).strip()
    }
    for item in image_order:
        if not isinstance(item, dict):
            continue
        artifact = str(item.get("artifact_id") or "").strip()
        if not artifact:
            continue
        item_check_ids = {
            str(value) for value in item.get("check_ids") or []
        }
        item_targets = {
            str(value)
            for value in [
                *list(item.get("target_ids") or []),
                *list(item.get("related_target_ids") or []),
            ]
        }
        if (
            check_id in item_check_ids
            or (not item_check_ids and bool(targets & item_targets))
        ):
            relevant.add(artifact)

    dynamic_paths = [
        path
        for path in paths
        if path not in baseline
        and (
            path in relevant
            or (not requested and not image_order)
        )
    ]
    selected = list(dict.fromkeys([*baseline, *dynamic_paths]))
    omitted = [path for path in paths if path not in selected]
    result["paths"] = selected
    resolution["functional_check_evidence_scope"] = {
        "policy": "fixed_group_seed_plus_check_bound_dynamic_v1",
        "check_id": check_id,
        "target_ids": sorted(targets),
        "fixed_paths": list(baseline),
        "selected_dynamic_paths": list(dynamic_paths),
        "omitted_unrelated_paths": omitted,
    }
    acquisition_budget = resolution.get("acquisition_budget")
    if isinstance(acquisition_budget, dict):
        acquisition_budget["initial_judge_evidence_count"] = len(
            evidence_artifact_refs(selected)
        )
    result["resolution"] = resolution
    result["camera_acquisition_ledger_after"] = (
        extend_acquisition_ledger(
            None,
            artifact_ids=evidence_artifact_refs(selected),
        )
    )
    return result


def _functional_visual_preflight(
    packet: dict[str, Any],
    *,
    required_checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Require one recovery view when machine-side direction binding failed.

    This is a routing guard, not a semantic sufficiency decision.  It covers
    the narrow fail-closed case where a check explicitly depends on a decoded
    usable side but no trusted surface observation or complete orientation
    binding reached the Judge packet.
    """

    if len(required_checks) != 1:
        return None
    check = required_checks[0]
    required = {
        str(item) for item in check.get("required_observations") or []
    }
    directional_observations = {
        "interaction_side_visible",
        "front_back_disambiguated",
    }
    if not (required & directional_observations):
        return None
    directed_targets = {
        str(item.get("target_id") or "")
        for item in check.get("target_affordances") or []
        if isinstance(item, dict)
        and item.get("directionality") == "directed"
        and str(item.get("target_id") or "")
    }
    if check.get("check_type") == "architecture_orientation":
        directed_targets.update(
            str(item) for item in check.get("target_ids") or []
        )
    if not directed_targets:
        return None

    functional = packet.get("functional_probe_evidence")
    functional = functional if isinstance(functional, dict) else {}
    observed_targets = _trusted_surface_observation_targets(functional)
    missing_targets = sorted(directed_targets - observed_targets)
    reason_codes: list[str] = []
    if missing_targets:
        reason_codes.append("usable_surface_not_machine_resolved")

    if check.get("check_type") == "architecture_orientation":
        bindings = (
            (functional.get("architecture_orientation_policy") or {}).get(
                "evidence_bindings"
            )
            if isinstance(
                functional.get("architecture_orientation_policy"), dict
            )
            else []
        )
        binding = next(
            (
                item
                for item in bindings or []
                if isinstance(item, dict)
                and str(item.get("check_id") or "")
                == str(check.get("check_id") or "")
            ),
            None,
        )
        if not isinstance(binding, dict) or binding.get("status") != "complete":
            reason_codes.append("side_conditioned_view_not_bound")

    if not reason_codes:
        return None
    missing_observations = [
        item
        for item in (
            "interaction_side_visible",
            "front_back_disambiguated",
        )
        if item in required
    ]
    return {
        "schema_version": "functional_evidence_preflight_v1",
        "active": True,
        "check_id": str(check.get("check_id") or ""),
        "target_ids": missing_targets or sorted(directed_targets),
        "missing_observations": missing_observations,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "initial_evidence_refs": list(
            dict.fromkeys(str(path) for path in packet.get("paths") or [])
        ),
        "resolution_policy": "acquire_before_binary_judgement",
        "decision_authority": "none",
    }


def _trusted_surface_observation_targets(
    functional: dict[str, Any],
) -> set[str]:
    hypotheses: list[Any] = []
    boundary = functional.get("boundary_clearance_evidence")
    if isinstance(boundary, dict):
        hypotheses.extend(boundary.get("usable_surface_hypotheses") or [])
    for request in functional.get("observation_requests") or []:
        if not isinstance(request, dict):
            continue
        usable = request.get("usable_surface")
        if isinstance(usable, dict):
            hypotheses.extend(usable.get("hypotheses") or [])
        elif isinstance(usable, list):
            hypotheses.extend(usable)
        geometry = request.get("functional_geometry")
        if isinstance(geometry, dict):
            hypotheses.extend(geometry.get("surface_observations") or [])
    return {
        str(item.get("target_id") or "")
        for item in hypotheses
        if isinstance(item, dict)
        and str(item.get("target_id") or "")
        and (
            item.get("status") == "identified"
            or bool(item.get("side_id"))
        )
    }


def _combine_functional_check_episodes(
    *,
    packets: list[dict[str, Any]],
    episode_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for result in episode_results:
        by_group.setdefault(str(result.get("group_id") or ""), []).append(
            deepcopy(result)
        )

    combined_results: list[dict[str, Any]] = []
    for packet in packets:
        group_id = str(packet["group"]["group_id"])
        episodes = by_group.get(group_id, [])
        functional = packet.get("functional_probe_evidence")
        required_checks = [
            deepcopy(item)
            for item in (
                functional.get("required_checks")
                if isinstance(functional, dict)
                else []
            )
            or []
            if isinstance(item, dict) and item.get("check_id")
        ]
        if not required_checks:
            if not episodes:
                continue
            baseline = deepcopy(episodes[0])
            baseline["functional_check_granularity"] = "per_check"
            baseline["check_episodes"] = deepcopy(episodes)
            baseline["judge_episode_count"] = sum(
                int(
                    item.get("judge_episode_count")
                    or (1 if item.get("vlm_invoked") else 0)
                )
                for item in episodes
            )
            combined_results.append(baseline)
            continue

        expected_ids = [str(item["check_id"]) for item in required_checks]
        observed_episode_ids = [
            str(item.get("functional_check_episode_id") or "")
            for item in episodes
            if item.get("functional_check_episode_id")
        ]
        if len(observed_episode_ids) != len(set(observed_episode_ids)):
            raise ValueError(
                "per-check Functional episodes contain duplicate check "
                f"identities for {group_id}"
            )
        episode_by_id = {
            str(item.get("functional_check_episode_id") or ""): item
            for item in episodes
            if item.get("functional_check_episode_id")
        }
        if set(episode_by_id) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(episode_by_id))
            duplicate_or_extra = sorted(set(episode_by_id) - set(expected_ids))
            raise ValueError(
                "per-check Functional episodes do not match the required "
                f"ledger for {group_id}: missing={missing}, "
                f"extra={duplicate_or_extra}"
            )
        ordered = [episode_by_id[check_id] for check_id in expected_ids]
        first = deepcopy(ordered[0])
        first["functional_check_episode_id"] = None
        first["functional_check_granularity"] = "per_check"
        first["check_episodes"] = deepcopy(ordered)
        first["judge_episode_count"] = sum(
            int(
                item.get("judge_episode_count")
                or (1 if item.get("vlm_invoked") else 0)
            )
            for item in ordered
        )
        first["vlm_invoked"] = first["judge_episode_count"] > 0
        first["camera_control_audits"] = [
            deepcopy(item["camera_control_audit"])
            for item in ordered
            if isinstance(item.get("camera_control_audit"), dict)
        ]
        first["functional_check_result_refs"] = {
            check_id: f"group_local_review:{group_id}:check:{check_id}"
            for check_id in expected_ids
        }
        first["claim_correspondence"] = [
            deepcopy(item)
            for episode in ordered
            for item in episode.get("claim_correspondence") or []
        ]
        failed = next(
            (item for item in ordered if item.get("status") != "evaluated"),
            None,
        )
        if failed is not None:
            first.update(
                status=failed.get("status"),
                terminal_state=failed.get("terminal_state"),
                score=None,
                reason=failed.get("reason"),
                judgement=deepcopy(failed.get("judgement")),
            )
            combined_results.append(first)
            continue

        judgements = [
            item.get("judgement")
            for item in ordered
            if isinstance(item.get("judgement"), dict)
        ]
        rows = [
            deepcopy(row)
            for judgement in judgements
            for row in judgement.get("functional_check_results") or []
            if isinstance(row, dict)
        ]
        defects = [
            deepcopy(defect)
            for judgement in judgements
            for defect in judgement.get("defects") or []
            if isinstance(defect, dict)
        ]
        invalid = any(item.get("score") == 0.0 for item in ordered)
        confidence = min(
            (
                float(judgement.get("confidence") or 0.0)
                for judgement in judgements
            ),
            default=0.0,
        )
        aggregate_judgement = {
            "evidence_status": "sufficient",
            "verdict": "invalid" if invalid else "valid",
            "confidence": confidence,
            "reason": (
                "At least one atomic group-local Functional check is invalid."
                if invalid
                else "Every atomic group-local Functional check is valid."
            ),
            "missing_evidence": [],
            "defects": defects if invalid else [],
            "evidence_request": None,
            "functional_check_results": rows,
            "aggregation": "atomic_group_local_checks",
        }
        check_resolution = validate_functional_check_results(
            aggregate_judgement,
            required_checks=required_checks,
        )
        first.update(
            status="evaluated",
            terminal_state=(
                "evaluated_degraded"
                if any(
                    item.get("terminal_state") == "evaluated_degraded"
                    for item in ordered
                )
                else "evaluated"
            ),
            score=0.0 if invalid else 1.0,
            reason=None,
            judgement=aggregate_judgement,
            functional_check_resolution=check_resolution,
        )
        combined_results.append(first)
    return combined_results


def _camera_acquisition_ledger_from_audit(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    audit = (
        record.get("audit")
        if isinstance(record.get("audit"), dict)
        else record
    )
    acquisition = (
        audit.get("camera_acquisition")
        if isinstance(audit, dict)
        and isinstance(audit.get("camera_acquisition"), dict)
        else {}
    )
    ledger = acquisition.get("ledger")
    return deepcopy(ledger) if isinstance(ledger, dict) else None


def _packet_episode_ledger(
    packet: dict[str, Any],
) -> dict[str, Any]:
    ledger = packet.get("camera_acquisition_ledger_after")
    if isinstance(ledger, dict):
        return deepcopy(ledger)
    return extend_acquisition_ledger(
        None,
        artifact_ids=evidence_artifact_refs(
            list(packet.get("paths") or [])
        ),
    )


def group_evidence_resolution_summary(
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    resolutions = [packet["resolution"] for packet in packets]
    statuses = {
        str(item.get("provider_status") or "unknown")
        for item in resolutions
    }
    return {
        "scope_satisfied": bool(resolutions)
        and all(
            item.get("scope_satisfied") is True
            for item in resolutions
        ),
        "source": "per_group_camera_evidence",
        "provider_invoked": any(
            item.get("provider_invoked") is True
            for item in resolutions
        ),
        "provider_status": (
            next(iter(statuses)) if len(statuses) == 1 else "mixed"
        ),
        "provider_reason": next(
            (
                str(item.get("provider_reason"))
                for item in resolutions
                if item.get("provider_reason")
            ),
            None,
        ),
        "global_context_count": sum(
            int(item.get("global_context_count") or 0)
            for item in resolutions
        ),
        "scoped_evidence_count": sum(
            int(item.get("scoped_evidence_count") or 0)
            for item in resolutions
        ),
        "missing_paths": list(
            dict.fromkeys(
                path
                for item in resolutions
                for path in item.get("missing_paths") or []
            )
        ),
    }


def group_packet_audit(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": packet["group"]["group_id"],
        "member_ids": list(packet["group_scope"].member_ids),
        "group_scope": packet["group_scope"].to_dict(),
        "evidence_paths": list(packet["paths"]),
        "evidence_resolution": deepcopy(packet["resolution"]),
        "routed_candidate_claims": deepcopy(
            packet.get("routed_candidate_claims") or []
        ),
        "functional_probe_evidence": deepcopy(
            packet.get("functional_probe_evidence")
        ),
        "functional_group_evidence_window": deepcopy(
            packet.get("functional_group_evidence_window")
        ),
        "functional_group_evidence_initial_window": deepcopy(
            packet.get("functional_group_evidence_initial_window")
        ),
        "placement_discovery": deepcopy(
            packet.get("placement_discovery")
        ),
        "required_placement_checks": deepcopy(
            packet.get("required_placement_checks") or []
        ),
        "functional_ownership_ledger": deepcopy(
            packet.get("functional_ownership_ledger")
        ),
        "budget_scope": packet.get("budget_scope"),
        "camera_acquisition_ledger_before": deepcopy(
            packet.get("camera_acquisition_ledger_before")
        ),
        "camera_acquisition_ledger_after": deepcopy(
            packet.get("camera_acquisition_ledger_after")
        ),
        "metric_camera_acquisition_ledger_before": deepcopy(
            packet.get("metric_camera_acquisition_ledger_before")
        ),
        "metric_camera_acquisition_ledger_after": deepcopy(
            packet.get("metric_camera_acquisition_ledger_after")
        ),
    }


def _registered_placement_checks_from_controller_audit(
    audit_records: Any,
    *,
    audit_start: int | None,
) -> list[dict[str, Any]]:
    if (
        audit_start is None
        or not isinstance(audit_records, list)
        or len(audit_records) <= audit_start
    ):
        return []
    latest = audit_records[-1]
    payload = (
        latest.get("audit")
        if isinstance(latest, dict)
        and isinstance(latest.get("audit"), dict)
        else latest
    )
    request = (
        payload.get("judge_request")
        if isinstance(payload, dict)
        and isinstance(payload.get("judge_request"), dict)
        else {}
    )
    context = (
        request.get("context")
        if isinstance(request.get("context"), dict)
        else {}
    )
    return [
        deepcopy(item)
        for item in context.get("required_placement_checks") or []
        if isinstance(item, dict)
    ]


def _group_control_audits(record: dict[str, Any]) -> list[Any]:
    audits = record.get("camera_control_audits")
    if isinstance(audits, list):
        return list(audits)
    audit = record.get("camera_control_audit")
    return [audit] if audit is not None else []


def _aggregate_group_results(
    base: dict[str, Any],
    group_results: list[dict[str, Any]],
    *,
    metric_name: str,
) -> dict[str, Any]:
    evaluated = [
        item
        for item in group_results
        if item["status"] == "evaluated"
    ]
    invalid = [item for item in evaluated if item["score"] == 0.0]
    all_resolved = bool(group_results) and len(evaluated) == len(
        group_results
    )
    base["group_results"] = group_results
    base["judge_call_count"] = sum(
        int(
            item.get("judge_episode_count")
            or (1 if item.get("vlm_invoked") else 0)
        )
        for item in group_results
    )
    base["vlm_invoked"] = bool(base["judge_call_count"])
    base["evidence_request"]["vlm_invoked"] = base["vlm_invoked"]
    control_summaries = [
        _control_audit_summary(audit)
        for item in group_results
        for audit in _group_control_audits(item)
    ]
    preview_count = sum(
        int(item["preview_render_count"])
        for item in control_summaries
    )
    controller_final_count = sum(
        int(item["final_render_count"])
        for item in control_summaries
    )
    initial_render_count = sum(
        _provider_resolution_render_count(
            item.get("evidence_resolution")
        )
        for item in group_results
    )
    final_count = initial_render_count + controller_final_count
    initial_rendered = any(
        _provider_resolution_rendered(
            item.get("evidence_resolution")
        )
        for item in group_results
    )
    base["renderer_invoked"] = bool(
        base.get("renderer_invoked")
        or initial_rendered
        or final_count
    )
    base["preview_renderer_invoked"] = preview_count > 0
    base["preview_render_count"] = preview_count
    base["final_render_count"] = final_count
    base["production_camera_selector_backend"] = next(
        (
            item["production_camera_selector_backend"]
            for item in control_summaries
            if item["production_camera_selector_backend"]
        ),
        None,
    )
    base["effective_vlm_selection_mode"] = next(
        (
            item["effective_vlm_selection_mode"]
            for item in control_summaries
            if item["effective_vlm_selection_mode"]
        ),
        None,
    )
    base["semantic_selection_triggered"] = any(
        item["semantic_selection_triggered"]
        for item in control_summaries
    )
    base["trusted_candidate_count"] = max(
        (
            item["trusted_candidate_count"]
            for item in control_summaries
        ),
        default=0,
    )
    base["evidence_request"]["renderer_invoked"] = base[
        "renderer_invoked"
    ]
    base["coverage"] = {
        "eligible_count": len(group_results),
        "resolved_count": len(evaluated),
        "fraction": (
            len(evaluated) / len(group_results)
            if group_results
            else None
        ),
        "complete": all_resolved,
    }

    infrastructure_failures = [
        failure
        for item in group_results
        if (
            failure := infrastructure_failure_from_scope(
                item,
                phase="group_local",
                scope_id=str(item.get("group_id") or "") or None,
            )
        )
        is not None
    ]
    if infrastructure_failures:
        base["infrastructure_failures"] = deepcopy(
            infrastructure_failures
        )
        base.update(
            status="failed",
            terminal_state="infrastructure_failure",
            reason="required_scope_infrastructure_failure",
            score=None,
            judgement={
                "evidence_status": "unavailable",
                "verdict": None,
                "confidence": 0.0,
                "reason": (
                    "One or more required group Judge scopes failed for an "
                    "engineering reason; no scientific verdict was fabricated."
                ),
                "missing_evidence": [
                    f"group_scoped_evidence:{item['group_id']}"
                    for item in group_results
                    if item.get("status") != "evaluated"
                ],
                "defects": [],
                "aggregation": "fail_closed_on_required_scope_failure",
                "infrastructure_failures": deepcopy(
                    infrastructure_failures
                ),
                "group_judgements": deepcopy(group_results),
            },
        )
        return base

    aggregate_terminal_state = (
        "evaluated_degraded"
        if any(
            item.get("terminal_state") == "evaluated_degraded"
            for item in group_results
        )
        else "evaluated"
    )

    defects = deduplicate_defects(
        metric_name,
        (
            defect
            for item in evaluated
            for defect in (item.get("judgement") or {}).get("defects")
            or []
        ),
    )
    placement_summary = (
        placement_severity_summary(defects)
        if metric_name == "semantic_placement_consistency"
        else None
    )
    if placement_summary is not None:
        base["placement_severity"] = deepcopy(placement_summary)
    base["final_defect_claims"] = claim_records(
        metric_name,
        defects,
        source_phase="group_visual",
        claim_status="final",
    )
    if invalid and all_resolved:
        aggregate = (
            deepcopy(invalid[0]["judgement"])
            if len(group_results) == 1
            else {
                "evidence_status": "sufficient",
                "verdict": "invalid",
                "confidence": min(
                    float(
                        (item.get("judgement") or {}).get(
                            "confidence"
                        )
                        or 0.0
                    )
                    for item in invalid
                ),
                "reason": (
                    "At least one group has a significant in-scope defect."
                ),
                "missing_evidence": [],
                "defects": defects,
            }
        )
        aggregate["defects"] = deepcopy(defects)
        if placement_summary is not None:
            aggregate["placement_severity"] = deepcopy(
                placement_summary
            )
        aggregate.update(
            aggregation="invalid_if_any_group_invalid",
            group_judgements=deepcopy(group_results),
        )
        base.update(
            status="evaluated",
            terminal_state=aggregate_terminal_state,
            reason=None,
            score=0.0,
            judgement=aggregate,
        )
        return base
    if all_resolved:
        aggregate = (
            deepcopy(evaluated[0]["judgement"])
            if len(group_results) == 1
            else {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "confidence": min(
                    float(
                        (item.get("judgement") or {}).get(
                            "confidence"
                        )
                        or 0.0
                    )
                    for item in evaluated
                ),
                "reason": (
                    "All eligible groups resolved without an in-scope "
                    "defect."
                ),
                "missing_evidence": [],
                "defects": [],
            }
        )
        if placement_summary is not None:
            aggregate["placement_severity"] = deepcopy(
                placement_summary
            )
        aggregate.update(
            aggregation="all_groups_must_resolve_valid",
            group_judgements=deepcopy(group_results),
        )
        base.update(
            status="evaluated",
            terminal_state=aggregate_terminal_state,
            reason=None,
            score=1.0,
            judgement=aggregate,
        )
        return base

    missing_groups = [
        item["group_id"]
        for item in group_results
        if item["status"] != "evaluated"
    ]
    contract_failure = {
        "phase": "metric_aggregation",
        "scope_id": metric_name,
        "failure_kind": "terminal_contract_failure",
        "reason": "required_scope_lacked_terminal_binary_result",
        "controller_stop_reason": None,
        "error_type": "TerminalContractError",
        "error": "A required group did not produce a terminal result.",
    }
    base["infrastructure_failures"] = [deepcopy(contract_failure)]
    base.update(
        status="failed",
        terminal_state="infrastructure_failure",
        reason="terminal_contract_failure",
        score=None,
        judgement={
            "evidence_status": "unavailable",
            "verdict": None,
            "confidence": 0.0,
            "reason": (
                "The required group terminal contract was not satisfied."
            ),
            "missing_evidence": [
                f"group_scoped_evidence:{group_id}"
                for group_id in missing_groups
            ],
            "defects": [],
            "aggregation": (
                "fail_closed_on_terminal_contract_violation"
            ),
            "infrastructure_failures": [deepcopy(contract_failure)],
            "group_judgements": deepcopy(group_results),
        },
    )
    return base


def _provider_resolution_rendered(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    usage = value.get("provider_usage")
    return (
        isinstance(usage, dict)
        and usage.get("cache_hit") is False
        and bool(usage.get("evidence_refs"))
    )


def _provider_resolution_render_count(value: Any) -> int:
    if not _provider_resolution_rendered(value):
        return 0
    usage = value.get("provider_usage")
    refs = usage.get("evidence_refs") if isinstance(usage, dict) else []
    return len(refs) if isinstance(refs, list) else 0


def _control_audit_summary(value: Any) -> dict[str, Any]:
    audit = (
        value.get("audit")
        if isinstance(value, dict)
        else None
    )
    telemetry = (
        audit.get("experiment_telemetry")
        if isinstance(audit, dict)
        else None
    )
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    events = telemetry.get("events")
    events = events if isinstance(events, list) else []
    selection_events = [
        item
        for item in events
        if isinstance(item, dict)
        and item.get("kind") == "camera_selection"
        and item.get("stage") == "vlm"
    ]
    trace = (
        audit.get("trace")
        if isinstance(audit, dict)
        and isinstance(audit.get("trace"), list)
        else []
    )
    bank_events = [
        item
        for item in trace
        if isinstance(item, dict)
        and item.get("stage") == "trusted_candidate_bank"
    ]
    return {
        "preview_render_count": int(
            telemetry.get("preview_render_count") or 0
        ),
        "final_render_count": int(
            telemetry.get("full_render_count") or 0
        ),
        "production_camera_selector_backend": (
            selection_events[-1].get("selector_backend")
            if selection_events
            else None
        ),
        "effective_vlm_selection_mode": (
            selection_events[-1].get("selection_mode")
            if selection_events
            else None
        ),
        "semantic_selection_triggered": any(
            item.get("reason") == "semantic_selection_required"
            for item in events
            if isinstance(item, dict)
            and item.get("kind") == "camera_escalation"
        ),
        "trusted_candidate_count": max(
            (
                int(item.get("candidate_count") or 0)
                for item in bank_events
            ),
            default=0,
        ),
    }


def _evidence_for_group(
    value: list[str] | dict[str, Any] | None,
    *,
    metric_name: str,
    scope: str,
    group_id: str,
    single_group: bool,
) -> list[str] | dict[str, Any] | None:
    if not isinstance(value, dict):
        return value
    global_paths = (
        value.get("global")
        or value.get("global_context")
        or value.get("default")
        or value.get("all")
    )
    group_paths: Any = None
    metric_value = value.get(metric_name)
    if isinstance(metric_value, dict):
        group_paths = metric_value.get(group_id)
    elif single_group and isinstance(metric_value, list):
        group_paths = metric_value
    if group_paths is None and isinstance(value.get(scope), dict):
        group_paths = value[scope].get(group_id)
    if group_paths is None and isinstance(value.get(group_id), list):
        group_paths = value[group_id]
    result: dict[str, Any] = {}
    if isinstance(global_paths, list):
        result["global"] = global_paths
    if isinstance(group_paths, list):
        result[scope] = group_paths
    return result


def _unavailable_group_scope(
    group: dict[str, Any],
) -> GroupCameraScope:
    """Audit-only placeholder; no selector or renderer receives it."""

    return GroupCameraScope(
        group_id=str(group.get("group_id") or "unknown"),
        member_ids=tuple(
            str(item) for item in group.get("object_ids") or []
        ),
        target_bounds_min=(0.0, 0.0, 0.0),
        target_bounds_max=(0.0, 0.0, 0.0),
        focus_center=(0.0, 0.0, 0.0),
        extent=(0.0, 0.0, 0.0),
        required_observations=(
            "joint_visibility",
            "group_context_visible",
            "limited_local_context",
        ),
        require_global_anchor=False,
    )
