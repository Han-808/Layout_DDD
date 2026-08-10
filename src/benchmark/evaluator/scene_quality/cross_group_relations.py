"""Cross-group functional-relation scheduling and Judge episodes.

This module owns the complete middle stage between scene-global discovery and
group-local review:

* project atomic correspondences into one schedule item per exact target set;
* require successful relation-specific acquisition before a Judge episode;
* execute one isolated Judge episode with one result row per atomic check; and
* fail closed when a Judge attributes a defect outside the relation scope.

The surrounding workflow remains in :mod:`global_group_first`; this module has
no authority over discovery, camera acquisition, group-local evaluation, or
scene-level aggregation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.evaluator.scene_quality.claim_identity import (
    canonical_target_ids,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    canonicalize_functional_defect_check_linkage,
    canonicalize_typed_invalid_envelope,
    checks_for_cross_group_relation,
    functional_relation_required_observations,
    validate_functional_check_results,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    functional_relation_judge_packet,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.functional_discovery_contract import (
    normalized_functional_relation_predicates,
)


def _cross_group_relation_episode_specs(
    *,
    acquisition_audit: dict[str, Any],
    groups: list[dict[str, Any]],
    global_paths: list[str],
) -> list[dict[str, Any]]:
    """Project atomic checks into one Judge episode per trusted target set."""

    group_by_object = {
        str(object_id): str(group.get("group_id") or "")
        for group in groups
        if isinstance(group, dict)
        for object_id in group.get("object_ids") or []
        if str(object_id).strip()
    }
    groups_by_id = {
        str(group.get("group_id") or ""): deepcopy(group)
        for group in groups
        if isinstance(group, dict) and group.get("group_id")
    }
    results_by_identity: dict[tuple[str, ...], dict[str, Any]] = {}
    for result in acquisition_audit.get("probe_results") or []:
        if (
            not isinstance(result, dict)
            or result.get("route_scope") != "cross_group"
            or result.get("kind") != "functional_correspondence"
        ):
            continue
        target_ids = _relation_target_ids(result)
        identity = tuple(sorted(target_ids))
        if len(identity) != 2:
            raise ValueError(
                "cross-group relation result requires exactly two target IDs"
            )
        current = results_by_identity.get(identity)
        results_by_identity[identity] = (
            _merge_relation_probe_results(current, result)
            if current is not None
            else deepcopy(result)
        )

    discovery = acquisition_audit.get("functional_discovery")
    discovery = discovery if isinstance(discovery, dict) else {}
    relations_by_identity: dict[
        tuple[str, ...],
        dict[str, dict[str, Any]],
    ] = {}
    for relation in discovery.get("cross_group_correspondences") or []:
        if not isinstance(relation, dict):
            continue
        target_ids = [
            str(item)
            for item in relation.get("target_ids") or []
            if str(item).strip()
        ]
        identity = tuple(sorted(dict.fromkeys(target_ids)))
        if len(identity) != 2:
            raise ValueError(
                "discovered cross-group relation requires exactly two target IDs"
            )
        for predicate in normalized_functional_relation_predicates(relation):
            by_predicate = relations_by_identity.setdefault(identity, {})
            if predicate in by_predicate:
                raise ValueError(
                    "duplicate discovered atomic relation identity"
                )
            by_predicate[predicate] = {
                **deepcopy(relation),
                "target_ids": target_ids,
                "predicate": predicate,
                "observation_kinds": [predicate],
            }

    # Frozen audits may contain acquired probes without the normalized
    # discovery copy. Reconstruct atomic compatibility records without making
    # the probe result a decision authority.
    if not relations_by_identity:
        for identity, result in results_by_identity.items():
            predicates = (
                list(result.get("relation_predicates") or [])
                or list(result.get("observation_kinds") or [])
            )
            compatibility_record = {
                "target_ids": list(identity),
                "observation_kinds": predicates,
            }
            normalized_predicates = normalized_functional_relation_predicates(
                compatibility_record
            )
            goals = list(result.get("observation_goals") or [])
            for index, predicate in enumerate(normalized_predicates):
                relations_by_identity.setdefault(identity, {})[predicate] = {
                    "discovery_id": str(
                        (result.get("discovery_ids") or [None])[0]
                        or result.get("probe_id")
                        or f"legacy_relation_{index + 1:02d}"
                    ),
                    "target_ids": list(identity),
                    "predicate": predicate,
                    "observation_kinds": [predicate],
                    "observation_goal": str(
                        (goals[index] if index < len(goals) else None)
                        or result.get("view_goal")
                        or "neutral atomic joint-use observation"
                    ),
                }

    specs: list[dict[str, Any]] = []
    functional_check_ledger = (
        acquisition_audit.get("functional_check_ledger")
        if isinstance(
            acquisition_audit.get("functional_check_ledger"),
            dict,
        )
        else None
    )
    for identity in sorted(relations_by_identity):
        atomic_relations = [
            relations_by_identity[identity][predicate]
            for predicate in sorted(relations_by_identity[identity])
        ]
        target_ids = list(identity)
        predicates = [
            str(item["predicate"]) for item in atomic_relations
        ]
        discovery_ids = list(
            dict.fromkeys(
                str(item.get("discovery_id") or "")
                for item in atomic_relations
                if str(item.get("discovery_id") or "").strip()
            )
        )
        observation_goals = list(
            dict.fromkeys(
                str(item.get("observation_goal") or "")
                for item in atomic_relations
                if str(item.get("observation_goal") or "").strip()
            )
        )
        result = results_by_identity.get(identity)
        pair_specific_available = bool(
            isinstance(result, dict)
            and result.get("status") == "available"
            and result.get("evidence_paths")
        )
        normalized_result = (
            deepcopy(result)
            if isinstance(result, dict)
            else {
                "status": "not_scheduled",
                "route_scope": "cross_group",
                "kind": "functional_correspondence",
                "target_ids": target_ids[:1],
                "related_target_ids": target_ids[1:],
                "discovery_ids": discovery_ids,
                "observation_kinds": predicates,
                "relation_predicates": predicates,
                "observation_goals": observation_goals,
                "view_goal": (
                    observation_goals[0]
                    if observation_goals
                    else "neutral atomic joint-use observation"
                ),
                "evidence_paths": [],
                "error_type": None,
                "error": (
                    "cross-group relation was not scheduled for acquisition"
                ),
            }
        )
        normalized_result["relation_predicates"] = predicates
        normalized_result["observation_kinds"] = predicates
        normalized_result["discovery_ids"] = discovery_ids
        normalized_result["observation_goals"] = observation_goals
        group_ids = list(
            dict.fromkeys(
                group_by_object.get(object_id, "")
                for object_id in target_ids
                if group_by_object.get(object_id)
            )
        )
        if len(group_ids) < 2:
            raise ValueError(
                "acquired cross-group relation does not span trusted groups"
            )
        evidence_paths = [
            str(path)
            for path in normalized_result.get("evidence_paths") or []
            if str(path).strip()
        ]
        relation_id = (
            discovery_ids[0]
            if len(discovery_ids) == 1
            else "functional_relation_episode:" + "+".join(discovery_ids)
            if discovery_ids
            else f"cross_group_relation_{len(specs) + 1:02d}"
        )
        if (functional_check_ledger or {}).get("checks"):
            required_checks = checks_for_cross_group_relation(
                functional_check_ledger,
                target_ids=target_ids,
                discovery_ids=discovery_ids,
                predicates=predicates,
            )
            delivered_predicates = {
                str(
                    check.get("predicate")
                    or check.get("check_type")
                    or ""
                )
                for check in required_checks
            }
            if delivered_predicates != set(predicates):
                raise ValueError(
                    "cross-group episode check predicates do not match "
                    "discovered atomic predicates"
                )
        else:
            required_checks = [
                _legacy_atomic_required_check(
                    index=(len(specs) * 100) + index,
                    predicate=relation["predicate"],
                    target_ids=target_ids,
                    group_ids=group_ids,
                    discovery_id=str(
                        relation.get("discovery_id") or relation_id
                    ),
                    observation_goal=str(
                        relation.get("observation_goal")
                        or "neutral atomic joint-use observation"
                    ),
                    acquisition_status=str(
                        normalized_result.get("status") or "failed"
                    ),
                    artifact_rendered=pair_specific_available,
                )
                for index, relation in enumerate(
                    atomic_relations,
                    start=1,
                )
            ]
        required_check_ids = [
            str(check.get("check_id") or "") for check in required_checks
        ]
        specs.append(
            {
                "relation_id": relation_id,
                "probe_id": normalized_result.get("probe_id"),
                "discovery_ids": discovery_ids,
                "target_ids": target_ids,
                "group_ids": group_ids,
                "groups": [
                    deepcopy(groups_by_id[group_id])
                    for group_id in group_ids
                    if group_id in groups_by_id
                ],
                "relation_predicates": predicates,
                "observation_kinds": predicates,
                "observation_goals": observation_goals,
                "evidence_paths": evidence_paths,
                "pair_specific_evidence_available": (
                    pair_specific_available
                ),
                "acquisition_status": str(
                    normalized_result.get("status") or "failed"
                ),
                "acquisition_error": (
                    {
                        "error_type": normalized_result.get("error_type"),
                        "error": normalized_result.get("error"),
                    }
                    if not pair_specific_available
                    else None
                ),
                "required_checks": required_checks,
                "required_check_ids": required_check_ids,
                "required_check": (
                    required_checks[0]
                    if len(required_checks) == 1
                    else None
                ),
                "required_check_id": (
                    required_check_ids[0]
                    if len(required_check_ids) == 1
                    else None
                ),
                "artifact_rendered": pair_specific_available,
                "view_coverage_complete": False,
                "observation_complete": False,
                "judge_packet": functional_relation_judge_packet(
                    global_paths=global_paths,
                    probe_result=normalized_result,
                    required_checks=required_checks,
                ),
            }
        )
    max_units = max(
        0,
        int(acquisition_audit.get("max_probe_units") or 0),
    )
    acquired_count = sum(
        bool(item.get("pair_specific_evidence_available"))
        for item in specs
    )
    if max_units and acquired_count > max_units:
        raise ValueError(
            "acquired cross-group relation episodes exceed max_probe_units"
        )
    return specs


def _relation_target_ids(value: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in value.get("target_ids") or []
                    if str(item).strip()
                ],
                *[
                    str(item)
                    for item in value.get("related_target_ids") or []
                    if str(item).strip()
                ],
            ]
        )
    )


def _merge_relation_probe_results(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge frozen duplicate probe rows for one exact relation target set."""

    if (
        current.get("route_scope") != incoming.get("route_scope")
        or current.get("kind") != incoming.get("kind")
    ):
        raise ValueError("relation probe rows disagree on routing identity")
    result = deepcopy(current)
    for key in (
        "evidence_paths",
        "discovery_ids",
        "check_ids",
        "required_observations",
        "observation_kinds",
        "relation_predicates",
        "observation_goals",
        "acquisition_triggers",
        "surface_targets",
    ):
        result[key] = _stable_unique(
            [
                *list(result.get(key) or []),
                *list(incoming.get(key) or []),
            ]
        )
    incoming_available = bool(
        incoming.get("status") == "available"
        and incoming.get("evidence_paths")
    )
    current_available = bool(
        result.get("status") == "available"
        and result.get("evidence_paths")
    )
    if incoming_available and not current_available:
        for key in (
            "probe_id",
            "target_ids",
            "related_target_ids",
            "view_goal",
            "evidence_scope",
            "owning_group_id",
        ):
            result[key] = deepcopy(incoming.get(key))
    if incoming_available or current_available:
        result["status"] = "available"
        result["error_type"] = None
        result["error"] = None
    return result


def _legacy_atomic_required_check(
    *,
    index: int,
    predicate: str,
    target_ids: list[str],
    group_ids: list[str],
    discovery_id: str,
    observation_goal: str,
    acquisition_status: str,
    artifact_rendered: bool,
) -> dict[str, Any]:
    """Build an audit-only obligation for frozen pre-ledger artifacts."""

    return {
        "check_id": f"functional_check_relation_{index:03d}",
        "check_type": predicate,
        "check_family": "cross_group_correspondence",
        "owner_stage": "cross_group_relation",
        "route_scope": "cross_group",
        "target_ids": list(target_ids),
        "group_ids": list(group_ids),
        "owning_group_id": None,
        "relation": predicate,
        "predicate": predicate,
        "required_observations": list(
            functional_relation_required_observations(predicate)
        ),
        "observation_goals": [observation_goal],
        "observation_kinds": [predicate],
        "routing_discovery_ids": [discovery_id],
        "source_discovery_ids": [discovery_id],
        "lifecycle_status": "routed",
        "acquisition_status": acquisition_status,
        "artifact_rendered": bool(artifact_rendered),
        "view_coverage_complete": False,
        "observation_complete": False,
        "judge_status": "pending",
        "decision_authority": "none",
    }


def _stable_unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _discovered_cross_group_target_sets(
    acquisition_audit: dict[str, Any],
) -> list[tuple[str, ...]]:
    discovery = acquisition_audit.get("functional_discovery")
    discovery = discovery if isinstance(discovery, dict) else {}
    return list(
        dict.fromkeys(
            tuple(
                sorted(
                    str(target_id)
                    for target_id in item.get("target_ids") or []
                    if str(target_id).strip()
                )
            )
            for item in discovery.get("cross_group_correspondences") or []
            if isinstance(item, dict)
            and len(item.get("target_ids") or []) >= 2
        )
    )


def _relation_schedule_audit(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": spec.get("relation_id"),
        "probe_id": spec.get("probe_id"),
        "discovery_ids": deepcopy(spec.get("discovery_ids") or []),
        "target_ids": deepcopy(spec.get("target_ids") or []),
        "group_ids": deepcopy(spec.get("group_ids") or []),
        "relation_predicates": deepcopy(
            spec.get("relation_predicates") or []
        ),
        "observation_kinds": deepcopy(
            spec.get("observation_kinds") or []
        ),
        "observation_goals": deepcopy(
            spec.get("observation_goals") or []
        ),
        "evidence_paths": deepcopy(spec.get("evidence_paths") or []),
        "pair_specific_evidence_available": bool(
            spec.get("pair_specific_evidence_available")
        ),
        "acquisition_status": spec.get("acquisition_status"),
        "acquisition_error": deepcopy(spec.get("acquisition_error")),
        "required_check_id": spec.get("required_check_id"),
        "required_check": deepcopy(spec.get("required_check")),
        "required_check_ids": deepcopy(
            spec.get("required_check_ids") or []
        ),
        "required_checks": deepcopy(spec.get("required_checks") or []),
        "artifact_rendered": bool(spec.get("artifact_rendered")),
        "view_coverage_complete": bool(
            spec.get("view_coverage_complete")
        ),
        "observation_complete": bool(spec.get("observation_complete")),
        "judge_episode": (
            "required"
            if spec.get("pair_specific_evidence_available") is True
            else "not_started_pair_specific_evidence_unavailable"
        ),
        "decision_authority": "none",
    }


def _evaluate_cross_group_relation_scopes(
    *,
    specs: list[dict[str, Any]],
    metric_name: str,
    scene: dict[str, Any],
    global_evidence: list[str],
    vlm_judge: Any,
    prompt: str | None,
    visual_style_spec: dict[str, Any] | None,
    authorized_deviations: list[dict[str, Any]],
    build_judge_request: Callable[..., dict[str, Any]],
    call_judge: Callable[[Any, dict[str, Any]], dict[str, Any]],
    apply_prompt_exemptions: Callable[..., dict[str, Any]],
    normalize_judgement: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for episode_index, spec in enumerate(specs, start=1):
        target_ids = list(spec.get("target_ids") or [])
        group_ids = list(spec.get("group_ids") or [])
        if spec.get("pair_specific_evidence_available") is not True:
            empty_ledger = _initial_camera_acquisition_ledger([])
            results.append(
                {
                    **_relation_schedule_audit(spec),
                    "episode_index": episode_index,
                    "evidence_phase": "cross_group_relation_review",
                    "evidence_paths": [],
                    "available_global_context_evidence_paths": list(
                        global_evidence
                    ),
                    "status": "unresolved",
                    "score": None,
                    "reason": "pair_specific_evidence_unavailable",
                    "vlm_invoked": False,
                    "judgement": {
                        "error_type": "RelationEvidenceUnavailable",
                        "error": (
                            "cross-group relation Judge was not started because "
                            "pair-specific evidence was not acquired"
                        ),
                        "acquisition_status": spec.get(
                            "acquisition_status"
                        ),
                        "acquisition_error": deepcopy(
                            spec.get("acquisition_error")
                        ),
                    },
                    "final_metric_verdict": False,
                    "camera_acquisition_episode": {
                        "scope": "cross_group_relation_judge_episode",
                        "status": "not_started",
                        "ledger_before_judge": deepcopy(empty_ledger),
                        "ledger_after_judge": deepcopy(empty_ledger),
                    },
                }
            )
            continue
        evidence_paths = list(
            dict.fromkeys(
                [
                    *global_evidence,
                    *list(spec.get("evidence_paths") or []),
                ]
            )
        )
        episode_ledger = _initial_camera_acquisition_ledger(
            evidence_paths
        )
        record: dict[str, Any] = {
            **_relation_schedule_audit(spec),
            "episode_index": episode_index,
            "evidence_phase": "cross_group_relation_review",
            "evidence_paths": evidence_paths,
            "status": "unresolved",
            "score": None,
            "reason": None,
            "vlm_invoked": True,
            "judgement": None,
            "camera_acquisition_episode": {
                "scope": "cross_group_relation_judge_episode",
                "ledger_before_judge": deepcopy(episode_ledger),
                "ledger_after_judge": deepcopy(episode_ledger),
            },
        }
        request = build_judge_request(
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            render_evidence=evidence_paths,
            selected_object_ids=target_ids,
            selected_group_ids=group_ids,
            groups=list(spec.get("groups") or []),
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            evidence_phase="cross_group_relation_review",
            decision_mode="final",
            functional_probe_evidence=deepcopy(
                spec.get("judge_packet")
            ),
        )
        request["functional_relation_scope"] = {
            "relation_id": spec.get("relation_id"),
            "target_ids": target_ids,
            "group_ids": group_ids,
            "relation_predicates": deepcopy(
                spec.get("relation_predicates") or []
            ),
            "observation_kinds": deepcopy(
                spec.get("observation_kinds") or []
            ),
            "observation_goals": deepcopy(
                spec.get("observation_goals") or []
            ),
            "required_check_ids": deepcopy(
                spec.get("required_check_ids") or []
            ),
            "required_check_id": spec.get("required_check_id"),
            "defect_target_policy": (
                "non_empty_subset_offending_objects_only"
            ),
        }
        request["camera_acquisition_ledger"] = deepcopy(
            episode_ledger
        )
        audit_records = getattr(vlm_judge, "audit_records", None)
        audit_start = (
            len(audit_records) if isinstance(audit_records, list) else None
        )
        try:
            raw = call_judge(vlm_judge, request)
            adjusted = apply_prompt_exemptions(
                raw,
                metric_name=metric_name,
                authorized_deviations=authorized_deviations,
            )
            adjusted = canonicalize_typed_invalid_envelope(adjusted)
            adjusted = canonicalize_functional_defect_check_linkage(
                adjusted,
                required_checks=deepcopy(
                    spec.get("required_checks") or []
                ),
            )
            check_resolution = validate_functional_check_results(
                adjusted,
                required_checks=deepcopy(
                    spec.get("required_checks") or []
                ),
                invalid_verdict_requires_invalid_check=True,
            )
            outcome = normalize_judgement(
                adjusted,
                metric_name=metric_name,
                valid_object_ids=set(target_ids),
            )
            violations = _relation_episode_defect_violations(
                adjusted.get("defects") or [],
                required_target_ids=target_ids,
            )
            if violations:
                raise ValueError(
                    "cross-group relation Judge returned an out-of-scope "
                    f"defect target set: {violations}"
                )
            record.update(
                status=outcome["status"],
                score=outcome["score"],
                reason=outcome["reason"],
                judgement=adjusted,
                functional_check_resolution=check_resolution,
                final_metric_verdict=(
                    outcome.get("status") == "evaluated"
                ),
            )
        except Exception as exc:
            schema_audit = response_schema_audit_from_exception(exc)
            record.update(
                status="unresolved",
                score=None,
                reason="vlm_cross_group_relation_judge_failed",
                final_metric_verdict=False,
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
            audit = deepcopy(audit_records[-1])
            record["camera_control_audit"] = audit
            next_ledger = _camera_acquisition_ledger_from_audit(
                audit
            )
            if next_ledger is not None:
                record["camera_acquisition_episode"][
                    "ledger_after_judge"
                ] = deepcopy(next_ledger)
        results.append(record)
    return results


def _forbidden_cross_group_defects(
    defects: list[Any],
    *,
    forbidden_target_sets: list[tuple[str, ...]],
) -> list[list[str]]:
    forbidden = set(forbidden_target_sets)
    return [
        list(target_ids)
        for defect in defects
        if isinstance(defect, dict)
        and (
            target_ids := tuple(
                sorted(canonical_target_ids(defect))
            )
        )
        in forbidden
    ]


def _relation_episode_defect_violations(
    defects: list[Any],
    *,
    required_target_ids: list[str],
) -> list[list[str]]:
    allowed = set(required_target_ids)
    return [
        list(target_ids)
        for defect in defects
        if isinstance(defect, dict)
        and (
            target_ids := tuple(
                sorted(canonical_target_ids(defect))
            )
        )
        and not set(target_ids) <= allowed
    ]


def _camera_acquisition_ledger_from_audit(
    record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
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


def _initial_camera_acquisition_ledger(
    paths: list[str],
) -> dict[str, Any]:
    artifact_ids = list(
        dict.fromkeys(evidence_artifact_refs(list(paths)))
    )
    return {
        "schema_version": "metric_camera_acquisition_ledger_v1",
        "artifact_ids": artifact_ids,
        "total_images_acquired": len(artifact_ids),
        "evidence_rounds": 0,
        "selector_calls": 0,
        "camera_actions": 0,
        "deterministic_rounds": 0,
        "vlm_rounds": 0,
    }
