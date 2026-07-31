"""Per-group evidence routing and scene-quality result aggregation.

This module keeps group-loop orchestration out of the metric contract module.
Metric rubrics, response validation, and provider adapters remain injected by
the caller so this layer cannot redefine benchmark semantics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.visual_judge.group_scope import (
    GroupCameraScope,
    build_group_camera_scope,
)


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
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
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
        paths, resolution = resolve_metric_evidence(
            group_value,
            metric_name=metric_name,
            policy=policy,
            scene=scene,
            prompt=prompt,
            selected_object_ids=list(scope.member_ids),
            selected_group_ids=[group_id],
            selected_groups=[group],
            camera_evidence_provider=camera_evidence_provider,
            group_scope=scope,
        )
        packets.append(
            {
                "group": deepcopy(group),
                "group_scope": scope,
                "paths": paths,
                "resolution": resolution,
            }
        )
    return packets


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
) -> dict[str, Any]:
    """Judge each local packet and aggregate without score averaging."""

    group_results: list[dict[str, Any]] = []
    for packet in packets:
        group = packet["group"]
        group_id = str(group["group_id"])
        members = [str(item) for item in group["object_ids"]]
        resolution = packet["resolution"]
        record: dict[str, Any] = {
            "group_id": group_id,
            "member_ids": members,
            "group_scope": packet["group_scope"].to_dict(),
            "evidence_paths": list(packet["paths"]),
            "evidence_resolution": deepcopy(resolution),
            "status": "unresolved",
            "score": None,
            "reason": resolution.get("provider_reason"),
            "vlm_invoked": False,
            "judgement": None,
        }
        if not resolution.get("scope_satisfied") or not packet["paths"]:
            record["reason"] = (
                resolution.get("provider_reason")
                or "group_local_render_evidence_unavailable"
            )
            group_results.append(record)
            continue

        request = build_judge_request(
            metric_name=metric_name,
            scene=scene,
            prompt=prompt,
            render_evidence=packet["paths"],
            selected_object_ids=members,
            selected_group_ids=[group_id],
            groups=[group],
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            group_scope=packet["group_scope"],
            evidence_phase=evidence_phase,
            decision_mode=decision_mode,
        )
        record["vlm_invoked"] = True
        try:
            raw = call_judge(vlm_judge, request)
            adjusted = apply_prompt_exemptions(
                raw,
                metric_name=metric_name,
                authorized_deviations=authorized_deviations,
            )
            outcome = normalize_judgement(
                adjusted,
                metric_name=metric_name,
                # A group-local Judge may only report defects on members of
                # this immutable evidence scope.  Scene-wide validation would
                # allow a defect from another group to leak into this result.
                valid_object_ids=set(members),
            )
            record.update(
                status=outcome["status"],
                score=outcome["score"],
                reason=outcome["reason"],
                judgement=adjusted,
            )
        except Exception as exc:
            record.update(
                status="unresolved",
                reason="vlm_judge_failed",
                judgement={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        group_results.append(record)

    return _aggregate_group_results(base, group_results)


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
    }


def _aggregate_group_results(
    base: dict[str, Any],
    group_results: list[dict[str, Any]],
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
        1 for item in group_results if item["vlm_invoked"]
    )
    base["vlm_invoked"] = bool(base["judge_call_count"])
    base["evidence_request"]["vlm_invoked"] = base["vlm_invoked"]
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

    defects = [
        deepcopy(defect)
        for item in evaluated
        for defect in (item.get("judgement") or {}).get("defects") or []
        if isinstance(defect, dict)
    ]
    if invalid:
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
        aggregate.update(
            aggregation="invalid_if_any_group_invalid",
            group_judgements=deepcopy(group_results),
        )
        base.update(
            status="evaluated",
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
        aggregate.update(
            aggregation="all_groups_must_resolve_valid",
            group_judgements=deepcopy(group_results),
        )
        base.update(
            status="evaluated",
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
    base.update(
        status="unresolved",
        reason=(
            group_results[0]["reason"]
            if len(group_results) == 1
            and not group_results[0]["vlm_invoked"]
            else "one_or_more_group_judgements_unresolved"
        ),
        score=None,
        judgement={
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.0,
            "reason": (
                "One or more group-scoped judgements remain unresolved."
            ),
            "missing_evidence": [
                f"group_scoped_evidence:{group_id}"
                for group_id in missing_groups
            ],
            "defects": [],
            "aggregation": (
                "unresolved_without_complete_group_coverage"
            ),
            "group_judgements": deepcopy(group_results),
        },
    )
    return base


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
