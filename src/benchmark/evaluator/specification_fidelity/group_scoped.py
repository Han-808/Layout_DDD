"""Group-scoped evidence decomposition for Functional Semantic Fidelity.

The caller owns claim semantics and Judge validation. This module only maps a
validated grouping partition to one camera request and one result per selected
group, then applies the existing all-valid/any-invalid aggregation rule.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from benchmark.visual_judge.group_scope import (
    GroupCameraScope,
    build_group_camera_scope,
)


def normalize_grouping_partition(
    value: Any,
    *,
    scene: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if (
            value.get("status") == "unavailable"
            and value.get("object_groups") is None
        ):
            return None
        groups = (
            value.get("object_groups")
            if value.get("object_groups") is not None
            else value.get("groups")
        )
    else:
        groups = value
    if not isinstance(groups, list):
        raise ValueError(
            "object_grouping_report must contain an object_groups list"
        )
    scene_ids = {
        str(item.get("id"))
        for item in scene.get("objects") or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    assigned: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(
                f"object_groups[{index}] must be a JSON object"
            )
        group_id = str(group.get("group_id") or "").strip()
        members = group.get("object_ids")
        if (
            not group_id
            or not isinstance(members, list)
            or not members
        ):
            raise ValueError(
                "every grouping record requires group_id and object_ids"
            )
        member_ids = [str(item) for item in members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError(
                f"group {group_id!r} contains duplicate object IDs"
            )
        unknown = sorted(set(member_ids) - scene_ids)
        overlap = sorted(set(member_ids) & assigned)
        if unknown:
            raise ValueError(
                f"group {group_id!r} references unknown IDs {unknown}"
            )
        if overlap:
            raise ValueError(
                f"grouping assigns object IDs more than once: {overlap}"
            )
        assigned.update(member_ids)
        normalized.append(
            {
                **deepcopy(group),
                "group_id": group_id,
                "object_ids": member_ids,
            }
        )
    missing = sorted(scene_ids - assigned)
    if missing:
        raise ValueError(
            "object_grouping_report must assign every scene object exactly "
            f"once; missing {missing}"
        )
    return normalized


def request_group_local_evidence(
    *,
    claim: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    trigger: str,
    camera_evidence_provider: Any,
    counters: dict[str, int],
    grouping_groups: list[dict[str, Any]] | None,
    grouping_report: dict[str, Any] | None,
    include_global_context: bool,
    metric: str,
    claim_target_ids: Callable[[dict[str, Any]], list[str]],
    request_local_evidence: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    target_ids = claim_target_ids(claim)
    missing_target_reason = {
        "local_functionality": (
            "prompt_scoped_local_functionality_targets_missing"
        ),
        "required_functional_areas": (
            "claim_scoped_required_area_targets_missing"
        ),
    }.get(str(claim.get("component") or ""))
    if missing_target_reason and not target_ids:
        return _unresolved(missing_target_reason)
    if grouping_groups is None:
        return _unresolved("object_grouping_unavailable")

    claim_targets = set(target_ids)
    selected = [
        group
        for group in grouping_groups
        if claim_targets & set(group.get("object_ids") or [])
    ]
    if not selected:
        return _unresolved("claim_targets_not_mapped_to_group")

    packets: list[dict[str, Any]] = []
    for group in selected:
        try:
            scope = build_group_camera_scope(
                scene,
                group,
                metric=metric,
                include_global_context=include_global_context,
                grouping_report=grouping_report,
            )
        except Exception as exc:
            return {
                "status": "evidence_provider_failed",
                "route": "evidence_provider_failed",
                "reason": "group_camera_scope_invalid",
                "error": f"{type(exc).__name__}: {exc}",
                "paths": [],
                "group_packets": packets,
            }
        result = request_local_evidence(
            claim=claim,
            scene=scene,
            prompt=prompt,
            trigger=trigger,
            camera_evidence_provider=camera_evidence_provider,
            counters=counters,
            group_scope=scope,
        )
        if result["status"] != "available":
            return {**result, "group_packets": packets}
        packets.append(
            {
                "group_id": scope.group_id,
                "member_ids": list(scope.member_ids),
                "group_scope": scope,
                "paths": list(result["paths"]),
            }
        )
    return {
        "status": "available",
        "route": "group_scoped_local_evidence",
        "reason": None,
        "paths": list(
            dict.fromkeys(
                path
                for packet in packets
                for path in packet["paths"]
            )
        ),
        "group_packets": packets,
    }


def aggregate_group_judgements(
    *,
    base: dict[str, Any],
    packets: list[dict[str, Any]],
    judge_group: Callable[
        [dict[str, Any], GroupCameraScope],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    group_results = [
        judge_group(packet, packet["group_scope"])
        for packet in packets
    ]
    checked = [
        result
        for result in group_results
        if result.get("status") == "checked"
    ]
    invalid = [
        result
        for result in checked
        if result.get("verdict") == "invalid"
    ]
    aggregate = {
        **base,
        "group_results": group_results,
        "group_judge_call_count": len(group_results),
    }
    if invalid:
        return {
            **aggregate,
            "status": "checked",
            "route": "vlm_adjudicated_per_group",
            "reason": invalid[0].get("reason"),
            "verdict": "invalid",
            "score": 0.0,
            "confidence": min(
                float(item.get("confidence") or 0.0)
                for item in invalid
            ),
        }
    if group_results and len(checked) == len(group_results):
        return {
            **aggregate,
            "status": "checked",
            "route": "vlm_adjudicated_per_group",
            "reason": "all_group_scoped_claim_checks_valid",
            "verdict": "valid",
            "score": 1.0,
            "confidence": min(
                float(item.get("confidence") or 0.0)
                for item in checked
            ),
        }
    return {
        **aggregate,
        "status": "requires_vlm",
        "route": "unresolved",
        "reason": "one_or_more_group_scoped_claim_checks_unresolved",
        "verdict": None,
        "score": None,
    }


def _unresolved(reason: str) -> dict[str, Any]:
    return {
        "status": "requires_vlm",
        "route": "unresolved",
        "reason": reason,
        "paths": [],
        "group_packets": [],
    }
