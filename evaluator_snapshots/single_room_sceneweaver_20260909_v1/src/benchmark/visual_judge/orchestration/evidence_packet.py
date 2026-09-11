from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from benchmark.visual_judge.interfaces.evidence import (
    EvidenceGateResult,
    EvidenceRenderResult,
)
from benchmark.visual_judge.interfaces.judge import (
    EvidenceRequest,
    JudgeRequest,
)
from benchmark.visual_judge.orchestration.audit import (
    evidence_content_identity,
)


def merge_evidence(
    previous: list[Any],
    rendered: EvidenceRenderResult,
    *,
    preserve_global_anchor: bool = False,
) -> list[Any]:
    """Compose a packet without discarding required global evidence."""

    additions = list(deepcopy(rendered.visual_evidence))
    if rendered.merge_policy == "replace" and not preserve_global_anchor:
        return additions
    combined = list(deepcopy(previous)) + additions
    result: list[Any] = []
    slots: dict[str, int] = {}
    seen: set[str] = set()
    for item in combined:
        slot = _evidence_slot_identity(item)
        if slot is not None and slot in slots:
            # A corrective render of the same view/representation supersedes
            # its deficient predecessor without changing packet ordering.
            replacement_index = slots[slot]
            previous_key = _content_key(result[replacement_index])
            result[replacement_index] = item
            seen.discard(previous_key)
            seen.add(_content_key(item))
            continue
        key = _content_key(item)
        if key in seen:
            continue
        if slot is not None:
            slots[slot] = len(result)
        result.append(item)
        seen.add(key)
    return result


def raw_render_evidence(value: Any) -> list[Any]:
    if isinstance(value, EvidenceRenderResult):
        return list(deepcopy(value.visual_evidence))
    if isinstance(value, (list, tuple)):
        return list(deepcopy(value))
    if not isinstance(value, dict):
        return []
    items = value.get("visual_evidence")
    if items is None:
        items = value.get("render_evidence")
    if not isinstance(items, (list, tuple)):
        return []
    return list(deepcopy(items))


def goal_from_judge_request(
    existing: dict[str, Any],
    request: EvidenceRequest,
) -> dict[str, Any]:
    result = deepcopy(existing)
    result.update(
        {
            "target_ids": list(request.target_ids),
            "missing_observations": list(request.missing_observations),
            "view_goal": request.view_goal,
            "judge_evidence_request": request.to_dict(),
        }
    )
    return result


def request_target_ids(request: JudgeRequest) -> tuple[str, ...]:
    values: list[Any] = []
    for source in (request.claim_or_event, request.context):
        for key in (
            "target_ids",
            "target_object_ids",
            "object_ids",
            "subject_ids",
            "member_ids",
        ):
            if isinstance(source.get(key), list):
                values.extend(source[key])
        for key in (
            "object_id",
            "subject_id",
            "anchor_id",
            "target_id",
        ):
            if source.get(key) is not None:
                values.append(source[key])
    resolved = tuple(
        dict.fromkeys(
            str(value) for value in values if str(value).strip()
        )
    )
    return resolved or ("scene",)


def gate_reason(result: EvidenceGateResult) -> str:
    if result.reason_codes:
        return "visual evidence is not technically ready: " + ", ".join(
            result.reason_codes
        )
    return "visual evidence is not technically ready"


def gate_stop_reason(result: EvidenceGateResult) -> str:
    codes = set(result.reason_codes)
    if any(code.startswith("evidence_manifest_") for code in codes):
        return "manifest_failure"
    if "corrupt_render_evidence" in codes:
        return "corrupt_evidence"
    if "blank_render" in codes:
        return "blank_evidence"
    if "undecodable_render" in codes:
        return "undecodable_evidence"
    if codes & {
        "visual_evidence_missing",
        "evidence_path_missing",
        "evidence_file_missing",
        "empty_render_file",
        "evidence_file_unreadable",
    }:
        return "evidence_missing"
    return "evidence_integrity_failure"


def validate_candidates(
    candidates: tuple[dict[str, Any], ...],
) -> None:
    ids = [str(item.get("id") or "") for item in candidates]
    if any(not value for value in ids):
        raise ValueError("candidate views require non-empty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate view IDs must be unique")


def _evidence_slot_identity(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    pose = item.get("pose")
    view_id = (
        item.get("view_id")
        or item.get("id")
        or (
            pose.get("id")
            if isinstance(pose, dict)
            else None
        )
    )
    if view_id is None or not str(view_id).strip():
        return None
    representation = (
        item.get("representation")
        or item.get("representation_type")
        or item.get("evidence_style")
        or item.get("role")
        or "default"
    )
    return f"{view_id}:{representation}"


def _content_key(item: Any) -> str:
    return json.dumps(
        evidence_content_identity(item),
        sort_keys=True,
        separators=(",", ":"),
    )
