"""Optional bounded active-evidence windows for controlled Judge calls.

The controller's acquisition ledger remains cumulative.  This module only
controls which already acquired artifacts are present in the current Judge
context.  Eviction therefore never deletes an artifact or erases provenance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from benchmark.visual_judge.orchestration.audit import (
    evidence_artifact_refs,
)


SHARED_GROUP_BANK_POLICY = "shared_group_bank"
EVIDENCE_WINDOW_SCHEMA_VERSION = "bounded_evidence_window_v1"


@dataclass(frozen=True)
class BoundedEvidenceWindow:
    policy: str
    max_active_images: int
    fixed_artifact_ids: tuple[str, ...]
    reusable_artifacts: tuple[dict[str, Any], ...]
    group_id: str
    check_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_WINDOW_SCHEMA_VERSION,
            "policy": self.policy,
            "max_active_images": self.max_active_images,
            "fixed_artifact_ids": list(self.fixed_artifact_ids),
            "reusable_artifacts": list(deepcopy(self.reusable_artifacts)),
            "group_id": self.group_id,
            "check_id": self.check_id,
        }


def evidence_artifact_id(item: Any) -> str:
    refs = evidence_artifact_refs([item])
    if len(refs) != 1 or not str(refs[0]).strip():
        raise ValueError("visual evidence requires one stable artifact identity")
    return str(refs[0])


def resolve_bounded_evidence_window(
    context: dict[str, Any],
    *,
    initial_evidence: list[Any],
) -> BoundedEvidenceWindow | None:
    raw = context.get("functional_group_evidence_window")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "functional_group_evidence_window must be a JSON object"
        )
    policy = str(raw.get("policy") or "").strip()
    if policy != SHARED_GROUP_BANK_POLICY:
        raise ValueError(
            "functional_group_evidence_window.policy must be exactly "
            f"'{SHARED_GROUP_BANK_POLICY}'"
        )
    maximum = raw.get("max_active_images")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise ValueError(
            "functional group active evidence window must contain at least "
            "two images"
        )
    fixed = _unique_strings(raw.get("fixed_artifact_ids"))
    if len(fixed) != 2:
        raise ValueError(
            "shared Functional group evidence requires exactly two fixed "
            "artifacts"
        )
    initial_ids = [evidence_artifact_id(item) for item in initial_evidence]
    if len(initial_ids) != len(set(initial_ids)):
        raise ValueError("initial evidence window contains duplicate artifacts")
    missing_fixed = [item for item in fixed if item not in set(initial_ids)]
    if missing_fixed:
        raise ValueError(
            "initial evidence window is missing fixed artifacts: "
            + ", ".join(missing_fixed)
        )
    if len(initial_ids) > maximum:
        raise ValueError(
            "initial evidence window exceeds max_active_images"
        )
    reusable = raw.get("reusable_artifacts") or []
    if not isinstance(reusable, list):
        raise ValueError(
            "functional group reusable_artifacts must be a list"
        )
    normalized: list[dict[str, Any]] = []
    reusable_ids: set[str] = set()
    for value in reusable:
        if not isinstance(value, dict):
            raise ValueError("reusable evidence records must be JSON objects")
        artifact = deepcopy(value.get("visual_evidence"))
        artifact_id = str(value.get("artifact_id") or "").strip()
        if not artifact_id:
            artifact_id = evidence_artifact_id(artifact)
        if artifact_id != evidence_artifact_id(artifact):
            raise ValueError(
                "reusable evidence artifact_id does not match its payload"
            )
        if artifact_id in reusable_ids:
            raise ValueError(
                "functional group reusable evidence IDs must be unique"
            )
        reusable_ids.add(artifact_id)
        normalized.append(
            {
                **deepcopy(value),
                "artifact_id": artifact_id,
                "visual_evidence": artifact,
                "check_ids": _unique_strings(value.get("check_ids")),
                "target_ids": _unique_strings(value.get("target_ids")),
                "required_observations": _unique_strings(
                    value.get("required_observations")
                ),
            }
        )
    return BoundedEvidenceWindow(
        policy=policy,
        max_active_images=maximum,
        fixed_artifact_ids=tuple(fixed),
        reusable_artifacts=tuple(normalized),
        group_id=str(raw.get("group_id") or "").strip(),
        check_id=str(raw.get("check_id") or "").strip(),
    )


def select_reusable_evidence(
    window: BoundedEvidenceWindow,
    *,
    active_evidence: list[Any],
    target_ids: tuple[str, ...] | list[str],
    missing_observations: tuple[str, ...] | list[str],
    excluded_artifact_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select relevant unused bank artifacts before a camera call.

    A generic observation match alone is deliberately insufficient.  Reuse
    requires either an explicit check binding or an overlapping target so a
    broad group bank cannot contaminate an unrelated atomic check.
    """

    active_ids = {evidence_artifact_id(item) for item in active_evidence}
    excluded_ids = set(excluded_artifact_ids or ())
    targets = set(_unique_strings(target_ids))
    observations = set(_unique_strings(missing_observations))
    ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for sequence, record in enumerate(window.reusable_artifacts):
        artifact_id = str(record["artifact_id"])
        if (
            artifact_id in active_ids
            or artifact_id in excluded_ids
            or artifact_id in window.fixed_artifact_ids
        ):
            continue
        check_match = bool(
            window.check_id
            and window.check_id in set(record.get("check_ids") or [])
        )
        target_overlap = len(targets & set(record.get("target_ids") or []))
        observation_overlap = len(
            observations & set(record.get("required_observations") or [])
        )
        if not check_match and target_overlap == 0:
            continue
        ranked.append(
            (
                (
                    1 if check_match else 0,
                    target_overlap,
                    observation_overlap,
                    -sequence,
                ),
                deepcopy(record),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    dynamic_capacity = (
        window.max_active_images - len(window.fixed_artifact_ids)
    )
    if len(active_evidence) < window.max_active_images:
        capacity = window.max_active_images - len(active_evidence)
    else:
        capacity = dynamic_capacity
    return [record for _, record in ranked[: max(0, capacity)]]


def compose_bounded_evidence_window(
    window: BoundedEvidenceWindow,
    *,
    previous: list[Any],
    additions: list[Any],
    trigger: str,
) -> tuple[list[Any], dict[str, Any]]:
    """Append evidence or flush non-fixed evidence when the window is full."""

    previous_items = _deduplicate_evidence(previous)
    fixed_set = set(window.fixed_artifact_ids)
    previous_by_id = {
        evidence_artifact_id(item): deepcopy(item) for item in previous_items
    }
    missing_fixed = [
        artifact_id
        for artifact_id in window.fixed_artifact_ids
        if artifact_id not in previous_by_id
    ]
    if missing_fixed:
        raise ValueError(
            "bounded evidence composition cannot evict fixed artifacts: "
            + ", ".join(missing_fixed)
        )

    unique_additions = _deduplicate_evidence(additions)
    added_ids: list[str] = []
    combined = list(previous_items)
    slots = {
        evidence_artifact_id(item): index
        for index, item in enumerate(combined)
    }
    for item in unique_additions:
        artifact_id = evidence_artifact_id(item)
        if artifact_id in fixed_set:
            continue
        if artifact_id in slots:
            combined[slots[artifact_id]] = deepcopy(item)
            continue
        slots[artifact_id] = len(combined)
        combined.append(deepcopy(item))
        added_ids.append(artifact_id)

    evicted_ids: list[str] = []
    overflow = len(combined) > window.max_active_images
    if overflow:
        retained = [
            deepcopy(previous_by_id[artifact_id])
            for artifact_id in window.fixed_artifact_ids
        ]
        evicted_ids = [
            evidence_artifact_id(item)
            for item in previous_items
            if evidence_artifact_id(item) not in fixed_set
        ]
        capacity = window.max_active_images - len(retained)
        new_dynamic = [
            deepcopy(item)
            for item in unique_additions
            if evidence_artifact_id(item) not in fixed_set
        ]
        combined = [*retained, *new_dynamic[:capacity]]
        added_ids = [evidence_artifact_id(item) for item in new_dynamic[:capacity]]

    after_ids = [evidence_artifact_id(item) for item in combined]
    if len(after_ids) > window.max_active_images:
        raise ValueError("bounded evidence window overflowed after composition")
    if any(item not in after_ids for item in window.fixed_artifact_ids):
        raise ValueError("bounded evidence window lost a fixed artifact")
    event = {
        "schema_version": EVIDENCE_WINDOW_SCHEMA_VERSION,
        "policy": window.policy,
        "group_id": window.group_id,
        "check_id": window.check_id,
        "trigger": str(trigger),
        "max_active_images": window.max_active_images,
        "fixed_artifact_ids": list(window.fixed_artifact_ids),
        "before_artifact_ids": [
            evidence_artifact_id(item) for item in previous_items
        ],
        "added_artifact_ids": added_ids,
        "evicted_artifact_ids": evicted_ids,
        "after_artifact_ids": after_ids,
        "overflow_flush_applied": overflow,
        "physical_artifacts_deleted": False,
    }
    return combined, event


def _deduplicate_evidence(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    slots: dict[str, int] = {}
    for item in items:
        artifact_id = evidence_artifact_id(item)
        if artifact_id in slots:
            result[slots[artifact_id]] = deepcopy(item)
            continue
        slots[artifact_id] = len(result)
        result.append(deepcopy(item))
    return result


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )
