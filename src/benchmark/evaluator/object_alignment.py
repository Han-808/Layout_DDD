"""Deterministic reference-to-generated object alignment (P0a).

This evaluator consumes a *frozen* reference annotation and the deterministic
object-mapping report. It classifies every confirmed reference object into one
of the frozen alignment states. It does not calculate prompt-fidelity scores;
P0c owns the penalty policy. It never calls a model and never resolves identity
with assets, placement, relations, or metric outcomes.
"""

from __future__ import annotations

from typing import Any

from benchmark.reference_annotation import annotation_scoring_gate, confirmed_objects


ALIGNMENT_STATES = ("resolved", "missing", "ambiguous", "low_confidence", "extra")

def evaluate_object_alignment(
    reference_annotation: dict,
    scene: dict,
    mapping_report: dict,
    *,
    config: dict | None = None,
) -> dict[str, Any]:
    _ = config
    gate = annotation_scoring_gate(reference_annotation)
    if not gate["official_scoreable"]:
        # A converter draft or a known-incomplete annotation is excluded from
        # official scoring rather than silently evaluated. This is never a
        # generator penalty.
        return {
            "evaluator_version": "object_alignment_v1",
            "official_scoreable": False,
            "status": "excluded_from_official_scoring",
            "reason": gate["reason"],
            "annotation_validation_status": reference_annotation.get("validation_status"),
            "metric_role": "alignment_only",
            "affects_benchmark_score": False,
            "score": None,
            "notes": [
                "Scoring consumes only a confirmed frozen reference annotation.",
                "Converter drafts and incomplete annotations are excluded, not scored.",
            ],
        }

    inventory_policy = str(reference_annotation.get("inventory_policy"))
    reference_objects = confirmed_objects(reference_annotation)

    slot_state = _slot_state_index(mapping_report)
    per_object: list[dict[str, Any]] = []
    state_counts = {state: 0 for state in ALIGNMENT_STATES}
    eligible_count = 0
    missing_count = 0
    for obj in reference_objects:
        object_id = str(obj.get("id"))
        count = _positive_count(obj.get("count"))
        slot_ids = _slot_ids(object_id, count)
        object_states: dict[str, int] = {state: 0 for state in ALIGNMENT_STATES if state != "extra"}
        for slot_id in slot_ids:
            state = slot_state.get(slot_id, "missing")
            object_states[state] = object_states.get(state, 0) + 1
            state_counts[state] += 1
            eligible_count += 1
            if state == "missing":
                missing_count += 1
        per_object.append(
            {
                "reference_object_id": object_id,
                "category": str(obj.get("category") or ""),
                "required_count": count,
                "states": object_states,
                "missing": object_states["missing"],
            }
        )

    extras = [
        str(item.get("generated_object_id") or "")
        for item in mapping_report.get("unmatched_generated", [])
        if isinstance(item, dict)
    ]
    extra_count = len(extras)
    state_counts["extra"] = extra_count

    resolved_count = state_counts["resolved"]
    unresolved_count = state_counts["ambiguous"] + state_counts["low_confidence"] + state_counts["missing"]

    return {
        "evaluator_version": "object_alignment_v1",
        "official_scoreable": True,
        "status": "complete" if unresolved_count == 0 else "partial",
        "reason": None,
        "annotation_validation_status": "confirmed",
        "inventory_policy": inventory_policy,
        "metric_role": "alignment_only",
        "affects_benchmark_score": False,
        "score": None,
        "state_counts": state_counts,
        "objects": per_object,
        "extras": extras,
        "presence_evidence": {
            "eligible_count": eligible_count,
            "resolved_count": resolved_count,
            "ambiguous_count": state_counts["ambiguous"],
            "low_confidence_count": state_counts["low_confidence"],
            "missing_count": missing_count,
        },
        "inventory_evidence": {
            "inventory_policy": inventory_policy,
            "extra_count": extra_count,
        },
        "coverage": {
            "eligible_count": eligible_count,
            "resolved_count": resolved_count,
            "unresolved_count": unresolved_count,
            "extra_count": extra_count,
            "coverage": (resolved_count / float(eligible_count)) if eligible_count else None,
        },
        "notes": [
            "Alignment is deterministic; no model or resolver is called.",
            "P0a reports identity and coverage only; P0c decides missing/extra fidelity penalties.",
            "Ambiguous and low-confidence identities are excluded from affected relation checks.",
        ],
    }


def _slot_state_index(mapping_report: dict) -> dict[str, str]:
    slot_state: dict[str, str] = {}
    for match in mapping_report.get("matches", []) if isinstance(mapping_report.get("matches"), list) else []:
        if not isinstance(match, dict):
            continue
        slot_id = str(match.get("plan_slot_id") or "")
        status = str(match.get("status") or "")
        if status == "matched":
            slot_state[slot_id] = "resolved"
        elif status == "ambiguous":
            slot_state[slot_id] = "ambiguous"
    for entry in mapping_report.get("low_confidence_mappings", []) if isinstance(mapping_report.get("low_confidence_mappings"), list) else []:
        if isinstance(entry, dict):
            slot_state.setdefault(str(entry.get("plan_slot_id") or ""), "low_confidence")
    missing_entries = mapping_report.get("missing_mappings")
    if not isinstance(missing_entries, list):
        missing_entries = mapping_report.get("unmatched_reference", [])
    for entry in missing_entries if isinstance(missing_entries, list) else []:
        if isinstance(entry, dict):
            slot_state.setdefault(str(entry.get("plan_slot_id") or ""), "missing")
    return slot_state


def _slot_ids(object_id: str, count: int) -> list[str]:
    if count == 1:
        return [object_id]
    return [f"{object_id}#{index + 1}" for index in range(count)]


def _positive_count(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    try:
        count = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, count)
