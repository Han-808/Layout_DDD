"""Read-only causal ownership projected from final Functional decisions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable


FUNCTIONAL_OWNERSHIP_LEDGER_VERSION = "functional_ownership_ledger_v2"
CROSS_METRIC_OWNERSHIP_AUDIT_VERSION = "cross_metric_ownership_audit_v2"


def build_functional_ownership_ledger(
    *,
    scene_object_ids: Iterable[str],
    global_record: dict[str, Any] | None,
    relation_results: list[dict[str, Any]],
    group_results: list[dict[str, Any]],
    functional_check_ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project final Functional defects into stable non-decisional events."""

    known_ids = _trusted_ids(scene_object_ids)
    checks_by_phase: dict[str, list[dict[str, Any]]] = {}
    for check in (functional_check_ledger or {}).get("checks") or []:
        if not isinstance(check, dict):
            raise TypeError("functional ownership checks must be objects")
        phase = _ownership_phase_ref(check)
        if not phase or check.get("check_conclusion") != "invalid":
            continue
        checks_by_phase.setdefault(phase, []).append(check)

    sources: list[tuple[str, dict[str, Any]]] = []
    if isinstance(global_record, dict) and _invalid_judgement(global_record):
        sources.append(("global_discovery", global_record))
    for record in relation_results:
        if not isinstance(record, dict) or record.get("score") != 0.0:
            continue
        judgement = record.get("judgement")
        if isinstance(judgement, dict):
            sources.append(
                (
                    "cross_group_relation_review:"
                    f"{record.get('relation_id')}",
                    judgement,
                )
            )
    for record in group_results:
        if not isinstance(record, dict) or record.get("score") != 0.0:
            continue
        judgement = record.get("judgement")
        if isinstance(judgement, dict):
            sources.append(
                (
                    f"group_local_review:{record.get('group_id')}",
                    judgement,
                )
            )

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    used_explicit_check_refs: dict[str, set[str]] = {}
    for phase, judgement in sources:
        defects = judgement.get("defects")
        if not isinstance(defects, list):
            raise ValueError(
                f"functional ownership source {phase!r} has invalid defects"
            )
        phase_checks = checks_by_phase.get(phase, [])
        phase_checks_by_id = {
            str(check.get("check_id") or ""): check
            for check in phase_checks
            if check.get("check_id")
        }
        for defect_index, defect in enumerate(defects):
            if not isinstance(defect, dict):
                raise TypeError("functional ownership defect must be an object")
            defect_targets = _validated_event_ids(
                defect.get("target_ids"),
                known_ids=known_ids,
                label="functional defect target_ids",
            )
            explicit_refs = defect.get("check_refs")
            if explicit_refs is not None:
                check_refs = _validated_check_refs(
                    explicit_refs,
                    phase=phase,
                    checks_by_id=phase_checks_by_id,
                )
                already_used = used_explicit_check_refs.setdefault(
                    phase,
                    set(),
                )
                duplicate_refs = sorted(set(check_refs) & already_used)
                if duplicate_refs:
                    raise ValueError(
                        "functional ownership check references are attached "
                        f"to multiple defects: {duplicate_refs}"
                    )
                already_used.update(check_refs)
                related_checks = [
                    phase_checks_by_id[check_ref]
                    for check_ref in check_refs
                ]
            else:
                # Compatibility path for reports produced before explicit
                # defect-to-check linkage.  New Judge responses are validated
                # against check_refs before reaching this projection.
                related_checks = [
                    check
                    for check in phase_checks
                    if _check_scoring_targets(check) == defect_targets
                ]
            if any(
                not _check_matches_defect_targets(
                    check,
                    defect_targets=defect_targets,
                )
                for check in related_checks
            ):
                raise ValueError(
                    "functional ownership defect targets do not satisfy "
                    "their explicit check references"
                )
            clearance_checks = [
                check
                for check in related_checks
                if check.get("check_type") == "clearance"
            ]
            if clearance_checks:
                causal = _clearance_ownership(
                    sorted(
                        clearance_checks,
                        key=lambda item: str(item.get("check_id") or ""),
                    )[0],
                    known_ids=known_ids,
                )
            else:
                causal = {
                    "affected_object_ids": list(defect_targets),
                    "cause_kind": "self_layout",
                    "causal_object_ids": list(defect_targets),
                    "scoring_target_ids": list(defect_targets),
                    "attribution_basis": "final_defect_targets",
                }
            check_refs = sorted(
                {
                    str(check.get("check_id"))
                    for check in related_checks
                    if check.get("check_id")
                }
            )
            related_object_ids = {
                str(object_id)
                for check in related_checks
                for object_id in check.get("target_ids") or []
            }
            related_object_ids.update(causal["affected_object_ids"])
            related_object_ids.update(causal["causal_object_ids"])
            counterpart_object_ids = sorted(
                related_object_ids - set(causal["scoring_target_ids"])
            )
            decision_ref = _decision_ref(phase, judgement)
            identity = {
                "phase": phase,
                "defect_index": defect_index,
                "scope": defect.get("scope"),
                "relation": defect.get("relation"),
                "scoring_target_ids": causal["scoring_target_ids"],
                "check_refs": check_refs,
                "decision_ref": decision_ref,
            }
            event_id = _stable_id("functional_event", identity)
            if event_id in seen_ids:
                raise ValueError(
                    "functional ownership generated a duplicate event"
                )
            seen_ids.add(event_id)
            events.append(
                {
                    "event_id": event_id,
                    "metric": "functional_consistency",
                    "owning_metric": "functional_consistency",
                    "source_phase": phase,
                    "scope": str(defect.get("scope") or ""),
                    "relation": str(defect.get("relation") or ""),
                    "reason": str(defect.get("reason") or ""),
                    **causal,
                    "counterpart_object_ids": counterpart_object_ids,
                    "check_refs": check_refs,
                    "decision_ref": decision_ref,
                    "defect_ref": _stable_id(
                        "functional_defect",
                        {
                            "phase": phase,
                            "index": defect_index,
                            "defect": defect,
                        },
                    ),
                    "lifecycle_status": "final",
                    "decision_authority": "none",
                }
            )
    events.sort(key=lambda item: str(item["event_id"]))
    return {
        "schema_version": FUNCTIONAL_OWNERSHIP_LEDGER_VERSION,
        "source_metric": "functional_consistency",
        "events": events,
        "event_count": len(events),
        "decision_authority": "none",
        "projection_mode": "posthoc_read_only",
    }


def build_cross_metric_ownership_audit(
    *,
    functional_ownership_ledger: dict[str, Any] | None,
    placement_check_ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    """Audit exact Function-owned deduplication without object-overlap guesses."""

    events = {
        str(item.get("event_id")): item
        for item in (functional_ownership_ledger or {}).get("events") or []
        if isinstance(item, dict) and item.get("event_id")
    }
    exclusions: list[dict[str, Any]] = []
    independent_invalids: list[str] = []
    for check in (placement_check_ledger or {}).get("checks") or []:
        if not isinstance(check, dict):
            raise TypeError("placement ownership audit check must be an object")
        check_id = str(check.get("check_id") or "").strip()
        conclusion = str(check.get("check_conclusion") or "")
        if conclusion == "invalid":
            independent_invalids.append(check_id)
            continue
        if conclusion != "excluded_function_owned":
            continue
        event_ref = str(check.get("function_event_ref") or "").strip()
        if event_ref not in events:
            raise ValueError(
                f"placement check {check_id!r} references an unknown "
                "functional ownership event"
            )
        result_row = check.get("result_row")
        same_physical_event = bool(
            check.get("same_physical_event") is True
            or (
                isinstance(result_row, dict)
                and result_row.get("same_physical_event") is True
            )
        )
        if not same_physical_event:
            raise ValueError(
                f"placement check {check_id!r} lacks explicit same-event "
                "confirmation"
            )
        exclusions.append(
            {
                "placement_check_id": check_id,
                "function_event_ref": event_ref,
                "subject_id": str(check.get("subject_id") or ""),
                "same_physical_event": True,
                "deduplication_basis": "explicit_stable_event_reference",
                "decision_authority": "none",
            }
        )
    return {
        "schema_version": CROSS_METRIC_OWNERSHIP_AUDIT_VERSION,
        "functional_event_count": len(events),
        "excluded_placement_checks": sorted(
            exclusions,
            key=lambda item: item["placement_check_id"],
        ),
        "independent_invalid_placement_check_ids": sorted(
            independent_invalids
        ),
        "deduplication_key": (
            "explicit_function_event_ref_and_same_physical_event"
        ),
        "runtime_cross_metric_suppression": bool(exclusions),
        "object_identity_alone_suppresses_placement": False,
        "placement_claims_judged_before_deduplication": True,
        "decision_authority": "none",
    }


def validate_functional_ownership_ledger(
    value: Any,
    *,
    known_object_ids: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("functional ownership ledger must be an object")
    if value.get("schema_version") != FUNCTIONAL_OWNERSHIP_LEDGER_VERSION:
        raise ValueError("unsupported functional ownership ledger version")
    if value.get("source_metric") != "functional_consistency":
        raise ValueError(
            "functional ownership ledger has the wrong source metric"
        )
    if value.get("decision_authority") != "none":
        raise ValueError(
            "functional ownership ledger cannot have decision authority"
        )
    if value.get("projection_mode") != "posthoc_read_only":
        raise ValueError(
            "functional ownership ledger must remain a read-only projection"
        )
    known = _trusted_ids(known_object_ids)
    events = value.get("events")
    if not isinstance(events, list):
        raise ValueError("functional ownership events must be a list")
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("functional ownership event must be an object")
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in seen:
            raise ValueError(
                "functional ownership event IDs must be unique and non-empty"
            )
        seen.add(event_id)
        for field in (
            "affected_object_ids",
            "causal_object_ids",
            "scoring_target_ids",
            "counterpart_object_ids",
        ):
            values = event.get(field)
            if field == "counterpart_object_ids" and values == []:
                continue
            _validated_event_ids(values, known_ids=known, label=f"functional ownership {field}")
        if event.get("owning_metric") != "functional_consistency":
            raise ValueError("functional ownership event has wrong owning_metric")
        if event.get("cause_kind") not in {
            "external_object",
            "self_layout",
        }:
            raise ValueError("functional ownership event has invalid cause_kind")
        if event.get("lifecycle_status") != "final":
            raise ValueError(
                "functional ownership events must be final before reuse"
            )
        if event.get("decision_authority") != "none":
            raise ValueError(
                "functional ownership events cannot have decision authority"
            )
        if not str(event.get("decision_ref") or "").strip():
            raise ValueError(
                "functional ownership event requires a decision reference"
            )
    if int(value.get("event_count") or 0) != len(events):
        raise ValueError("functional ownership event_count is inconsistent")
    return deepcopy(value)


def _clearance_ownership(
    check: dict[str, Any],
    *,
    known_ids: set[str],
) -> dict[str, Any]:
    result_row = check.get("result_row")
    if not isinstance(result_row, dict):
        raise ValueError(
            "invalid clearance check requires its validated result row"
        )
    affected = _validated_event_ids(
        result_row.get("affected_object_ids"),
        known_ids=known_ids,
        label="clearance affected_object_ids",
    )
    causal = _validated_event_ids(
        result_row.get("causal_object_ids"),
        known_ids=known_ids,
        label="clearance causal_object_ids",
    )
    scoring = _validated_event_ids(
        result_row.get("scoring_target_ids"),
        known_ids=known_ids,
        label="clearance scoring_target_ids",
    )
    cause_kind = str(result_row.get("cause_kind") or "")
    if cause_kind not in {"external_object", "self_layout"}:
        raise ValueError("clearance ownership has invalid cause_kind")
    return {
        "affected_object_ids": affected,
        "cause_kind": cause_kind,
        "causal_object_ids": causal,
        "scoring_target_ids": scoring,
        "attribution_basis": "validated_clearance_check_result",
    }


def _validated_check_refs(
    value: Any,
    *,
    phase: str,
    checks_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item.strip()
            for item in value
        )
    ):
        raise ValueError(
            "functional ownership defect check_refs must contain check IDs"
        )
    refs = sorted(str(item).strip() for item in value)
    if len(refs) != len(set(refs)):
        raise ValueError(
            "functional ownership defect check_refs contain duplicates"
        )
    unknown = sorted(set(refs) - set(checks_by_id))
    if unknown:
        raise ValueError(
            f"functional ownership phase {phase!r} references checks not "
            f"resolved invalid in that phase: {unknown}"
        )
    return refs


def _ownership_phase_ref(check: dict[str, Any]) -> str:
    """Map an atomic Judge episode back to its aggregate ownership phase."""

    result_ref = str(check.get("judge_result_ref") or "").strip()
    marker = ":check:"
    if (
        not result_ref.startswith("group_local_review:")
        or marker not in result_ref
    ):
        return result_ref
    phase, separator, episode_check_id = result_ref.partition(marker)
    check_id = str(check.get("check_id") or "").strip()
    if (
        not separator
        or not phase.removeprefix("group_local_review:")
        or not check_id
        or episode_check_id != check_id
    ):
        raise ValueError(
            "functional ownership has a malformed per-check Judge result "
            f"reference: {result_ref!r}"
        )
    return phase


def _check_scoring_targets(check: dict[str, Any]) -> list[str]:
    values = (
        check.get("scoring_target_ids")
        if check.get("check_type") == "clearance"
        and check.get("check_conclusion") == "invalid"
        else check.get("target_ids")
    )
    return sorted(str(item) for item in values or [])


def _check_matches_defect_targets(
    check: dict[str, Any],
    *,
    defect_targets: list[str],
) -> bool:
    expected = set(_check_scoring_targets(check))
    actual = set(defect_targets)
    if not actual:
        return False
    if check.get("check_type") == "clearance":
        return actual == expected
    return actual <= expected


def _invalid_judgement(value: dict[str, Any]) -> bool:
    return value.get("verdict") == "invalid" or value.get("score") == 0.0


def _decision_ref(phase: str, judgement: dict[str, Any]) -> str:
    return _stable_id(
        "functional_decision",
        {
            "phase": phase,
            "verdict": judgement.get("verdict"),
            "confidence": judgement.get("confidence"),
            "reason": judgement.get("reason"),
        },
    )


def _trusted_ids(values: Iterable[str]) -> set[str]:
    normalized = [str(item).strip() for item in values]
    if (
        not normalized
        or any(not item for item in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError(
            "functional ownership requires unique non-empty scene object IDs"
        )
    return set(normalized)


def _validated_event_ids(
    value: Any,
    *,
    known_ids: set[str],
    label: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} must contain non-empty object IDs")
    result = sorted(str(item).strip() for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate object IDs")
    unknown = sorted(set(result) - known_ids)
    if unknown:
        raise ValueError(f"{label} references unknown IDs: {unknown}")
    return result


def _stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


__all__ = [
    "CROSS_METRIC_OWNERSHIP_AUDIT_VERSION",
    "FUNCTIONAL_OWNERSHIP_LEDGER_VERSION",
    "build_cross_metric_ownership_audit",
    "build_functional_ownership_ledger",
    "validate_functional_ownership_ledger",
]
