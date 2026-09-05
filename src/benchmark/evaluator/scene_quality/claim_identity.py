"""Stable identities and correspondence records for L3 metric claims."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Iterable

from benchmark.evaluator.scoring import L3_SEVERITY_BURDENS
from benchmark.evaluator.scene_quality.placement_severity import (
    placement_severity_rank,
)


OBJECT_LEVEL_ATTRIBUTION_METRICS = frozenset(
    {
        "style_consistency",
        "functional_consistency",
        "semantic_placement_consistency",
    }
)


def claim_record(
    metric_name: str,
    defect: dict[str, Any],
    *,
    source_phase: str,
    claim_status: str,
) -> dict[str, Any]:
    """Return an auditable, stable identity for one metric-scoped claim."""

    scope, target_ids, relation = canonical_claim_key(
        metric_name,
        defect,
    )[1:]
    payload = {
        "metric": str(metric_name),
        "scope": scope,
        "target_ids": list(target_ids),
        "relation": relation,
    }
    identity_key = _claim_identity_key(metric_name, defect)
    digest_payload: Any = (
        payload
        if len(identity_key) > 1 and identity_key[1] == "claim"
        else identity_key
    )
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    record = {
        "claim_id": f"l3_claim_{digest}",
        **payload,
        "reason": str(defect.get("reason") or "").strip(),
        "source_phase": str(source_phase),
        "claim_status": str(claim_status),
    }
    for field in ("ownership_event_id", "check_id", "finding_id"):
        value = str(defect.get(field) or "").strip()
        if value:
            record[field] = value
    check_refs = defect.get("check_refs")
    if isinstance(check_refs, (list, tuple)):
        normalized_refs = sorted(
            {
                str(item).strip()
                for item in check_refs
                if str(item).strip()
            }
        )
        if normalized_refs:
            record["check_refs"] = normalized_refs
    if (
        metric_name == "semantic_placement_consistency"
        and placement_severity_rank(defect.get("severity"))
    ):
        record["severity"] = str(defect["severity"])
    return record


def claim_records(
    metric_name: str,
    defects: Iterable[Any],
    *,
    source_phase: str,
    claim_status: str,
) -> list[dict[str, Any]]:
    """Build unique claim records while preserving first-seen order."""

    records: list[dict[str, Any]] = []
    for defect in deduplicate_defects(metric_name, defects):
        records.append(
            claim_record(
                metric_name,
                defect,
                source_phase=source_phase,
                claim_status=claim_status,
            )
        )
    return records


def deduplicate_defects(
    metric_name: str,
    defects: Iterable[Any],
) -> list[dict[str, Any]]:
    """Merge duplicate final defects and retain the strongest observation."""

    retained: list[dict[str, Any]] = []
    retained_index: dict[tuple[Any, ...], int] = {}
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        key = _claim_identity_key(metric_name, defect)
        if key in retained_index:
            if _severity_rank(metric_name, defect.get("severity")) > (
                _severity_rank(
                    metric_name,
                    retained[retained_index[key]].get("severity"),
                )
            ):
                retained[retained_index[key]] = deepcopy(defect)
            continue
        retained_index[key] = len(retained)
        retained.append(deepcopy(defect))
    return retained


def object_level_finding_records(
    metric_name: str,
    observations: Iterable[tuple[str, Any]],
) -> list[dict[str, Any]]:
    """Merge defect observations into one finding per metric/object.

    Raw global and local defects remain available in their original judgement
    records.  This projection defines the object-level penalty/audit unit for
    metrics that independently judge both global and group-local visual
    scopes, so a local observation of an object already flagged globally
    cannot become a second penalty unit.
    """

    if metric_name not in OBJECT_LEVEL_ATTRIBUTION_METRICS:
        return []
    findings: dict[tuple[str, str], dict[str, Any]] = {}
    for source_phase, raw_defect in observations:
        if not isinstance(raw_defect, dict):
            continue
        target_ids = canonical_target_ids(raw_defect)
        if not target_ids:
            continue
        observation = {
            "source_phase": str(source_phase),
            "scope": _normalize_token(raw_defect.get("scope")),
            "target_ids": list(target_ids),
            "relation": _normalize_token(raw_defect.get("relation")),
            "reason": str(raw_defect.get("reason") or "").strip(),
        }
        if (
            metric_name == "semantic_placement_consistency"
            and placement_severity_rank(raw_defect.get("severity"))
        ):
            observation["severity"] = str(raw_defect["severity"])
        for object_id in target_ids:
            key = (_normalize_token(metric_name), object_id)
            finding = findings.get(key)
            if finding is None:
                payload = {
                    "metric": str(metric_name),
                    "object_id": object_id,
                }
                digest = hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:16]
                finding = {
                    "finding_id": f"l3_object_{digest}",
                    **payload,
                    "attribution_unit": "object",
                    "source_phases": [],
                    "observations": [],
                }
                findings[key] = finding
            if source_phase not in finding["source_phases"]:
                finding["source_phases"].append(str(source_phase))
            if observation not in finding["observations"]:
                finding["observations"].append(deepcopy(observation))

    records = list(findings.values())
    for finding in records:
        observation_count = len(finding["observations"])
        finding["observation_count"] = observation_count
        finding["merged_duplicate_observation_count"] = max(
            0,
            observation_count - 1,
        )
        finding["observed_in_global_and_local"] = bool(
            any(
                phase == "global_discovery"
                for phase in finding["source_phases"]
            )
            and any(
                phase.startswith("group_local_review")
                for phase in finding["source_phases"]
            )
        )
        if metric_name == "semantic_placement_consistency":
            severity_values = [
                str(observation.get("severity") or "")
                for observation in finding["observations"]
                if placement_severity_rank(
                    observation.get("severity")
                )
            ]
            finding["highest_severity"] = max(
                severity_values,
                key=placement_severity_rank,
                default="none",
            )
    return records


def match_final_defects_to_routed_claims(
    metric_name: str,
    defects: Iterable[Any],
    routed_claims: Iterable[Any],
) -> list[dict[str, Any]]:
    """Classify final defects as confirmations or distinct new claims."""

    candidates = [
        deepcopy(item)
        for item in routed_claims
        if isinstance(item, dict)
    ]
    matches: list[dict[str, Any]] = []
    matched_candidate_ids: set[str] = set()
    for defect in deduplicate_defects(metric_name, defects):
        final_claim = claim_record(
            metric_name,
            defect,
            source_phase="visual_confirmation",
            claim_status="final",
        )
        exact = next(
            (
                candidate
                for candidate in candidates
                if _claim_identity_key(metric_name, candidate)
                == _claim_identity_key(metric_name, defect)
            ),
            None,
        )
        related = exact or next(
            (
                candidate
                for candidate in candidates
                if _scope_and_targets(metric_name, candidate)
                == _scope_and_targets(metric_name, defect)
            ),
            None,
        )
        if exact is not None:
            relationship = "confirmed_routed_candidate"
        elif related is not None:
            relationship = "same_targets_distinct_relation"
        else:
            relationship = "new_final_defect"
        related_id = (
            str(related.get("claim_id") or "")
            if isinstance(related, dict)
            else ""
        )
        if related_id:
            matched_candidate_ids.add(related_id)
        matches.append(
            {
                "final_claim": final_claim,
                "routed_candidate_id": related_id or None,
                "relationship": relationship,
            }
        )
    matches.extend(
        {
            "final_claim": None,
            "routed_candidate_id": candidate_id,
            "relationship": "routed_candidate_not_confirmed",
        }
        for candidate in candidates
        if (
            candidate_id := str(candidate.get("claim_id") or "")
        )
        and candidate_id not in matched_candidate_ids
    )
    return matches


def canonical_claim_key(
    metric_name: str,
    defect: dict[str, Any],
) -> tuple[str, str, tuple[str, ...], str]:
    """Canonical ``metric/scope/targets/relation`` identity tuple."""

    targets = canonical_target_ids(defect)
    return (
        _normalize_token(metric_name),
        _normalize_token(defect.get("scope")),
        targets,
        _normalize_token(defect.get("relation")),
    )


def canonical_target_ids(defect: dict[str, Any]) -> tuple[str, ...]:
    """Return stable exact object IDs named by one defect."""

    target_ids = defect.get("target_ids")
    return (
        tuple(
            sorted(
                {
                    str(item).strip()
                    for item in target_ids
                    if str(item).strip()
                }
            )
        )
        if isinstance(target_ids, (list, tuple))
        else ()
    )


def _claim_identity_key(
    metric_name: str,
    defect: dict[str, Any],
) -> tuple[Any, ...]:
    """Prefer validated workflow identity over scope/relation wording.

    Ownership events, required checks, and findings are generated by the
    deterministic orchestration layer and survive global/local retries.  When
    one is present it is a safer duplicate key than VLM-authored scope or
    relation text.  Target IDs remain part of the key so a malformed reused ID
    cannot collapse findings for different objects.
    """

    metric = _normalize_token(metric_name)
    targets = _claim_responsibility_targets(defect)
    for field in ("ownership_event_id", "check_id", "finding_id"):
        value = str(defect.get(field) or "").strip()
        if value:
            return (metric, "stable", field, value, targets)
    check_refs = defect.get("check_refs")
    if isinstance(check_refs, (list, tuple)):
        refs = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in check_refs
                    if str(item).strip()
                }
            )
        )
        if refs:
            return (metric, "stable", "check_refs", refs, targets)
    return (metric, "claim", *canonical_claim_key(metric_name, defect)[1:])


def _claim_responsibility_targets(defect: dict[str, Any]) -> tuple[str, ...]:
    """Use explicit scoring owners when context-rich target lists differ."""

    scoring_targets = defect.get("scoring_target_ids")
    if isinstance(scoring_targets, (list, tuple)):
        normalized = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in scoring_targets
                    if str(item).strip()
                }
            )
        )
        if normalized:
            return normalized
    return canonical_target_ids(defect)


def _severity_rank(metric_name: str, value: Any) -> float:
    """Return the canonical burden rank used to keep stronger duplicates."""

    severity = str(value or "").strip()
    burden = L3_SEVERITY_BURDENS.get(metric_name, {}).get(severity)
    if burden is not None:
        return float(burden)
    if metric_name == "semantic_placement_consistency":
        placement_rank = placement_severity_rank(severity)
        if placement_rank == 1:
            return 0.4
        if placement_rank == 2:
            return 1.0
    return 0.0


def _scope_and_targets(
    metric_name: str,
    defect: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    key = canonical_claim_key(metric_name, defect)
    return key[:3]


def _normalize_token(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().lower(),
    ).strip("_")
