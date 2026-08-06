"""Stable identities and correspondence records for L3 metric claims."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Iterable


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
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "claim_id": f"l3_claim_{digest}",
        **payload,
        "reason": str(defect.get("reason") or "").strip(),
        "source_phase": str(source_phase),
        "claim_status": str(claim_status),
    }


def claim_records(
    metric_name: str,
    defects: Iterable[Any],
    *,
    source_phase: str,
    claim_status: str,
) -> list[dict[str, Any]]:
    """Build unique claim records while preserving first-seen order."""

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        key = canonical_claim_key(metric_name, defect)
        if key in seen:
            continue
        seen.add(key)
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
    """Drop exact duplicate final defects by canonical claim identity."""

    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        key = canonical_claim_key(metric_name, defect)
        if key in seen:
            continue
        seen.add(key)
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
                if canonical_claim_key(metric_name, candidate)
                == canonical_claim_key(metric_name, defect)
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
