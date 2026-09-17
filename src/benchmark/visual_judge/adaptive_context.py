"""Lossless mandatory Judge context and deterministic atomic-check sharding."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable

from benchmark.visual_judge.evidence_resolution import (
    ContextCapacityError, AdaptiveJudgeError, EvidenceIntegrityError, adaptive_enabled,
)


MANDATORY_KEYS = frozenset({
    "vlm_role", "decision_contract", "judge_method", "metric", "metric_rubric",
    "metric_boundary_rules", "rubric", "response_contract", "phase_instruction",
    "required_functional_checks", "required_placement_checks", "deferred_placement_checks",
    "functional_ownership_ledger", "placement_residual_context", "allowed_missing_observations",
    "allowed_evidence_request_target_ids", "allowed_defect_target_ids", "target_scope",
    "metric_scope", "defect_attribution", "functional_measurements", "functional_measurement_policy",
    "placement_check_policy", "placement_severity_policy", "functional_relation_scope",
    "event", "objects", "architecture", "detector_evidence", "adaptive_evidence",
    "view_names", "view_evidence", "evidence_phase", "decision_mode",
    "authorized_deviations", "object_groups", "context_object_ids", "causal_object_catalog",
    "deterministic_evidence", "functional_probe_evidence",
})


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def budget_adaptive_context(context: dict[str, Any], limit: int) -> str:
    """Do not replace complete checks/facts with JSON-prefix placeholders."""
    result = {key: deepcopy(value) for key, value in context.items()
              if key in MANDATORY_KEYS and value is not None}
    evidence = result.get("adaptive_evidence")
    if isinstance(evidence, dict):
        # Local artifact paths/provenance belong in the audit, not model text.
        for key in ("available_images", "excluded_images"):
            evidence.pop(key, None)
    marker = {"mandatory_complete": True, "omitted_optional_keys": []}
    result["_adaptive_context_budget"] = marker
    if len(compact_json(result)) > limit:
        raise ContextCapacityError(
            "mandatory metric context exceeds the configured character budget",
            audit={"limit": limit, "mandatory_chars": len(compact_json(result)),
                   "mandatory_keys": sorted(set(result) - {"_adaptive_context_budget"})},
        )
    for key, value in context.items():
        if key in result or value is None:
            continue
        trial = {**result, key: value}
        if len(compact_json(trial)) <= limit:
            result[key] = deepcopy(value)
        else:
            marker["omitted_optional_keys"].append(key)
    # The audit marker itself must fit; never trim the mandatory payload.
    if len(compact_json(result)) > limit:
        result["_adaptive_context_budget"] = {"mandatory_complete": True,
                                              "optional_fields_omitted": True}
    if len(compact_json(result)) > limit:
        raise ContextCapacityError("mandatory context and budget audit do not fit")
    return compact_json(result)


def _checks(request: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    if request.get("required_placement_checks"):
        return "placement", list(request["required_placement_checks"])
    if request.get("required_functional_checks"):
        return "functional", list(request["required_functional_checks"])
    probe = request.get("functional_probe_evidence") or {}
    if probe.get("required_checks"):
        return "functional", list(probe["required_checks"])
    return None, []


def _slice_request(request: dict[str, Any], kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    value = deepcopy(request)
    value["adaptive_context_shard"] = True
    ids = {str(row.get("check_id")) for row in rows}
    if kind == "placement":
        value["required_placement_checks"] = rows
    else:
        if "required_functional_checks" in value:
            value["required_functional_checks"] = rows
        if isinstance(value.get("functional_probe_evidence"), dict):
            value["functional_probe_evidence"]["required_checks"] = rows
            for key in ("checks", "measurements", "results"):
                items = value["functional_probe_evidence"].get(key)
                if isinstance(items, list):
                    value["functional_probe_evidence"][key] = [
                        item for item in items if not isinstance(item, dict)
                        or not item.get("check_id") or str(item["check_id"]) in ids
                    ]
    # Ownership and room-global inventory are deliberately NOT sliced.
    return value


def call_with_atomic_shards(
    request: dict[str, Any], call: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if adaptive_enabled(request):
        _, required = _checks(request)
        ids = [str(row.get("check_id") or "") for row in required]
        if not all(ids) or len(ids) != len(set(ids)):
            raise EvidenceIntegrityError("required checks contain duplicate or missing IDs")
    try:
        return call(request)
    except ContextCapacityError:
        kind, checks = _checks(request)
        if not adaptive_enabled(request) or not kind or len(checks) < 2:
            raise
        if request.get("metric") == "object_pairing_consistency":
            raise
    ordered = sorted(deepcopy(checks), key=lambda row: str(row.get("check_id") or ""))
    expected_ids = [str(row.get("check_id") or "") for row in ordered]
    if not all(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise AdaptiveJudgeError("cannot shard duplicate or missing required check IDs")
    if any(set(row.get("depends_on_check_ids") or []) & set(expected_ids) for row in ordered):
        raise ContextCapacityError("interdependent checks cannot be independently sharded")
    results: list[dict[str, Any]] = []

    def resolve(rows: list[dict[str, Any]]) -> None:
        try:
            results.append(call(_slice_request(request, kind, rows)))
        except ContextCapacityError:
            if len(rows) < 2:
                raise
            middle = len(rows) // 2
            resolve(rows[:middle])
            resolve(rows[middle:])

    middle = len(ordered) // 2
    resolve(ordered[:middle])
    resolve(ordered[middle:])
    key = "placement_check_results" if kind == "placement" else "functional_check_results"
    merged = deepcopy(results[0])
    rows = [deepcopy(row) for result in results for row in result.get(key) or []]
    actual_ids = [str(row.get("check_id") or "") for row in rows]
    if sorted(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise AdaptiveJudgeError("shard results must resolve each exact check ID once")
    if any(result.get("verdict") not in {"valid", "invalid"} for result in results):
        raise AdaptiveJudgeError("atomic context shards require completed binary decisions")
    merged[key] = rows
    merged["verdict"] = "invalid" if any(result["verdict"] == "invalid" for result in results) else "valid"
    merged["evidence_status"] = "sufficient"
    merged["confidence"] = min(float(result["confidence"]) for result in results)
    merged["reason"] = " ".join(str(result.get("reason") or "") for result in results)
    defects: dict[str, dict[str, Any]] = {}
    for result in results:
        for defect in result.get("defects") or []:
            defects.setdefault(compact_json(defect), deepcopy(defect))
    merged["defects"] = list(defects.values())
    if any(result.get("group_global_observations") != results[0].get("group_global_observations") for result in results):
        raise AdaptiveJudgeError("context shards disagree on indivisible global observations")
    merged["evidence_request"] = None
    if kind == "placement":
        # Preserve all per-check sidecars, not just the first shard's audit.
        placement_audits = {
            check_id: deepcopy(audit)
            for result in results
            for check_id, audit in (result.get("placement_check_resolutions") or {}).items()
        }
        if placement_audits:
            merged["placement_check_resolutions"] = placement_audits
    if isinstance(merged.get("evidence_resolution"), dict):
        merged["evidence_resolution"]["model_call_count"] = sum(
            int((result.get("evidence_resolution") or {}).get("model_call_count") or 0) for result in results
        )
    merged.setdefault("request_metadata", {})["atomic_check_sharding"] = {
        "policy": "stable_check_id_bisection_v1", "shard_count": len(results),
        "required_check_ids": expected_ids, "resolved_check_ids": sorted(actual_ids),
        "ownership_and_global_inventory_preserved": True,
    }
    return merged
