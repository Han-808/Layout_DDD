"""One real same-evidence inventory retry; failed discovery is not no relations.

Discovery proposes obligations, never a metric verdict. Preserve legal earlier
atoms and require unresolved anchored atoms; a retry may discover additional
relations only as an explicitly new complete model inventory, not schema repair.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
from typing import Any

from benchmark.models import parse_json_object
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.functional_discovery_validation import validate_functional_relation_response

POLICY = "bounded_same_evidence_relation_inventory_v1"
RECOVERY_MAX_TOKENS = 16384


class FunctionalRelationInventoryError(ResponseSchemaRepairError):
    """Unresolved inventory must not degrade into an empty relation plan."""

    failure_category = "judge_response_failure"


def relation_inventory_issues(discovery: Any) -> list[str]:
    """Read explicit unresolved contract obligations, including old reports.

    Successful discovery is not an additional scored judgement. Only unresolved
    discovery is retained as a hard response-contract failure, never default valid.
    Absence of newer metadata in a legacy report is not fabricated failure evidence.
    """
    if not isinstance(discovery, dict):
        return []
    call = (discovery.get("provenance") or {}).get("calls", {}).get("relations") or {}
    audit = call.get("schema_validation") or {}
    salvage = (discovery.get("coverage") or {}).get("relations") or audit.get("salvage") or {}
    issues = []
    if salvage.get("consideration_contract_valid") is False:
        issues.append("full_object_consideration_unresolved")
    for anchor in salvage.get("dropped_relation_anchors") or []:
        identity = json.dumps(anchor, sort_keys=True, separators=(",", ":"))
        issues.append("unresolved_anchor_" + hashlib.sha256(identity.encode()).hexdigest()[:16])
    if salvage.get("dropped_relation_count", 0) and not salvage.get("dropped_relation_anchors"):
        issues.append("unresolved_relation_anchors")
    for row in salvage.get("rejected_items") or []:
        reason = str(row.get("reason") or "")
        if reason.startswith("invalid_identity_anchor") or reason == "relation_has_no_initial_identity_anchor":
            issues.append("unresolved_candidate_" + str(row.get("source")) + "_" + str(row.get("index")))
    recovery = audit.get("inventory_recovery") or {}
    if not recovery.get("complete"):
        if audit.get("attempt_count", 0) > 1 and audit.get("recovered") is False:
            issues.append("relation_inventory_schema_unresolved")
        attempts = audit.get("attempts") or []
        last_meta = (attempts[-1].get("request_metadata") or call) if attempts else call
        if last_meta.get("finish_reason") == "length":
            issues.append("relation_inventory_output_truncated")
    return list(dict.fromkeys(issues))


def complete_relation_inventory(*, model, normalized: dict, messages: list,
                                initial_metadata: dict, relations: dict, audit: dict,
                                response_format_json: bool, initial_raw: str = "") -> tuple[dict, dict]:
    """At most one additional model call, without acquiring or inventing evidence."""
    envelope = {"provenance": {"calls": {"relations": {
        **initial_metadata, "schema_validation": audit}}}}
    issues = relation_inventory_issues(envelope)
    if not issues:
        return relations, audit
    audit = deepcopy(audit)
    if audit.get("attempts") and "raw_response" not in audit["attempts"][0]:
        audit["attempts"][0].update(raw_response=str(initial_raw)[:20000],
            request_metadata=deepcopy(initial_metadata),
            raw_response_sha256=hashlib.sha256(str(initial_raw).encode()).hexdigest())
    retained = deepcopy(relations.get("relations") or [])
    pending = deepcopy((relations.get("item_salvage") or {}).get("dropped_relation_anchors") or [])
    call_type = "vlm_camera_pose.functional_discovery.relations.inventory_recovery"
    instruction = (
        "The bounded schema repair did not establish a complete relation inventory. "
        "This is one NEW complete non-judging discovery using the SAME original visual "
        "evidence and trusted object list. Inspect all objects; return the original exact "
        "considered_object_ids/relations/reason JSON contract. Preserve every retained "
        "legal row below exactly, including ordered roles and predicate. Resolve each "
        "pending identity exactly once; do not delete or retarget it. You may add real "
        "joint-use relations supported by this full-set review. Empty relations are "
        "allowed only after your complete review and only if no retained/pending atoms "
        "exist. Do not output a score, verdict, defect or camera request. Be concise and "
        "reserve output for the JSON object.\n" + json.dumps({
            "retained_legal_relations": retained, "pending_relation_identities": pending,
            "unresolved_contracts": issues}, ensure_ascii=False, sort_keys=True)
    )
    request = [*deepcopy(messages), {"role": "user", "content": instruction}]
    recovery = {"policy": POLICY, "complete": False, "call_type": call_type,
        "same_original_messages": request[:-1] == messages, "additional_model_calls": 1,
        "original_messages_sha256": hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest(),
        "trigger_issues": issues, "retained_legal_relations": retained,
        "pending_relation_identities": pending}
    started = time.perf_counter()
    try:
        raw = model.chat_messages(request, response_format_json=response_format_json,
            call_type=call_type, max_tokens=max(RECOVERY_MAX_TOKENS, int(getattr(model,"max_tokens",0) or 0)),
            max_tokens_source=POLICY, case={"case_id":str(normalized.get("scene_id") or "functional_discovery"),
                "scene_id":str(normalized.get("scene_id") or ""), "objects":deepcopy(normalized["objects"])})
    except Exception as exc:
        recovery.update(failure_kind="transport", error_type=type(exc).__name__, error=str(exc))
        audit["inventory_recovery"] = recovery
        audit["attempts"].append({"attempt":len(audit["attempts"])+1,
            "call_type":call_type,"failure_kind":"transport","validation_error_type":type(exc).__name__})
        audit["attempt_count"] = len(audit["attempts"])
        raise FunctionalRelationInventoryError("Functional relation inventory request failed",schema_audit=audit) from exc
    meta = deepcopy(getattr(model,"last_request_metadata",{}) or {})
    attempt = {"attempt":len(audit["attempts"])+1,"call_type":call_type,
        "raw_response":str(raw)[:20000], "raw_response_sha256":hashlib.sha256(str(raw).encode()).hexdigest(),
        "request_metadata":meta,"latency_seconds":round(time.perf_counter()-started,6)}
    audit["attempts"].append(attempt)
    audit["attempt_count"] = len(audit["attempts"])
    try:
        if meta.get("finish_reason") == "length":
            raise ValueError("relation inventory recovery output truncated")
        result = validate_functional_relation_response(parse_json_object(raw),
            object_ids=tuple(item["id"] for item in normalized["objects"]))
        def identity(row):
            return tuple(sorted(row["target_ids"])), row["predicate"]
        by_id = {identity(row):row for row in result["relations"]}
        for row in retained:
            if by_id.get(identity(row)) != row:
                raise ValueError("relation inventory recovery changed or deleted a retained legal atom")
        if any(identity(row) not in by_id for row in pending):
            raise ValueError("relation inventory recovery dropped a pending anchored atom")
    except (ValueError, TypeError, KeyError) as exc:
        attempt.update(validation_error_type=type(exc).__name__,validation_error=str(exc))
        recovery.update(failure_kind="response_contract",error=str(exc))
        audit["inventory_recovery"] = recovery
        raise FunctionalRelationInventoryError("Functional relation inventory remains unresolved",schema_audit=audit) from exc
    attempt["validation_error"] = None
    recovery.update(complete=True,model_inventory=result)
    audit.update(inventory_recovery=recovery, previous_salvage=deepcopy(audit.get("salvage")),
        salvage={"policy":POLICY,"consideration_contract_valid":True,
            "anchored_relation_count":len(result["relations"]),"accepted_relation_count":len(result["relations"]),
            "dropped_relation_count":0,"dropped_relation_anchors":[],"rejected_relation_count":0,"rejected_items":[]})
    # Original schema-repair recovered flag remains unchanged; the new call is
    # independently identified as a real inventory, not a successful old repair.
    result["item_salvage"] = deepcopy(audit["salvage"])
    return result, audit


def unresolved_inventory_units(report: dict) -> list[dict]:
    """Keep missing discovery obligations visible after typed checks exist."""
    issues = relation_inventory_issues(report.get("functional_discovery"))
    acquisition = report.get("functional_probe_acquisition") or {}
    recovery = (acquisition.get("response_schema_validation") or {}).get("inventory_recovery") or {}
    category = "judge_response_failure"
    if recovery.get("complete") is False:
        issues.append("bounded_inventory_recovery_failed")
        category = (acquisition.get("failure") or {}).get("failure_category") or category
    return [{"unit_id":"functional_relation_discovery:"+issue,
        "status":"failed","score":None,"accepted":False,"execution_complete":True,
        "visual_observation_complete":False,"model_judgement_completed":False,
        "failure_category":category,"fallback_reason":issue,
        "fallback_source":"unresolved_relation_discovery_contract"} for issue in issues]
