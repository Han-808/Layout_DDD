"""Route validated out-of-stage findings without scoring them here."""
from copy import deepcopy


def pending_handoff(value, check, original):
    """Caller has validated the entire finding, ignoring only its stage mismatch."""
    items = original.get("judge_originated_placement_results") or []
    if len(items) != 1 or len(value.get("defects") or []) != 1:
        raise ValueError("stage handoff requires one isolated validated finding")
    item = items[0]
    from benchmark.evaluator.scene_quality.placement_severity import validate_placement_defect_severity
    for defect in original.get("defects") or []:
        validate_placement_defect_severity(defect)
    proposal = {key: deepcopy(item[key]) for key in (
        "proposal_id", "subject_id", "context_ids", "check_type", "observation_goal")}
    observation = "group_context_visible" if check["owner_stage"] == "group_local" else "global_context_preserved"
    audit = {"policy": "validated_placement_stage_handoff_v1",
             "owner_stage": check["owner_stage"], "accepted_in_current_stage": False,
             "prior_finding": deepcopy(item), "prior_defects": deepcopy(value["defects"]),
             "prior_confidence": value["confidence"]}
    pending = {
        "evidence_status": "insufficient", "verdict": "ambiguous", "confidence": value["confidence"],
        "reason": "A validated candidate belongs to another stage; defer it for that owner's judgement, not a score here.",
        "missing_evidence": [observation], "defects": [],
        "evidence_request": {"target_ids": [proposal["subject_id"]],
                             "missing_observations": [observation],
                             "view_goal": proposal["observation_goal"],
                             "metadata": {"placement_check_proposal": proposal,
                                          "validated_stage_handoff": audit}},
    }
    if "group_global_observations" in value:
        pending["group_global_observations"] = deepcopy(value["group_global_observations"])
    return pending


def pending_handoffs(value, checks, original):
    """Keep every prior claim; each trusted owner must make a new judgement."""
    items = original.get("judge_originated_placement_results") or []
    if len(checks) == 1:
        return pending_handoff(value, checks[0], original)
    if len(checks) != len(items) or len(value.get("defects") or []) != len(checks):
        raise ValueError("batch handoff requires one linked defect per distinct finding")
    by_id = {item["proposal_id"]: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("batch handoff requires distinct proposal identities")
    children = []
    for check in checks:
        refs = check["source_discovery_refs"]
        if len(refs) != 1 or refs[0] not in by_id:
            raise ValueError("batch handoff cannot lose a source finding")
        defects = [d for d in value["defects"] if d.get("check_id") == check["check_id"]]
        if len(defects) != 1:
            raise ValueError("batch handoff requires an exact defect/check mapping")
        item = by_id.pop(refs[0])
        child = pending_handoff(
            {**value, "defects": defects}, check,
            {**original, "judge_originated_placement_results": [item], "defects": defects})
        children.append(child)
    if by_id:
        raise ValueError("batch handoff left an untransferred source finding")
    pending = deepcopy(children[0])
    request = pending["evidence_request"]
    request["target_ids"] = sorted({t for c in children for t in c["evidence_request"]["target_ids"]})
    request["metadata"]["placement_check_handoffs"] = [c["evidence_request"] for c in children]
    return pending


def handoff_requests(evidence_request):
    """Validate a flat batch envelope; children still undergo trusted validation."""
    metadata = evidence_request.get("metadata") or {}
    children = metadata.get("placement_check_handoffs")
    if not isinstance(children, list) or len(children) < 2:
        raise ValueError("stage handoff batch must contain multiple requests")
    seen, targets = set(), set()
    for child in children:
        if not isinstance(child, dict) or not isinstance(child.get("metadata"), dict):
            raise ValueError("invalid stage handoff child")
        inner = child["metadata"]
        if "placement_check_handoffs" in inner:
            raise ValueError("nested stage handoff batches are forbidden")
        proposal = inner.get("placement_check_proposal")
        if not isinstance(proposal, dict):
            raise ValueError("stage handoff child requires a typed proposal")
        identity = proposal.get("proposal_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("stage handoff batch requires distinct proposal IDs")
        seen.add(identity)
        targets.update(child.get("target_ids") or [])
    if targets != set(evidence_request.get("target_ids") or []):
        raise ValueError("stage handoff batch target inventory mismatch")
    if metadata.get("placement_check_proposal") != children[0]["metadata"]["placement_check_proposal"]:
        raise ValueError("stage handoff primary proposal mismatch")
    return children
