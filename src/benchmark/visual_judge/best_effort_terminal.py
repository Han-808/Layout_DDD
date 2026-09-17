"""Opt-in final judgement under bounded evidence; never a policy-made score."""
from copy import deepcopy
from typing import Any, Mapping

POLICY = "best_available_final_v1"
DEFAULT_POLICY = "bounded_abstention_v2"
INSTRUCTION = """Evidence acquisition has ended. This is a best-available-evidence
FINAL JUDGEMENT, not another acquisition request. Apply the original metric
definition and thresholds to every exact required check using the supplied
legal images and metric-owned facts. Choose the more defensible valid/invalid
conclusion; do not abstain solely because another view would be preferable.
Mark inferred typed rows inferred_under_budget and explain uncertainty in
confidence and reason. Never claim an unobserved feature was observed.
Missing evidence alone is neither a defect nor proof of validity. Do not use
generator intent as ground truth. Preserve owners, targets and already resolved
typed conclusions. Do not invent new checks or defects to satisfy a schema.
For canonical metrics return evidence_status=sufficient, verdict valid/invalid,
missing_evidence=[], evidence_request=null, and every required typed row resolved.
Here sufficient means a completed constrained judgement, not complete visual
observation. For binary metrics return valid/invalid, never need_more_evidence.
No more acquisition requests will execute."""


def validate_policy(policy: str) -> str:
    if policy not in {DEFAULT_POLICY, POLICY}:
        raise ValueError("unsupported terminal_evidence_policy")
    return policy


def enabled(request: Mapping[str, Any]) -> bool:
    from .evidence_gap_v2 import enabled as fallback_enabled
    return fallback_enabled(request) and request.get("terminal_evidence_policy") == POLICY


def terminal(request: Mapping[str, Any]) -> bool:
    return enabled(request) and request.get("adaptive_terminal") is True


def inject(request: dict, policy: str) -> dict:
    value = deepcopy(request)
    value["terminal_evidence_policy"] = validate_policy(
        str(value.get("terminal_evidence_policy", policy)))
    return value


class TerminalDecisionRequired(ValueError):
    """A structurally valid unresolved reply needs a NEW terminal decision."""
    def __init__(self, response: dict):
        super().__init__("best-available terminal judgement requires a final conclusion")
        self.response = deepcopy(response)


def require_final(value: dict) -> dict:
    label = value.get("verdict", value.get("status"))
    unresolved = any(
        row.get("conclusion") not in {"valid", "invalid", "excluded_function_owned"}
        for field in ("functional_check_results", "placement_check_results", "judge_originated_placement_results")
        for row in value.get(field) or []
    )
    if unresolved or label not in {"valid", "invalid"} or value.get("evidence_request") is not None:
        raise TerminalDecisionRequired(value)
    return value


def preserve_resolved_rows(initial: dict, result: dict) -> None:
    for field in ("functional_check_results", "placement_check_results"):
        prior = {row["check_id"]: row for row in initial.get(field) or []
                 if row.get("conclusion") in {"valid", "invalid", "excluded_function_owned"}}
        actual = {row["check_id"]: row for row in result.get(field) or []}
        for check_id, before in prior.items():
            after = actual.get(check_id) or {}
            # Explanations/confidence may elaborate, not reverse known answers.
            for key in ("conclusion", "observation_status", "subject_id", "context_ids",
                        "target_ids", "check_type", "owner_stage"):
                if before.get(key) != after.get(key):
                    raise ValueError("terminal decision changed an already resolved typed check")


def retry_decision(*, model, messages, raw, response_format_json, call_type,
                   judge_label, validator, initial_error):
    """One model decision retry, separate from locked semantic format repair."""
    from benchmark.models import parse_json_object
    from .contracts import ResponseSchemaRepairError
    audit = {"policy": POLICY, "attempt_count": 2, "repair_retry_count": 0,
             "decision_retry_count": 1, "recovered": False,
             "attempts": [{"attempt": 1, "call_type": call_type,
                           "validation_error_type": "TerminalDecisionRequired",
                           "validation_error": str(initial_error)}]}
    retry_type = call_type + ".terminal_decision_retry"
    retry_messages = [*deepcopy(messages), {"role": "assistant", "content": raw},
                      {"role": "user", "content": INSTRUCTION +
                       "\nYour previous reply was a legal uncertainty report. Make the required final judgement now; this is not schema-only repair."}]
    try:
        answer = model.chat_messages(retry_messages, response_format_json=response_format_json,
                                     call_type=retry_type)
        result = validator(parse_json_object(answer))
        preserve_resolved_rows(initial_error.response, result)
    except Exception as exc:
        audit["attempts"].append({"attempt": 2, "call_type": retry_type,
                                   "validation_error_type": type(exc).__name__})
        raise ResponseSchemaRepairError(
            judge_label + " did not complete its bounded terminal decision",
            schema_audit=audit) from exc
    audit["recovered"] = True
    audit["attempts"].append({"attempt": 2, "call_type": retry_type, "validation_error": None})
    return result, audit
