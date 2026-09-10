"""Lossless exception observation plus bounded, source-grounded reason details.

No validator is relaxed. Record identifiers, not prompts or complete responses.
Ambiguous historical messages remain ambiguous unless observed inputs resolve them.
"""
from __future__ import annotations

from collections import Counter
import functools
import importlib
import inspect
import re
import sys

from .safety import best_effort_record, exception_record, identity, redact

EXACT = {
    "placement_check_results must cover every required check exactly once": "placement.check_coverage_mismatch",
    "functional_check_results must cover every required check exactly once": "functional.check_coverage_mismatch",
    "placement Judge response requires placement_check_results": "placement.results_missing_or_wrong_shape",
    "functional Judge response requires functional_check_results": "functional.results_missing_or_wrong_shape",
    "required placement checks contain invalid IDs": "placement.required_check_ids_invalid",
    "required functional checks contain invalid IDs": "functional.required_check_ids_invalid",
    "camera selector selected an untrusted candidate ID": "camera.untrusted_candidate_id",
    "camera selector selected an untrusted repair-plan ID": "camera.untrusted_repair_plan_id",
    "candidate_only response requires unique selected_view_ids": "camera.selected_view_ids_invalid_or_duplicate",
    "candidate_only requires successful candidate previews": "camera.candidate_preview_unavailable",
    "candidate_only accepts only technically feasible views": "camera.candidate_not_feasible",
    "camera selector response requires reason": "camera.reason_missing",
    "camera selector response must be a JSON object": "camera.response_not_object",
    "freeform_pose response requires camera_proposal": "camera.proposal_missing",
}
PATTERNS = (
    (r"^placement check (?P<check_id>\S+) has invalid observation_status$", "placement.observation_status_invalid", "observation_status"),
    (r"^placement check (?P<check_id>\S+) has invalid conclusion$", "placement.conclusion_invalid", "conclusion"),
    (r"^placement check (?P<check_id>\S+) requires a non-empty reason$", "placement.reason_missing", "reason"),
    (r"^placement check (?P<check_id>\S+) must reference one exact known functional ownership event$", "placement.functional_event_reference_invalid", "function_event_ref"),
    (r"^placement check (?P<check_id>\S+) must explicitly confirm same_physical_event before deduplication$", "placement.same_event_confirmation_missing", "same_physical_event"),
    (r"^invalid placement check (?P<check_id>\S+) requires exactly one mapped defect$", "placement.defect_mapping_cardinality", "defects"),
    (r"^placement defect references unknown check (?P<check_id>.+)$", "placement.unknown_check_reference", "check_id"),
    (r"^camera selector response contains forbidden or unknown fields:", "camera.response_fields_invalid", "response_fields"),
)
SCOPE_FIELDS = ("phase", "scope_id", "check_id", "subject_id", "group_id", "target_id", "field_path", "stop_reason")


def reason_detail(kind: str, message: str, *, stage=None, metric=None) -> dict:
    code = EXACT.get(message)
    fields = {}
    for pattern, candidate, field in PATTERNS:
        match = re.search(pattern, message)
        if match:
            code, fields = candidate, {**match.groupdict(), "field": field}
            break
    return {"error_type": kind, "reason_code": code or f"unclassified.{kind}",
            "classification_basis": "verified_validator_message" if code else "unclassified",
            "message": redact(message)[:2048], "stage": stage, "metric": metric, **fields}


def _ids(value):
    return [str(row.get("check_id") or "") for row in value if isinstance(row, dict)]


def check_details(arguments: dict, field: str) -> dict:
    required, result = arguments.get("required_checks"), arguments.get("result")
    if not isinstance(required, list):
        return {"input_details_available": False}
    rows = result.get(field) if isinstance(result, dict) else None
    expected = _ids(required)
    returned = _ids(rows) if isinstance(rows, list) else []
    e, a = set(expected) - {""}, set(returned) - {""}
    duplicates = sorted(k for k, n in Counter(returned).items() if k and n > 1)
    def bounded(values): return [redact(str(v))[:256] for v in sorted(values)[:256]]
    violations = []
    if e - a: violations.append("missing_required_check_ids")
    if a - e: violations.append("unexpected_check_ids")
    if duplicates: violations.append("duplicate_check_ids")
    if "" in returned: violations.append("empty_check_id")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        violations.append("results_missing_or_wrong_shape")
    return {"input_details_available": True, "field": field, "violations": violations,
            "required_check_count": len(required), "returned_row_count": len(rows) if isinstance(rows, list) else None,
            "missing_count": len(e - a), "unexpected_count": len(a - e), "duplicate_id_count": len(duplicates),
            "missing_check_ids": bounded(e - a), "unexpected_check_ids": bounded(a - e),
            "duplicate_check_ids": bounded(duplicates),
            "id_lists_truncated": max(len(e - a), len(a - e), len(duplicates)) > 256}


def wrap_validator(function, current, *, check_field=None):
    @functools.wraps(function)
    def observed(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError) as exc:
            ctx = current.get()
            if ctx is not None:
                try:
                    detail = reason_detail(type(exc).__name__, str(exc), stage="validation")
                    detail["validator"] = {"module": function.__module__, "function": function.__name__}
                    detail["exception_chain"] = exception_record(exc)["exception_chain"]
                    if check_field:
                        bound = inspect.signature(function).bind(*args, **kwargs)
                        detail["check_inventory"] = check_details(bound.arguments, check_field)
                    key = identity({"type": type(exc).__name__, "message": str(exc)})
                    entries = ctx.validator_origins.setdefault(key, [])
                    if detail not in entries and len(entries) < 16:
                        entries.append(detail)
                    best_effort_record((ctx.attempt or ctx.root) / "execution_diagnostics" /
                                       f"validator_{identity(detail)}.json", detail)
                except Exception:
                    # Diagnosis itself must never replace the original error.
                    pass
            raise
    return observed


def install(current):
    targets = (
        ("benchmark.evaluator.scene_quality.placement_checks", "validate_placement_check_results", "placement_check_results"),
        ("benchmark.evaluator.scene_quality.functional_checks", "validate_functional_check_results", "functional_check_results"),
        ("benchmark.visual_judge.openai_camera_selector", "_validated_payload", None),
        ("benchmark.visual_judge.openai_camera_selector", "_validated_response", None),
        ("benchmark.visual_judge.openai_camera_selector", "_validated_evidence_readiness_response", None),
    )
    for module_name, name, field in targets:
        module = importlib.import_module(module_name)
        original = getattr(module, name)
        wrapped = wrap_validator(original, current, check_field=field)
        # Update already imported aliases as well as the definition used by
        # later lazy imports. Match by object identity, not merely function name.
        for loaded_name, loaded in list(sys.modules.items()):
            if loaded is not None and loaded_name.startswith("benchmark.") and getattr(loaded, name, None) is original:
                setattr(loaded, name, wrapped)


def describe(raw, ctx, *, stage, metric=None) -> list[dict]:
    rows = []
    def walk(value, scope):
        if isinstance(value, dict):
            scope = {**scope, **{k: value[k] for k in SCOPE_FIELDS if k in value}}
            kind = value.get("error_type") or value.get("validation_error_type")
            message = value.get("error") or value.get("validation_error")
            if kind and isinstance(message, str):
                row = {**reason_detail(str(kind), message, stage=stage, metric=metric), **scope}
                key = identity({"type": str(kind), "message": message})
                matches = ctx.validator_origins.get(key, [])
                if matches:
                    row["validator_observations"] = matches
                    row["observation_match"] = "unique_type_message_in_attempt" if len(matches) == 1 else "ambiguous_type_message_candidates"
                if row not in rows and len(rows) < 64: rows.append(row)
            elif value.get("stop_reason") and len(rows) < 64:
                stop = redact(str(value["stop_reason"]))[:256]
                row = {"reason_code": f"stop.{stop}", "classification_basis": "observed_stop_reason",
                       "stage": stage, "metric": metric, **scope}
                if row not in rows: rows.append(row)
            for k, child in value.items():
                if k not in {"raw_response", "raw_request", "request_metadata", "frames"}:
                    walk(child, scope)
        elif isinstance(value, (list, tuple)):
            for child in value: walk(child, scope)
    walk(raw, {})
    return rows
