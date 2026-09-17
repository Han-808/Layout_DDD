"""Versioned, opt-in evidence resolution; never an implicit valid verdict.

The acquisition controller owns budgets. Metric evaluators own conclusions.
This module transports evidence/provenance and distinguishes model judgement
from the sole authorised no-deduction policy (unobservable Style).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


ADAPTIVE_POLICY = "evidence_adaptive_judgement_v1"
FALLBACK_POLICY = "evidence_consistency_fallback_v2"
LEGACY_POLICY = "legacy"
RESOLUTION_SCHEMA = "adaptive_evidence_resolution_v1"
ADAPTIVE_METRICS = frozenset({
    "collision", "support", "oob", "scale_consistency",
    "object_pairing_consistency", "style_consistency",
    "functional_consistency", "semantic_placement_consistency",
})
_POLICY: ContextVar[str | None] = ContextVar("evidence_resolution_policy", default=None)

TERMINAL_INSTRUCTION = """This is a terminal best-available-evidence judgement.
No further images can be acquired. Use every relevant supplied observation,
trusted asset fact and metric-owned measurement; do not discard an available
image merely because the calibrated packet is incomplete. Apply the original
metric definition and defect thresholds. Missing evidence itself is neither
validity nor invalidity. Choose the best-supported valid/invalid conclusion,
distinguish observations from inferences, and state remaining uncertainty in
confidence and reason. Do not pretend that an unobserved material, functional
side, contact or surface is visible. Resolve every exact required check once;
use inferred_under_budget for inferred rows, not a fabricated observation.
Do not request further evidence. evidence_status=sufficient denotes completion
of this constrained judgement, not complete visual observation."""


class AdaptiveEvidenceError(RuntimeError):
    failure_category = "evidence_resolution_failure"

    def __init__(self, message: str, *, audit: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit = deepcopy(audit or {})


class EvidenceUnavailableError(AdaptiveEvidenceError):
    failure_category = "evidence_unavailable"


class EvidenceIntegrityError(AdaptiveEvidenceError):
    failure_category = "input_integrity_failure"


class ContextCapacityError(AdaptiveEvidenceError):
    failure_category = "context_capacity_failure"


class AdaptiveJudgeError(AdaptiveEvidenceError):
    failure_category = "judge_response_failure"


def _declared_policy(value: Any) -> str | None:
    """Read a declaration, without mistaking a nested Judge default for a call override."""
    if isinstance(value, Mapping):
        policy = value.get("evidence_resolution_policy")
    else:
        control = getattr(value, "control", None)
        policy = (control.get("evidence_resolution_policy") if isinstance(control, Mapping)
                  else getattr(control, "evidence_resolution_policy", None))
        if policy is None:
            policy = getattr(value, "evidence_resolution_policy", None)
    if policy is None:
        return None
    if policy not in {LEGACY_POLICY, ADAPTIVE_POLICY, FALLBACK_POLICY}:
        raise ValueError(f"unsupported evidence_resolution_policy: {policy!r}")
    return str(policy)


def policy_of(value: Any = None) -> str:
    """Observe the effective call policy; a bound call also overrides legacy defaults."""
    scoped = _POLICY.get()
    return scoped if scoped is not None else (_declared_policy(value) or LEGACY_POLICY)


def resolve_call_evidence_policy(control: Any = None, judge: Any = None) -> str:
    """Explicit call control > enclosing call > controlled/raw Judge > legacy.

    A budget-only patch inherits policy; explicitly spelling ``legacy`` is an
    actual override. Shared Judge objects are never mutated to bind a call.
    """
    return _declared_policy(control) or policy_of(judge)


def adaptive_enabled(value: Any = None, *, metric: str | None = None) -> bool:
    return policy_of(value) in {ADAPTIVE_POLICY, FALLBACK_POLICY} and (
        metric is None or metric in ADAPTIVE_METRICS
    )


@contextmanager
def evidence_policy_scope(value: Any = None) -> Iterator[None]:
    token = _POLICY.set(_declared_policy(value) or policy_of())
    try:
        yield
    finally:
        _POLICY.reset(token)


def with_evidence_policy(function: Callable[..., Any]) -> Callable[..., Any]:
    """Bind per-call policy without mutating injected/shared judge objects."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs).arguments
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                bound = {**bound, **bound.get(name, {})}
        source = bound.get("vlm_judge", bound.get("judge"))
        if source is None:
            source = bound.get("self")
        policy = resolve_call_evidence_policy(bound.get("vlm_evaluation_control"), source)
        with evidence_policy_scope({"evidence_resolution_policy": policy}):
            return function(*args, **kwargs)

    return wrapped


def finite_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    )


def failure_record(error: BaseException, *, phase: str) -> dict[str, Any]:
    """Classify by structured category/type and phase, never arbitrary prose."""
    if policy_of() == FALLBACK_POLICY:
        from benchmark.visual_judge.evidence_gap_v2 import classify_failure
        return classify_failure(error, phase=phase)
    category = getattr(error, "failure_category", None)
    names = {cls.__name__ for cls in type(error).__mro__}
    response_failure = bool(names & {"ResponseSchemaRepairError", "JSONDecodeError"})
    if not category and response_failure:
        category = "judge_response_failure"
    if error.__cause__ is not None and error.__cause__ is not error:
        cause = failure_record(error.__cause__, phase=phase)
        # Response validation intentionally wraps ValueError/TypeError/KeyError.
        # That generic cause must not erase the response boundary; a typed
        # service/integrity failure during repair must still pass through.
        generic_response_cause = (
            response_failure
            and cause["failure_category"] == "implementation_or_input_failure"
        )
        if not category or (not cause["recoverable_acquisition"] and not generic_response_cause):
            category = cause["failure_category"]
    if not category:
        if names & {"ResponseSchemaRepairError", "JSONDecodeError"}:
            category = "judge_response_failure"
        elif isinstance(error, (TimeoutError, ConnectionError)) or names & {
            "HTTPError", "URLError", "AuthenticationError", "PermissionError",
            "EndpointPreflightError", "ModelRequestError", "OpenAICompatibleError",
            "OpenAICompatibleModelError", "EndpointStabilityPreflightError",
        }:
            category = "model_service_failure"
        elif phase == "acquisition" and (
            names & {"NonRectangularCameraEvidenceExhausted", "EvidenceControlUnresolvedError"}
            or type(error) in {RuntimeError, FileNotFoundError, OSError}
        ):
            category = "evidence_unavailable"
        else:
            category = "implementation_or_input_failure"
    return {
        "failure_category": str(category), "phase": phase,
        "error_type": type(error).__name__,
        # Exception prose can contain transport URLs/credentials; do not persist it.
        "recoverable_acquisition": category == "evidence_unavailable",
    }


def adaptive_provider_payload(value: Any) -> tuple[list[Any], dict[str, Any]]:
    """Preserve returned artifacts even when acquisition is only partially available.

    This parses a provider's evidence/status, not the metric verdict. Callers
    still validate each image and account for the existing acquisition budget.
    """
    status = "available"
    category = None
    if isinstance(value, Mapping):
        status = str(value.get("status") or "available").strip().lower()
        failure = value.get("failure") or {}
        if not isinstance(failure, Mapping):
            raise EvidenceIntegrityError("provider failure record must be an object")
        category = value.get("failure_category") or failure.get("failure_category")
        if policy_of() == FALLBACK_POLICY and category is None and (
            status in {"failed", "error"} or value.get("error")
        ):
            category = "unclassified_provider_failure"
        recoverable = {None, "evidence_gap", "evidence_unavailable", "acquisition_unavailable",
                       "no_feasible_camera"}
        if policy_of() != FALLBACK_POLICY:
            recoverable.add("render_failure")
        if category not in recoverable:
            error = AdaptiveEvidenceError("provider reported a non-recoverable failure", audit={
                "failure_category": str(category), "phase": "acquisition",
                "recoverable_acquisition": False,
                "available_items": deepcopy(next((value[key] for key in (
                    "visual_evidence", "render_evidence_items", "render_evidence", "paths",
                ) if isinstance(value.get(key), (list, tuple)) and value[key]), [])),
            })
            error.failure_category = str(category)
            raise error
        values = next((value[key] for key in (
            "visual_evidence", "render_evidence_items", "render_evidence", "paths",
        ) if isinstance(value.get(key), (list, tuple)) and value[key]), [])
        incomplete = status in {"failed", "error", "insufficient", "unavailable", "not_available", "partial"} or bool(value.get("error")) or category is not None
    elif isinstance(value, (list, tuple)):
        values, incomplete = value, False
    elif isinstance(value, (str, Path)):
        values, incomplete = [value], False
    else:
        raise EvidenceIntegrityError("provider evidence must be an object, list or path")
    audit = {"provider_status": status, "packet_incomplete": bool(incomplete or not values)}
    if incomplete or not values:
        audit["failure"] = {"failure_category": "evidence_unavailable", "phase": "acquisition",
                            "recoverable_acquisition": True}
    return deepcopy(list(values)), audit


def visual_references(paths: list[Any], metadata: list[Any]) -> list[Any]:
    """Rebind path projections without dropping duplicate/conflicting provenance claims."""
    def reference(value: Any) -> str | None:
        raw = value.get("path") or value.get("image_path") if isinstance(value, Mapping) else value
        return str(Path(raw).expanduser().resolve()) if isinstance(raw, (str, Path)) and str(raw).strip() else None

    records: dict[str, list[dict[str, Any]]] = {}
    for item in metadata:
        if isinstance(item, dict) and (key := reference(item)) is not None:
            records.setdefault(key, []).append(item)
    result: list[Any] = []
    for item in paths:
        matches = records.get(reference(item) or "", [])
        if isinstance(item, dict) or not matches:
            result.append(deepcopy(item))
        result.extend(deepcopy(matches))
    return result


def usable_visuals(
    values: list[Any], *, target_ids: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Quarantine unusable artifacts, preserving exact paths and real roles."""
    from PIL import Image, ImageStat

    accepted: list[str] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        metadata = value if isinstance(value, dict) else {}
        raw = metadata.get("path") or metadata.get("image_path") if metadata else value
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            excluded.append({"reason": "missing_artifact_path"})
            continue
        path = Path(raw).expanduser()
        reference = str(path.resolve())
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = metadata.get("sha256") or metadata.get("content_sha256")
            if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise EvidenceIntegrityError("artifact hash mismatch")
            explicit_ids = metadata.get("target_ids") or metadata.get("object_ids")
            if explicit_ids and (not isinstance(explicit_ids, (list, tuple)) or
                                 not all(isinstance(item, str) and item.strip() for item in explicit_ids)):
                raise EvidenceIntegrityError("image target IDs must be a list of non-empty strings")
            if explicit_ids and target_ids and not set(map(str, explicit_ids)) & set(target_ids):
                excluded.append({"path": reference, "reason": "unrelated_target_scope"})
                continue
            with Image.open(path) as visual:
                visual.load()
                if visual.width < 2 or visual.height < 2:
                    excluded.append({"path": reference, "reason": "empty_visual"})
                    continue
                extrema = ImageStat.Stat(visual.convert("RGB")).extrema
                if all(high == low for low, high in extrema):
                    excluded.append({"path": reference, "reason": "blank_visual"})
                    continue
        except EvidenceIntegrityError:
            # Hash drift is a provenance failure, not merely a missing view.
            raise
        except (OSError, ValueError) as error:
            if policy_of() == FALLBACK_POLICY:
                raise EvidenceIntegrityError("supplied visual artifact is missing or undecodable", audit={
                    "path": reference, "error_type": type(error).__name__,
                }) from None
            excluded.append({"path": reference, "reason": "missing_or_undecodable_visual",
                             "error_type": type(error).__name__})
            continue
        if reference not in seen:
            accepted.append(reference)
            seen.add(reference)
    return accepted, excluded


def request_target_ids(request: Mapping[str, Any]) -> list[str]:
    candidates = request.get("target_object_ids") or request.get("selected_object_ids")
    scope = request.get("target_scope")
    if not candidates and isinstance(scope, Mapping):
        candidates = scope.get("target_ids") or scope.get("object_ids")
    event = request.get("event") or {}
    if not candidates and isinstance(event, Mapping):
        candidates = event.get("object_ids") or [event.get("object_id"),
                                                     event.get("object_a"), event.get("object_b")]
    candidates = [item for item in candidates or [] if item is not None and str(item).strip()]
    if not candidates:
        candidates = [item.get("id") for item in _scene_from_request(request).get("objects", [])
                      if isinstance(item, Mapping)]
    return list(dict.fromkeys(str(item) for item in candidates or [] if item is not None and str(item)))


def request_evidence_target_ids(request: Mapping[str, Any]) -> list[str]:
    """Camera context may include real scene entities outside the scored event.

    Match the controller's existing target authority, without treating arbitrary
    metadata or model-authored IDs as scene entities. Scoring targets remain
    exclusively those returned by request_target_ids.
    """
    from benchmark.visual_judge.camera_targets import merge_authoritative_target_ids

    return list(merge_authoritative_target_ids(request_target_ids(request), _scene_from_request(request)))


def _scene_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("camera_scene_context", "scene_summary", "scene"):
        if isinstance(request.get(key), dict):
            return request[key]
    architecture = request.get("architecture") or {}
    return {"objects": request.get("objects") or [],
            "boundary": architecture.get("boundary"),
            "scene_height": architecture.get("height")}


def _finite_vector(value: Any, *, positive: bool = False) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        and math.isfinite(float(item)) and (not positive or item > 0) for item in value
    )


def structured_evidence(request: Mapping[str, Any]) -> dict[str, Any]:
    from benchmark.evaluator.context_projection import project_scene_for_evaluator_context

    metric = str(request.get("metric") or "")
    scene = project_scene_for_evaluator_context(deepcopy(_scene_from_request(request)))
    targets = request_target_ids(request)
    known_ids = [str(item.get("id")) for item in scene.get("objects") or [] if isinstance(item, dict) and item.get("id")]
    if len(known_ids) != len(set(known_ids)):
        raise EvidenceIntegrityError("duplicate canonical object IDs")
    unknown_ids = sorted(set(targets) - set(known_ids) - {"scene"})
    if unknown_ids:
        raise EvidenceIntegrityError("requested target IDs are absent from canonical facts", audit={"unknown_target_ids": unknown_ids})
    objects: list[dict[str, Any]] = []
    appearance: list[dict[str, Any]] = []
    for raw in scene.get("objects") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        item = {key: deepcopy(raw[key]) for key in (
            "id", "category", "retrieval_category", "description", "desc", "short_desc",
            "center", "size", "rotation", "asset_proxy", "semantic_role",
        ) if raw.get(key) is not None}
        for vector in ("center", "size", "rotation"):
            if vector in item and not _finite_vector(item[vector], positive=vector == "size"):
                raise EvidenceIntegrityError(f"non-finite or invalid canonical {vector}")
        # Appearance must have a traceable asset source, not a generator style wish.
        for source_key in ("asset_metadata", "catalog_metadata", "asset_proxy"):
            source = raw.get(source_key)
            if not isinstance(source, dict):
                continue
            attrs = {key: deepcopy(source[key]) for key in (
                "material", "materials", "color", "colors", "palette", "style_tags",
                "form_description", "shape_description", "design_era",
            ) if source.get(key)}
            if attrs:
                appearance.append({"object_id": str(raw["id"]), "source": source_key,
                                   "attributes": attrs})
        objects.append(item)
    probe = request.get("functional_probe_evidence")
    probe = probe if isinstance(probe, Mapping) else {}
    functional_measurements = deepcopy(request.get("functional_measurements") or {})
    if policy_of(request) == FALLBACK_POLICY:
        functional_measurements = functional_measurements or deepcopy(probe.get("functional_measurements") or {})
        check_ids = {row.get("check_id") for row in request.get("required_functional_checks") or probe.get("required_checks") or []
                     if isinstance(row, Mapping)}
        if isinstance(functional_measurements, dict):
            functional_measurements["check_measurements"] = [
                row for row in functional_measurements.get("check_measurements") or []
                if isinstance(row, dict) and row.get("check_id") in check_ids
                and row.get("status") in {"complete", "partial"}
            ]
    packet = {
        "schema_version": "metric_structured_evidence_v1", "metric": metric,
        "target_ids": targets, "objects": objects,
        "room": {"boundary": deepcopy(scene.get("boundary")),
                 "scene_height": scene.get("scene_height"),
                 "room_type": scene.get("room_type") or scene.get("scene_type")},
        "measurements": deepcopy(request.get("detector_evidence") or
                                 request.get("deterministic_evidence") or {}),
        "functional_measurements": functional_measurements,
        "asset_appearance": appearance,
        "geometry_is_proxy_unless_measurement_provenance_says_otherwise": True,
        "generator_private_intent_removed": True,
    }
    packet["source_sha256"] = hashlib.sha256(json.dumps(
        packet, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return packet


def has_related_facts(packet: Mapping[str, Any]) -> bool:
    objects = packet.get("objects") or []
    if packet.get("metric") == "style_consistency":
        return bool(packet.get("asset_appearance"))
    if packet.get("metric") == "object_pairing_consistency":
        return bool(objects) and all(any(obj.get(key) for key in (
            "category", "retrieval_category", "description", "desc", "semantic_role",
        )) for obj in objects)
    if packet.get("measurements") or packet.get("functional_measurements"):
        return bool(objects)
    relevant = [obj for obj in objects if not packet.get("target_ids") or obj.get("id") in packet["target_ids"]]
    return bool(relevant) and all(
        _finite_vector(obj.get("center")) and _finite_vector(obj.get("size"), positive=True)
        for obj in relevant
    )


def terminal_structured_facts(packet: Mapping[str, Any]) -> bool:
    """Admit a metric-owned text-only judgement, never certify its verdict.

    Proxy dimensions can inform Scale, but cannot establish mesh contact,
    functional visibility, appearance, or a semantic Placement conclusion.
    Those metrics require their own measurements or asset-grounded facts.
    Existing response/check validation still owns acceptance and may keep gaps.
    """
    metric = packet.get("metric")
    targets = set(packet.get("target_ids") or [])
    objects = [obj for obj in packet.get("objects") or [] if obj.get("id") in targets]
    if not objects:
        return False
    if metric == "style_consistency":
        return any(item.get("object_id") in targets for item in packet.get("asset_appearance") or [])
    if metric == "scale_consistency":
        return all(_finite_vector(obj.get("size"), positive=True)
                   and (obj.get("category") or obj.get("retrieval_category")) for obj in objects)
    if metric == "object_pairing_consistency":
        return all(obj.get("category") or obj.get("retrieval_category") for obj in objects)
    if metric == "functional_consistency":
        return bool((packet.get("functional_measurements") or {}).get("check_measurements"))
    return bool(packet.get("measurements"))


def prepare_adaptive_request(
    request: Mapping[str, Any], *, terminal: bool, trigger: str,
) -> dict[str, Any]:
    value = deepcopy(dict(request))
    policy = FALLBACK_POLICY if policy_of(request) == FALLBACK_POLICY else ADAPTIVE_POLICY
    value["evidence_resolution_policy"] = policy
    value["adaptive_terminal"] = bool(terminal)
    failure = value.get("acquisition_failure") or {}
    if policy == FALLBACK_POLICY and failure and not failure.get("recoverable_acquisition", False):
        error = AdaptiveEvidenceError("acquisition failed; terminal judgement is prohibited", audit={
            "failure": deepcopy(failure), "available_images": deepcopy(value.get("render_evidence") or []),
        })
        error.failure_category = str(failure.get("failure_category") or "implementation_or_input_failure")
        raise error
    candidates = list(value.get("render_evidence") or [])
    if terminal and policy == FALLBACK_POLICY:
        # These were already acquired, but omitted by an earlier packet budget.
        # Preserve the anchor, then let the adapter prioritize unsent local views.
        candidates.extend(value.get("adaptive_unselected_images") or [])
    visual_refs = visual_references(candidates,
                                   list(value.get("local_render_evidence_metadata") or []))
    paths, excluded = usable_visuals(visual_refs, target_ids=request_target_ids(value))
    excluded = deepcopy((value.get("adaptive_evidence") or {}).get("excluded_images") or []) + excluded
    excluded.extend(deepcopy(value.get("adaptive_excluded_images") or []))
    if policy == FALLBACK_POLICY and any(item.get("reason") in {
        "missing_artifact_path", "missing_or_undecodable_visual",
    } for item in excluded):
        raise EvidenceIntegrityError("supplied visual artifact is missing or undecodable",
                                     audit={"excluded_images": excluded})
    value["render_evidence"] = paths
    packet = structured_evidence(value)
    tier = "structured_fallback" if not paths else (
        "partial_visual" if terminal or excluded or value.get("adaptive_packet_incomplete")
        else "full_visual"
    )
    value["adaptive_evidence"] = {
        "schema_version": RESOLUTION_SCHEMA, "policy": policy,
        "metric": str(value.get("metric") or ""), "evidence_tier": tier,
        "terminal": bool(terminal), "trigger_reason": str(trigger),
        "available_images": list(dict.fromkeys(paths + list(value.get("adaptive_unselected_images") or []))), "excluded_images": excluded,
        "structured_evidence": packet,
        "missing_observations": deepcopy(value.get("adaptive_missing_observations") or []),
    }
    # Do not simultaneously activate historical geometry/default-valid forcing.
    if terminal:
        value.pop("structured_geometry_finalization", None)
        value.pop("budget_exhaustion_finalization", None)
        if policy == FALLBACK_POLICY and not paths and not terminal_structured_facts(packet):
            from benchmark.visual_judge.evidence_gap_v2 import EvidenceGapError
            # No images is not itself a reason to skip judgement. The metric's
            # evidence boundary, not generic bounding boxes, controls admission.
            raise EvidenceGapError(
                "no_usable_visual_evidence_for_remaining_judgement",
                audit=value["adaptive_evidence"],
            )
        if not paths and value.get("metric") == "style_consistency" and not packet.get("objects"):
            raise EvidenceUnavailableError("Style policy cannot replace a missing canonical object inventory")
        if not paths and value.get("metric") != "style_consistency" and not has_related_facts(packet):
            raise EvidenceUnavailableError("no metric-relevant evidence for terminal judgement",
                                           audit=value["adaptive_evidence"])
    return value


def bind_acquisition_resolution(request: dict[str, Any], resolution: Mapping[str, Any]) -> None:
    if not adaptive_enabled(request):
        return
    incomplete = (not resolution.get("scope_satisfied") or bool(resolution.get("missing_paths"))
                  or bool(resolution.get("excluded_images"))
                  or resolution.get("provider_status") in {"failed", "insufficient"})
    request["adaptive_packet_incomplete"] = bool(request.get("adaptive_packet_incomplete") or incomplete)
    request["adaptive_missing_observations"] = list(dict.fromkeys(
        list(request.get("adaptive_missing_observations") or []) + (["calibrated_visual_packet_incomplete"] if incomplete else [])
    ))
    request["adaptive_unselected_images"] = list(dict.fromkeys(
        list(request.get("adaptive_unselected_images") or []) + list(resolution.get("images_not_delivered") or [])
    ))
    request["adaptive_excluded_images"] = list(request.get("adaptive_excluded_images") or []) + [
        {"path": path, "reason": "missing_or_undecodable_visual"} for path in resolution.get("missing_paths") or []
    ]
    request["adaptive_excluded_images"].extend(deepcopy(resolution.get("excluded_images") or []))
    request["local_render_evidence_metadata"] = list(request.get("local_render_evidence_metadata") or []) + deepcopy(resolution.get("evidence_metadata") or [])
    if resolution.get("failure"):
        request["acquisition_failure"] = deepcopy(resolution["failure"])
        if policy_of(request) == FALLBACK_POLICY:
            request["adaptive_terminal"] = True
            request["adaptive_trigger"] = "metric_acquisition_exhausted"


def bind_provider_acquisition(request: dict[str, Any], acquisition: Mapping[str, Any]) -> None:
    """Apply the same evidence audit to initial acquisition and internal repair."""
    bind_acquisition_resolution(request, {
        "scope_satisfied": not acquisition.get("packet_incomplete", False),
        "provider_status": "insufficient" if acquisition.get("packet_incomplete") else "available",
        # The final Judge audit, not the acquisition selector, determines which
        # available images were actually delivered (including budget rejection).
        "images_not_delivered": list(acquisition.get("available_paths") or []) + list(acquisition.get("images_not_delivered") or []),
        "evidence_metadata": acquisition.get("evidence_metadata") or [],
        "excluded_images": acquisition.get("excluded_images") or [],
        "failure": acquisition.get("failure"),
    })


def style_policy_result(request: Mapping[str, Any]) -> dict[str, Any] | None:
    if policy_of(request) == FALLBACK_POLICY:
        return None
    adaptive = request.get("adaptive_evidence") or {}
    if (
        request.get("metric") != "style_consistency" or not adaptive.get("terminal")
        or request.get("render_evidence")
        or has_related_facts(adaptive.get("structured_evidence") or {})
    ):
        return None
    return attach_resolution({
        "evidence_status": "sufficient", "verdict": "valid", "confidence": None,
        "reason": "Style has no usable appearance evidence; the explicit policy makes no deduction.",
        "missing_evidence": [], "defects": [], "evidence_request": None,
        "images_used": [],
    }, request, source="style_policy_no_deduction", model_invoked=False)


def attach_resolution(
    result: dict[str, Any], request: Mapping[str, Any], *, source: str = "model",
    model_invoked: bool = True,
) -> dict[str, Any]:
    value = deepcopy(result)
    evidence = request.get("adaptive_evidence") or {}
    if not evidence:
        return value
    accepted = value.get("verdict", value.get("status")) in {"valid", "invalid"}
    actual_images = list(value.get("images_used", request.get("render_evidence") or []))
    available = list(evidence.get("available_images") or [])
    packet = evidence.get("structured_evidence") or {}
    identity = {
        "metric": request.get("metric"), "target_ids": packet.get("target_ids"),
        "phase": request.get("evidence_phase"), "event": request.get("event"),
        "group_ids": request.get("selected_group_ids") or request.get("group_ids"),
    }
    schema_audit = (value.get("request_metadata") or {}).get("response_schema_validation") or {}
    call_count = int(schema_audit.get("attempt_count") or 1) if model_invoked else 0
    if model_invoked and (value.get("request_metadata") or {}).get("atomic_check_sharding"):
        call_count = int((value.get("evidence_resolution") or {}).get("model_call_count") or call_count)
    value["evidence_resolution"] = {
        "schema_version": RESOLUTION_SCHEMA, "policy": evidence.get("policy", ADAPTIVE_POLICY),
        "unit_key": hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest(),
        "metric": str(request.get("metric") or ""),
        "evidence_tier": evidence.get("evidence_tier"), "decision_source": source,
        "fallback": evidence.get("evidence_tier") != "full_visual" or source == "style_policy_no_deduction",
        "accepted": accepted, "model_invoked": bool(model_invoked),
        "model_judgement_completed": bool(model_invoked and accepted),
        "model_call_count": call_count,
        "visual_observation_complete": evidence.get("evidence_tier") == "full_visual" and len(actual_images) == len(available),
        "trigger_reason": evidence.get("trigger_reason"),
        "images_used": actual_images,
        "images_not_delivered": [path for path in available if path not in actual_images],
        "excluded_images": deepcopy(evidence.get("excluded_images") or []),
        "missing_observations": deepcopy(evidence.get("missing_observations") or [])
        + (["style_appearance_unavailable"] if source == "style_policy_no_deduction" else []),
        "structured_source_sha256": packet.get("source_sha256"),
        "target_ids": list(packet.get("target_ids") or []),
        "inference_under_constraints": evidence.get("evidence_tier") != "full_visual",
    }
    value["decision_source"] = source
    from .best_effort_terminal import terminal as best_effort_terminal, POLICY as TERMINAL_POLICY
    if best_effort_terminal(request):
        value["evidence_resolution"]["terminal_evidence_policy"] = TERMINAL_POLICY
        value["evidence_resolution"]["decision_retry_count"] = int(schema_audit.get("decision_retry_count") or 0)
    for row in value.get("functional_check_results") or []:
        if isinstance(row, dict):
            row["evidence_resolution"] = deepcopy(value["evidence_resolution"])
            row["evidence_resolution"]["unit_key"] = str(row.get("check_id") or "")
    # Placement rows have an exact model contract, revalidated by every
    # evaluator scope after this audit is attached. Keep evaluator-owned
    # provenance outside those rows; do not strip or whitelist model extras.
    # Keep the sidecar outside the scope audit too: coverage expands that
    # audit per check, so nesting this map there would cause quadratic copies.
    # Rebuild it rather than trusting a model-supplied or stale audit map.
    value.pop("placement_check_resolutions", None)
    placement_audits: dict[str, dict[str, Any]] = {}
    for row in value.get("placement_check_results") or []:
        if isinstance(row, dict):
            check_id = str(row.get("check_id") or "")
            placement_audits[check_id] = {
                **deepcopy(value["evidence_resolution"]), "unit_key": check_id,
            }
    if placement_audits:
        value["placement_check_resolutions"] = placement_audits
    return value


def resolution_of(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    record = value.get("evidence_resolution")
    if isinstance(record, dict) and record.get("policy") in {ADAPTIVE_POLICY, FALLBACK_POLICY}:
        return record
    for key in ("judgement", "judge_result"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            record = resolution_of(nested)
            if record is not None:
                return record
    return None


def resolution_accepted(value: Any) -> bool:
    resolution = resolution_of(value)
    return bool(resolution and resolution.get("accepted") is True)


def bind_resolution(record: dict[str, Any], judgement: Mapping[str, Any]) -> None:
    resolution = resolution_of(judgement)
    if resolution is not None:
        record["evidence_resolution"] = deepcopy(resolution)
        record["evidence_resolution_policy"] = resolution["policy"]


__all__ = [
    "ADAPTIVE_POLICY", "LEGACY_POLICY", "ADAPTIVE_METRICS", "TERMINAL_INSTRUCTION",
    "AdaptiveEvidenceError", "EvidenceUnavailableError", "EvidenceIntegrityError",
    "ContextCapacityError", "AdaptiveJudgeError", "adaptive_enabled", "policy_of",
    "with_evidence_policy", "evidence_policy_scope", "failure_record", "finite_score",
    "usable_visuals", "request_target_ids", "structured_evidence", "has_related_facts",
    "prepare_adaptive_request", "style_policy_result", "attach_resolution",
    "resolution_of", "resolution_accepted", "bind_resolution",
]
