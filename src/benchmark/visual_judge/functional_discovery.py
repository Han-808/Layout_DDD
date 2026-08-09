"""Strict, non-judging discovery for functional visual evidence.

The discovery model answers *what should be inspected*, never whether the
scene is valid.  Object-to-group ownership is recomputed from the trusted
grouping partition so a VLM cannot silently redefine evidence scopes.
"""

from __future__ import annotations

import base64
import json
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from benchmark.visual_judge.functional_discovery_contract import (
    FUNCTIONAL_AFFORDANCE_PROMPT_VERSION,
    FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION,
    FUNCTIONAL_DIRECTIONALITY,
    FUNCTIONAL_DISCOVERY_PROMPT_VERSION,
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
    FUNCTIONAL_RELATION_PREDICATES,
    FUNCTIONAL_RELATION_PROMPT_VERSION,
    FUNCTIONAL_RELATION_SCHEMA_VERSION,
    FUNCTIONAL_REVIEW_STATES,
    FUNCTIONAL_SURFACE_ROLES,
    FunctionalDiscoveryResult,
)
from benchmark.visual_judge.functional_discovery_validation import (
    compose_functional_discovery_result,
    validate_functional_affordance_response,
    validate_functional_discovery_request,
    validate_functional_discovery_response,
    validate_functional_relation_response,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


FUNCTIONAL_DISCOVERY_MAX_TOKENS = 4096
FUNCTIONAL_DISCOVERY_REPAIR_POLICY = (
    "single_nonjudging_discovery_contract_repair_v1"
)
_RAW_RESPONSE_LIMIT = 20_000


FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT = """Inventory functional observability;
do not judge validity. For every input object, classify ordinary use as
directed or non_directed. Directed means intended use depends on a particular
functional side; non_directed covers direction-independent use and objects
without an ordinary operation. This classification asks whether a functional
side exists, not which physical side it is. If a directional use is plausible
but the exact side is not visible, use directed; later usable-side localization
resolves the side. Record only the allowed surface roles and whether the
object's own ordinary use requires a dedicated free-space region. Separately
record whether an already-required directed-side or clearance observation
must include the supplied logical room boundary. Boundary context only refines
framing; it never creates an independent check. Therefore an object that is
both non_directed and need_clearance=false must use
boundary_review_state=routine. Category is a navigation hint; visible geometry
and appearance remain relevant.
Set need_clearance=true only when ordinary approach, opening, or operation
requires dedicated free space around the object. Do not classify the kind of
clearance. This is not generic room circulation: passive objects without a
dedicated user or operating zone normally use need_clearance=false.

The conditional fields must agree:
- directed requires one or more surface_roles;
- non_directed requires surface_roles=[];
- boundary_review_state=routine requires boundary_observation_goal="";
- a non-routine boundary_review_state requires a non-empty
  boundary_observation_goal and requires either directionality=directed or
  need_clearance=true.

Copy every object ID exactly once in input order. Return exactly:
{"objects":[{"object_id":"id",
"directionality":"directed",
"surface_roles":["access_side"],
"need_clearance":true,
"boundary_review_state":"routine",
"review_state":"routine",
"observation_goal":"neutral observable fact",
"boundary_observation_goal":""}],
"reason":"brief coverage summary"}

Use only the supplied vocabulary. Never return a defect, score, verdict, pose,
vector, camera action, or scene edit. Return no other fields."""


FUNCTIONAL_RELATION_SYSTEM_PROMPT = """Inventory direct ordinary joint-use
checks; do not judge the current arrangement. Every row is one atomic relation
between exactly two objects and names exactly one predicate:
- directional_correspondence: joint use depends on compatible functional-side
  or facing directions;
- relative_use_geometry: joint use depends on relative position, distance,
  reach, contact, or connection geometry.

Create a row only when ordinary joint use imposes that observable condition.
Broad cooperation, co-presence, similarity, style compatibility, or possible
usefulness together is insufficient. If one focal object has several direct
participants, emit one focal-participant row for each real direct relation; do
not emit one multi-object row or invent every pair in the set. The same pair
may occur once per predicate with distinct observation goals.

An object's own approach, opening, or operating free space is object-level
clearance in affordance_prior.need_clearance_object_ids, never an object
relation. Distance and group membership affect framing and scope only; the
trusted_group_partition is not semantic ground truth and cannot suppress a
real relation. Current bad geometry must not suppress the check that tests it.

The validated affordance_prior is a sparse positive hint, not a candidate whitelist.
Use it to sharpen checks, while allowing the images and full object
list to recover omitted or misclassified relations. Test each object for a
direct ordinary joint-use counterpart before returning none.

Copy every object ID exactly once in considered_object_ids. Return exactly:
{"considered_object_ids":["id"],
"relations":[{"target_ids":["id1","id2"],
"predicate":"directional_correspondence",
"observation_goal":"neutral atomic joint-use observation"}],
"reason":"brief coverage summary"}

Never return a defect, score, verdict, pose, vector, camera action, scene edit,
or object-level clearance relation. Return no other fields."""


_FUNCTIONAL_AFFORDANCE_REPAIR_PROMPT = """Your previous response violated the
functional affordance ledger contract. This is one contract-repair attempt
using exactly the same image and trusted object list; it is not a metric
judgment and must not output any defect, validity, score, camera action, or
scene edit.

Copy every object ID exactly once in the original order and preserve every
already compliant row. Resolve contradictory rows from the same visual
evidence:
- directed requires one or more allowed surface_roles;
- non_directed requires surface_roles=[];
- need_clearance must be a JSON boolean, never a clearance subtype;
- boundary_review_state=routine requires boundary_observation_goal="";
- non-routine boundary review requires a non-empty boundary_observation_goal.
- boundary context may enrich only an existing directed-side or clearance
  observation; non_directed with need_clearance=false must use routine.

Return exactly the affordance-ledger JSON object required by the original
request, with no additional fields. Return JSON only."""


_FUNCTIONAL_RELATION_REPAIR_PROMPT = """Your previous response violated the
functional relation audit contract. This is one contract-repair attempt using
exactly the same image and trusted object list; it is not a metric judgment.
Copy every trusted object ID exactly once in considered_object_ids and in the
original order. Preserve compliant atomic checks, remove duplicate
(target_ids, predicate) identities, use exactly two trusted target IDs per
relation, and use exactly one allowed predicate per row. Split a multi-object
set only into supported direct focal-participant relations; do not generate a
complete pair graph. Object-level
approach, opening, or operating clearance is not a relation. Do not output any
defect, validity, score, camera action, or scene edit. Return exactly the
relation-audit JSON object required by the original request, with no
additional fields. Return JSON only."""


def discover_openai_compatible_functional_evidence(
    *,
    model: OpenAICompatibleModel,
    request: dict[str, Any],
    max_context_chars: int = 30000,
    response_format_json: bool | None = None,
) -> dict[str, Any]:
    """Run ordered non-judging discovery audits and compose one result."""

    normalized = validate_functional_discovery_request(request)
    common = {
        "metric": "functional_consistency",
        "decision_authority": "none",
        "scene_access": "read_only",
        "scene_id": normalized.get("scene_id"),
        "scene_type": normalized.get("scene_type"),
        "object_list": deepcopy(normalized["objects"]),
        "identity_grounding": {
            "status": normalized["identity_grounding"],
            "image_role": "global_identity_overlay",
            "legend": deepcopy(normalized["identity_legend"]),
        },
    }
    use_json_response = (
        bool(getattr(model, "response_format_json", True))
        if response_format_json is None
        else bool(response_format_json)
    )
    affordance_raw, affordance_meta, affordance_messages = (
        _run_discovery_call(
            model=model,
            normalized=normalized,
            context={
                **common,
                **vlm_audit_metadata(
                    VLMRole.FUNCTIONAL_AFFORDANCE_DISCOVERY,
                    decision_contract=(
                        DecisionContract.FUNCTIONAL_AFFORDANCE_DISCOVERY
                    ),
                    judge_method="discover_functional_affordances",
                ),
                "role": "functional_affordance_ledger",
                "prompt_version": FUNCTIONAL_AFFORDANCE_PROMPT_VERSION,
                "schema_version": FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION,
                "architecture_context": deepcopy(
                    normalized["architecture_context"]
                ),
                "allowed_surface_roles": sorted(
                    FUNCTIONAL_SURFACE_ROLES
                ),
                "allowed_directionality": sorted(
                    FUNCTIONAL_DIRECTIONALITY
                ),
                "clearance_contract": {
                    "field": "need_clearance",
                    "type": "boolean",
                    "semantics": (
                        "ordinary use requires a dedicated free-space region"
                    ),
                },
                "allowed_review_states": sorted(
                    FUNCTIONAL_REVIEW_STATES
                ),
            },
            system_prompt=FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT,
            call_type=(
                "vlm_camera_pose.functional_discovery.affordance"
            ),
            role=VLMRole.FUNCTIONAL_AFFORDANCE_DISCOVERY,
            decision_contract=(
                DecisionContract.FUNCTIONAL_AFFORDANCE_DISCOVERY
            ),
            max_context_chars=max_context_chars,
            response_format_json=use_json_response,
        )
    )
    object_ids = tuple(item["id"] for item in normalized["objects"])
    affordance, affordance_schema_audit = (
        _validate_discovery_response_with_single_repair(
            model=model,
            normalized=normalized,
            messages=affordance_messages,
            initial_raw=affordance_raw,
            initial_metadata=affordance_meta,
            response_format_json=use_json_response,
            call_type=(
                "vlm_camera_pose.functional_discovery.affordance"
            ),
            role=VLMRole.FUNCTIONAL_AFFORDANCE_DISCOVERY,
            decision_contract=(
                DecisionContract.FUNCTIONAL_AFFORDANCE_DISCOVERY
            ),
            label="functional affordance ledger",
            repair_prompt=_FUNCTIONAL_AFFORDANCE_REPAIR_PROMPT,
            validator=lambda value: validate_functional_affordance_response(
                value,
                object_ids=object_ids,
            ),
        )
    )
    affordance_meta["schema_validation"] = deepcopy(
        affordance_schema_audit
    )
    relation_affordance_prior = _relation_affordance_prior(
        affordance,
        object_ids=object_ids,
    )
    relation_prior_hint_object_count = len(
        {
            *relation_affordance_prior[
                "directed_surface_roles"
            ],
            *relation_affordance_prior["need_clearance_object_ids"],
        }
    )
    relation_context = {
        **common,
        **vlm_audit_metadata(
            VLMRole.FUNCTIONAL_RELATION_DISCOVERY,
            decision_contract=(
                DecisionContract.FUNCTIONAL_RELATION_DISCOVERY
            ),
            judge_method="discover_functional_relations",
        ),
        "role": "functional_relation_audit",
        "prompt_version": FUNCTIONAL_RELATION_PROMPT_VERSION,
        "schema_version": FUNCTIONAL_RELATION_SCHEMA_VERSION,
        "allowed_relation_predicates": sorted(
            FUNCTIONAL_RELATION_PREDICATES
        ),
        "trusted_group_partition": deepcopy(normalized["groups"]),
        "group_partition_semantics": (
            "framing_and_scope_context_not_semantic_ground_truth"
        ),
        "affordance_prior": relation_affordance_prior,
    }
    relation_raw, relation_meta, relation_messages = _run_discovery_call(
        model=model,
        normalized=normalized,
        context=relation_context,
        system_prompt=FUNCTIONAL_RELATION_SYSTEM_PROMPT,
        call_type="vlm_camera_pose.functional_discovery.relations",
        role=VLMRole.FUNCTIONAL_RELATION_DISCOVERY,
        decision_contract=(
            DecisionContract.FUNCTIONAL_RELATION_DISCOVERY
        ),
        max_context_chars=max_context_chars,
        response_format_json=use_json_response,
    )
    relations, relation_schema_audit = (
        _validate_discovery_response_with_single_repair(
            model=model,
            normalized=normalized,
            messages=relation_messages,
            initial_raw=relation_raw,
            initial_metadata=relation_meta,
            response_format_json=use_json_response,
            call_type=(
                "vlm_camera_pose.functional_discovery.relations"
            ),
            role=VLMRole.FUNCTIONAL_RELATION_DISCOVERY,
            decision_contract=(
                DecisionContract.FUNCTIONAL_RELATION_DISCOVERY
            ),
            label="functional relation audit",
            repair_prompt=_FUNCTIONAL_RELATION_REPAIR_PROMPT,
            validator=lambda value: validate_functional_relation_response(
                value,
                object_ids=object_ids,
            ),
        )
    )
    relation_meta["schema_validation"] = deepcopy(relation_schema_audit)
    relation_meta["affordance_prior"] = {
        "policy": relation_affordance_prior["policy"],
        "source_status": relation_affordance_prior["source_status"],
        "input_object_count": len(object_ids),
        "hint_object_count": relation_prior_hint_object_count,
    }
    relation_input_contract = {
        "visual_evidence_roles": [
            "scene_global",
            *(
                ["global_identity_overlay"]
                if normalized["identity_image_path"] is not None
                else []
            ),
        ],
        "visual_evidence": [
            {
                "role": "scene_global",
                "path": normalized["global_image_path"],
            },
            *(
                [
                    {
                        "role": "global_identity_overlay",
                        "path": normalized["identity_image_path"],
                    }
                ]
                if normalized["identity_image_path"] is not None
                else []
            ),
        ],
        "structured_context_fields": [
            "scene_id",
            "scene_type",
            "object_list",
            "identity_grounding",
            "trusted_group_partition",
            "affordance_prior",
        ],
        "structured_context": {
            "scene_id": relation_context["scene_id"],
            "scene_type": relation_context["scene_type"],
            "object_list": deepcopy(relation_context["object_list"]),
            "identity_grounding": deepcopy(
                relation_context["identity_grounding"]
            ),
            "trusted_group_partition": deepcopy(
                relation_context["trusted_group_partition"]
            ),
            "affordance_prior": deepcopy(
                relation_context["affordance_prior"]
            ),
        },
        "object_fields": ["id", "category"],
        "group_fields": ["group_id", "object_ids"],
        "excluded_object_fields": [
            "center",
            "size",
            "rotation",
            "description",
        ],
        "group_partition_semantics": relation_context[
            "group_partition_semantics"
        ],
        "image_count": 1
        + int(normalized["identity_image_path"] is not None),
        "object_count": len(object_ids),
        "group_count": len(normalized["groups"]),
    }
    relation_meta["input_contract"] = deepcopy(relation_input_contract)
    result = compose_functional_discovery_result(
        affordance=affordance,
        relations=relations,
        object_ids=tuple(item["id"] for item in normalized["objects"]),
        groups=normalized["groups"],
    )
    return FunctionalDiscoveryResult(
        **result,
        provenance={
            "prompt_version": FUNCTIONAL_DISCOVERY_PROMPT_VERSION,
            "schema_version": FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
            "backend": "openai_compatible",
            "affordance_prompt_version": (
                FUNCTIONAL_AFFORDANCE_PROMPT_VERSION
            ),
            "affordance_schema_version": (
                FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION
            ),
            "relation_prompt_version": FUNCTIONAL_RELATION_PROMPT_VERSION,
            "relation_schema_version": FUNCTIONAL_RELATION_SCHEMA_VERSION,
            "calls": {
                "affordance": affordance_meta,
                "relations": relation_meta,
            },
            "discovery_order": [
                "affordance",
                "relations",
            ],
            "relation_affordance_prior": {
                "policy": relation_affordance_prior["policy"],
                "source_status": relation_affordance_prior[
                    "source_status"
                ],
                "input_object_count": len(object_ids),
                "hint_object_count": (
                    relation_prior_hint_object_count
                ),
            },
            "relation_input_contract": deepcopy(
                relation_input_contract
            ),
            "vlm_roles": [
                VLMRole.FUNCTIONAL_AFFORDANCE_DISCOVERY.value,
                VLMRole.FUNCTIONAL_RELATION_DISCOVERY.value,
            ],
            "decision_contracts": [
                DecisionContract.FUNCTIONAL_AFFORDANCE_DISCOVERY.value,
                DecisionContract.FUNCTIONAL_RELATION_DISCOVERY.value,
            ],
            "normalization_warnings": deepcopy(
                affordance.get("normalization_warnings") or []
            ),
            # Additive compatibility: points at the most recent call while the
            # complete per-call records above remain authoritative.
            "request_metadata": deepcopy(relation_meta),
        },
    ).to_dict()


def _relation_affordance_prior(
    affordance: dict[str, Any],
    *,
    object_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Build a compact, non-filtering prior from a validated ledger."""

    rows = affordance.get("objects")
    if not isinstance(rows, list):
        raise TypeError("validated affordance objects must be a list")
    row_ids = tuple(str(item.get("object_id") or "") for item in rows)
    if row_ids != object_ids:
        raise ValueError(
            "validated affordance prior must cover every input object "
            "exactly once in supplied order"
        )
    directed_surface_roles: dict[str, list[str]] = {}
    need_clearance_object_ids: list[str] = []
    for item in rows:
        object_id = str(item["object_id"])
        directionality = str(item["directionality"])
        roles = list(item.get("surface_roles") or [])
        if directionality == "directed" and roles:
            directed_surface_roles[object_id] = roles
        if item["need_clearance"]:
            need_clearance_object_ids.append(object_id)
    return {
        "policy": "soft_non_exhaustive_positive_prior",
        "source_status": "validated",
        "coverage_requirement": (
            "objects omitted from this prior remain full relation candidates"
        ),
        "directed_surface_roles": directed_surface_roles,
        "need_clearance_object_ids": need_clearance_object_ids,
    }


def _run_discovery_call(
    *,
    model: OpenAICompatibleModel,
    normalized: dict[str, Any],
    context: dict[str, Any],
    system_prompt: str,
    call_type: str,
    role: VLMRole,
    decision_contract: DecisionContract,
    max_context_chars: int,
    response_format_json: bool,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    context_text = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(context_text) > max(1000, int(max_context_chars)):
        raise ValueError(
            f"{call_type} context exceeds max_context_chars; "
            "implicit truncation is forbidden"
        )
    content = [
        {
            "type": "text",
            "text": "Perform the supplied non-judging audit.\n" + context_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(
                    Path(normalized["global_image_path"]),
                    alias="scene_global",
                )
            },
        },
    ]
    if normalized["identity_image_path"] is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(
                        Path(normalized["identity_image_path"]),
                        alias="global_identity_overlay",
                    )
                },
            }
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    raw = model.chat_messages(
        messages,
        response_format_json=response_format_json,
        call_type=call_type,
        max_tokens=max(
            FUNCTIONAL_DISCOVERY_MAX_TOKENS,
            int(getattr(model, "max_tokens", None) or 0),
        ),
        max_tokens_source=f"{call_type}.minimum",
        case={
            "case_id": str(
                normalized.get("scene_id") or "functional_discovery"
            ),
            "scene_id": str(normalized.get("scene_id") or ""),
            "objects": deepcopy(normalized["objects"]),
        },
    )
    metadata = dict(getattr(model, "last_request_metadata", {}))
    metadata.update(
        vlm_audit_metadata(
            role,
            decision_contract=decision_contract,
            judge_method=call_type,
        )
    )
    metadata.update(
        call_type=call_type,
        status="complete",
        latency_seconds=round(time.perf_counter() - started, 6),
    )
    return raw, metadata, messages


def _validate_discovery_response_with_single_repair(
    *,
    model: OpenAICompatibleModel,
    normalized: dict[str, Any],
    messages: list[dict[str, Any]],
    initial_raw: str,
    initial_metadata: dict[str, Any],
    response_format_json: bool,
    call_type: str,
    role: VLMRole,
    decision_contract: DecisionContract,
    label: str,
    repair_prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a non-judging inventory, then permit one bounded repair."""

    try:
        initial_value = parse_json_object(initial_raw)
        validated = validator(initial_value)
    except (TypeError, ValueError, KeyError) as first_error:
        first_attempt = {
            "attempt": 1,
            "call_type": call_type,
            "raw_response": _bounded_raw_response(initial_raw),
            "validation_error_type": type(first_error).__name__,
            "validation_error": str(first_error),
            "request_metadata": deepcopy(initial_metadata),
        }
    else:
        return validated, {
            "policy": FUNCTIONAL_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 1,
            "repair_retry_count": 0,
            "recovered": False,
            "attempts": [
                {
                    "attempt": 1,
                    "call_type": call_type,
                    "validation_error": None,
                }
            ],
        }

    repair_call_type = f"{call_type}.schema_repair"
    repair_messages = [
        *deepcopy(messages),
        {"role": "assistant", "content": initial_raw},
        {
            "role": "user",
            "content": (
                repair_prompt
                + "\n\nValidator error to correct: "
                + str(first_attempt["validation_error"])
            ),
        },
    ]
    started = time.perf_counter()
    try:
        repaired_raw = model.chat_messages(
            repair_messages,
            response_format_json=response_format_json,
            call_type=repair_call_type,
            max_tokens=max(
                FUNCTIONAL_DISCOVERY_MAX_TOKENS,
                int(getattr(model, "max_tokens", None) or 0),
            ),
            max_tokens_source=f"{repair_call_type}.minimum",
            case={
                "case_id": str(
                    normalized.get("scene_id")
                    or "functional_discovery"
                ),
                "scene_id": str(normalized.get("scene_id") or ""),
                "objects": deepcopy(normalized["objects"]),
            },
        )
    except Exception as repair_transport_error:
        audit = {
            "policy": FUNCTIONAL_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": None,
                    "validation_error_type": type(
                        repair_transport_error
                    ).__name__,
                    "validation_error": str(repair_transport_error),
                    "failure_kind": "transport",
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{label} repair request failed: "
            f"{repair_transport_error}",
            schema_audit=audit,
        ) from repair_transport_error

    repair_metadata = dict(
        getattr(model, "last_request_metadata", {})
    )
    repair_metadata.update(
        vlm_audit_metadata(
            role,
            decision_contract=decision_contract,
            judge_method=repair_call_type,
        )
    )
    repair_metadata.update(
        call_type=repair_call_type,
        status="complete",
        latency_seconds=round(time.perf_counter() - started, 6),
    )
    try:
        repaired_value = parse_json_object(repaired_raw)
        repaired = validator(repaired_value)
    except (TypeError, ValueError, KeyError) as second_error:
        audit = {
            "policy": FUNCTIONAL_DISCOVERY_REPAIR_POLICY,
            "attempt_count": 2,
            "repair_retry_count": 1,
            "recovered": False,
            "attempts": [
                first_attempt,
                {
                    "attempt": 2,
                    "call_type": repair_call_type,
                    "raw_response": _bounded_raw_response(repaired_raw),
                    "validation_error_type": type(second_error).__name__,
                    "validation_error": str(second_error),
                    "request_metadata": repair_metadata,
                },
            ],
        }
        raise ResponseSchemaRepairError(
            f"{label} remained invalid after one repair retry: "
            f"{second_error}",
            schema_audit=audit,
        ) from second_error

    return repaired, {
        "policy": FUNCTIONAL_DISCOVERY_REPAIR_POLICY,
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": True,
        "attempts": [
            first_attempt,
            {
                "attempt": 2,
                "call_type": repair_call_type,
                "raw_response": _bounded_raw_response(repaired_raw),
                "validation_error": None,
                "request_metadata": repair_metadata,
            },
        ],
    }


def _bounded_raw_response(value: Any) -> str:
    raw = str(value)
    if len(raw) <= _RAW_RESPONSE_LIMIT:
        return raw
    return raw[:_RAW_RESPONSE_LIMIT] + "\n...[truncated]"


def _image_data_url(path: Path, *, alias: str) -> str:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required to sanitize discovery images"
        ) from exc
    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"functional discovery image {alias!r} is not decodable"
        ) from exc
    return (
        "data:image/png;base64,"
        + base64.b64encode(output.getvalue()).decode("ascii")
    )


__all__ = [
    "FUNCTIONAL_AFFORDANCE_PROMPT_VERSION",
    "FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION",
    "FUNCTIONAL_AFFORDANCE_SYSTEM_PROMPT",
    "FUNCTIONAL_DIRECTIONALITY",
    "FUNCTIONAL_DISCOVERY_MAX_TOKENS",
    "FUNCTIONAL_DISCOVERY_PROMPT_VERSION",
    "FUNCTIONAL_DISCOVERY_REPAIR_POLICY",
    "FUNCTIONAL_DISCOVERY_SCHEMA_VERSION",
    "FUNCTIONAL_RELATION_PREDICATES",
    "FUNCTIONAL_RELATION_PROMPT_VERSION",
    "FUNCTIONAL_RELATION_SCHEMA_VERSION",
    "FUNCTIONAL_RELATION_SYSTEM_PROMPT",
    "FUNCTIONAL_REVIEW_STATES",
    "FUNCTIONAL_SURFACE_ROLES",
    "FunctionalDiscoveryResult",
    "compose_functional_discovery_result",
    "discover_openai_compatible_functional_evidence",
    "validate_functional_affordance_response",
    "validate_functional_discovery_request",
    "validate_functional_discovery_response",
    "validate_functional_relation_response",
]
