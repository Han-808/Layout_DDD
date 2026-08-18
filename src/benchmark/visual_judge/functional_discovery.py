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
    FUNCTIONAL_COUNTERPART_MODES,
    FUNCTIONAL_DIRECTIONALITY,
    FUNCTIONAL_DISCOVERY_PROMPT_VERSION,
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
    FUNCTIONAL_ORDINARY_MOBILITY,
    FUNCTIONAL_RELATION_DEPENDENCIES,
    FUNCTIONAL_RELATION_PREDICATES,
    FUNCTIONAL_RELATION_PROMPT_VERSION,
    FUNCTIONAL_RELATION_SCHEMA_VERSION,
    FUNCTIONAL_REVIEW_STATES,
    FUNCTIONAL_SURFACE_ROLES,
    FunctionalDiscoveryResult,
)
from benchmark.visual_judge.functional_discovery_validation import (
    compose_functional_discovery_result,
    salvage_functional_affordance_response,
    salvage_functional_relation_response,
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
directed iff use depends on a horizontally facing side relative to accessible
architecture or a joint-use counterpart. Use non_directed otherwise, including
top/bottom/gravity-axis controls and objects without ordinary operation. This
asks whether such a side exists, not which side. If plausible but unseen, use
directed; later localization resolves it. Set need_clearance=true only when
blocking a dedicated approach, opening, or operation region prevents ordinary
use. Mere nearby posture, generic circulation, or use of a seat, table, or work
surface is insufficient; passive objects normally use false. Do not subtype
clearance. Boundary context may refine an already-required side or clearance
view, never create a check; thus non_directed with need_clearance=false requires
boundary_review_state=routine. Category is a hint; visible geometry remains
relevant.

The conditional fields must agree:
- directed requires one or more surface_roles;
- non_directed requires surface_roles=[];
- top/bottom/gravity-axis access alone is non_directed even when an ordinary
  control or interaction exists;
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


FUNCTIONAL_RELATION_SYSTEM_PROMPT = """Inventory ordinary joint-use role
assignments; never judge validity. Analyze all objects jointly. Each row uses
target_ids exactly [focal_id,counterpart_id] and one predicate:
- directional_correspondence: use depends on compatible functional-side or
  facing directions;
- relative_use_geometry: use depends on relative position, distance, reach,
  coordinated operation, or an operational connection.

Use dependency=required only if ordinary use of the focal materially depends
on this counterpart condition; otherwise contextual. Use counterpart_mode:
dedicated when the counterpart serves this focal assignment, shared only when
it genuinely serves multiple focals, or alternative when interchangeable. Use
ordinary_mobility for the counterpart: fixed; movable_companion when ordinary
use expects repositioning; portable_unrelated when it is not an assigned
participant. Movable_companion is context, never by itself evidence of a
permanent clearance blocker.

Resolve roles across the full set. Never assign one dedicated counterpart to
multiple focals, promote optional associations to required, or emit every
possible pair. Put the strongest dedicated claim first if claims conflict.
Broad cooperation, co-presence, similarity, style, or possible usefulness is
contextual at most. Object-level approach/opening/operation clearance is not a
relation. Static support/contact is L1 or Placement. Distance and group membership
affect framing; they cannot suppress a real relation. Bad current
geometry cannot suppress the check. The affordance_prior is a sparse positive
hint, not a candidate whitelist; inspect every object before returning none.

Copy every object ID exactly once in considered_object_ids. Return exactly:
{"considered_object_ids":["id"],
"relations":[{"target_ids":["id1","id2"],
"predicate":"directional_correspondence",
"dependency":"required",
"counterpart_mode":"dedicated",
"ordinary_mobility":"fixed",
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
- top/bottom/gravity-axis access without horizontal facing dependence is
  non_directed;
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
relation in [focal,counterpart] order, and use exactly one allowed predicate,
dependency, counterpart_mode, and ordinary_mobility per row. Keep full-set role
assignment coherent: a dedicated counterpart cannot serve multiple focals;
shared and alternative are explicit, and portable_unrelated cannot create a
required obligation. Split a multi-object set only into supported direct
focal-participant relations; do not generate a complete pair graph. Object-level
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
            salvage=lambda value, fallback: salvage_functional_affordance_response(
                value,
                object_ids=object_ids,
                fallback_value=fallback,
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
        "allowed_relation_dependencies": sorted(
            FUNCTIONAL_RELATION_DEPENDENCIES
        ),
        "allowed_counterpart_modes": sorted(
            FUNCTIONAL_COUNTERPART_MODES
        ),
        "allowed_ordinary_mobility": sorted(
            FUNCTIONAL_ORDINARY_MOBILITY
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
            salvage=lambda value, fallback: salvage_functional_relation_response(
                value,
                object_ids=object_ids,
                fallback_value=fallback,
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
            "allowed_relation_predicates",
            "allowed_relation_dependencies",
            "allowed_counterpart_modes",
            "allowed_ordinary_mobility",
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
            "allowed_relation_predicates": deepcopy(
                relation_context["allowed_relation_predicates"]
            ),
            "allowed_relation_dependencies": deepcopy(
                relation_context["allowed_relation_dependencies"]
            ),
            "allowed_counterpart_modes": deepcopy(
                relation_context["allowed_counterpart_modes"]
            ),
            "allowed_ordinary_mobility": deepcopy(
                relation_context["allowed_ordinary_mobility"]
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
    relations, relation_admission_audit = _apply_relation_admission_gate(
        relations,
        objects=normalized["objects"],
        groups=normalized["groups"],
    )
    relation_meta["relation_admission"] = deepcopy(
        relation_admission_audit
    )
    result = compose_functional_discovery_result(
        affordance=affordance,
        relations=relations,
        object_ids=tuple(item["id"] for item in normalized["objects"]),
        groups=normalized["groups"],
    )
    result["relation_admission_audit"] = relation_admission_audit
    affordance_salvage = (
        affordance.get("item_salvage")
        if isinstance(affordance.get("item_salvage"), dict)
        else {}
    )
    relation_salvage = (
        relations.get("item_salvage")
        if isinstance(relations.get("item_salvage"), dict)
        else {}
    )
    expected_object_count = len(object_ids)
    accepted_object_count = int(
        affordance_salvage.get(
            "accepted_object_count",
            expected_object_count,
        )
    )
    relation_contract_valid = bool(
        relation_salvage.get("consideration_contract_valid", True)
    )
    anchored_relation_count = int(
        relation_salvage.get(
            "anchored_relation_count",
            len(relations.get("relations") or []),
        )
    )
    accepted_relation_count = int(
        relation_salvage.get(
            "accepted_relation_count",
            len(relations.get("relations") or []),
        )
    )
    coverage_eligible = expected_object_count + 1 + anchored_relation_count
    coverage_grounded = accepted_object_count + int(
        relation_contract_valid
    ) + accepted_relation_count
    return FunctionalDiscoveryResult(
        **result,
        coverage={
            "unit": "discovery_contract_obligation",
            "eligible_count": coverage_eligible,
            "grounded_count": coverage_grounded,
            "fraction": (
                coverage_grounded / coverage_eligible
                if coverage_eligible
                else 0.0
            ),
            "complete": coverage_grounded == coverage_eligible,
            "affordance": deepcopy(
                affordance_salvage
                or {
                    "expected_object_count": expected_object_count,
                    "accepted_object_count": expected_object_count,
                    "defaulted_object_count": 0,
                    "coverage_fraction": 1.0,
                }
            ),
            "relations": deepcopy(
                relation_salvage
                or {
                    "consideration_contract_valid": True,
                    "anchored_relation_count": len(
                        relations.get("relations") or []
                    ),
                    "accepted_relation_count": len(
                        relations.get("relations") or []
                    ),
                    "dropped_relation_count": 0,
                    "rejected_relation_count": 0,
                }
            ),
        },
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


def _apply_relation_admission_gate(
    value: dict[str, Any],
    *,
    objects: list[dict[str, str]],
    groups: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign relation roles jointly, without another model call.

    The model call inventories candidates for the entire scene.  This gate then
    treats those rows as one assignment problem, rather than independently
    promoting every plausible pair to an obligation.  It has no metric-decision
    authority: contextual, alternative, and losing role claims remain in the
    audit instead of becoming Functional checks.
    """

    result = deepcopy(value)
    category_by_id = {
        str(item["id"]): str(item["category"])
        for item in objects
    }
    group_by_id: dict[str, str] = {}
    for group in groups:
        group_id = str(group["group_id"])
        for object_id in group.get("object_ids") or []:
            group_by_id[str(object_id)] = group_id

    proposals: list[dict[str, Any]] = []
    for index, relation in enumerate(result.get("relations") or [], start=1):
        target_ids = [str(item) for item in relation["target_ids"]]
        focal_id, counterpart_id = target_ids
        goal = str(relation.get("observation_goal") or "").strip()
        predicate = str(relation.get("predicate") or "")
        dependency = str(relation.get("dependency") or "")
        counterpart_mode = str(relation.get("counterpart_mode") or "")
        ordinary_mobility = str(
            relation.get("ordinary_mobility") or ""
        )
        reasons: list[str] = []
        state = "admitted"
        if dependency == "contextual":
            state = "context_only"
            reasons.append("contextual_dependency_is_not_function_required")
        elif counterpart_mode == "alternative":
            state = "context_only"
            reasons.append("alternative_requires_disjunctive_obligation")
        elif ordinary_mobility == "portable_unrelated":
            state = "context_only"
            reasons.append("portable_unrelated_is_not_assigned_counterpart")
        proposals.append(
            {
                "proposal_ref": f"relation_proposal_{index:03d}",
                "relation": deepcopy(relation),
                "target_ids": target_ids,
                "focal_id": focal_id,
                "counterpart_id": counterpart_id,
                "focal_category": category_by_id.get(focal_id),
                "counterpart_category": category_by_id.get(counterpart_id),
                "focal_group_id": group_by_id.get(focal_id),
                "counterpart_group_id": group_by_id.get(counterpart_id),
                "predicate": predicate,
                "dependency": dependency,
                "counterpart_mode": counterpart_mode,
                "ordinary_mobility": ordinary_mobility,
                "observation_goal": goal,
                "admission_state": state,
                "reasons": reasons,
            }
        )

    # A dedicated counterpart has one focal owner across the entire accepted
    # inventory.  Multiple predicates for the same focal-counterpart assignment
    # remain valid.  Proposal order is used only as the deterministic tie-break
    # for contradictory model claims; the prompt requires strongest fit first.
    dedicated_owner: dict[str, str] = {}
    for proposal in proposals:
        if (
            proposal["admission_state"] == "admitted"
            and proposal["counterpart_mode"] == "dedicated"
        ):
            dedicated_owner.setdefault(
                proposal["counterpart_id"],
                proposal["focal_id"],
            )

    admitted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for proposal in proposals:
        state = str(proposal["admission_state"])
        reasons = list(proposal["reasons"])
        counterpart_id = str(proposal["counterpart_id"])
        focal_id = str(proposal["focal_id"])
        owner = dedicated_owner.get(counterpart_id)
        if state == "admitted" and owner is not None and owner != focal_id:
            state = "context_only"
            reasons.append("dedicated_counterpart_assigned_to_other_focal")
        if state == "admitted":
            reasons.append("atomic_direct_joint_use_assignment_admitted")
            if proposal["ordinary_mobility"] == "movable_companion":
                reasons.append(
                    "movable_companion_retained_for_clearance_semantics"
                )
            admitted.append(deepcopy(proposal["relation"]))
        audit_rows.append(
            {
                key: deepcopy(item)
                for key, item in proposal.items()
                if key != "relation"
            }
            | {
                "admission_state": state,
                "reasons": reasons,
                "selected_dedicated_focal_id": owner,
                "same_group": bool(
                    proposal["focal_group_id"]
                    and proposal["focal_group_id"]
                    == proposal["counterpart_group_id"]
                ),
                "decision_authority": "none",
            }
        )
    result["relations"] = admitted
    audit = {
        "schema_version": "functional_relation_assignment_v2",
        "policy": "deterministic_group_aware_role_assignment",
        "proposal_count": len(audit_rows),
        "admitted_count": sum(
            row["admission_state"] == "admitted" for row in audit_rows
        ),
        "context_only_count": sum(
            row["admission_state"] == "context_only" for row in audit_rows
        ),
        "rejected_count": sum(
            row["admission_state"] == "rejected" for row in audit_rows
        ),
        "dedicated_assignments": [
            {
                "counterpart_id": counterpart_id,
                "focal_id": focal_id,
            }
            for counterpart_id, focal_id in dedicated_owner.items()
        ],
        "context_only_relations": [
            {
                key: deepcopy(row.get(key))
                for key in (
                    "proposal_ref",
                    "target_ids",
                    "focal_id",
                    "counterpart_id",
                    "focal_group_id",
                    "counterpart_group_id",
                    "predicate",
                    "dependency",
                    "counterpart_mode",
                    "ordinary_mobility",
                    "observation_goal",
                    "reasons",
                )
            }
            for row in audit_rows
            if row["admission_state"] == "context_only"
        ],
        "rejected_relations": [
            {
                key: deepcopy(row.get(key))
                for key in (
                    "proposal_ref",
                    "target_ids",
                    "focal_id",
                    "counterpart_id",
                    "predicate",
                    "dependency",
                    "counterpart_mode",
                    "ordinary_mobility",
                    "observation_goal",
                    "reasons",
                )
            }
            for row in audit_rows
            if row["admission_state"] == "rejected"
        ],
        "rows": audit_rows,
        "decision_authority": "none",
        "additional_api_calls": 0,
    }
    return result, audit


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
    salvage: Callable[
        [dict[str, Any], dict[str, Any]],
        dict[str, Any],
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a non-judging inventory, then permit one bounded repair."""

    initial_value: Any = None
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
        salvaged = salvage(
            repaired_value if isinstance(repaired_value, dict) else {},
            initial_value if isinstance(initial_value, dict) else {},
        )
        audit.update(
            item_level_salvage=True,
            fallback_mode="valid_items_plus_neutral_defaults",
            salvage=deepcopy(salvaged.get("item_salvage") or {}),
        )
        return salvaged, audit

    # Even a fully schema-valid repair is not allowed to revise a legal
    # initial atom or invent a relation identity.  The atomic merger keeps
    # initial-valid rows first and uses the repair only for missing/malformed
    # atoms with a trusted identity anchor.
    merged = salvage(
        repaired_value if isinstance(repaired_value, dict) else repaired,
        initial_value if isinstance(initial_value, dict) else {},
    )
    return merged, {
        "policy": FUNCTIONAL_DISCOVERY_REPAIR_POLICY,
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": True,
        "item_level_salvage": True,
        "fallback_mode": "initial_valid_atoms_plus_anchored_repair",
        "salvage": deepcopy(merged.get("item_salvage") or {}),
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
    "FUNCTIONAL_COUNTERPART_MODES",
    "FUNCTIONAL_DIRECTIONALITY",
    "FUNCTIONAL_DISCOVERY_MAX_TOKENS",
    "FUNCTIONAL_DISCOVERY_PROMPT_VERSION",
    "FUNCTIONAL_DISCOVERY_REPAIR_POLICY",
    "FUNCTIONAL_DISCOVERY_SCHEMA_VERSION",
    "FUNCTIONAL_ORDINARY_MOBILITY",
    "FUNCTIONAL_RELATION_DEPENDENCIES",
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
