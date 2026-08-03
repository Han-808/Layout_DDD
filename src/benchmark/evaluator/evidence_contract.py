"""Shared evidence vocabulary and canonical L0--L4 ownership.

This module is the single hierarchy source for the active non-Game evaluator.
It also validates the global/local/text evidence plans consumed by the L2
Functional Semantic and L3 Scene Quality runtimes. Rendering, selection, and
judgement stay in their dedicated runtime modules; this file only defines their
shared contract.

Canonical hierarchy (single source of truth for ownership labels):

- L1 Physical Plausibility: Collision, OOB, Support, Navigability/Accessibility
  (when applicable), and other prompt-independent physical validity.
- L2 Specification / Intent Fidelity ("Did the scene follow the prompt?"):
  - OOR and OAR are explicit-relation fidelity, NOT L1 Physical Plausibility.
  - high-level: ``functional_semantic_fidelity``. Room/scene type, broad
    visual-functional intent, required functional areas, and an explicitly
    requested local function are components of this one family rather than
    separately weighted metrics. Global evidence is primary; group-local
    evidence is requested only when the prompt explicitly specifies a local
    function.
- L3 Scene Quality ("Is the scene coherent, except prompt-authorized
  deviations?"):
  - L3a Semantic Coherence: scale_consistency, object_pairing_consistency.
    Object pairing is category/role compatibility over groups supplied by the
    grouping algorithm; position, angle, and functional arrangement are outside
    its verdict.
  - L3b Perceptual Visual Quality: style_consistency.

Rules encoded here:

- ``suspicious`` and ``insufficient_evidence`` may request local evidence.
- Unknown/missing evidence must never become valid by default.
- ``failed`` routing remains failed/unresolved.
- Global evidence and local evidence are complementary, not interchangeable.
- Every future VLM judge must have access to the prompt or a benchmark-owned
  structured representation of it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


# Active grouping is a VLM-produced downstream visual-evidence partition. The
# deterministic topology/anchor implementations remain explicit deprecated
# replay backends but are not active defaults or silent fallbacks.
GROUPING_POLICY_ID = "vlm_visual_evidence_scope_v2"
GROUPING_IMPLEMENTATION = "src/benchmark/grouping/vlm.py"
GROUPING_CONFIG_PATH = "configs/grouping/vlm_visual_evidence_scope_v2.yaml"
GROUPING_ROLE = "evidence_partition_not_metric_verdict"

# Adaptive evidence strategies.
EVIDENCE_STRATEGIES = (
    "global_only",
    "global_screen_then_local",
    "json_screen_then_visual",
    "script_screen_then_local",
    "global_and_local",
)

# Router lifecycle states. Unknown/missing evidence is never valid by default;
# failed remains failed.
ROUTER_STATES = (
    "not_run",
    "not_suspicious",
    "suspicious",
    "insufficient_evidence",
    "failed",
)

# States that may trigger a local-evidence request.
LOCAL_TRIGGERS = (
    "suspicious",
    "insufficient_evidence",
    "deterministic_candidate",
    "unknown_coverage",
    "prompt_specified_local_functionality",
)
# Only these router states may request local evidence.
LOCAL_REQUESTING_STATES = ("suspicious", "insufficient_evidence")

# Superset of trigger tokens a candidate router may emit (VLM screen or
# deterministic statistics/matrix/geometry).
ROUTER_TRIGGER_STATES = (
    "suspicious",
    "insufficient_evidence",
    "deterministic_candidate",
    "unknown_coverage",
    "statistical_outlier",
    "rare_pair",
    "unsupported_pair",
)

# Camera scopes shared by evidence policies and local policies.
CAMERA_SCOPES = ("global", "object_local", "group_local", "pair_local")
LOCAL_CAMERA_SCOPES = ("object_local", "group_local", "pair_local")

# Declarative text-context tokens an evidence plan may require.
TEXT_CONTEXT_FIELDS = (
    "original_prompt",
    "parsed_prompt_requirements",
    "parsed_functional_semantic_requirements",
    "parsed_room_scene_type",
    "parsed_broad_intent",
    "parsed_required_functional_areas",
    "parsed_local_functionality_requirements",
    "authorized_deviations",
    "asset_policy",
    "deterministic_router_evidence",
    "object_grouping_report",
    "scene_metadata",
    "target_ids",
    "group_ids",
)

# What a future final VLM adjudication must always receive. Documented contract
# only; no judge is invoked here.
FINAL_VLM_CONTEXT_CONTRACT = (
    "original_prompt",
    "parsed_prompt_requirements",
    "authorized_deviations",
    "target_ids_or_group_ids",
    "relevant_global_visual_evidence",
    "relevant_local_visual_evidence",
    "asset_policy",
    "deterministic_router_evidence_when_script_router",
    "object_grouping_report_when_group_scoped",
)

# The high-level L2 family deliberately exposes components for auditability
# without turning each component into a separately weighted metric.
FUNCTIONAL_SEMANTIC_COMPONENTS = (
    "room_scene_type",
    "visual_functional_intent",
    "required_functional_areas",
    "local_functionality",
)

# A group-local functionality request is legal only when the benchmark-owned
# prompt contract explicitly contains such a requirement. It is not a generic
# "suspicious group" router.
LOCAL_ACTIVATION_CONDITIONS = ("prompt_specified_local_functionality",)

# Canonical hierarchy ownership. This is the single source of truth for active
# benchmark metrics. Prompt granularity and asset strategy are descriptive
# metadata; neither selects a parallel evaluator workflow.
CANONICAL_HIERARCHY: dict[str, Any] = {
    "l0_structural_validity": {
        "question": "Can the submission be parsed and evaluated under the canonical contracts?",
        "metrics": [],
        "scoring": False,
        "role": "execution_gate",
        "checks": [
            "schema",
            "normalization",
            "coordinate_and_unit_consistency",
            "required_input_coverage",
        ],
    },
    "l1_physical_plausibility": {
        "question": "Is the scene physically plausible, independent of the prompt?",
        "metrics": ["collision", "oob", "support", "navigability", "accessibility"],
        "default_disabled": ["navigability", "accessibility"],
        "note": "Generic Collision/OOB/Support stay L1 even when they help verify an explicit relation.",
    },
    "l2_specification_fidelity": {
        "question": "Did the scene follow the prompt?",
        "metrics": ["oor", "oar", "functional_semantic_fidelity"],
        "functional_semantic_components": list(FUNCTIONAL_SEMANTIC_COMPONENTS),
        "note": (
            "OOR/OAR are explicit-relation fidelity. Room type, broad functional "
            "intent, required areas, and prompt-specified local functionality share "
            "one functional-semantic family. Object presence/count/attributes are "
            "not benchmark metrics."
        ),
    },
    "l3_scene_quality": {
        "question": "Is the scene coherent, except for prompt-authorized deviations?",
        "semantic_coherence": ["scale_consistency", "object_pairing_consistency"],
        "perceptual_visual_quality": ["style_consistency"],
        "object_pairing_scope": "group_member_category_and_role_compatibility_only",
    },
    "l4_downstream_task_functionality": {
        "question": "Does the environment support the downstream task?",
        "metrics": [],
        "status": "tbd",
        "scoring": False,
    },
}


class EvidenceContractError(ValueError):
    """Raised when a structural evidence plan is malformed."""


def grouping_policy_provenance(role: str = GROUPING_ROLE) -> dict[str, Any]:
    """Declarative provenance for the default evidence-partition grouping policy."""

    return {
        "policy_id": GROUPING_POLICY_ID,
        "implementation": GROUPING_IMPLEMENTATION,
        "config": GROUPING_CONFIG_PATH,
        "role": role,
    }


def canonical_hierarchy() -> dict[str, Any]:
    """Return a copy of the canonical hierarchy ownership labels."""

    return deepcopy(CANONICAL_HIERARCHY)


def validate_evidence_strategy(value: Any, *, where: str = "evidence_strategy") -> str:
    if value not in EVIDENCE_STRATEGIES:
        raise EvidenceContractError(
            f"{where} must be one of {list(EVIDENCE_STRATEGIES)}, got {value!r}"
        )
    return value


def validate_trigger_states(
    values: Any,
    *,
    allowed: Iterable[str] = LOCAL_TRIGGERS,
    where: str = "trigger_states",
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise EvidenceContractError(f"{where} must be a non-empty list")
    allowed_set = set(allowed)
    for token in values:
        if token not in allowed_set:
            raise EvidenceContractError(
                f"{where} token {token!r} must be one of {sorted(allowed_set)}"
            )
    return list(values)


def validate_global_policy(policy: Any, *, where: str = "global_policy") -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise EvidenceContractError(f"{where} must be a JSON object")
    view_family = policy.get("view_family")
    if not isinstance(view_family, str) or not view_family.strip():
        raise EvidenceContractError(f"{where}.view_family must be a non-empty string")
    budget = policy.get("image_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise EvidenceContractError(f"{where}.image_budget must be a positive integer")
    if not isinstance(policy.get("top_down"), bool):
        raise EvidenceContractError(f"{where}.top_down must be boolean")
    for optional_bool in ("perspective_diversity_required", "preserve_boundary_cues"):
        if optional_bool in policy and not isinstance(policy[optional_bool], bool):
            raise EvidenceContractError(f"{where}.{optional_bool} must be boolean")
    if "occluder_policy" in policy and not isinstance(policy["occluder_policy"], str):
        raise EvidenceContractError(f"{where}.occluder_policy must be a string")
    return policy


def validate_local_policy(policy: Any, *, where: str = "local_policy") -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise EvidenceContractError(f"{where} must be a JSON object")
    scope = policy.get("camera_scope")
    if scope not in LOCAL_CAMERA_SCOPES:
        raise EvidenceContractError(
            f"{where}.camera_scope must be one of {list(LOCAL_CAMERA_SCOPES)}, got {scope!r}"
        )
    grouping_policy_id = policy.get("grouping_policy_id")
    if not isinstance(grouping_policy_id, str) or not grouping_policy_id.strip():
        raise EvidenceContractError(f"{where}.grouping_policy_id must be a non-empty string")
    if "trigger_states" in policy:
        validate_trigger_states(
            policy["trigger_states"], allowed=LOCAL_TRIGGERS, where=f"{where}.trigger_states"
        )
    if "activation_condition" in policy:
        condition = policy["activation_condition"]
        if condition not in LOCAL_ACTIVATION_CONDITIONS:
            raise EvidenceContractError(
                f"{where}.activation_condition must be one of "
                f"{list(LOCAL_ACTIVATION_CONDITIONS)}, got {condition!r}"
            )
    if (
        "include_global_context" in policy
        and not isinstance(policy["include_global_context"], bool)
    ):
        raise EvidenceContractError(
            f"{where}.include_global_context must be boolean"
        )
    for budget_name in ("image_budget", "max_packet_images"):
        if budget_name not in policy:
            continue
        budget = policy[budget_name]
        if (
            isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 1
        ):
            raise EvidenceContractError(
                f"{where}.{budget_name} must be a positive integer"
            )
    return policy


def validate_router_options(options: Any, *, where: str = "router_options") -> dict[str, Any]:
    if not isinstance(options, dict) or not options:
        raise EvidenceContractError(f"{where} must be a non-empty JSON object")
    for strategy, spec in options.items():
        if strategy not in EVIDENCE_STRATEGIES:
            raise EvidenceContractError(
                f"{where} key {strategy!r} must be one of {list(EVIDENCE_STRATEGIES)}"
            )
        if not isinstance(spec, dict):
            raise EvidenceContractError(f"{where}.{strategy} must be a JSON object")
        router = spec.get("router")
        if not isinstance(router, str) or not router.strip():
            raise EvidenceContractError(f"{where}.{strategy}.router must be a non-empty string")
        if "source" in spec and not isinstance(spec["source"], str):
            raise EvidenceContractError(f"{where}.{strategy}.source must be a string")
        if "trigger_states" in spec:
            validate_trigger_states(
                spec["trigger_states"],
                allowed=ROUTER_TRIGGER_STATES,
                where=f"{where}.{strategy}.trigger_states",
            )
        # A candidate router is referenced, never executed by these placeholders.
        if spec.get("executes_router", False) is not False:
            raise EvidenceContractError(
                f"{where}.{strategy}.executes_router must be false; routers are referenced, not invoked"
            )
    return options


def validate_text_context(values: Any, *, where: str = "text_context") -> list[str]:
    if not isinstance(values, list) or not values:
        raise EvidenceContractError(f"{where} must be a non-empty list")
    for token in values:
        if not isinstance(token, str) or not token.strip():
            raise EvidenceContractError(f"{where} entries must be non-empty strings")
    if "original_prompt" not in values:
        raise EvidenceContractError(
            f"{where} must include 'original_prompt'; every L2/L3 evidence plan must carry prompt context"
        )
    return list(values)


def validate_evidence_plan(plan: Any, *, where: str = "evidence_plan") -> dict[str, Any]:
    """Validate a declarative per-metric evidence plan.

    A plan may declare a single ``evidence_strategy`` and/or a ``router_options``
    map of candidate routers, plus optional ``global_policy``, ``local_policy``,
    and a required ``text_context`` that always carries the prompt. Unknown,
    future-compatible fields are preserved.
    """

    if not isinstance(plan, dict):
        raise EvidenceContractError(f"{where} must be a JSON object")
    if plan.get("evidence_strategy") is not None:
        validate_evidence_strategy(plan["evidence_strategy"], where=f"{where}.evidence_strategy")
    if "available_strategies" in plan and plan["available_strategies"] is not None:
        strategies = plan["available_strategies"]
        if not isinstance(strategies, list) or not strategies:
            raise EvidenceContractError(f"{where}.available_strategies must be a non-empty list")
        for strategy in strategies:
            validate_evidence_strategy(strategy, where=f"{where}.available_strategies")
    if plan.get("global_policy") is not None:
        validate_global_policy(plan["global_policy"], where=f"{where}.global_policy")
    if plan.get("local_policy") is not None:
        validate_local_policy(plan["local_policy"], where=f"{where}.local_policy")
    if plan.get("router_options") is not None:
        validate_router_options(plan["router_options"], where=f"{where}.router_options")
    if plan.get("text_context") is not None:
        validate_text_context(plan["text_context"], where=f"{where}.text_context")
    return plan
