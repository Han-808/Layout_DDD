"""Benchmark-owned specification contract and claim-driven canonical L2.

The canonical L2 score has exactly three metric families: OOR, OAR, and
``functional_semantic_fidelity``. Prompt granularity remains descriptive
metadata / a reporting slice; it never suppresses a claim present in the frozen
benchmark-owned contract.

This module provides:

- :func:`validate_specification_contract` - structural + trust validation;
- :func:`specification_contract_from_reference_annotation` - a compiler that maps
  confirmed OOR/OAR annotations into the canonical contract;
- :func:`compile_specification_evaluation_plan` - claim-driven activation plan;
- :func:`build_specification_fidelity_report` - the canonical scored L2 report
  that references already-executed OOR/OAR and functional-semantic outputs.

The canonical runtime accepts only the three canonical claim-family keys.
Frozen construction-time ``cal_dataset2`` v0 artifacts have a separate,
read-only validator helper; that compatibility boundary never enters
``run_evaluate``. Missing evidence, failed evaluators, and unresolved claims are
never converted into valid, zero, or full score.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.evaluator.evidence_contract import FUNCTIONAL_SEMANTIC_COMPONENTS
from benchmark.reference_annotation import (
    confirmed_oar_relations,
    confirmed_oor_relations,
    ensure_reference_relation_ids,
    validate_reference_annotation,
)


SPECIFICATION_CONTRACT_VERSION = "specification_contract_v1"

FINE_DETAIL_CLAIM_FAMILIES = ("oor", "oar")
FUNCTIONAL_SEMANTIC_FIDELITY = "functional_semantic_fidelity"
HIGH_LEVEL_CLAIM_FAMILIES = (FUNCTIONAL_SEMANTIC_FIDELITY,)
SPECIFICATION_CLAIM_FAMILIES = FINE_DETAIL_CLAIM_FAMILIES + HIGH_LEVEL_CLAIM_FAMILIES

# Canonical acceptance is intentionally exact.  The name remains as a public
# compatibility constant, but no longer includes retired aliases/provenance.
ACCEPTED_SPECIFICATION_CLAIM_FAMILIES = SPECIFICATION_CLAIM_FAMILIES

# Read-only compatibility inventory for the already frozen cal_dataset2 v0
# construction artifacts.  These constants are not consulted by canonical
# validation, activation, reporting, run_evaluate, or the public JSON schema.
LEGACY_NON_SCORING_CLAIM_FAMILIES = (
    "object_presence",
    "object_count",
    "explicit_attributes",
)
LEGACY_HIGH_LEVEL_CLAIM_FAMILIES = (
    "room_scene_type",
    "broad_semantic_intent",
    "required_functional_areas",
)
FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES = (
    FINE_DETAIL_CLAIM_FAMILIES
    + LEGACY_HIGH_LEVEL_CLAIM_FAMILIES
    + LEGACY_NON_SCORING_CLAIM_FAMILIES
)

# Sources that may own an official contract. Runtime VLM prompt parsing and
# public generator output are intentionally excluded.
TRUSTED_CONTRACT_SOURCES = ("benchmark_annotation", "benchmark_owned", "trusted_case_bundle")
CONTRACT_SOURCES = TRUSTED_CONTRACT_SOURCES + ("manual", "programmatic", "diagnostic")

# Which evaluator module resolves each canonical scoring family.
FAMILY_MODULE_ROUTING = {
    "oor": {"module": "oor", "implemented": True},
    "oar": {"module": "oar", "implemented": True},
    FUNCTIONAL_SEMANTIC_FIDELITY: {
        "module": FUNCTIONAL_SEMANTIC_FIDELITY,
        "implemented": True,
    },
}


class SpecificationContractError(ValueError):
    """Raised when a specification contract is structurally invalid or untrusted."""


def _empty_claims() -> dict[str, list]:
    return {family: [] for family in SPECIFICATION_CLAIM_FAMILIES}


def validate_specification_contract(
    contract: Any,
    *,
    valid_object_ids: set[str] | None = None,
    require_trusted: bool = False,
    require_frozen: bool = False,
) -> dict[str, Any]:
    """Structurally validate a specification contract.

    Enforces the contract version, a known ``source``, a boolean ``frozen``
    flag, a claims object keyed only by known families, unique ``claim_id`` values
    across the whole contract, and per-family required content. When
    ``require_trusted`` / ``require_frozen`` are set (official v2 protocol), the
    source must be benchmark-owned and ``frozen`` must be true. When
    ``valid_object_ids`` is supplied, referenced object/relation targets must
    exist.
    """

    return _validate_specification_contract(
        contract,
        accepted_families=SPECIFICATION_CLAIM_FAMILIES,
        valid_object_ids=valid_object_ids,
        require_trusted=require_trusted,
        require_frozen=require_frozen,
    )


def validate_frozen_cal_dataset2_v0_specification_contract(
    contract: Any,
    *,
    valid_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact frozen cal_dataset2 v0 family inventory, read-only.

    This helper exists only so the historical human-reviewed dataset can still
    be integrity-checked without rewriting its files.  It does not normalize
    the contract, does not make legacy families scoreable, and is never called
    by canonical evaluation.
    """

    return _validate_specification_contract(
        contract,
        accepted_families=FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES,
        required_family_set=set(FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES),
        valid_object_ids=valid_object_ids,
        require_trusted=False,
        require_frozen=False,
        boundary_name="frozen cal_dataset2 v0 read-only compatibility",
    )


def _validate_specification_contract(
    contract: Any,
    *,
    accepted_families: tuple[str, ...],
    valid_object_ids: set[str] | None,
    require_trusted: bool,
    require_frozen: bool,
    required_family_set: set[str] | None = None,
    boundary_name: str = "canonical specification contract",
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise SpecificationContractError("specification_contract must be a JSON object")
    if contract.get("contract_version") != SPECIFICATION_CONTRACT_VERSION:
        raise SpecificationContractError(
            f"specification_contract.contract_version must be {SPECIFICATION_CONTRACT_VERSION!r}"
        )
    source = contract.get("source")
    if source not in CONTRACT_SOURCES:
        raise SpecificationContractError(
            f"specification_contract.source must be one of {list(CONTRACT_SOURCES)}, got {source!r}"
        )
    if require_trusted and source not in TRUSTED_CONTRACT_SOURCES:
        raise SpecificationContractError(
            "official specification_contract must be benchmark-owned "
            f"(source in {list(TRUSTED_CONTRACT_SOURCES)}); runtime/generated claims are not scoring truth"
        )
    frozen = contract.get("frozen")
    if not isinstance(frozen, bool):
        raise SpecificationContractError("specification_contract.frozen must be boolean")
    if require_frozen and frozen is not True:
        raise SpecificationContractError("official specification_contract must be frozen")

    claims = contract.get("claims")
    if not isinstance(claims, dict):
        raise SpecificationContractError("specification_contract.claims must be a JSON object")
    unknown_families = set(claims) - set(accepted_families)
    if unknown_families:
        raise SpecificationContractError(
            f"specification_contract.claims has unknown families {sorted(unknown_families)}"
        )
    if required_family_set is not None and set(claims) != required_family_set:
        raise SpecificationContractError(
            f"{boundary_name} requires exactly claim families "
            f"{sorted(required_family_set)}, got {sorted(claims)}"
        )

    seen_claim_ids: set[str] = set()
    for family in accepted_families:
        family_claims = claims.get(family, [])
        if family_claims is None:
            continue
        if not isinstance(family_claims, list):
            raise SpecificationContractError(
                f"specification_contract.claims.{family} must be a JSON list"
            )
        for index, claim in enumerate(family_claims):
            path = f"specification_contract.claims.{family}[{index}]"
            if not isinstance(claim, dict):
                raise SpecificationContractError(f"{path} must be a JSON object")
            claim_id = claim.get("claim_id")
            if not isinstance(claim_id, str) or not claim_id.strip():
                raise SpecificationContractError(f"{path}.claim_id must be a non-empty string")
            if claim_id in seen_claim_ids:
                raise SpecificationContractError(
                    f"{path}.claim_id duplicates claim id {claim_id!r}; claim ids must be unique"
                )
            seen_claim_ids.add(claim_id)
            declared_family = claim.get("claim_family")
            if declared_family is not None and declared_family != family:
                raise SpecificationContractError(
                    f"{path}.claim_family {declared_family!r} does not match its family key {family!r}"
                )
            required = claim.get("required")
            if required is not None and not isinstance(required, bool):
                raise SpecificationContractError(f"{path}.required must be boolean when present")
            if family == FUNCTIONAL_SEMANTIC_FIDELITY:
                component = claim.get("component")
                if component not in FUNCTIONAL_SEMANTIC_COMPONENTS:
                    raise SpecificationContractError(
                        f"{path}.component must be one of "
                        f"{list(FUNCTIONAL_SEMANTIC_COMPONENTS)}, got {component!r}"
                    )
            if valid_object_ids is not None:
                _validate_claim_targets(
                    claim,
                    valid_object_ids=valid_object_ids,
                    path=path,
                )
    return contract


def specification_contract_from_reference_annotation(
    annotation: dict[str, Any],
    *,
    source: str = "benchmark_annotation",
    frozen: bool = True,
) -> dict[str, Any]:
    """Compile a confirmed reference annotation into the canonical contract.

    This is a normalization layer over the existing frozen truth: it does not
    invent a second authoritative copy. Only confirmed OOR/OAR relations become
    canonical L2 claims. Presence, count, and attribute annotations are not L2
    metrics and are therefore not compiled here. ``functional_semantic_fidelity``
    is not present in the current reference annotation schema and remains empty;
    a directly authored benchmark-owned contract may populate it.
    """

    annotation = ensure_reference_relation_ids(annotation)
    validate_reference_annotation(annotation)

    claims = _empty_claims()

    for relation in confirmed_oor_relations(annotation):
        relation_id = str(relation.get("relation_id") or "")
        claims["oor"].append(
            {
                "claim_id": relation_id or f"oor::{len(claims['oor'])}",
                "claim_family": "oor",
                "relation_id": relation_id,
                "relation_type": str(relation.get("type") or ""),
                "target_ids": sorted(_oor_relation_ids(relation)),
                "source_ref": {"relation_id": relation_id},
            }
        )
    for relation in confirmed_oar_relations(annotation):
        relation_id = str(relation.get("relation_id") or "")
        subject_id = str(relation.get("subject_id") or "")
        claims["oar"].append(
            {
                "claim_id": relation_id or f"oar::{len(claims['oar'])}",
                "claim_family": "oar",
                "relation_id": relation_id,
                "relation_type": str(relation.get("type") or ""),
                "subject_id": subject_id,
                "architectural_element": str(relation.get("architectural_element") or ""),
                "target_ids": [subject_id] if subject_id else [],
                "source_ref": {"relation_id": relation_id},
            }
        )

    contract = {
        "contract_version": SPECIFICATION_CONTRACT_VERSION,
        "source": source,
        "frozen": bool(frozen),
        "compiled_from": "reference_annotation_v1",
        "request_id": str(annotation.get("request_id") or ""),
        "scene_type": str(annotation.get("scene_type") or ""),
        "claims": claims,
    }
    return validate_specification_contract(contract)


def compile_specification_evaluation_plan(
    contract: dict[str, Any] | None,
    prompt_granularity: str | None,
    evaluation_profile: dict[str, Any] | None = None,
    *,
    activation_mode: str = "specification_contract",
) -> dict[str, Any]:
    """Compile the claim-driven L2 activation plan.

    A non-empty frozen claim family activates its module; an empty family is
    not_applicable; a family whose module is a placeholder is active-but-unavailable.
    Prompt granularity is recorded as metadata only and never suppresses a claim.
    """

    plan: dict[str, Any] = {
        "category": "specification_fidelity",
        "activation_source": (
            "benchmark_owned_specification_contract"
            if activation_mode == "specification_contract"
            else "legacy_prompt_granularity_gate"
        ),
        "activation_mode": activation_mode,
        "prompt_granularity": prompt_granularity,
        "prompt_granularity_role": "metadata_and_reporting_slice",
        "active_claim_families": [],
        "inactive_claim_families": [],
        "ignored_optional_claims": {},
        "modules": {},
    }
    if activation_mode != "specification_contract":
        plan["accepted_non_scoring_claim_families"] = list(
            LEGACY_NON_SCORING_CLAIM_FAMILIES
        )
        plan["ignored_non_scoring_claims"] = {}
    if contract is None:
        plan["contract_present"] = False
        plan["status"] = "missing_specification_contract"
        plan["inactive_claim_families"] = list(SPECIFICATION_CLAIM_FAMILIES)
        for family in SPECIFICATION_CLAIM_FAMILIES:
            plan["modules"][family] = {"active": False, "reason": "missing_specification_contract"}
        return plan

    validate_specification_contract(contract)
    plan["contract_present"] = True
    plan["contract_version"] = contract.get("contract_version")
    plan["contract_source"] = contract.get("source")
    plan["contract_frozen"] = bool(contract.get("frozen"))
    claims = _canonical_claims(contract)
    plan["ignored_optional_claims"] = {
        family: [
            str(claim.get("claim_id"))
            for claim in family_claims
            if claim.get("required") is False
        ]
        for family, family_claims in claims.items()
        if any(claim.get("required") is False for claim in family_claims)
    }
    for family in SPECIFICATION_CLAIM_FAMILIES:
        family_claims = _required_claims(claims.get(family) or [])
        routing = FAMILY_MODULE_ROUTING[family]
        if family_claims:
            plan["active_claim_families"].append(family)
            module_entry = {
                "active": True,
                "module": routing["module"],
                "implemented": bool(routing["implemented"]),
                "claim_ids": [str(claim.get("claim_id")) for claim in family_claims],
            }
            if family == FUNCTIONAL_SEMANTIC_FIDELITY:
                module_entry["components"] = sorted(
                    {
                        str(claim.get("component"))
                        for claim in family_claims
                        if claim.get("component") in FUNCTIONAL_SEMANTIC_COMPONENTS
                    }
                )
                module_entry["local_functionality_claim_ids"] = [
                    str(claim.get("claim_id"))
                    for claim in family_claims
                    if claim.get("component") == "local_functionality"
                ]
                module_entry["local_evidence_condition"] = (
                    "prompt_specified_local_functionality"
                )
                module_entry["required_area_local_fallback_claim_ids"] = [
                    str(claim.get("claim_id"))
                    for claim in family_claims
                    if claim.get("component") == "required_functional_areas"
                ]
                module_entry["required_area_local_fallback_condition"] = (
                    "global_screen_suspicious_or_insufficient_evidence"
                )
                module_entry["generic_pairing_scan"] = False
            if not routing["implemented"]:
                module_entry["reason"] = "evaluator_placeholder_not_implemented"
            plan["modules"][family] = module_entry
        else:
            plan["inactive_claim_families"].append(family)
            plan["modules"][family] = {"active": False, "reason": "no_claims"}
    plan["status"] = "compiled"
    return plan


def build_specification_fidelity_report(
    *,
    contract: dict[str, Any] | None,
    prompt_granularity: str | None,
    activation_mode: str = "specification_contract",
    oor_report: dict[str, Any] | None = None,
    oar_report: dict[str, Any] | None = None,
    functional_semantic_report: dict[str, Any] | None = None,
    object_alignment_report: dict[str, Any] | None = None,
    official: bool = False,
    legacy_category_alias: str | None = None,
) -> dict[str, Any]:
    """Build the canonical L2 ``specification_fidelity`` report.

    The report references already-executed evaluator outputs; it never re-runs
    them. Its canonical score is the micro-average of required active L2 claim
    scores within each family, followed by an equal macro-average across active
    OOR/OAR/functional-semantic families, only when coverage is complete. Any
    unresolved or failed claim keeps the canonical score ``None``.
    ``object_alignment_report`` is accepted only
    for old call-site compatibility and is deliberately ignored because
    presence/count/attribute are not canonical L2 metrics.
    """

    if contract is not None and official:
        validate_specification_contract(
            contract,
            require_trusted=True,
            require_frozen=True,
        )
    plan = compile_specification_evaluation_plan(
        contract, prompt_granularity, activation_mode=activation_mode
    )
    base = {
        "category": "specification_fidelity",
        "score": None,
        "partial_score": None,
        "score_role": (
            "equal_macro_average_across_active_metric_families_when_complete;"
            "required_claim_micro_average_within_family"
        ),
        "activation_source": plan["activation_source"],
        "activation_mode": activation_mode,
        "prompt_granularity": prompt_granularity,
        "prompt_granularity_role": "metadata_and_reporting_slice",
    }
    if activation_mode != "specification_contract" and legacy_category_alias is not None:
        base["legacy_category_alias"] = legacy_category_alias

    if contract is None:
        base.update(
            {
                "status": "not_evaluable" if (official or activation_mode == "specification_contract") else "not_applicable",
                "reason": "missing_specification_contract",
                "contract_version": None,
                "contract_source": None,
                "contract_frozen": None,
                "active_claim_families": [],
                "inactive_claim_families": list(SPECIFICATION_CLAIM_FAMILIES),
                "claim_family_reports": {},
                "coverage": {
                    "eligible_claim_count": 0,
                    "resolved_claim_count": 0,
                    "unresolved_claim_count": 0,
                    "failed_claim_count": 0,
                    "not_applicable_family_count": len(SPECIFICATION_CLAIM_FAMILIES),
                    "complete": False,
                },
                "activation_plan": plan,
            }
        )
        return base

    claims = _canonical_claims(contract)
    relation_resolution = {
        "oor": _relation_resolution_map(oor_report),
        "oar": _relation_resolution_map(oar_report),
    }
    functional_resolution = _functional_semantic_resolution_map(
        functional_semantic_report
    )

    family_reports: dict[str, dict[str, Any]] = {}
    totals = {"eligible": 0, "resolved": 0, "unresolved": 0, "failed": 0}
    resolved_scores: list[float] = []
    resolved_family_scores: list[float] = []
    family_partial_scores: list[float] = []
    active_families: list[str] = []
    inactive_families: list[str] = []

    for family in SPECIFICATION_CLAIM_FAMILIES:
        family_claims = _required_claims(claims.get(family) or [])
        routing = FAMILY_MODULE_ROUTING[family]
        if not family_claims:
            inactive_families.append(family)
            family_reports[family] = {
                "family": family,
                "module": routing["module"],
                "active": False,
                "status": "not_applicable",
                "reason": "no_claims",
                "score": None,
                "eligible_claim_count": 0,
                "resolved_claim_count": 0,
                "unresolved_claim_count": 0,
                "failed_claim_count": 0,
                "claims": [],
            }
            continue

        active_families.append(family)
        claim_results = [
            _claim_result(
                claim,
                family=family,
                routing=routing,
                relation_resolution=relation_resolution,
                functional_resolution=functional_resolution,
                oor_report=oor_report,
                oar_report=oar_report,
                functional_semantic_report=functional_semantic_report,
            )
            for claim in family_claims
        ]
        resolved = sum(1 for result in claim_results if result["resolution"] == "resolved")
        failed = sum(1 for result in claim_results if result["resolution"] == "failed")
        unresolved = len(claim_results) - resolved - failed
        totals["eligible"] += len(claim_results)
        totals["resolved"] += resolved
        totals["unresolved"] += unresolved
        totals["failed"] += failed
        family_scores = [
            float(result["score"])
            for result in claim_results
            if _is_score(result.get("score"))
        ]
        resolved_scores.extend(family_scores)

        if not routing["implemented"]:
            family_status = "not_implemented"
            family_reason = "evaluator_placeholder_not_implemented"
        elif failed:
            family_status = "failed"
            family_reason = "module_reported_failure"
        elif unresolved:
            family_status = "incomplete"
            family_reason = "claims_unresolved"
        else:
            family_status = "evaluated"
            family_reason = None
        family_complete = bool(
            claim_results and resolved == len(claim_results) and not failed
        )
        family_score = (
            sum(family_scores) / len(family_scores)
            if family_complete and len(family_scores) == len(claim_results)
            else None
        )
        family_partial_score = (
            sum(family_scores) / len(family_scores)
            if family_scores
            else None
        )
        family_reports[family] = {
            "family": family,
            "module": routing["module"],
            "module_implemented": bool(routing["implemented"]),
            "active": True,
            "status": family_status,
            "reason": family_reason,
            "score": family_score,
            "partial_score": family_partial_score,
            "score_role": "required_claim_micro_average_when_complete",
            "eligible_claim_count": len(claim_results),
            "resolved_claim_count": resolved,
            "unresolved_claim_count": unresolved,
            "failed_claim_count": failed,
            "module_report_ref": routing["module"] if routing["implemented"] else None,
            "claims": claim_results,
        }
        if _is_score(family_score):
            resolved_family_scores.append(float(family_score))
        if _is_score(family_partial_score):
            family_partial_scores.append(float(family_partial_score))
        if family == FUNCTIONAL_SEMANTIC_FIDELITY:
            family_reports[family]["components"] = sorted(
                {
                    str(claim.get("component"))
                    for claim in family_claims
                    if claim.get("component") in FUNCTIONAL_SEMANTIC_COMPONENTS
                }
            )
            family_reports[family]["component_scoring"] = (
                "single_family_no_separate_component_denominators"
            )
            family_reports[family]["local_functionality_activation"] = (
                "prompt_specified_local_functionality"
            )

    eligible = totals["eligible"]
    if eligible == 0:
        status = "not_applicable"
        reason = "no_active_claims"
    elif totals["failed"]:
        status = "incomplete"
        reason = "module_failures_present"
    elif totals["unresolved"]:
        status = "incomplete"
        reason = "unresolved_or_placeholder_claims"
    else:
        status = "evaluated"
        reason = None
    coverage_complete = bool(
        eligible > 0 and totals["unresolved"] == 0 and totals["failed"] == 0
    )
    partial_score = (
        sum(family_partial_scores) / len(family_partial_scores)
        if family_partial_scores
        else None
    )
    canonical_score = (
        partial_score
        if coverage_complete
        and len(resolved_family_scores) == len(active_families)
        else None
    )

    base.update(
        {
            "score": canonical_score,
            "partial_score": partial_score,
            "status": status,
            "reason": reason,
            "contract_version": contract.get("contract_version"),
            "contract_source": contract.get("source"),
            "contract_frozen": bool(contract.get("frozen")),
            "active_claim_families": active_families,
            "active_family_signature": list(active_families),
            "inactive_claim_families": inactive_families,
            "claim_family_reports": family_reports,
            "ignored_optional_claims": plan.get("ignored_optional_claims", {}),
            "coverage": {
                "eligible_claim_count": eligible,
                "resolved_claim_count": totals["resolved"],
                "unresolved_claim_count": totals["unresolved"],
                "failed_claim_count": totals["failed"],
                "not_applicable_family_count": len(inactive_families),
                "complete": coverage_complete,
            },
            "resolved_claim_micro_average_diagnostic": (
                sum(resolved_scores) / len(resolved_scores)
                if resolved_scores
                else None
            ),
            "resolved_family_count": len(resolved_family_scores),
            "family_with_partial_evidence_count": len(family_partial_scores),
            "comparability": {
                "case_level_scalar_is_comparable": False,
                "note": (
                    "Active-claim composition is case-specific. Comparable scores require a frozen prompt "
                    "suite, frozen contracts, fixed family composition, benchmark-owned family weights, and a "
                    "fixed aggregation protocol (Phase B)."
                ),
            },
            "activation_plan": plan,
            "notes": [
                "L2 Specification Fidelity is claim-driven; prompt granularity is metadata only.",
                "OOR/OAR are L2 explicit-relation Fidelity modules, not L1 Physical Plausibility.",
                "Room type, visual-functional intent, required areas, and prompt-specified local functionality fold into one functional_semantic_fidelity family.",
                "Each family micro-averages its required claims; canonical L2 equally macro-averages active OOR/OAR/functional-semantic families.",
                "The canonical score is emitted only when every active required claim is resolved; incomplete coverage keeps score=None.",
                "Missing/failed/unresolved claims never become valid, zero, or full score.",
            ],
        }
    )
    if activation_mode != "specification_contract":
        base["accepted_non_scoring_claim_families"] = list(
            LEGACY_NON_SCORING_CLAIM_FAMILIES
        )
        base["ignored_non_scoring_claims"] = plan.get(
            "ignored_non_scoring_claims", {}
        )
        base["ignored_legacy_inputs"] = {
            "object_alignment_report": bool(
                isinstance(object_alignment_report, dict)
            ),
            "reason": "legacy_game_compatibility_only",
        }
    return base


def _canonical_claims(contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return canonical L2 families without legacy normalization."""

    source = contract.get("claims") or {}
    result: dict[str, list[dict[str, Any]]] = {
        family: [
            deepcopy(claim)
            for claim in (source.get(family) or [])
            if isinstance(claim, dict)
        ]
        for family in SPECIFICATION_CLAIM_FAMILIES
    }
    return result


def canonical_specification_claims(
    contract: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return defensive copies of canonical scoring claims.

    Only ``oor``, ``oar``, and ``functional_semantic_fidelity`` are accepted.
    """

    validate_specification_contract(contract)
    return _canonical_claims(contract)


def _required_claims(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [claim for claim in claims if claim.get("required") is not False]


# --- resolution helpers -------------------------------------------------------


def _claim_result(
    claim: dict[str, Any],
    *,
    family: str,
    routing: dict[str, Any],
    relation_resolution: dict[str, dict[str, dict[str, Any]]],
    functional_resolution: dict[str, dict[str, Any]],
    oor_report: dict[str, Any] | None,
    oar_report: dict[str, Any] | None,
    functional_semantic_report: dict[str, Any] | None,
) -> dict[str, Any]:
    claim_id = str(claim.get("claim_id"))
    if not routing["implemented"]:
        result = {
            "claim_id": claim_id,
            "family": family,
            "resolution": "unresolved",
            "reason": "evaluator_placeholder_not_implemented",
        }
        if family == FUNCTIONAL_SEMANTIC_FIDELITY:
            result["component"] = claim.get("component")
        return result

    if family in ("oor", "oar"):
        report = oor_report if family == "oor" else oar_report
        if not isinstance(report, dict) or not isinstance(report.get("checks"), list):
            return {
                "claim_id": claim_id,
                "family": family,
                "resolution": "unresolved",
                "reason": "evaluator_not_executed",
                "score": None,
            }
        relation_id = str(claim.get("relation_id") or claim_id)
        module_result = relation_resolution[family].get(relation_id)
        if module_result is None:
            module_result = {
                "resolution": "unresolved",
                "reason": "not_resolved_by_module",
                "score": None,
            }
        return {
            "claim_id": claim_id,
            "family": family,
            **module_result,
            "module_relation_id": relation_id,
        }

    if family == FUNCTIONAL_SEMANTIC_FIDELITY:
        if not isinstance(functional_semantic_report, dict):
            module_result = {
                "resolution": "unresolved",
                "reason": "evaluator_not_executed",
                "score": None,
            }
        else:
            module_result = functional_resolution.get(
                claim_id,
                {
                    "resolution": "unresolved",
                    "reason": "not_resolved_by_module",
                    "score": None,
                },
            )
        result = {
            "claim_id": claim_id,
            "family": family,
            "component": claim.get("component"),
            **module_result,
        }
        return result

    raise AssertionError(f"unexpected canonical L2 claim family {family!r}")


def _relation_resolution_map(
    report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    resolution: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return resolution
    for check in report.get("checks", []) if isinstance(report.get("checks"), list) else []:
        if not isinstance(check, dict):
            continue
        relation_id = str(check.get("relation_id") or "")
        if not relation_id:
            continue
        status = check.get("status")
        score = check.get("score")
        if status == "checked" and _is_score(score):
            resolution[relation_id] = {
                "resolution": "resolved",
                "reason": None,
                "score": float(score),
            }
        elif status in {"invalid_input", "vlm_adjudication_failed"}:
            resolution[relation_id] = {
                "resolution": "failed",
                "reason": (
                    "module_invalid_input"
                    if status == "invalid_input"
                    else "module_reported_failure"
                ),
                "score": None,
            }
        else:
            resolution.setdefault(
                relation_id,
                {
                    "resolution": "unresolved",
                    "reason": "not_resolved_by_module",
                    "score": None,
                },
            )
    return resolution


def _functional_semantic_resolution_map(
    report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    resolution: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return resolution
    metric_report = report
    nested_metrics = report.get("metrics")
    if isinstance(nested_metrics, dict) and isinstance(
        nested_metrics.get(FUNCTIONAL_SEMANTIC_FIDELITY), dict
    ):
        metric_report = nested_metrics[FUNCTIONAL_SEMANTIC_FIDELITY]
    checks = metric_report.get("checks")
    if not isinstance(checks, list):
        return resolution
    for check in checks:
        if not isinstance(check, dict):
            continue
        claim_id = str(check.get("claim_id") or "")
        if not claim_id:
            continue
        status = check.get("status")
        score = check.get("score")
        if status == "checked" and _is_score(score):
            resolution[claim_id] = {
                "resolution": "resolved",
                "reason": None,
                "score": float(score),
            }
        elif status in {
            "vlm_adjudication_failed",
            "camera_evidence_failed",
            "evidence_provider_failed",
        }:
            resolution[claim_id] = {
                "resolution": "failed",
                "reason": str(check.get("reason") or "module_reported_failure"),
                "score": None,
            }
        else:
            resolution[claim_id] = {
                "resolution": "unresolved",
                "reason": str(check.get("reason") or "not_resolved_by_module"),
                "score": None,
            }
    return resolution


def _is_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _oor_relation_ids(relation: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ("subject_id", "object_id"):
        if relation.get(key) is not None:
            values.append(relation[key])
    for key in ("subject_ids", "object_ids", "member_ids"):
        if isinstance(relation.get(key), list):
            values.extend(relation[key])
    return {str(value) for value in values if str(value).strip()}


def _validate_claim_targets(
    claim: dict[str, Any],
    *,
    valid_object_ids: set[str],
    path: str,
) -> None:
    for key in ("target_ids", "object_ids", "member_ids"):
        target_ids = claim.get(key)
        if target_ids is None:
            continue
        if not isinstance(target_ids, list):
            raise SpecificationContractError(f"{path}.{key} must be a list when present")
        for target in target_ids:
            if str(target) not in valid_object_ids:
                raise SpecificationContractError(
                    f"{path}.{key} references unknown object id {target!r}"
                )
    for key in ("target_id", "object_id", "subject_id"):
        if claim.get(key) is not None and str(claim[key]) not in valid_object_ids:
            raise SpecificationContractError(
                f"{path}.{key} references unknown object id {claim[key]!r}"
            )
