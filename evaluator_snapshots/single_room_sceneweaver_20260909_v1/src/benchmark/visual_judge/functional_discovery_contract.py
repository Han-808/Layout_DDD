"""Stable data contract for non-judging functional discovery.

This module owns only version identifiers, finite vocabularies, and the
serialized result type.  Transport, prompting, and response validation live
behind separate modules so downstream code can depend on the contract without
depending on an OpenAI-compatible model client.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


FUNCTIONAL_DISCOVERY_SCHEMA_VERSION = "functional_discovery_v6"
FUNCTIONAL_DISCOVERY_PROMPT_VERSION = "functional_discovery_v15"
FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION = "functional_affordance_ledger_v4"
FUNCTIONAL_AFFORDANCE_PROMPT_VERSION = "functional_affordance_ledger_v9"
FUNCTIONAL_RELATION_SCHEMA_VERSION = "functional_relation_audit_v4"
FUNCTIONAL_RELATION_PROMPT_VERSION = "functional_relation_audit_v10"

FUNCTIONAL_SURFACE_ROLES = frozenset(
    {
        "access_side",
        "opening_side",
        "control_side",
        "display_side",
        "seating_side",
        "interaction_side",
        "reflective_side",
        "service_side",
    }
)
FUNCTIONAL_RELATION_PREDICATES = frozenset(
    {
        "directional_correspondence",
        "relative_use_geometry",
    }
)
FUNCTIONAL_RELATION_DEPENDENCIES = frozenset(
    {
        "required",
        "contextual",
    }
)
FUNCTIONAL_COUNTERPART_MODES = frozenset(
    {
        "dedicated",
        "shared",
        "alternative",
    }
)
FUNCTIONAL_ORDINARY_MOBILITY = frozenset(
    {
        "fixed",
        "movable_companion",
        "portable_unrelated",
    }
)
# Frozen discovery-v3 records are normalized through this map before current
# check obligations are built. New model responses may not emit these tokens.
LEGACY_FUNCTIONAL_RELATION_PREDICATE_MAP = {
    "mutual_orientation": "directional_correspondence",
    "cooperative_operation": "relative_use_geometry",
    "operational_access": "relative_use_geometry",
    "shared_task_reach": "relative_use_geometry",
    "attachment_or_service_relation": "relative_use_geometry",
}
# Import compatibility for callers using the previous constant name. The
# vocabulary intentionally exposes only current atomic predicates.
FUNCTIONAL_RELATION_OBSERVATIONS = FUNCTIONAL_RELATION_PREDICATES
FUNCTIONAL_DIRECTIONALITY = frozenset(
    {
        "directed",
        "non_directed",
    }
)
FUNCTIONAL_REVIEW_STATES = frozenset(
    {"routine", "local_confirmation", "uncertain"}
)


def normalized_functional_relation_predicates(
    value: Mapping[str, Any],
    *,
    allow_implicit_legacy: bool = True,
) -> tuple[str, ...]:
    """Return atomic predicates from current or frozen relation records."""

    explicit = str(value.get("predicate") or "").strip()
    if explicit:
        if explicit not in FUNCTIONAL_RELATION_PREDICATES:
            raise ValueError(
                f"unsupported functional relation predicate {explicit!r}"
            )
        return (explicit,)
    raw = value.get("observation_kinds")
    if raw is None:
        raw = ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "functional relation observation_kinds must be a JSON list"
        )
    predicates: list[str] = []
    for item in raw:
        token = str(item or "").strip()
        predicate = (
            token
            if token in FUNCTIONAL_RELATION_PREDICATES
            else LEGACY_FUNCTIONAL_RELATION_PREDICATE_MAP.get(token)
        )
        if predicate is None:
            raise ValueError(
                f"unsupported legacy functional relation token {token!r}"
            )
        if predicate not in predicates:
            predicates.append(predicate)
    if predicates:
        return tuple(predicates)
    if allow_implicit_legacy:
        # Early discovery-v3 fixtures represented only directional
        # correspondence and omitted an observation token.
        return ("directional_correspondence",)
    raise ValueError("functional relation requires one atomic predicate")


@dataclass(frozen=True)
class FunctionalDiscoveryResult:
    """Validated discovery output with no metric-decision authority."""

    inspected_object_ids: tuple[str, ...]
    object_affordance_ledger: tuple[dict[str, Any], ...] = ()
    directed_surface_targets: tuple[dict[str, Any], ...] = ()
    within_group_correspondences: tuple[dict[str, Any], ...] = ()
    cross_group_correspondences: tuple[dict[str, Any], ...] = ()
    approach_clearance_targets: tuple[dict[str, Any], ...] = ()
    boundary_sensitive_targets: tuple[dict[str, Any], ...] = ()
    unusual_unconfirmed: tuple[dict[str, Any], ...] = ()
    relation_admission_audit: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        affordance_coverage = (
            self.coverage.get("affordance")
            if isinstance(self.coverage.get("affordance"), dict)
            else {}
        )
        defaulted_ids = {
            str(item)
            for item in affordance_coverage.get("defaulted_object_ids") or []
        }
        return {
            "schema_version": FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
            "inspected_object_ids": list(self.inspected_object_ids),
            "object_coverage": [
                {
                    "object_id": object_id,
                    "inspected": object_id not in defaulted_ids,
                    "defaulted": object_id in defaulted_ids,
                    **deepcopy(
                        next(
                            (
                                {
                                    key: value
                                    for key, value in item.items()
                                    if key != "object_id"
                                }
                                for item in self.object_affordance_ledger
                                if item.get("object_id") == object_id
                            ),
                            {},
                        )
                    ),
                }
                for object_id in self.inspected_object_ids
            ],
            "object_affordance_ledger": list(
                deepcopy(self.object_affordance_ledger)
            ),
            "directed_surface_targets": list(
                deepcopy(self.directed_surface_targets)
            ),
            "within_group_correspondences": list(
                deepcopy(self.within_group_correspondences)
            ),
            "cross_group_correspondences": list(
                deepcopy(self.cross_group_correspondences)
            ),
            "approach_clearance_targets": list(
                deepcopy(self.approach_clearance_targets)
            ),
            "boundary_sensitive_targets": list(
                deepcopy(self.boundary_sensitive_targets)
            ),
            "unusual_unconfirmed": list(
                deepcopy(self.unusual_unconfirmed)
            ),
            "relation_admission_audit": deepcopy(
                self.relation_admission_audit
            ),
            "reason": self.reason,
            "coverage": deepcopy(self.coverage),
            "decision_authority": "none",
            "provenance": deepcopy(self.provenance),
        }


__all__ = [
    "FUNCTIONAL_AFFORDANCE_PROMPT_VERSION",
    "FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION",
    "FUNCTIONAL_DIRECTIONALITY",
    "FUNCTIONAL_COUNTERPART_MODES",
    "FUNCTIONAL_DISCOVERY_PROMPT_VERSION",
    "FUNCTIONAL_DISCOVERY_SCHEMA_VERSION",
    "FUNCTIONAL_RELATION_PREDICATES",
    "FUNCTIONAL_RELATION_OBSERVATIONS",
    "FUNCTIONAL_RELATION_DEPENDENCIES",
    "FUNCTIONAL_RELATION_PROMPT_VERSION",
    "FUNCTIONAL_RELATION_SCHEMA_VERSION",
    "FUNCTIONAL_REVIEW_STATES",
    "FUNCTIONAL_ORDINARY_MOBILITY",
    "FUNCTIONAL_SURFACE_ROLES",
    "LEGACY_FUNCTIONAL_RELATION_PREDICATE_MAP",
    "FunctionalDiscoveryResult",
    "normalized_functional_relation_predicates",
]
