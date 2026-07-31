"""Asset policy: an orthogonal dimension to prompt granularity.

Prompt granularity (``fine_grained`` / ``coarse_grained``) describes how much the
prompt specifies. Asset policy describes *who owns* the assets and their
identity, category selection, scale, appearance, and arrangement. The two are
independent dimensions: asset policy is never inferred from prompt granularity,
and every prompt-granularity x asset-mode combination is structurally legal
unless an unrelated explicit constraint forbids it.

This module only validates the structure and derives *applicability-only*
metadata for the implemented L3 Scene Quality metrics. It never produces a
valid/invalid metric verdict. ``relevant`` includes a metric in the canonical
L3 denominator, ``not_relevant`` excludes it, and ``pending`` keeps the metric
explicitly unresolved. Asset policy therefore changes applicability within the
single canonical workflow; it never selects another workflow.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ASSET_MODES = (
    "benchmark_provided",
    "fixed_catalog_selection",
    "retrieval_allowed",
    "generated_or_open_assets",
)
ASSET_OWNERS = ("benchmark", "generator")
ASSET_OWNER_ROLES = (
    "identity_owner",
    "category_selection_owner",
    "scale_owner",
    "appearance_owner",
    "arrangement_owner",
)

# Conservative default when a policy is absent: the benchmark owns everything and
# provides the assets. Callers that omit ``asset_policy`` are treated as legacy
# and this default is only used for declarative metadata, never to change scores.
DEFAULT_ASSET_POLICY: dict[str, Any] = {
    "mode": "benchmark_provided",
    "identity_owner": "benchmark",
    "category_selection_owner": "benchmark",
    "scale_owner": "benchmark",
    "appearance_owner": "benchmark",
    "arrangement_owner": "benchmark",
}


class AssetPolicyError(ValueError):
    """Raised when an asset policy structure is malformed."""


def validate_asset_policy(value: Any) -> dict[str, Any]:
    """Validate and normalize an asset policy structure.

    ``mode`` must be one of :data:`ASSET_MODES`; each owner role must be one of
    :data:`ASSET_OWNERS`. Missing owner roles default to ``benchmark``. Unknown,
    future-compatible fields are preserved. This never infers any field from
    prompt granularity.
    """

    if not isinstance(value, dict):
        raise AssetPolicyError("asset_policy must be a JSON object")
    resolved = deepcopy(DEFAULT_ASSET_POLICY)

    mode = value.get("mode", resolved["mode"])
    if mode not in ASSET_MODES:
        raise AssetPolicyError(
            f"asset_policy.mode must be one of {list(ASSET_MODES)}, got {mode!r}"
        )
    resolved["mode"] = mode

    for role in ASSET_OWNER_ROLES:
        if role in value:
            owner = value[role]
            if owner not in ASSET_OWNERS:
                raise AssetPolicyError(
                    f"asset_policy.{role} must be one of {list(ASSET_OWNERS)}, got {owner!r}"
                )
            resolved[role] = owner

    for key, extra in value.items():
        if key not in resolved and key != "mode":
            resolved[key] = deepcopy(extra)
    return resolved


def resolve_asset_policy(value: Any | None) -> dict[str, Any] | None:
    """Return a validated asset policy, or ``None`` when none is supplied.

    A ``None`` result preserves backward compatibility for jobs that never
    declared an asset policy.
    """

    if value is None:
        return None
    return validate_asset_policy(value)


def _owner(policy: dict[str, Any], role: str) -> str:
    return str(policy.get(role, DEFAULT_ASSET_POLICY[role]))


def scene_quality_applicability(policy: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Declarative L3 metric applicability derived from an asset policy.

    Returns applicability metadata only (``relevant`` / ``not_relevant`` /
    ``pending``); it is never a pass/fail verdict. When no policy is supplied,
    all metrics are conservatively ``pending`` rather than implicitly relevant
    or valid.
    """

    if policy is None:
        return {
            metric: {
                "applicability": "pending",
                "basis": [],
                "reason": "asset_policy_not_declared",
                "decision_role": "applicability_only",
                "workflow": "canonical_l0_l4",
            }
            for metric in (
                "scale_consistency",
                "object_pairing_consistency",
                "style_consistency",
                "functional_consistency",
            )
        }

    mode = str(policy.get("mode", DEFAULT_ASSET_POLICY["mode"]))
    generator_controls_assets = mode in ("generated_or_open_assets",)
    selection_modes = ("fixed_catalog_selection", "retrieval_allowed", "generated_or_open_assets")

    # scale_consistency: relevant when the generator controls scale, instance
    # transforms, or asset generation.
    scale_basis: list[str] = []
    if _owner(policy, "scale_owner") == "generator":
        scale_basis.append("scale_owner=generator")
    if _owner(policy, "arrangement_owner") == "generator":
        scale_basis.append("arrangement_owner=generator")
    if generator_controls_assets:
        scale_basis.append(f"mode={mode}")

    # object_pairing_consistency: category/role compatibility only. Arrangement
    # ownership does not activate it; position/orientation/function are outside
    # the canonical pairing verdict.
    pairing_basis: list[str] = []
    if _owner(policy, "category_selection_owner") == "generator":
        pairing_basis.append("category_selection_owner=generator")

    # style_consistency: relevant when the generator controls asset appearance,
    # materials, generation, or asset selection.
    style_basis: list[str] = []
    if _owner(policy, "appearance_owner") == "generator":
        style_basis.append("appearance_owner=generator")
    if _owner(policy, "category_selection_owner") == "generator":
        style_basis.append("category_selection_owner=generator")
    if mode in selection_modes:
        style_basis.append(f"mode={mode}")

    # Functional consistency concerns generic real-world usability of the
    # generated arrangement, independent of whether the prompt requested a
    # specific function.
    functional_basis: list[str] = []
    if _owner(policy, "arrangement_owner") == "generator":
        functional_basis.append("arrangement_owner=generator")
    if _owner(policy, "category_selection_owner") == "generator":
        functional_basis.append("category_selection_owner=generator")

    return {
        "scale_consistency": _applicability_record(scale_basis),
        "object_pairing_consistency": _applicability_record(pairing_basis),
        "style_consistency": _applicability_record(style_basis),
        "functional_consistency": _applicability_record(
            functional_basis
        ),
    }


def _applicability_record(basis: list[str]) -> dict[str, Any]:
    return {
        "applicability": "relevant" if basis else "not_relevant",
        "basis": basis,
        "decision_role": "applicability_only",
        "workflow": "canonical_l0_l4",
    }
