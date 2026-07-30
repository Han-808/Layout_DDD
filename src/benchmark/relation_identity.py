"""Stable identity helpers for benchmark-owned relationship claims."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


RELATION_FAMILIES = {"oor", "oar"}


def normalize_relation_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def provisional_relation_id(family: str, index: int) -> str:
    family_name = str(family).strip().lower()
    if family_name not in RELATION_FAMILIES:
        raise ValueError(f"unsupported relationship family {family_name!r}")
    if isinstance(index, bool) or int(index) < 0:
        raise ValueError("relationship index must be a non-negative integer")
    return f"{family_name}_{int(index):03d}"


def with_relation_ids(relations: Iterable[dict], *, family: str) -> list[dict[str, Any]]:
    """Return copied claims with unique IDs, upgrading legacy positional claims.

    New benchmark annotations write IDs at authoring time. This compatibility
    path keeps older frozen fixtures usable while making the upgrade explicit in
    each relation rather than silently relying on a downstream list index.
    """

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            continue
        claim = deepcopy(relation)
        relation_id = normalize_relation_id(claim.get("relation_id"))
        if relation_id is None:
            relation_id = provisional_relation_id(family, index)
            claim["relation_id_generated"] = True
            claim["relation_id_provenance"] = "legacy_family_index"
        if relation_id in seen:
            raise ValueError(f"duplicate relationship relation_id {relation_id!r}")
        seen.add(relation_id)
        claim["relation_id"] = relation_id
        normalized.append(claim)
    return normalized


def copy_relation_identity(result: dict[str, Any], relation: dict | None) -> dict[str, Any]:
    """Attach claim identity to an evaluator result without changing its verdict."""

    if not isinstance(relation, dict):
        return result
    relation_id = normalize_relation_id(relation.get("relation_id"))
    if relation_id is not None:
        result["relation_id"] = relation_id
    if relation.get("relation_id_generated") is True:
        result["relation_id_generated"] = True
        result["relation_id_provenance"] = str(
            relation.get("relation_id_provenance") or "legacy_family_index"
        )
    return result
