from __future__ import annotations

from typing import Any

OOR_RELATION_TYPES = {
    "left",
    "right",
    "in_front",
    "behind",
    "above",
    "below",
    "near",
    "far",
    "contact",
    "on_top_of",
    "within",
    "contains",
    "aligned",
    "parallel",
    "perpendicular",
    "between",
    "ordered",
    "around",
    None,
}

OAR_RELATION_TYPES = {
    "on_floor",
    "against_wall",
    "near_wall",
    "at_corner",
    "near_corner",
    "room_center",
    "room_region",
    "along_wall",
    "mounted_on_wall",
    "attached_to_ceiling",
    "hung_from_ceiling",
    None,
}

RELATION_SOURCES = {"explicit_text", "inferred", "manual", "unknown"}


def oor_relation_intent(
    *,
    relation_type: str | None,
    subject_id: str,
    anchor_id: str,
    raw_relation: str = "",
    confidence: float | None = None,
    source: str = "unknown",
    reason: str = "",
) -> dict[str, Any]:
    canonical_type = _lossless_relation_type(relation_type)
    return {
        "family": "oor",
        "type": canonical_type,
        "subject_id": subject_id,
        "anchor_id": anchor_id,
        "raw_relation": raw_relation,
        "confidence": confidence,
        "source": source if source in RELATION_SOURCES else "unknown",
        "reason": reason,
    }


def oar_relation_intent(
    *,
    relation_type: str | None,
    subject_id: str,
    architectural_element_type: str = "unknown",
    wall: str | None = None,
    corner: str | None = None,
    raw_relation: str = "",
    confidence: float | None = None,
    source: str = "unknown",
    reason: str = "",
) -> dict[str, Any]:
    canonical_type = _lossless_relation_type(relation_type)
    return {
        "family": "oar",
        "type": canonical_type,
        "subject_id": subject_id,
        "architectural_element_type": architectural_element_type,
        "wall": wall,
        "corner": corner,
        "raw_relation": raw_relation,
        "confidence": confidence,
        "source": source if source in RELATION_SOURCES else "unknown",
        "reason": reason,
    }


def relationship_intent_document(
    *,
    request_id: str,
    status: str = "mapped",
    oor_relations: list[dict[str, Any]] | None = None,
    oar_relations: list[dict[str, Any]] | None = None,
    unsupported_relations: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": status,
        "oor_relations": oor_relations or [],
        "oar_relations": oar_relations or [],
        "unsupported_relations": unsupported_relations or [],
        "notes": notes
        or [
            "The converter maps explicit text to the frozen relation registry.",
            "Unknown explicit predicates are preserved for evaluator-side VLM adjudication.",
        ],
    }


def _lossless_relation_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
    return normalized or None
