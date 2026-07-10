from __future__ import annotations

from typing import Any

OOR_RELATION_TYPES = {
    "near",
    "left",
    "right",
    "in_front",
    "behind",
    "above",
    "below",
    "aligned_with",
    "contact",
    "face_to",
    "within",
    "out_of",
    None,
}

OAR_RELATION_TYPES = {
    "on_floor",
    "against_wall",
    "near_wall",
    "below_wall",
    "at_corner",
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
    return {
        "family": "oor",
        "type": relation_type if relation_type in OOR_RELATION_TYPES else None,
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
    return {
        "family": "oar",
        "type": relation_type if relation_type in OAR_RELATION_TYPES else None,
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
    status: str = "tbd_passthrough",
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
        or ["Relationship mapping is a TODO. This module defines the interface only."],
    }
