from __future__ import annotations

from typing import Any

from benchmark.relationship_mapper.schemas import relationship_intent_document


def map_relationships(
    *,
    scene_request: dict,
    object_plan: dict,
    model_config: dict | None = None,
    mode: str = "passthrough",
) -> dict[str, Any]:
    """Separate converter-mapped explicit claims into OOR/OAR families.

    The NL converter performs the semantic mapping. This boundary is deliberately
    lossless: predicates outside the frozen registry remain intact so the
    evaluator can adjudicate them with visual evidence.
    """

    if mode == "vlm":
        raise NotImplementedError("Use the NL converter for VLM relation mapping; this boundary is lossless passthrough only.")
    if mode != "passthrough":
        raise ValueError(f"Unsupported relationship mapper mode: {mode}")
    _ = model_config
    request_id = str(object_plan.get("request_id") or scene_request.get("request_id") or "request_001")
    oor_relations: list[dict[str, Any]] = []
    oar_relations: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for relation in object_plan.get("relations", []) if isinstance(object_plan.get("relations"), list) else []:
        if not isinstance(relation, dict):
            unsupported.append({"raw_relation": relation, "reason": "non-dict relation intent"})
            continue
        family = relation.get("family")
        if family == "oor":
            oor_relations.append(dict(relation))
        elif family == "oar":
            oar_relations.append(dict(relation))
        else:
            unsupported.append({"raw_relation": relation, "reason": "relation family must be oor or oar"})
    return relationship_intent_document(
        request_id=request_id,
        oor_relations=oor_relations,
        oar_relations=oar_relations,
        unsupported_relations=unsupported,
    )
