from __future__ import annotations

import json


LAYOUT_JSON_VERSION = "layout_json_v1"

SYSTEM_PROMPT = """You generate explicit 3D room layouts as one JSON object.
Return JSON only, without Markdown or explanation. Use meters. The floor is z=0.
Object center is [x, y, z], size is [width, depth, height], and rotation is
[x_degrees, y_degrees, z_degrees]. Keep every object inside the room and avoid
unintended collisions. Preserve the requested object descriptions and spatial relationships."""

OUTPUT_CONTRACT = {
    "schema_version": LAYOUT_JSON_VERSION,
    "scene_type": "room",
    "room": {"size": [5.0, 5.0, 2.9]},
    "objects": [
        {
            "id": "object_1",
            "category": "chair",
            "description": "requested chair appearance and role",
            "center": [1.0, 1.0, 0.45],
            "size": [0.6, 0.6, 0.9],
            "rotation": [0.0, 0.0, 0.0],
        }
    ],
    "relationships": [
        {"subject": "object_id", "predicate": "near|left|right|in_front|behind|above|below|contact|face_to|within", "object": "object_id"}
    ],
}


def build_layout_json_method_input(generation_input: dict) -> dict:
    """Build the model-facing request without exposing hidden benchmark structure."""

    contract = generation_input.get("generation_contract") if isinstance(generation_input.get("generation_contract"), dict) else {}
    input_mode = str(contract.get("input_mode") or "natural_language_direct")
    user_payload = _generator_payload(generation_input, input_mode)
    reflection = generation_input.get("self_reflection")
    if isinstance(reflection, dict) and reflection.get("enabled"):
        user_payload["repair_context"] = {
            "instruction": "Return a complete revised layout that fixes the reported deterministic evaluation failures.",
            "previous_generated_scene": reflection.get("previous_generated_scene"),
            "previous_evaluation": reflection.get("previous_evaluation"),
        }
    user_prompt = (
        "Generate the requested scene using this output-shape example. Replace all example values and add every requested object:\n"
        f"{json.dumps(OUTPUT_CONTRACT, ensure_ascii=True, separators=(',', ':'))}\n\n"
        "Scene request:\n"
        f"{json.dumps(user_payload, ensure_ascii=True, separators=(',', ':'))}"
    )
    return {
        "adapter": "layout_json",
        "provider": "openai_compatible",
        "output_schema": LAYOUT_JSON_VERSION,
        "input_mode": input_mode,
        "request_id": str(generation_input.get("request_id") or "request_001"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def _generator_payload(generation_input: dict, input_mode: str) -> dict:
    generator_input = generation_input.get("generator_input") if isinstance(generation_input.get("generator_input"), dict) else {}
    scene_request = generation_input.get("scene_request") if isinstance(generation_input.get("scene_request"), dict) else {}
    instruction = str(generator_input.get("instruction") or scene_request.get("instruction") or "")
    if input_mode == "natural_language_direct":
        return {"natural_language": instruction}

    payload: dict = {
        "natural_language": instruction,
        "room": generator_input.get("room") or scene_request.get("room"),
        "object_plan": generator_input.get("object_plan") or generation_input.get("object_plan"),
    }
    if input_mode == "structured_assets":
        payload["asset_selection"] = generation_input.get("asset_selection")
    return payload
