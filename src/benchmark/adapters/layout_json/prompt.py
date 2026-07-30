from __future__ import annotations

import json
from copy import deepcopy

from benchmark.nl_scene.generation_input import build_generator_visible_payload


LAYOUT_JSON_VERSION = "layout_json_v1"
MIN_CORNER_ORIGIN = "room_min_corner_floor"
CENTER_ORIGIN = "room_center_floor"
AXIS_CONVENTION = "x_width_y_depth_z_up"
COORDINATE_UNIT = "meter"
ROTATION_UNIT = "degree"

SYSTEM_PROMPT = """You generate explicit 3D room layouts as one JSON object.
Return JSON only, without Markdown or explanation. Use meters and declare coordinate_frame,
including rotation_unit="degree".
The axes are fixed: +x follows room width, +y follows room depth, and +z is up; the floor is z=0.
The benchmark environment supplies the resolved room for this case. Do not choose or resize it. You may use
room_min_corner_floor, where a room of size [W,D,H] spans x=[0,W] and y=[0,D],
or room_center_floor, where it spans x=[-W/2,W/2] and y=[-D/2,D/2]. Prefer
room_min_corner_floor and use the declared origin consistently for the room and every object.
Object center is [x, y, z], size is [width, depth, height], and rotation is
[x_degrees, y_degrees, z_degrees]. Account for object half-sizes when keeping objects inside
the room. Avoid unintended collisions and preserve requested descriptions and relationships.
Object id values are local identifiers for the generated layout; they do not need to copy
internal object-plan ids.
Every relationship must declare family="oor" for object-object or family="oar" for
object-architecture. It must be binary: subject, predicate, and object must each be one string.
Never put an array or object in a relationship field. When the same predicate applies to
multiple targets, emit one relationship entry per target. subject must reference an emitted
object id. object must reference either an emitted object id or one of the benchmark architecture
tokens: north_wall, south_wall, east_wall, west_wall, floor, or ceiling. The benchmark room has no doors,
windows, columns, partitions, or other architecture. Never invent region pseudo-ids such as
foot_of_object_1."""

OUTPUT_CONTRACT = {
    "schema_version": LAYOUT_JSON_VERSION,
    "scene_type": "room",
    "coordinate_frame": {
        "origin": MIN_CORNER_ORIGIN,
        "axes": AXIS_CONVENTION,
        "unit": COORDINATE_UNIT,
        "rotation_unit": ROTATION_UNIT,
    },
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
        {
            "family": "oor",
            "subject": "object_id",
            "predicate": "left|right|in_front|behind|above|below|near|far|contact|within|contains|aligned|parallel|perpendicular",
            "object": "object_id",
        }
    ],
}


def build_layout_json_method_input(generation_input: dict) -> dict:
    """Build the model-facing request without exposing hidden benchmark structure."""

    contract = generation_input.get("generation_contract") if isinstance(generation_input.get("generation_contract"), dict) else {}
    input_mode = str(contract.get("input_mode") or "natural_language_direct")
    visible = build_generator_visible_payload(generation_input)
    user_payload = {
        "natural_language": visible["natural_language"],
        "benchmark_environment": visible["benchmark_environment"],
    }
    if isinstance(visible.get("structure"), dict):
        user_payload.update(visible["structure"])
    assistance = visible.get("assistance")
    asset_grounding_instructions = ""
    output_contract = deepcopy(OUTPUT_CONTRACT)
    if isinstance(assistance, dict) and assistance.get("asset_selection") is not None:
        user_payload["asset_selection"] = assistance["asset_selection"]
        output_contract["objects"][0]["asset_id"] = "selected_asset.jid"
        asset_grounding_instructions = (
            "\nAsset-grounded mode:\n"
            "- Match each generated object semantically to asset_selection.object_spec using category and description.\n"
            "- Every emitted object must explicitly set asset_id to a selected_asset.jid present in asset_selection.\n"
            "- The adapter performs exact asset_id lookup only; it will not infer a missing or incorrect binding.\n"
            "- Do not match or retrieve assets using object ids. object_id is internal provenance only, and your output id may differ.\n"
            "- Create the requested count of instances; repeated instances may reuse the same selected asset.\n"
        )
    reflection = visible.get("self_reflection")
    if isinstance(reflection, dict):
        user_payload["repair_context"] = {
            "instruction": "Return a complete revised layout that fixes the reported deterministic evaluation failures.",
            "previous_generated_scene": reflection.get("previous_generated_scene"),
            "previous_evaluation": reflection.get("previous_evaluation"),
        }
    user_prompt = (
        "Generate the requested scene using this output-shape example. Replace all example values and add every requested object:\n"
        f"{json.dumps(output_contract, ensure_ascii=True, separators=(',', ':'))}\n"
        f"{asset_grounding_instructions}\n"
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
