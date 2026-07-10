from __future__ import annotations

from typing import Any


STRUCTURED_ASSETS_INPUT_MODE = "structured_assets"
STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_structured"
DIRECT_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_direct"


def build_scene_request(
    *,
    request_id: str,
    instruction: str,
    scene_type: str,
    room: dict,
    structure: bool = True,
    metadata: dict | None = None,
) -> dict:
    """Build the natural-language request artifact used by the scene harness."""

    return {
        "request_id": str(request_id),
        "instruction": str(instruction),
        "scene_type": str(scene_type),
        "room": room,
        "structure": bool(structure),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def build_generation_input(
    *,
    scene_request: dict,
    object_plan: dict,
    asset_selection: dict | None = None,
) -> dict:
    """Build generation_input without coupling structure to asset retrieval."""

    structure_value = scene_request.get("structure", True)
    if not isinstance(structure_value, bool):
        raise ValueError("scene_request.structure must be boolean")
    structure = structure_value
    has_assets = asset_selection is not None
    if structure and has_assets:
        input_mode = STRUCTURED_ASSETS_INPUT_MODE
    elif structure:
        input_mode = STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE
    else:
        input_mode = DIRECT_NATURAL_LANGUAGE_INPUT_MODE
    generation_input: dict[str, Any] = {
        "request_id": str(scene_request.get("request_id") or object_plan.get("request_id") or "request_001"),
        "scene_request": scene_request,
        "object_plan": object_plan,
        "generation_contract": {
            "output_format": "canonical_generated_scene_v1",
            "requires_pose": True,
            "input_mode": input_mode,
            "requires_asset_selection": structure and has_assets,
        },
    }
    if structure and has_assets:
        generation_input["asset_selection"] = asset_selection
    elif structure:
        generation_input["generator_input"] = build_structured_generator_input(scene_request, object_plan)
        generation_input["evaluation_context"] = {
            "object_plan": object_plan,
            "asset_retrieval_skipped": True,
            "asset_selection_required": False,
            "structure_available_to_generator": True,
        }
    else:
        generation_input["generator_input"] = build_natural_language_generator_input(scene_request)
        generation_input["evaluation_context"] = {
            "object_plan": object_plan,
            "asset_retrieval_skipped": True,
            "asset_selection_required": False,
        }
    return generation_input


def build_direct_natural_language_generation_input(
    *,
    request_id: str,
    instruction: str,
    scene_type: str,
    room: dict,
    object_plan: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Interface-only helper for generators that expect raw natural language."""

    scene_request = build_scene_request(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=room,
        structure=False,
        metadata=metadata,
    )
    plan = object_plan if isinstance(object_plan, dict) else _empty_object_plan(scene_request)
    return build_generation_input(scene_request=scene_request, object_plan=plan, asset_selection=None)


def build_natural_language_generator_input(scene_request: dict) -> dict:
    """Return the method-facing direct natural-language payload."""

    return {
        "input_mode": DIRECT_NATURAL_LANGUAGE_INPUT_MODE,
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "instruction": str(scene_request.get("instruction") or ""),
        "scene_type": str(scene_request.get("scene_type") or "room"),
        "room": scene_request.get("room"),
    }


def build_structured_generator_input(scene_request: dict, object_plan: dict) -> dict:
    """Return method-facing natural language plus benchmark structure."""

    return {
        "input_mode": STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE,
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "instruction": str(scene_request.get("instruction") or ""),
        "scene_type": str(scene_request.get("scene_type") or "room"),
        "room": scene_request.get("room"),
        "object_plan": object_plan,
    }


def _empty_object_plan(scene_request: dict) -> dict:
    instruction = str(scene_request.get("instruction") or "")
    return {
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "scene_type": str(scene_request.get("scene_type") or "room"),
        "scene_description": instruction,
        "objects": [],
        "global_constraints": [],
        "relations": [],
    }
