from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.architecture_policy import validate_architecture_contract
from benchmark.io_contracts import I2_NATURAL_LANGUAGE_STRUCTURE, O1_OBJECT_STATE, input_type_for_mode
from benchmark.nl_scene.converter import FINE_GRAINED
from benchmark.task_contract import architecture_contract_for_room


STRUCTURED_ASSETS_INPUT_MODE = "structured_assets"
STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_structured"
DIRECT_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_direct"
GENERATION_COMPARISON_PUBLIC_KEYS = {
    "schema_version",
    "protocol_id",
    "protocol_version",
    "mode",
    "case_id",
    "architecture",
    "architecture_sha256",
    "object_inventory_policy",
    "objects",
    "object_inventory_sha256",
    "asset_policy",
    "scale_policy",
    "retrieval_policy",
    "generation",
    "catalog",
    "method_materialization",
}
GENERATION_COMPARISON_PRIVATE_KEYS = {
    "reference_annotation",
    "evaluation_context",
    "evaluation_report",
    "previous_evaluation",
    "benchmark_score",
    "hidden_metric_weights",
    "private_vlm_judgments",
    "private_evidence",
}


def build_scene_request(
    *,
    request_id: str,
    instruction: str,
    scene_type: str,
    room: dict | None,
    structure: bool = True,
    prompt_granularity: str = FINE_GRAINED,
    metadata: dict | None = None,
) -> dict:
    """Build the natural-language request artifact used by the scene harness."""

    request = {
        "request_id": str(request_id),
        "instruction": str(instruction),
        "scene_type": str(scene_type),
        "structure": bool(structure),
        "prompt_granularity": str(prompt_granularity),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }
    if room is not None:
        request["room"] = room
    return request


def build_generation_input(
    *,
    scene_request: dict,
    object_plan: dict | None = None,
    asset_selection: dict | None = None,
    evaluator_output_type: str = O1_OBJECT_STATE,
    architecture_contract: dict | None = None,
) -> dict:
    """Build the generator-facing input from public benchmark artifacts only.

    ``object_plan`` is public generator structure, not evaluator ground truth.
    It is required for structured modes and forbidden for direct NL mode. Frozen
    reference annotations are passed separately to the evaluator and must never
    be added to this artifact.
    """

    structure_value = scene_request.get("structure", True)
    if not isinstance(structure_value, bool):
        raise ValueError("scene_request.structure must be boolean")
    structure = structure_value
    if structure and not isinstance(object_plan, dict):
        raise ValueError("structured generator input requires an explicit public object_plan")
    if not structure and object_plan is not None:
        raise ValueError(
            "direct natural-language generator input must not carry object_plan; "
            "use a private reference_annotation for evaluation"
        )
    has_assets = asset_selection is not None
    if structure and has_assets:
        input_mode = STRUCTURED_ASSETS_INPUT_MODE
    elif structure:
        input_mode = STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE
    else:
        input_mode = DIRECT_NATURAL_LANGUAGE_INPUT_MODE
    architecture = (
        validate_architecture_contract(architecture_contract)
        if architecture_contract is not None
        else architecture_contract_for_room(scene_request.get("room"))
    )
    generation_input: dict[str, Any] = {
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "scene_request": scene_request,
        "generation_contract": {
            "output_format": "canonical_generated_scene_v1",
            "requires_pose": True,
            "input_mode": input_mode,
            "input_type": input_type_for_mode(input_mode),
            "evaluator_output_type": str(evaluator_output_type),
            "requires_asset_selection": structure and has_assets,
            "architecture": architecture,
        },
    }
    if structure:
        generation_input["object_plan"] = object_plan
    if structure and has_assets:
        generation_input["asset_selection"] = asset_selection
    elif structure:
        generation_input["generator_input"] = build_structured_generator_input(scene_request, object_plan)
    else:
        generation_input["generator_input"] = build_natural_language_generator_input(scene_request)
    return generation_input


def build_direct_natural_language_generation_input(
    *,
    request_id: str,
    instruction: str,
    scene_type: str,
    room: dict,
    object_plan: dict | None = None,
    metadata: dict | None = None,
    prompt_granularity: str = FINE_GRAINED,
    evaluator_output_type: str = O1_OBJECT_STATE,
    architecture_contract: dict | None = None,
) -> dict:
    """Interface-only helper for generators that expect raw natural language."""

    if object_plan is not None:
        raise ValueError(
            "object_plan is not allowed in direct natural-language mode; "
            "pass it only as public structured generator input"
        )

    scene_request = build_scene_request(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=room,
        structure=False,
        prompt_granularity=prompt_granularity,
        metadata=metadata,
    )
    return build_generation_input(
        scene_request=scene_request,
        object_plan=None,
        asset_selection=None,
        evaluator_output_type=evaluator_output_type,
        architecture_contract=architecture_contract,
    )


def build_natural_language_generator_input(scene_request: dict) -> dict:
    """Return the method-facing direct natural-language payload."""

    return {
        "input_mode": DIRECT_NATURAL_LANGUAGE_INPUT_MODE,
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "instruction": str(scene_request.get("instruction") or ""),
        "scene_type": str(scene_request.get("scene_type") or "room"),
        "room": scene_request.get("room"),
        "prompt_granularity": str(scene_request.get("prompt_granularity") or FINE_GRAINED),
    }


def build_structured_generator_input(scene_request: dict, object_plan: dict) -> dict:
    """Return method-facing natural language plus benchmark structure."""

    return {
        "input_mode": STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE,
        "request_id": str(scene_request.get("request_id") or "request_001"),
        "instruction": str(scene_request.get("instruction") or ""),
        "scene_type": str(scene_request.get("scene_type") or "room"),
        "room": scene_request.get("room"),
        "prompt_granularity": str(scene_request.get("prompt_granularity") or FINE_GRAINED),
        "object_plan": object_plan,
    }


def build_generator_visible_payload(generation_input: dict) -> dict:
    """Project canonical input onto the fields a generator is allowed to see."""

    contract = generation_input.get("generation_contract")
    if not isinstance(contract, dict):
        raise ValueError("generation_input.generation_contract must be a JSON object")
    input_mode = str(contract.get("input_mode") or DIRECT_NATURAL_LANGUAGE_INPUT_MODE)
    input_type = input_type_for_mode(input_mode)
    scene_request = generation_input.get("scene_request")
    if not isinstance(scene_request, dict):
        raise ValueError("generation_input.scene_request must be a JSON object")
    payload: dict[str, Any] = {
        "input_type": input_type,
        "natural_language": str(scene_request.get("instruction") or ""),
        "benchmark_environment": {
            "architecture": contract.get("architecture")
            or architecture_contract_for_room(scene_request.get("room")),
        },
    }
    if input_type == I2_NATURAL_LANGUAGE_STRUCTURE:
        payload["structure"] = {
            "room": scene_request.get("room"),
            "object_plan": generation_input.get("object_plan"),
        }
    if input_mode == STRUCTURED_ASSETS_INPUT_MODE:
        payload["assistance"] = {"asset_selection": generation_input.get("asset_selection")}
    comparison = generation_input.get("generation_comparison")
    if comparison is not None:
        if not isinstance(comparison, dict):
            raise ValueError("generation_input.generation_comparison must be a JSON object")
        _validate_public_comparison_projection(comparison)
        # This contract is built exclusively from public room/inventory/catalog
        # inputs. Keeping the explicit projection here prevents adapter configs
        # or evaluator-private state from leaking into external runners.
        payload["generation_comparison"] = deepcopy(comparison)
    reflection = generation_input.get("self_reflection")
    if isinstance(reflection, dict) and reflection.get("enabled") is True:
        payload["self_reflection"] = reflection
    return payload


def _validate_public_comparison_projection(value: dict) -> None:
    unknown = sorted(set(value) - GENERATION_COMPARISON_PUBLIC_KEYS)
    if unknown:
        raise ValueError(
            "generation_input.generation_comparison contains non-public fields: "
            f"{unknown}"
        )

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                name = str(key)
                if name.casefold() in GENERATION_COMPARISON_PRIVATE_KEYS:
                    raise ValueError(
                        f"{path}.{name} is evaluator-private and cannot be generator-visible"
                    )
                walk(child, f"{path}.{name}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "generation_input.generation_comparison")
