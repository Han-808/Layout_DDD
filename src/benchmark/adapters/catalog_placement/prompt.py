from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from benchmark.nl_scene.generation_input import (
    STRUCTURED_ASSETS_INPUT_MODE,
    build_generator_visible_payload,
)


CATALOG_PLACEMENT_VERSION = "catalog_placement_v1"

SYSTEM_PROMPT = """You place instances of benchmark-provided frozen catalog assets.
Return exactly one JSON object and no explanation or Markdown. The only root fields are
schema_version and instances. schema_version, when present, is "catalog_placement_v1".
Each instance has exactly instance_id, asset_id, center_m, target_size_m,
rotation_euler_xyz_deg, and optionally slot_id. Never emit category, description,
scene type, coordinate declarations, relationships, evaluator IDs, metric claims,
verdicts, scores, lighting, materials, cameras, or rendering settings.

instance_id is generator-owned identity. It must be unique and remain stable if the
instances array is reordered. Reusing one asset for multiple instances is allowed, but
every instance still needs a distinct instance_id. asset_id must exactly equal a
selected_asset.jid from the supplied frozen catalog.

Coordinates are fixed and must not be declared in the output: meters, z-up, +x along
room width, +y along room depth, with the room-min-corner floor as origin. center_m is
the world position of the scaled catalog canonical local-bbox center. target_size_m is
a positive pre-rotation envelope along the asset's canonical local axes. The benchmark
uses uniform contain-fit scaling: min(target_size_m[i] / catalog_bbox_size_m[i]).
rotation_euler_xyz_deg is intrinsic XYZ Euler degrees, applied to column vectors as
Rz @ Ry @ Rx about the canonical bbox center. Express floor or support placement
directly through center_m; there is no vertical-anchor field.

slot_id is optional provenance. It may be copied only from public_slot_ids supplied
below. It does not bind the asset, choose evaluator identity or category, or make any
metric claim."""

OUTPUT_CONTRACT = {
    "schema_version": CATALOG_PLACEMENT_VERSION,
    "instances": [
        {
            "instance_id": "chair_left",
            "asset_id": "selected_asset.jid",
            "center_m": [1.0, 2.0, 0.5],
            "target_size_m": [0.8, 0.8, 1.0],
            "rotation_euler_xyz_deg": [0.0, 0.0, 90.0],
            "slot_id": "optional_public_slot_id",
        }
    ],
}


def public_slot_ids_from_generation_input(generation_input: dict) -> list[str]:
    """Return only slots derivable from the public structured object plan."""

    object_plan = generation_input.get("object_plan")
    if not isinstance(object_plan, dict):
        return []
    slots: list[str] = []
    for item in object_plan.get("objects", []):
        if not isinstance(item, dict):
            continue
        # Use only a literal identifier already present in generator-visible
        # structure. If a future object-plan schema exposes slot_id directly,
        # it supersedes the object's public id; count never creates new names.
        literal_slot_id = str(item.get("slot_id") or item.get("id") or "").strip()
        if not literal_slot_id:
            continue
        if literal_slot_id not in slots:
            slots.append(literal_slot_id)
    return slots


def build_catalog_placement_method_input(generation_input: dict) -> dict:
    """Build the model-facing fixed-catalog placement request."""

    contract = (
        generation_input.get("generation_contract")
        if isinstance(generation_input.get("generation_contract"), dict)
        else {}
    )
    input_mode = str(contract.get("input_mode") or "")
    if input_mode != STRUCTURED_ASSETS_INPUT_MODE:
        raise ValueError(
            "catalog_placement requires generation_contract.input_mode='structured_assets'"
        )
    visible = build_generator_visible_payload(generation_input)
    assistance = visible.get("assistance")
    asset_selection = (
        assistance.get("asset_selection") if isinstance(assistance, dict) else None
    )
    if not isinstance(asset_selection, dict):
        raise ValueError("catalog_placement requires generator-visible asset_selection")

    public_slots = public_slot_ids_from_generation_input(generation_input)
    user_payload: dict[str, Any] = {
        "natural_language": visible["natural_language"],
        "benchmark_environment": visible["benchmark_environment"],
        "structure": visible.get("structure"),
        "asset_selection": deepcopy(asset_selection),
        "public_slot_ids": public_slots,
    }
    reflection = visible.get("self_reflection")
    if isinstance(reflection, dict):
        user_payload["repair_context"] = {
            "instruction": (
                "Return a complete revised placement that fixes the reported deterministic "
                "evaluation failures while preserving stable instance_id values."
            ),
            "previous_generated_scene": reflection.get("previous_generated_scene"),
            "previous_evaluation": reflection.get("previous_evaluation"),
        }

    user_prompt = (
        "Output-shape example (replace every example value and include every intended instance):\n"
        f"{json.dumps(OUTPUT_CONTRACT, ensure_ascii=True, separators=(',', ':'))}\n"
        "Generator-visible request and frozen selected-asset catalog:\n"
        f"{json.dumps(user_payload, ensure_ascii=True, separators=(',', ':'))}"
    )
    return {
        "adapter": "catalog_placement",
        "provider": "openai_compatible",
        "output_schema": CATALOG_PLACEMENT_VERSION,
        "input_mode": input_mode,
        "request_id": str(generation_input.get("request_id") or "request_001"),
        "public_slot_ids": public_slots,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
