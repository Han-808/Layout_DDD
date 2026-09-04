"""Evaluator-owned normalization of Agent intent and shared-DB asset identity."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from benchmark.non_rectangular import (
    NonRectangularEvaluationInput,
    prepare_non_rectangular_evaluation,
    validate_multi_room_scene,
)
from benchmark.scene_generation.non_rectangular_multi_room.contracts import (
    materialize_generated_scene,
)

from .contracts import ValidatedAgentSubmission


def materialize_agent_scene(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    validated: ValidatedAgentSubmission,
    generation_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create canonical output while refusing Agent-authored asset semantics."""

    scene, _ = materialize_generated_scene(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=validated.object_plan,
        asset_selection=validated.asset_selection,
        placement=validated.global_placement,
        generation_mode=generation_mode,
    )
    selected = {
        (str(room["room_id"]), str(item["slot_id"])): item["selected_asset"]
        for room in validated.asset_selection["rooms"]
        for item in room["objects"]
    }
    normalized = deepcopy(scene)
    for room in normalized["rooms"]:
        room_id = str(room["room_id"])
        for instance in room["objects"]:
            slot_id = str(instance["slot_id"])
            asset = selected[(room_id, slot_id)]
            metadata = (
                dict(instance["metadata"])
                if isinstance(instance.get("metadata"), Mapping)
                else {}
            )
            metadata["agent_intended_task_slot"] = deepcopy(
                metadata.get("task_slot")
            )
            metadata["asset_identity_source"] = (
                "evaluator_owned_frozen_shared_database"
            )
            instance["category"] = str(asset["category"])
            instance["description"] = str(asset["desc"])
            instance["metadata"] = metadata
    validate_multi_room_scene(normalized)
    preflight = prepare_non_rectangular_evaluation(
        NonRectangularEvaluationInput.from_artifacts(
            room_layout=room_layout,
            room_program=room_program,
            object_plan=validated.object_plan,
            generated_scene=normalized,
        )
    )
    if not preflight.should_run_room_evaluation:
        raise ValueError(
            "Agent-normalized scene unexpectedly failed evaluation preflight"
        )
    return normalized, preflight.public_dict()


__all__ = ["materialize_agent_scene"]
