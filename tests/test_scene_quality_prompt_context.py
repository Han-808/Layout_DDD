from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.evaluator.scene_quality.interfaces import (
    evaluate_scene_quality_interfaces,
)
from benchmark.evaluator.scene_quality.prompt_context import (
    METRIC_PROMPT_CONTEXT_VERSION,
    metric_prompt_context,
    resolve_scene_quality_prompt_context,
)


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "context-scene",
        "request_id": "context-request",
        "scene_type": "game room",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.8,
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "center": [1.0, 1.0, 0.5],
                "size": [0.6, 0.6, 1.0],
                "rotation": [0, 0, 0],
            },
            {
                "id": "table",
                "category": "table",
                "center": [2.0, 1.0, 0.4],
                "size": [1.2, 0.8, 0.8],
                "rotation": [0, 0, 0],
            },
        ],
    }


def test_default_context_is_short_and_metric_scoped() -> None:
    full_prompt = (
        "Create a game room with a chair facing a specific table and obey "
        "several detailed relations."
    )
    resolved = resolve_scene_quality_prompt_context(
        scene=_scene(),
        original_prompt=full_prompt,
    )

    pairing = metric_prompt_context(
        resolved,
        "object_pairing_consistency",
    )
    style = metric_prompt_context(resolved, "style_consistency")
    functional = metric_prompt_context(
        resolved,
        "functional_consistency",
    )

    assert pairing["values"] == {"room_type": "game room"}
    assert style["values"] == {"room_type": "game room"}
    assert "stereotypical" in pairing["rendered_prompt"]
    assert "mandatory aesthetic" in style["rendered_prompt"]
    assert full_prompt not in pairing["rendered_prompt"]
    assert functional["rendered_prompt"] is None
    assert pairing["original_prompt_included"] is False


def test_context_interface_can_opt_in_fields_per_metric() -> None:
    full_prompt = "Create a quiet hybrid game and reading room."
    resolved = resolve_scene_quality_prompt_context(
        scene=_scene(),
        original_prompt=full_prompt,
        override={
            "schema_version": METRIC_PROMPT_CONTEXT_VERSION,
            "values": {"task_summary": "Hybrid leisure space."},
            "metric_fields": {
                "semantic_placement_consistency": [
                    "room_type",
                    "task_summary",
                ],
                "functional_consistency": ["original_prompt"],
            },
            "metric_instructions": {
                "semantic_placement_consistency": (
                    "Use this context only to disambiguate scene zones."
                )
            },
            "source": "frozen_public_brief",
        },
    )

    placement = metric_prompt_context(
        resolved,
        "semantic_placement_consistency",
    )
    functional = metric_prompt_context(
        resolved,
        "functional_consistency",
    )
    assert placement["values"] == {
        "room_type": "game room",
        "task_summary": "Hybrid leisure space.",
    }
    assert placement["source"] == "frozen_public_brief"
    assert functional["values"] == {"original_prompt": full_prompt}
    assert functional["original_prompt_included"] is True


def test_context_interface_rejects_unfrozen_or_missing_fields() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        resolve_scene_quality_prompt_context(
            scene=_scene(),
            original_prompt=None,
            override={"metric_fields": {"made_up_metric": []}},
        )
    with pytest.raises(ValueError, match="references missing values"):
        resolve_scene_quality_prompt_context(
            scene=_scene(),
            original_prompt=None,
            override={
                "metric_fields": {
                    "style_consistency": ["style_descriptor"]
                }
            },
        )


def test_evaluator_routes_short_context_not_full_prompt(tmp_path: Path) -> None:
    image = tmp_path / "pair.png"
    image.write_bytes(b"test-image")
    requests: list[dict] = []

    def judge(request: dict) -> dict:
        requests.append(request)
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "No object-pairing defect.",
            "missing_evidence": [],
            "defects": [],
        }

    config = {
        "enabled": True,
        "metrics": {
            name: {"enabled": name == "object_pairing_consistency"}
            for name in (
                "style_consistency",
                "scale_consistency",
                "object_pairing_consistency",
                "functional_consistency",
                "semantic_placement_consistency",
            )
        },
    }
    config["metrics"]["object_pairing_consistency"]["evidence_plan"] = {
        "evidence_strategy": "global_and_local",
        "router_options": None,
    }
    full_prompt = "Create a game room and place the chair north of the table."
    report = evaluate_scene_quality_interfaces(
        _scene(),
        config=config,
        object_grouping_report={
            "object_groups": [
                {"group_id": "group", "object_ids": ["chair", "table"]}
            ]
        },
        render_evidence={
            "object_pairing_consistency": {"group": [str(image)]}
        },
        vlm_judge=judge,
        prompt=full_prompt,
        metric_applicability={
            "object_pairing_consistency": {
                "applicability": "relevant",
                "basis": ["test"],
            }
        },
    )

    assert len(requests) == 1
    outbound = requests[0]
    assert outbound["prompt"] != full_prompt
    assert "room_type: game room" in outbound["prompt"]
    assert "chair north of the table" not in outbound["prompt"]
    metric = report["metrics"]["object_pairing_consistency"]
    assert metric["metric_prompt_context"]["values"] == {
        "room_type": "game room"
    }
