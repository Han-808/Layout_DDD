from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.assets.generation import AssetGenerationError, MCPAssetGenerationTool
from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_object_plan
from benchmark.nl_scene.converter import (
    COARSE_GRAINED,
    FINE_GRAINED,
    classify_prompt_granularity,
    convert_nl_to_object_plan,
    extract_room_dimension_claims,
    parse_json_object_from_text,
    validate_object_plan_json,
)
from benchmark.scene_io.validate import validate_asset_selection, validate_object_plan


OBJECT_PLAN = {
    "request_id": "demo_001",
    "scene_type": "living room",
    "scene_description": "A cozy reading living room.",
    "objects": [
        {
            "id": "obj_000",
            "role": "main seating",
            "category": "sofa",
            "description": "comfortable dark modern sofa",
            "estimated_size": [2.2, 0.9, 0.8],
            "count": 1,
            "placement_intent": {"absolute_relations": [], "relative_relations": []},
            "metadata": {},
        }
    ],
    "global_constraints": ["cozy", "walkable"],
    "relations": [],
}


def test_converter_json_parser_strips_markdown_fences() -> None:
    parsed = parse_json_object_from_text('```json\n{"scene_type":"living room","objects":[]}\n```')

    assert parsed == {"scene_type": "living room", "objects": []}


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "A room surrounded by walls, of size 8m by 8m by 2.8 m height.",
            {"width": 8.0, "depth": 8.0, "height": 2.8},
        ),
        (
            "A room enclosed by four walls measures 8 m × 8 m with a height of 2.8 m.",
            {"width": 8.0, "depth": 8.0, "height": 2.8},
        ),
        (
            "A 7 m × 9 m bedroom with a ceiling height of 3 m contains one bed.",
            {"width": 7.0, "depth": 9.0, "height": 3.0},
        ),
        (
            "A square room measuring 10 m × 10 m and 3.2 m high contains two beds.",
            {"width": 10.0, "depth": 10.0, "height": 3.2},
        ),
        ("Create a room with a ceiling height of 2.8 m.", {"height": 2.8}),
        ("Create a cozy room.", {}),
    ],
)
def test_room_dimension_claim_extraction_is_literal(
    instruction: str,
    expected: dict,
) -> None:
    assert extract_room_dimension_claims(instruction) == expected


@pytest.mark.parametrize(
    "instruction",
    [
        "Create a room containing a bed measuring 2 m by 1 m by 0.6 m.",
        "Create a room with a bed that is 2 m high.",
        "Create a room containing a cabinet with a height of 1.8 m.",
    ],
)
def test_room_dimension_claim_extraction_ignores_object_measurements(instruction: str) -> None:
    assert extract_room_dimension_claims(instruction) == {}


def test_converter_outputs_object_plan_with_no_pose_or_assets() -> None:
    result = convert_nl_to_object_plan(
        "Create a cozy living room.",
        request_id="demo_001",
        scene_type="living room",
        model_config={"mock_response": json.dumps(OBJECT_PLAN)},
    )

    assert result["request_id"] == "demo_001"
    assert result["objects"][0]["id"] == "obj_000"
    forbidden = {"center", "position", "rotation", "pose", "target_pose", "jid", "asset_id", "asset_ref"}
    assert forbidden.isdisjoint(result["objects"][0])


def test_converter_rejects_pose_and_asset_fields() -> None:
    bad = {**OBJECT_PLAN, "objects": [{**OBJECT_PLAN["objects"][0], "jid": "asset_a"}]}

    with pytest.raises(Exception, match="forbidden"):
        convert_nl_to_object_plan(
            "Create a cozy living room.",
            request_id="demo_001",
            scene_type="living room",
            model_config={"mock_response": json.dumps(bad)},
        )


def test_converter_normalizes_architecture_relation_alias_before_hard_gate() -> None:
    plan = {
        **OBJECT_PLAN,
        "relations": [
            {
                "subject": "obj_000",
                "predicate": "against wall",
                "object": "right_wall",
            }
        ],
    }

    result = validate_object_plan_json(plan)

    assert result["relations"] == [
        {
            "family": "oar",
            "subject_id": "obj_000",
            "type": "against_wall",
            "target": "east_wall",
            "raw_target": "right_wall",
        }
    ]
    assert validate_object_plan(result) is result


def test_converter_normalizes_missing_object_relation_family() -> None:
    plan = {
        **OBJECT_PLAN,
        "objects": [
            OBJECT_PLAN["objects"][0],
            {
                **OBJECT_PLAN["objects"][0],
                "id": "obj_001",
                "category": "table",
                "description": "coffee table",
            },
        ],
        "relations": [
            {
                "subject": "obj_000",
                "predicate": "near",
                "object": "obj_001",
            }
        ],
    }

    result = validate_object_plan_json(plan)

    assert result["relations"] == [
        {
            "family": "oor",
            "subject_id": "obj_000",
            "type": "near",
            "object_id": "obj_001",
        }
    ]
    assert validate_object_plan(result) is result


def test_converter_recognizes_on_top_of_as_frozen_oor_predicate() -> None:
    plan = {
        **OBJECT_PLAN,
        "objects": [
            OBJECT_PLAN["objects"][0],
            {
                **OBJECT_PLAN["objects"][0],
                "id": "obj_001",
                "category": "table",
                "description": "coffee table",
            },
        ],
        "relations": [
            {
                "subject": "obj_000",
                "predicate": "on top of",
                "object": "obj_001",
            }
        ],
    }

    result = validate_object_plan_json(plan)

    assert result["relations"] == [
        {
            "family": "oor",
            "subject_id": "obj_000",
            "type": "on_top_of",
            "object_id": "obj_001",
        }
    ]
    assert validate_object_plan(result) is result


def test_converter_preserves_group_relation_contracts_and_unknown_predicates() -> None:
    objects = [
        {**OBJECT_PLAN["objects"][0], "id": object_id, "category": "chair", "description": object_id}
        for object_id in ["left", "middle", "right", "chairs"]
    ]
    plan = {
        **OBJECT_PLAN,
        "objects": objects,
        "relations": [
            {"family": "oor", "type": "between", "subject_id": "middle", "object_ids": ["left", "right"]},
            {
                "family": "oor",
                "type": "ordered",
                "object_ids": ["left", "middle", "right"],
                "direction": "left_to_right",
            },
            {"family": "oor", "type": "around", "subject_ids": ["chairs"], "object_id": "middle"},
            {
                "family": "oor",
                "type": "mirrors",
                "subject_id": "left",
                "object_id": "right",
                "raw_relation": "the left chair mirrors the right chair",
            },
        ],
    }

    result = validate_object_plan_json(plan)

    assert result["relations"][0]["object_ids"] == ["left", "right"]
    assert result["relations"][1]["direction"] == "left_to_right"
    assert result["relations"][2]["subject_ids"] == ["chairs"]
    assert result["relations"][3]["type"] == "mirrors"
    assert result["relations"][3]["raw_relation"].startswith("the left chair")
    assert validate_object_plan(result) is result


def test_converter_normalizes_equivalent_group_relation_field_shapes() -> None:
    objects = [
        {**OBJECT_PLAN["objects"][0], "id": object_id, "category": "chair", "description": object_id}
        for object_id in ["left", "middle", "right"]
    ]
    plan = {
        **OBJECT_PLAN,
        "objects": objects,
        "relations": [
            {"predicate": "between", "subject": "middle", "object": ["left", "right"]},
            {
                "family": "oor",
                "predicate": "ordered",
                "objects": ["left", "middle", "right"],
                "direction": "left_to_right",
            },
            {"predicate": "around", "subject": ["left", "right"], "object": "middle"},
        ],
    }

    result = validate_object_plan_json(plan)

    assert result["relations"] == [
        {
            "family": "oor",
            "type": "between",
            "subject_id": "middle",
            "object_ids": ["left", "right"],
        },
        {
            "family": "oor",
            "type": "ordered",
            "object_ids": ["left", "middle", "right"],
            "direction": "left_to_right",
        },
        {
            "family": "oor",
            "type": "around",
            "subject_ids": ["left", "right"],
            "object_id": "middle",
        },
    ]
    assert validate_object_plan(result) is result


def test_coarse_converter_does_not_request_scene_completion() -> None:
    captured_messages = []

    def coarse_response(messages: list[dict]) -> str:
        captured_messages.extend(messages)
        return json.dumps(
            {
                "request_id": "demo_coarse",
                "scene_type": "bedroom",
                "scene_description": "Create a cozy bedroom.",
                "prompt_granularity": COARSE_GRAINED,
                "explicit_claims": ["bedroom", "cozy"],
                "implicit_intents": ["should contain a bed"],
                "objects": [],
                "global_constraints": [],
                "relations": [],
            }
        )

    result = convert_nl_to_object_plan(
        "Create a cozy bedroom.",
        request_id="demo_coarse",
        scene_type="bedroom",
        prompt_granularity=COARSE_GRAINED,
        model_config={"chat_model": coarse_response},
    )

    assert result["prompt_granularity"] == COARSE_GRAINED
    assert result["objects"] == []
    assert "implicit_intents" not in result
    assert "Do not complete the scene" in captured_messages[0]["content"]
    assert "Include an object only when it is explicitly named" in captured_messages[0]["content"]
    assert "does not imply a required bed" in captured_messages[0]["content"]


@pytest.mark.parametrize(
    ("instruction", "classification"),
    [
        (
            "Create a cozy room with a bed.",
            {
                "prompt_granularity": COARSE_GRAINED,
                "object_evidence": ["a bed"],
                "relationship_evidence": [],
                "reason": "one object mention without an explicit layout relationship",
            },
        ),
        (
            "Put a blue velvet bed in the center, with a desk to its right.",
            {
                "prompt_granularity": FINE_GRAINED,
                "object_evidence": ["blue velvet bed", "desk"],
                "relationship_evidence": ["bed in the center", "desk to its right"],
                "reason": "multiple described objects and explicit OA/OO relationships",
            },
        ),
    ],
)
def test_granularity_classifier_uses_object_and_relationship_evidence(
    instruction: str,
    classification: dict,
) -> None:
    result = classify_prompt_granularity(
        instruction,
        model_config={"mock_response": json.dumps(classification)},
    )

    assert result == classification


def test_retriever_wrapper_selects_top1_asset_selection(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    result = retrieve_assets_for_object_plan(
        OBJECT_PLAN,
        asset_index_path=str(index_path),
        retrieval_k=1,
        retriever_module_path=str(module_path),
    )

    item = result["objects"][0]
    assert result["request_id"] == "demo_001"
    assert item["object_id"] == "obj_000"
    assert item["retrieval_query"] == {
        "description": "comfortable dark modern sofa",
        "category": "sofa",
        "size_constraint": [2.2, 0.9, 0.8],
    }
    assert item["object_spec"]["count"] == 1
    assert item["selected_asset"]["jid"] == "asset_a"
    assert item["selected_asset"]["asset_ref"]["asset_key"] == "asset_a"
    assert item["selected_asset"]["asset_proxy"]["bbox_size"] == [2.0, 0.9, 0.8]
    assert item["selection_action"] == "select"
    assert item["selection_reason"] == "top retrieval result; VLM selector disabled"


def test_retriever_wrapper_uses_configured_selector_for_topk(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    result = retrieve_assets_for_object_plan(
        OBJECT_PLAN,
        asset_index_path=str(index_path),
        retrieval_k=2,
        retriever_module_path=str(module_path),
        use_vlm_selector=True,
        model_config={"selector_response": {"selected_jid": "asset_b", "reason": "better color match"}},
    )

    item = result["objects"][0]
    assert [candidate["jid"] for candidate in item["candidates"]] == ["asset_a", "asset_b"]
    assert item["selected_asset"]["jid"] == "asset_b"
    assert item["selection_action"] == "select"
    assert item["selection_reason"] == "better color match"


def test_retriever_vlm_can_generate_when_no_candidate_is_suitable(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)
    calls = []

    def generate_asset(request: dict) -> dict:
        calls.append(request)
        return {
            "jid": "generated_sofa_001",
            "description": "custom comfortable dark modern sofa",
            "size": [2.2, 0.9, 0.8],
            "mesh_uri": "outputs/generated_sofa_001.glb",
        }

    result = retrieve_assets_for_object_plan(
        OBJECT_PLAN,
        asset_index_path=str(index_path),
        retrieval_k=2,
        retriever_module_path=str(module_path),
        use_vlm_selector=True,
        model_config={
            "selector_response": {
                "action": "generate",
                "selected_jid": None,
                "reason": "neither candidate has the requested dark upholstery",
                "generation_request": {
                    "prompt": "a comfortable dark modern sofa",
                    "category": "sofa",
                    "target_size": [2.2, 0.9, 0.8],
                },
            }
        },
        asset_generation_tool=generate_asset,
    )

    item = result["objects"][0]
    selected = item["selected_asset"]
    assert item["selection_action"] == "generate"
    assert item["selection_decision"]["action"] == "generate"
    assert selected["jid"] == "generated_sofa_001"
    assert selected["asset_ref"]["source_db"] == "generated"
    assert selected["asset_ref"]["mesh_uri"] == "outputs/generated_sofa_001.glb"
    assert selected["metadata"]["generated"] is True
    assert calls[0]["object_id"] == "obj_000"
    assert [candidate["jid"] for candidate in calls[0]["rejected_candidates"]] == ["asset_a", "asset_b"]
    assert validate_asset_selection(result) is result


def test_retriever_fails_if_vlm_requests_generation_without_tool(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    with pytest.raises(AssetGenerationError, match="not enabled/configured"):
        retrieve_assets_for_object_plan(
            OBJECT_PLAN,
            asset_index_path=str(index_path),
            retrieval_k=2,
            retriever_module_path=str(module_path),
            use_vlm_selector=True,
            model_config={
                "selector_response": {
                    "action": "generate",
                    "reason": "no suitable candidate",
                }
            },
        )


def test_mcp_asset_generation_adapter_calls_configured_tool() -> None:
    class FakeMCPClient:
        def __init__(self) -> None:
            self.calls = []

        def call_tool(self, name: str, arguments: dict) -> dict:
            self.calls.append((name, arguments))
            return {
                "asset": {
                    "jid": "generated_chair_001",
                    "size": [0.6, 0.6, 0.9],
                    "mesh_uri": "outputs/generated_chair_001.glb",
                }
            }

    client = FakeMCPClient()
    tool = MCPAssetGenerationTool(client, tool_name="create_3d_asset")

    asset = tool.generate_asset({"object_id": "chair_001"})

    assert asset["jid"] == "generated_chair_001"
    assert client.calls == [("create_3d_asset", {"object_id": "chair_001"})]


def _fake_index_path(tmp_path: Path) -> Path:
    index_path = tmp_path / "asset_index"
    index_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    return index_path


def _fake_retriever_module(tmp_path: Path) -> Path:
    module_path = tmp_path / "fake_retriever.py"
    module_path.write_text(
        """
class AssetRetriever:
    def __init__(self, index_path):
        self.index_path = index_path

    def retrieve(self, description, category=None, size_constraint=None, top_k=1, **kwargs):
        candidates = [
            {"jid": "asset_a", "short_desc": "dark modern sofa", "category": category or "sofa", "size": [2.0, 0.9, 0.8], "score": 0.8},
            {"jid": "asset_b", "short_desc": "warm brown sofa", "category": category or "sofa", "size": [2.1, 0.9, 0.8], "score": 0.7},
        ]
        return candidates[:top_k]
""".strip() + chr(10),
        encoding="utf-8",
    )
    return module_path
