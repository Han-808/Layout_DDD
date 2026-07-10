from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.assets.generation import AssetGenerationError, MCPAssetGenerationTool
from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_object_plan
from benchmark.nl_scene.converter import convert_nl_to_object_plan, parse_json_object_from_text
from benchmark.scene_io.validate import validate_asset_selection


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
