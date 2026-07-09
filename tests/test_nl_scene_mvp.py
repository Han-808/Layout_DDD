from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_scene_spec
from benchmark.nl_scene.converter import parse_json_object_from_text
from benchmark.nl_scene.dummy_evaluator import evaluate_scene
from benchmark.nl_scene.workflow import run_nl_scene_workflow
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]


SCENE_SPEC = {
    "scene_type": "living room",
    "scene_description": "A cozy reading living room.",
    "objects": [
        {
            "id": 0,
            "role": "main seating",
            "category": "sofa",
            "description": "comfortable dark modern sofa",
            "estimated_size": [2.2, 0.9, 0.8],
            "count": 1,
        }
    ],
    "global_constraints": ["cozy", "walkable"],
    "relations": [],
}


def test_converter_json_parser_strips_markdown_fences() -> None:
    parsed = parse_json_object_from_text('```json\n{"scene_type":"living room","objects":[]}\n```')

    assert parsed == {"scene_type": "living room", "objects": []}


def test_retriever_wrapper_selects_top1(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    result = retrieve_assets_for_scene_spec(
        SCENE_SPEC,
        asset_index_path=str(index_path),
        retrieval_k=1,
        retriever_module_path=str(module_path),
        use_vlm_selector=False,
    )

    item = result["objects"][0]
    assert item["selected_asset"]["jid"] == "asset_a"
    assert item["candidates"] == [item["selected_asset"]]
    assert item["selection_reason"] == "selected top-1 retriever result"


def test_retriever_wrapper_uses_fake_vlm_selector_for_topk(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    result = retrieve_assets_for_scene_spec(
        SCENE_SPEC,
        asset_index_path=str(index_path),
        retrieval_k=2,
        retriever_module_path=str(module_path),
        use_vlm_selector=True,
        model_config={"selector_response": {"selected_jid": "asset_b", "reason": "better color match"}},
    )

    item = result["objects"][0]
    assert [candidate["jid"] for candidate in item["candidates"]] == ["asset_a", "asset_b"]
    assert item["selected_asset"]["jid"] == "asset_b"
    assert item["selection_reason"] == "better color match"


def test_dummy_evaluator_is_reproducible() -> None:
    scene = {"scene_id": "generated_scene", "assets": []}

    first = evaluate_scene(scene, instruction="make a room", seed=7)
    second = evaluate_scene(scene, instruction="make a room", seed=7)
    other = evaluate_scene(scene, instruction="make a room", seed=8)

    assert first["evaluator_version"] == "dummy_v0"
    assert first["dummy"] is True
    assert first["metrics"] == second["metrics"]
    assert first["metrics"] != other["metrics"]


def test_workflow_stops_after_retrieval_when_generation_missing(tmp_path: Path) -> None:
    module_path = _fake_retriever_module(tmp_path)
    index_path = _fake_index_path(tmp_path)

    result = run_nl_scene_workflow(
        instruction="Create a cozy living room.",
        scene_type="living room",
        asset_index_path=str(index_path),
        retrieval_k=1,
        retriever_module_path=str(module_path),
        use_vlm_selector=False,
        model_config={"mock_response": json.dumps(SCENE_SPEC)},
        out_dir=tmp_path / "out",
        seed=0,
    )

    out_dir = tmp_path / "out"
    assert read_json(out_dir / "scene_spec.json")["scene_type"] == "living room"
    assert read_json(out_dir / "asset_retrieval.json")["objects"][0]["selected_asset"]["jid"] == "asset_a"
    generation_input = read_json(out_dir / "generation_input.json")
    assert generation_input["original_instruction"] == "Create a cozy living room."
    assert read_json(out_dir / "workflow_status.json") == {
        "status": "generation_skipped",
        "reason": "generation stage is not implemented yet",
        "next_expected_input": "generated_scene.json",
    }
    assert result["workflow_status"]["status"] == "generation_skipped"


def test_direct_dummy_evaluator_cli_writes_report(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "dummy_scene", "assets": []}), encoding="utf-8")
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_scene.py"),
            "--scene",
            str(scene_path),
            "--instruction",
            "Create a room.",
            "--out",
            str(report_path),
            "--seed",
            "3",
            "--dummy",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = read_json(report_path)
    assert "evaluation_report:" in completed.stdout
    assert report["scene_id"] == "dummy_scene"
    assert report["dummy"] is True
    assert set(report["metrics"]) == {"physical_validity", "text_fidelity", "scene_plausibility", "game_usability", "overall"}


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
