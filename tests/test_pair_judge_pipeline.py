from __future__ import annotations

from pathlib import Path

from benchmark.workflow.evaluation import evaluate_layout_v0


def _case() -> dict:
    return {
        "case_id": "pair_case",
        "schema_version": "2.0",
        "input_level": "structured_relation",
        "description": {"text": "Create a small office."},
        "room": {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "unit": "meter"},
        "objects": [
            {"id": "chair_001", "category": "chair"},
            {"id": "desk_001", "category": "desk"},
            {"id": "lamp_001", "category": "lamp"},
        ],
        "relations": [{"id": "rel_001", "type": "near", "subject": "chair_001", "object": "desk_001"}],
        "attachments": [{"id": "att_001", "type": "support", "child": "lamp_001", "parent": "desk_001"}],
    }


def _layout() -> dict:
    return {
        "scene_id": "pair_case",
        "unit": "meter",
        "objects": [
            {"object_id": "chair_001", "category": "chair", "center": [1, 1, 0.45], "size": [0.6, 0.6, 0.9], "yaw": 0},
            {"object_id": "desk_001", "category": "desk", "center": [1.8, 1, 0.4], "size": [1.2, 0.7, 0.8], "yaw": 0},
            {"object_id": "lamp_001", "category": "lamp", "center": [1.8, 1, 1.1], "size": [0.2, 0.2, 0.4], "yaw": 0},
        ],
    }


def test_explicit_relation_and_attachment_create_pair_views(tmp_path: Path) -> None:
    report, metrics = evaluate_layout_v0(
        case=_case(),
        layout=_layout(),
        out_dir=tmp_path,
        model_name="mock",
        benchmark_config={},
    )

    assert metrics["specified_relation_pass_rate"] == 1.0
    assert metrics["specified_attachment_pass_rate"] == 1.0
    assert report["specified_relations"]["results"][0]["pass"]
    assert report["specified_attachments"]["results"][0]["pass"]
    groups = report["debug_evidence"]["object_groups"]
    assert any({"chair_001", "desk_001"}.issubset(set(group["object_ids"])) for group in groups)
    assert any({"lamp_001", "desk_001"}.issubset(set(group["object_ids"])) for group in groups)
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_yz.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xz.png").exists()
