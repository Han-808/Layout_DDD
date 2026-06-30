from __future__ import annotations

from pathlib import Path

from benchmark.models.base_model import BaseLayoutModel, ModelResponseError
from benchmark.utils.io import read_json
from benchmark.workflow import run_workflow
from benchmark.workflow.evaluation import evaluate_layout_v0


ROOT = Path(__file__).resolve().parents[1]
HSSD_CASE = ROOT / "data" / "benchmark_cases" / "hssd_small" / "102343992_structured_relation.json"


class _UnparseableModel(BaseLayoutModel):
    def __init__(self) -> None:
        super().__init__(name="unparseable")

    def generate_layout(self, bm_instance: dict, layout_schema: dict) -> dict:
        raise ModelResponseError("Model response does not contain a JSON object.")

    def repair_layout(self, bm_instance: dict, current_layout: dict, feedback: dict, layout_schema: dict) -> dict:
        raise AssertionError("repair should not run")


def _case() -> dict:
    return {
        "case_id": "bad_layout_case",
        "schema_version": "2.0",
        "input_level": "structured_basic",
        "description": {"text": "Create a room with boxes."},
        "room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0},
        "objects": [{"id": "box_good", "category": "box"}, {"id": "box_bad", "category": "box"}],
    }


def test_parseable_invalid_layout_still_reaches_vlm_judge(tmp_path: Path) -> None:
    layout = {
        "scene_id": "bad_layout_case",
        "unit": "meter",
        "objects": [
            {"object_id": "box_good", "category": "box", "center": [1, 1, 0.5], "size": [1, 1, 1], "yaw": 0},
            {"object_id": "box_bad", "category": "box", "center": [2, 1, 0.5], "size": [-1, 1, 1], "yaw": 0},
        ],
    }

    report, metrics = evaluate_layout_v0(
        case=_case(),
        layout=layout,
        out_dir=tmp_path,
        model_name="mock",
        layout_schema=read_json(ROOT / "schemas" / "layout.schema.json"),
    )

    assert report["overall_valid"] is True
    assert report["validity_gate"]["passed"] is True
    assert report["summary"]["schema_valid"] is False
    assert report["debug_evidence"]["sanity_flags"]
    assert report["debug_evidence"]["render_skipped_objects"][0]["objects"] == ["box_bad"]
    assert report["vlm_judgement"]["short_reason"] == "Mock VLM judge returned deterministic v1 score."
    assert metrics["primary_score"] == 0.75
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xy.png").exists()


def test_serious_collision_is_flag_only_and_does_not_skip_judge(tmp_path: Path) -> None:
    layout = {
        "scene_id": "bad_layout_case",
        "unit": "meter",
        "coordinate_system": {
            "origin": "front-left floor corner",
            "x_axis": "room width",
            "y_axis": "room depth",
            "z_axis": "height",
            "rotation_unit": "degree",
        },
        "objects": [
            {"object_id": "box_a", "category": "box", "center": [1, 1, 0.5], "size": [1, 1, 1], "yaw": 0},
            {"object_id": "box_b", "category": "box", "center": [1.1, 1, 0.5], "size": [1, 1, 1], "yaw": 0},
        ],
    }

    report, _ = evaluate_layout_v0(case=_case(), layout=layout, out_dir=tmp_path, model_name="mock")

    assert any(flag["type"] == "serious_collision" for flag in report["debug_evidence"]["physical_flags"])
    assert report["overall_valid"] is True
    assert report["judge_skipped_reason"] == ""


def test_no_renderable_objects_still_produces_global_evidence(tmp_path: Path) -> None:
    layout = {
        "scene_id": "bad_layout_case",
        "unit": "meter",
        "objects": [{"object_id": "box_bad", "category": "box", "center": [1, 1, 0.5], "size": [0, 1, 1], "yaw": 0}],
    }

    report, _ = evaluate_layout_v0(case=_case(), layout=layout, out_dir=tmp_path, model_name="mock")

    assert any(flag["type"] == "no_renderable_objects" for flag in report["debug_evidence"]["view_flags"])
    assert report["debug_evidence"]["object_groups"] == []
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert report["overall_valid"] is True


def test_unparseable_generation_writes_structured_failure(tmp_path: Path) -> None:
    state = run_workflow(
        {
            "case_path": str(HSSD_CASE),
            "out_dir": str(tmp_path),
            "model": _UnparseableModel(),
            "model_name": "unparseable",
            "layout_schema": read_json(ROOT / "schemas" / "layout.schema.json"),
            "benchmark_config": {"benchmark": {"save_viewer_scene": True}},
            "max_repair_iterations": 0,
        }
    )

    report = state["current_evaluation"]
    assert report["overall_valid"] is False
    assert report["judgement_status"] == "unparseable_layout"
    assert report["vlm_judgement"]["judgement_status"] == "unparseable_layout"
    assert report["judge_skipped_reason"] == "model_output_unparseable"
    assert report["debug_evidence"]["sanity_flags"][0]["type"] == "model_output_unparseable"
    assert state["case_metrics"]["primary_score"] == 0.0
    assert (tmp_path / "evaluation_report.json").exists()
