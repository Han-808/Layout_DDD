from __future__ import annotations

from typing import Any, TypedDict


class BenchmarkState(TypedDict, total=False):
    task_id: str
    model_name: str
    model: Any
    case_path: str
    out_dir: str
    input_json: dict
    layout_schema: dict
    layout_schema_path: str
    benchmark_config: dict
    current_layout: dict
    current_layout_path: str
    current_evaluation: dict
    current_evaluation_path: str
    current_feedback: dict
    current_feedback_path: str
    iteration: int
    max_repair_iterations: int
    history: list[dict]
    evaluation_reports: list[dict]
    metrics: dict
    case_metrics: dict
    case_metrics_path: str
    per_case_result: dict
    per_case_result_path: str
    viewer_scene: dict
    viewer_scene_path: str
