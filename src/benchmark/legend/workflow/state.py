from __future__ import annotations

from typing import Any, TypedDict


class BenchmarkState(TypedDict, total=False):
    task_id: str
    pipeline_mode: str
    generation_used: bool
    model_name: str
    model: Any
    judge_model_name: str
    judge_model: Any
    case_path: str
    input_scene_path: str
    out_dir: str
    input_json: dict
    normalized_case: dict
    prompt_payload: dict
    eval_context: dict
    prompt_payload_path: str
    eval_context_summary_path: str
    visibility_audit_path: str
    input_quality_path: str
    layout_schema: dict
    layout_schema_path: str
    benchmark_config: dict
    resolved_run_config: dict
    resolved_run_config_path: str
    current_layout: dict
    current_layout_path: str
    current_scene: dict
    current_scene_path: str
    normalized_scene_path: str
    candidate_scene_path: str
    generated_layout_path: str
    generation_error: str
    generation_request_metadata_path: str
    generation_prompt_path: str
    generation_prompt_budget_report_path: str
    generation_prompt_sections_path: str
    generation_raw_response_path: str
    model_request_metadata_paths: list[str]
    current_evaluation: dict
    current_evaluation_path: str
    current_feedback: dict
    current_feedback_path: str
    repair_error: str
    repair_prompt_path: str
    repair_prompt_budget_report_path: str
    repair_prompt_sections_path: str
    repair_raw_response_path: str
    prompt_budget_exceeded: bool
    prompt_budget_error_stage: str
    iteration: int
    max_repair_iterations: int
    history: list[dict]
    evaluation_reports: list[dict]
    metrics: dict
    case_metrics: dict
    current_case_metrics_path: str
    case_metrics_path: str
    per_case_result: dict
    per_case_result_path: str
    viewer_scene: dict
    viewer_scene_path: str
