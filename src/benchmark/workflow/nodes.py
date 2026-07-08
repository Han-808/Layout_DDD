from __future__ import annotations

import json
from pathlib import Path

from benchmark.input_modes import representation_mode_for_level, resolve_input_representation_mode
from benchmark.models.base_model import ModelResponseError, build_generation_prompt
from benchmark.models.prompt_budget import PromptBudgetError
from benchmark.feedback import build_feedback
from benchmark.metrics import compute_case_metrics
from benchmark.data.scene_adapters import layout_to_scene
from benchmark.utils.io import ensure_dir, read_json, write_json
from benchmark.visualization import export_viewer_scene
from benchmark.workflow.evaluate import evaluate_scene
from benchmark.workflow.layout_normalization import enforce_layout_object_set
from benchmark.object_aliasing import ALIAS_MAP_KEY
from benchmark.workflow.payloads import build_input_payloads, eval_context_summary
from benchmark.workflow.scoring import infer_input_level
from benchmark.workflow.artifacts import (
    attach_feedback_to_history,
    build_workflow_metadata,
    compact_history,
    make_history_entry,
    per_case_filename,
    save_intermediate_artifacts,
    save_viewer_scene,
)
from benchmark.workflow.state import BenchmarkState
from benchmark.workflow.trace import write_workflow_trace


class NormalizeInputNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        if "input_json" not in next_state:
            next_state["input_json"] = read_json(next_state["case_path"])
        if "layout_schema" not in next_state and next_state.get("layout_schema_path"):
            next_state["layout_schema"] = read_json(next_state["layout_schema_path"])
        input_json = dict(next_state["input_json"])
        input_json.setdefault("input_level", infer_input_level(input_json))
        mode = resolve_input_representation_mode(
            input_json,
            default=representation_mode_for_level(input_json.get("input_level")),
        )
        input_json.setdefault("scene_representation_mode", mode)
        if isinstance(input_json.get("source"), dict):
            source = dict(input_json["source"])
            source.setdefault("input_representation_mode", mode)
            source.setdefault("scene_representation_mode", mode)
            input_json["source"] = source
        case_id = input_json.get("case_id") or input_json.get("task_id")
        if case_id:
            input_json.setdefault("case_id", case_id)
            input_json.setdefault("task_id", case_id)
        payloads = build_input_payloads(input_json)
        input_json = payloads["normalized_case"]
        next_state["input_json"] = input_json
        next_state["normalized_case"] = input_json
        next_state["prompt_payload"] = payloads["prompt_payload"]
        next_state["eval_context"] = payloads["eval_context"]
        next_state["task_id"] = input_json.get("task_id") or input_json.get("case_id")
        next_state.setdefault("iteration", 0)
        next_state.setdefault("history", [])
        next_state.setdefault("evaluation_reports", [])
        next_state.setdefault("model_request_metadata_paths", [])
        ensure_dir(next_state["out_dir"])
        if isinstance(next_state.get("resolved_run_config"), dict):
            resolved_path = Path(next_state["out_dir"]) / "resolved_run_config.json"
            write_json(resolved_path, next_state["resolved_run_config"])
            hash_value = str(next_state["resolved_run_config"].get("config_hash", ""))
            if hash_value:
                (Path(next_state["out_dir"]) / "config_hash.txt").write_text(hash_value + "\n", encoding="utf-8")
            next_state["resolved_run_config_path"] = str(resolved_path)
        next_state["normalized_case_path"] = str(write_json(Path(next_state["out_dir"]) / "normalized_case.json", next_state["normalized_case"]))
        next_state["prompt_payload_path"] = str(write_json(Path(next_state["out_dir"]) / "prompt_payload.json", next_state["prompt_payload"]))
        next_state["eval_context_summary_path"] = str(write_json(Path(next_state["out_dir"]) / "eval_context_summary.json", eval_context_summary(next_state["eval_context"])))
        if isinstance(next_state["normalized_case"].get(ALIAS_MAP_KEY), dict):
            next_state["object_alias_map_path"] = str(write_json(Path(next_state["out_dir"]) / "object_alias_map.json", next_state["normalized_case"][ALIAS_MAP_KEY]))
        next_state["visibility_audit_path"] = str(write_json(Path(next_state["out_dir"]) / "visibility_audit.json", payloads["visibility_audit"]))
        next_state["input_quality_path"] = str(write_json(Path(next_state["out_dir"]) / "input_quality.json", payloads["input_quality"]))
        return next_state


class GenerateLayoutNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        next_state.pop("generation_error", None)
        generation_input = next_state.get("prompt_payload") or next_state["input_json"]
        try:
            layout = next_state["model"].generate_layout(generation_input, next_state["layout_schema"])
        except PromptBudgetError as exc:
            layout = _empty_layout(next_state["input_json"])
            next_state["generation_error"] = str(exc)
            next_state["prompt_budget_exceeded"] = True
            next_state["prompt_budget_error_stage"] = "generation"
        except ModelResponseError as exc:
            layout = _empty_layout(next_state["input_json"])
            next_state["generation_error"] = str(exc)
        if not next_state.get("generation_error"):
            layout, normalization = enforce_layout_object_set(layout, next_state["input_json"], stage="generation")
            next_state["current_layout_normalization"] = normalization
        path = Path(next_state["out_dir"]) / "initial_layout.json"
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        if save_intermediate:
            write_json(path, layout)
        next_state["current_layout"] = layout
        next_state["current_layout_path"] = str(path) if save_intermediate else ""
        next_state["generated_layout_path"] = next_state["current_layout_path"]
        metadata_path = _write_model_request_metadata(
            next_state,
            filename="generation_request_metadata.json",
            metadata=getattr(next_state.get("model"), "last_request_metadata", None),
        )
        if metadata_path:
            next_state["generation_request_metadata_path"] = metadata_path
            next_state["model_request_metadata_paths"] = [*next_state.get("model_request_metadata_paths", []), metadata_path]
        raw_response_path = _write_model_text_artifact(
            next_state,
            filename="generation_raw_response.txt",
            text=getattr(next_state.get("model"), "last_response_text", "") or (json.dumps(layout, ensure_ascii=False, indent=2) if not next_state.get("generation_error") else ""),
        )
        if raw_response_path:
            next_state["generation_raw_response_path"] = raw_response_path
        prompt_path = _write_model_text_artifact(
            next_state,
            filename="generation_prompt.txt",
            text=getattr(next_state.get("model"), "last_prompt_text", "") or build_generation_prompt(generation_input, next_state["layout_schema"]),
        )
        if prompt_path:
            next_state["generation_prompt_path"] = prompt_path
        budget_path = _write_prompt_budget_report(
            next_state,
            filename="generation_prompt_budget_report.json",
            metadata=getattr(next_state.get("model"), "last_request_metadata", None),
        )
        if budget_path:
            next_state["generation_prompt_budget_report_path"] = budget_path
        sections_path = _write_prompt_sections(
            next_state,
            filename="generation_prompt_sections.json",
            sections=getattr(next_state.get("model"), "last_prompt_sections", None),
        )
        if sections_path:
            next_state["generation_prompt_sections_path"] = sections_path
        next_state["iteration"] = 0
        return next_state


class EvaluateLayoutNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        iteration = int(next_state.get("iteration", 0))
        candidate_scene = layout_to_scene(next_state["current_layout"], next_state["input_json"])
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        scene_path = Path(next_state["out_dir"]) / ("candidate_scene.json" if iteration == 0 else f"candidate_scene_iter_{iteration}.json")
        if save_intermediate:
            write_json(scene_path, candidate_scene)
        next_state["current_scene"] = candidate_scene
        next_state["current_scene_path"] = str(scene_path) if save_intermediate else ""
        next_state["candidate_scene_path"] = next_state["current_scene_path"]
        report, case_metrics = evaluate_scene(
            candidate_scene,
            case=next_state["input_json"],
            out_dir=next_state["out_dir"],
            model_name=next_state.get("model_name", getattr(next_state.get("model"), "name", "unknown_model")),
            benchmark_config=next_state.get("benchmark_config"),
            layout_schema=next_state.get("layout_schema"),
            iteration=iteration,
            generator_model=next_state.get("model"),
            judge_model=next_state.get("judge_model", next_state.get("model")),
            judge_model_name=next_state.get("judge_model_name"),
            generation_error=next_state.get("generation_error"),
            eval_context=next_state.get("eval_context"),
        )
        pipeline_mode = str(next_state.get("pipeline_mode") or "generation")
        generation_used = bool(next_state.get("generation_used", pipeline_mode == "generation"))
        report["pipeline_mode"] = pipeline_mode
        report["generation_used"] = generation_used
        report["generated_layout_path"] = next_state.get("generated_layout_path", next_state.get("current_layout_path", ""))
        report["candidate_scene_path"] = next_state.get("candidate_scene_path", next_state.get("current_scene_path", ""))
        case_metrics.update({"pipeline_mode": pipeline_mode, "generation_used": generation_used})
        if isinstance(report.get("metrics"), dict):
            report["metrics"].update({"pipeline_mode": pipeline_mode, "generation_used": generation_used})
        path = Path(next_state["out_dir"]) / "evaluation_report.json"
        write_json(path, report)
        iter_path = Path(next_state["out_dir"]) / f"evaluation_report_iter_{iteration}.json"
        if iter_path != path and save_intermediate_artifacts(next_state.get("benchmark_config")):
            write_json(iter_path, report)

        history = list(next_state.get("history", []))
        history.append(
            make_history_entry(
                iteration=iteration,
                layout_path=next_state.get("current_layout_path", ""),
                evaluation_path=str(path),
                report=report,
                layout=next_state["current_layout"],
                evaluation=report,
            )
        )
        reports = list(next_state.get("evaluation_reports", []))
        reports.append(report)

        next_state["current_evaluation"] = report
        next_state["current_evaluation_path"] = str(path)
        next_state["case_metrics"] = case_metrics
        next_state["current_case_metrics_path"] = str(Path(next_state["out_dir"]) / f"case_metrics_iter_{iteration}.json")
        next_state["history"] = history
        next_state["evaluation_reports"] = reports
        return next_state


class BuildFeedbackNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        feedback = build_feedback(
            next_state["current_evaluation"],
            next_state["current_layout"],
            next_state["input_json"],
            benchmark_config=next_state.get("benchmark_config"),
        )
        iteration = int(next_state.get("iteration", 0))
        path = Path(next_state["out_dir"]) / f"feedback_iter_{iteration}.json"
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        if save_intermediate:
            write_json(path, feedback)
        next_state["current_feedback"] = feedback
        next_state["current_feedback_path"] = str(path) if save_intermediate else ""
        next_state["history"] = attach_feedback_to_history(
            list(next_state.get("history", [])),
            iteration,
            next_state["current_feedback_path"],
            feedback,
        )
        return next_state


class RepairLayoutNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        next_iteration = int(next_state.get("iteration", 0)) + 1
        previous_layout = next_state["current_layout"]
        deterministic_layout, deterministic_summary = _apply_deterministic_repair_actions(
            previous_layout,
            next_state.get("current_feedback", {}),
        )
        repair_input = _model_repair_input(next_state)
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        try:
            layout = next_state["model"].repair_layout(
                repair_input,
                deterministic_layout,
                next_state["current_feedback"],
                next_state["layout_schema"],
            )
        except (PromptBudgetError, ModelResponseError) as exc:
            next_state["repair_error"] = str(exc)
            if isinstance(exc, PromptBudgetError):
                next_state["prompt_budget_exceeded"] = True
                next_state["prompt_budget_error_stage"] = "repair"
            next_state["iteration"] = next_iteration
            layout = previous_layout
            path = Path(next_state.get("current_layout_path", ""))
        else:
            layout, normalization = enforce_layout_object_set(
                layout,
                next_state["input_json"],
                previous_layout=previous_layout,
                stage=f"repair_iter_{next_iteration}",
            )
            if deterministic_summary["num_changed_objects"]:
                layout["_deterministic_repair_summary"] = deterministic_summary
            layout["_repair_change_summary"] = _layout_change_summary(
                previous_layout,
                layout,
                next_state.get("current_feedback", {}).get("repair_targets", []),
            )
            next_state["current_layout_normalization"] = normalization
            next_state["iteration"] = next_iteration
            path = Path(next_state["out_dir"]) / f"repaired_layout_iter_{next_iteration}.json"
            if save_intermediate:
                write_json(path, layout)
        next_state["current_layout"] = layout
        if not next_state.get("repair_error"):
            next_state["current_layout_path"] = str(path) if save_intermediate else ""
        metadata_path = _write_model_request_metadata(
            next_state,
            filename=f"repair_request_metadata_iter_{next_iteration}.json",
            metadata=getattr(next_state.get("model"), "last_request_metadata", None),
        )
        if metadata_path:
            next_state["model_request_metadata_paths"] = [*next_state.get("model_request_metadata_paths", []), metadata_path]
        raw_response_path = _write_model_text_artifact(
            next_state,
            filename=f"repair_raw_response_iter_{next_iteration}.txt",
            text=getattr(next_state.get("model"), "last_response_text", ""),
        )
        if raw_response_path:
            next_state["repair_raw_response_path"] = raw_response_path
        prompt_path = _write_model_text_artifact(
            next_state,
            filename=f"repair_prompt_iter_{next_iteration}.txt",
            text=getattr(next_state.get("model"), "last_prompt_text", ""),
        )
        if prompt_path:
            next_state["repair_prompt_path"] = prompt_path
        budget_path = _write_prompt_budget_report(
            next_state,
            filename=f"repair_prompt_budget_report_iter_{next_iteration}.json",
            metadata=getattr(next_state.get("model"), "last_request_metadata", None),
        )
        if budget_path:
            next_state["repair_prompt_budget_report_path"] = budget_path
        sections_path = _write_prompt_sections(
            next_state,
            filename=f"repair_prompt_sections_iter_{next_iteration}.json",
            sections=getattr(next_state.get("model"), "last_prompt_sections", None),
        )
        if sections_path:
            next_state["repair_prompt_sections_path"] = sections_path
        return next_state


class ComputeMetricsNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        metrics = compute_case_metrics(
            next_state["history"],
            max_repair_iterations=int(next_state.get("max_repair_iterations", 0)),
        )
        pipeline_mode = str(next_state.get("pipeline_mode") or "generation")
        generation_used = bool(next_state.get("generation_used", pipeline_mode == "generation"))
        metrics.update({"pipeline_mode": pipeline_mode, "generation_used": generation_used})
        final_eval_metrics = next_state.get("current_evaluation", {}).get("metrics", {})
        if isinstance(final_eval_metrics, dict):
            for key in [
                "scene_id",
                "scene_asset_count",
                "geometry_asset_count",
                "non_geometric_asset_count",
                "asset_ref_asset_count",
                "asset_ref_available_rate",
                "local_asset_ref_count",
                "local_asset_available_rate",
                "local_scene_ref_available",
                "local_scene_id",
                "local_scene_json_path",
                "geometry_available_rate",
                "renderable",
                "num_renderable_objects",
                "judge_success",
                "vlm_valid",
                "vlm_score",
                "evidence_flag_counts",
                "physical_flag_confidence_counts",
                "physical_flag_source_kind_counts",
                "vlm_judge_input_mode",
                "render_evidence_used",
                "json_scene_used",
            ]:
                if key in final_eval_metrics:
                    metrics.setdefault(key, final_eval_metrics[key])
        feedback = next_state.get("current_feedback", {})
        metrics["feedback_issue_count"] = len(feedback.get("issues", [])) if isinstance(feedback.get("issues"), list) else 0
        metrics["feedback_suggested_action_count"] = (
            len(feedback.get("suggested_actions", [])) if isinstance(feedback.get("suggested_actions"), list) else 0
        )
        case_metrics_path = Path(next_state["out_dir"]) / "case_metrics.json"
        write_json(case_metrics_path, metrics)
        next_state["case_metrics"] = metrics
        next_state["case_metrics_path"] = str(case_metrics_path)
        input_source = next_state["input_json"].get("source") if isinstance(next_state["input_json"].get("source"), dict) else {}
        result = {
            "task_id": next_state["task_id"],
            "model_name": next_state.get("model_name", getattr(next_state.get("model"), "name", "unknown_model")),
            "pipeline_mode": pipeline_mode,
            "generation_used": generation_used,
            "max_repair_iterations": int(next_state.get("max_repair_iterations", 0)),
            "input_level": next_state["input_json"].get("input_level"),
            "scene_representation_mode": next_state["input_json"].get("scene_representation_mode"),
            "object_aliasing": next_state.get("eval_context", {}).get("object_aliasing", {}),
            "object_alias_map_path": next_state.get("object_alias_map_path", ""),
            "input_source_summary": {
                key: input_source[key]
                for key in [
                    "dataset",
                    "scene_instance",
                    "raw_object_instance_count",
                    "imported_object_count",
                    "truncated",
                    "mesh_imported",
                    "mesh_free_import",
                    "room_boundary_source_kind",
                    "room_geometry_fidelity",
                    "room_is_proxy_geometry",
                ]
                if key in input_source
            },
            "metrics": metrics,
            "case_metrics_path": next_state.get("case_metrics_path", ""),
            "metrics_path": next_state.get("case_metrics_path", ""),
            "generated_layout_path": next_state.get("generated_layout_path", ""),
            "candidate_scene_path": next_state.get("candidate_scene_path", next_state.get("current_scene_path", "")),
            "evaluation_report_path": next_state.get("current_evaluation_path", ""),
            "feedback_path": next_state.get("current_feedback_path", ""),
            "history": compact_history(next_state["history"]),
            "final_evaluation": next_state["current_evaluation"],
            "config_refs": next_state["current_evaluation"].get("config_refs", {}),
            "config_hash": next_state["current_evaluation"].get("config_hash", ""),
        }
        result_path = Path(next_state["out_dir"]) / per_case_filename(next_state.get("benchmark_config"))
        write_json(result_path, result)
        next_state["metrics"] = metrics
        next_state["per_case_result"] = result
        next_state["per_case_result_path"] = str(result_path)

        trace_path, graph_path = write_workflow_trace(next_state, next_state["out_dir"])
        next_state["workflow_trace_path"] = str(trace_path)
        next_state["workflow_graph_path"] = str(graph_path)

        if save_viewer_scene(next_state.get("benchmark_config")):
            viewer_scene = export_viewer_scene(
                next_state["input_json"],
                next_state["current_layout"],
                next_state["current_evaluation"],
                next_state["history"],
                next_state.get("benchmark_config"),
            )
            workflow_metadata = build_workflow_metadata(
                {
                    **next_state,
                    "per_case_result": result,
                    "per_case_result_path": str(result_path),
                    "viewer_scene_path": str(Path(next_state["out_dir"]) / "viewer_scene.json"),
                },
                include_data=False,
            )
            viewer_scene["workflow"] = workflow_metadata
            viewer_scene["workflow_steps"] = workflow_metadata.get("artifacts", [])
            viewer_scene["artifacts"] = workflow_metadata.get("artifacts", [])
            viewer_scene["feedback"] = next_state.get("current_feedback", {})
            viewer_scene["metrics"] = metrics
            viewer_scene["metrics_summary"] = next_state.get("case_metrics", metrics)
            viewer_scene["view_artifacts"] = {
                "room": next_state["current_evaluation"].get("room_consistency", {}).get("view_artifacts", []),
                "global": next_state["current_evaluation"].get("room_consistency", {}).get("view_artifacts", []),
                "groups": next_state["current_evaluation"].get("debug_evidence", {}).get("group_view_artifacts", []),
                "relations": next_state["current_evaluation"].get("specified_relations", {}).get("results", []),
                "attachments": next_state["current_evaluation"].get("specified_attachments", {}).get("results", []),
                "vlm_judge": next_state["current_evaluation"].get("vlm_judge_artifacts", {}),
            }
            viewer_scene["workflow_trace_path"] = "workflow_trace.json"
            viewer_scene["history"] = result["history"]
            viewer_scene_path = Path(next_state["out_dir"]) / "viewer_scene.json"
            write_json(viewer_scene_path, viewer_scene)
            next_state["viewer_scene"] = viewer_scene
            next_state["viewer_scene_path"] = str(viewer_scene_path)
        return next_state


def route_after_eval(state: BenchmarkState) -> str:
    overall_valid = state["current_evaluation"].get("overall_valid")
    has_budget = int(state.get("iteration", 0)) < int(state.get("max_repair_iterations", 0))
    if overall_valid is True:
        metrics = state["current_evaluation"].get("metrics", {})
        relation_rate = metrics.get("specified_relation_pass_rate")
        attachment_rate = metrics.get("specified_attachment_pass_rate")
        low_explicit_score = any(value is not None and float(value) < 0.5 for value in [relation_rate, attachment_rate])
        if has_budget and low_explicit_score:
            return "repair"
        return "metrics"
    if overall_valid is False and has_budget:
        return "repair"
    return "metrics"


def _empty_layout(input_json: dict) -> dict:
    scene_id = input_json.get("task_id") or input_json.get("case_id") or "unparseable_scene"
    return {
        "scene_id": str(scene_id),
        "unit": "meter",
        "coordinate_system": {
            "origin": "case floor-plan coordinate frame; HSSD cases may use negative x/y values",
            "x_axis": "floor-plan x coordinate",
            "y_axis": "floor-plan y/depth coordinate",
            "z_axis": "height",
            "rotation_unit": "degree",
        },
        "objects": [],
        "relations": [],
        "hierarchy": {"regions": [], "floor_objects": [], "supported_objects": []},
    }


def _write_model_request_metadata(state: BenchmarkState, *, filename: str, metadata: object) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return ""
    path = Path(state["out_dir"]) / filename
    write_json(path, metadata)
    return str(path)


def _write_model_text_artifact(state: BenchmarkState, *, filename: str, text: object) -> str:
    if not isinstance(text, str) or not text:
        return ""
    path = Path(state["out_dir"]) / filename
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_prompt_budget_report(state: BenchmarkState, *, filename: str, metadata: object) -> str:
    if not isinstance(metadata, dict):
        return ""
    report = metadata.get("prompt_budget_report")
    if not isinstance(report, dict):
        return ""
    path = Path(state["out_dir"]) / filename
    write_json(path, report)
    return str(path)


def _write_prompt_sections(state: BenchmarkState, *, filename: str, sections: object) -> str:
    if not isinstance(sections, list) or not sections:
        return ""
    path = Path(state["out_dir"]) / filename
    write_json(path, sections)
    return str(path)


def _model_repair_input(state: BenchmarkState) -> dict:
    base = state.get("prompt_payload") or state["input_json"]
    repair_input = dict(base)
    eval_context = state.get("eval_context")
    alias_map = eval_context.get(ALIAS_MAP_KEY) if isinstance(eval_context, dict) else None
    if not isinstance(alias_map, dict):
        input_json = state.get("input_json")
        alias_map = input_json.get(ALIAS_MAP_KEY) if isinstance(input_json, dict) else None
    if isinstance(alias_map, dict):
        repair_input[ALIAS_MAP_KEY] = alias_map
    return repair_input


REPAIR_POSITION_TOLERANCE_M = 0.01
REPAIR_SIZE_TOLERANCE_M = 0.01
REPAIR_YAW_TOLERANCE_DEG = 1.0


def _apply_deterministic_repair_actions(previous_layout: dict, feedback: dict) -> tuple[dict, dict]:
    actions = feedback.get("repair_actions") if isinstance(feedback, dict) else None
    return previous_layout, _deterministic_repair_summary([], action_count=len(actions) if isinstance(actions, list) else 0)


def _deterministic_repair_summary(changes: list[dict], *, action_count: int = 0) -> dict:
    return {
        "enabled": False,
        "reason": "Repair actions are advisory cues for model generation; no deterministic geometry rewrite is applied.",
        "candidate_action_count": action_count,
        "applied_actions": changes,
        "changed_object_ids": sorted({str(change["object_id"]) for change in changes if change.get("object_id")}),
        "num_changed_objects": len({str(change["object_id"]) for change in changes if change.get("object_id")}),
        "position_tolerance_m": REPAIR_POSITION_TOLERANCE_M,
    }


def _layout_change_summary(previous_layout: dict, current_layout: dict, repair_targets: object) -> dict:
    targets = {str(item) for item in repair_targets} if isinstance(repair_targets, list) else set()
    previous = {
        obj.get("object_id"): obj
        for obj in previous_layout.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    }
    changed = []
    for obj in current_layout.get("objects", []):
        if not isinstance(obj, dict) or not isinstance(obj.get("object_id"), str):
            continue
        object_id = obj["object_id"]
        before = previous.get(object_id)
        if before is None:
            changed.append(object_id)
            continue
        if _meaningful_layout_object_change(before, obj):
            changed.append(object_id)
    changed_targets = sorted(object_id for object_id in changed if object_id in targets)
    return {
        "changed_object_ids": sorted(changed),
        "changed_repair_targets": changed_targets,
        "num_changed_objects": len(changed),
        "num_changed_repair_targets": len(changed_targets),
        "repair_targets": sorted(targets),
        "repair_noop": bool(targets and not changed_targets),
        "meaningful_change_tolerances": {
            "position_m": REPAIR_POSITION_TOLERANCE_M,
            "size_m": REPAIR_SIZE_TOLERANCE_M,
            "yaw_degrees": REPAIR_YAW_TOLERANCE_DEG,
        },
    }


def _meaningful_layout_object_change(before: dict, after: dict) -> bool:
    if _vector_changed(before.get("center"), after.get("center"), REPAIR_POSITION_TOLERANCE_M):
        return True
    if _vector_changed(before.get("size"), after.get("size"), REPAIR_SIZE_TOLERANCE_M):
        return True
    if _float_changed(before.get("yaw", 0), after.get("yaw", 0), REPAIR_YAW_TOLERANCE_DEG):
        return True
    return any(before.get(key) != after.get(key) for key in ["support_parent", "region_id"])


def _vector_changed(left: object, right: object, tolerance: float) -> bool:
    left_vector = _numeric_vector(left)
    right_vector = _numeric_vector(right)
    if left_vector is None or right_vector is None or len(left_vector) != len(right_vector):
        return left != right
    return any(abs(left_vector[index] - right_vector[index]) > tolerance for index in range(len(left_vector)))


def _float_changed(left: object, right: object, tolerance: float) -> bool:
    try:
        return abs(float(left) - float(right)) > tolerance
    except (TypeError, ValueError):
        return left != right


def _numeric_vector(value: object, *, length: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    if length is not None and len(value) != length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None
