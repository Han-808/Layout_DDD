from __future__ import annotations

from pathlib import Path

from benchmark.feedback import build_feedback
from benchmark.metrics import compute_case_metrics
from benchmark.utils.io import ensure_dir, read_json, write_json
from benchmark.visualization import export_viewer_scene
from benchmark.workflow.evaluation import evaluate_layout_v0
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
        case_id = input_json.get("case_id") or input_json.get("task_id")
        if case_id:
            input_json.setdefault("case_id", case_id)
            input_json.setdefault("task_id", case_id)
        next_state["input_json"] = input_json
        next_state["task_id"] = input_json.get("task_id") or input_json.get("case_id")
        next_state.setdefault("iteration", 0)
        next_state.setdefault("history", [])
        next_state.setdefault("evaluation_reports", [])
        ensure_dir(next_state["out_dir"])
        return next_state


class GenerateLayoutNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        layout = next_state["model"].generate_layout(next_state["input_json"], next_state["layout_schema"])
        path = Path(next_state["out_dir"]) / "initial_layout.json"
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        if save_intermediate:
            write_json(path, layout)
        next_state["current_layout"] = layout
        next_state["current_layout_path"] = str(path) if save_intermediate else ""
        next_state["iteration"] = 0
        return next_state


class EvaluateLayoutNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        iteration = int(next_state.get("iteration", 0))
        report, case_metrics = evaluate_layout_v0(
            case=next_state["input_json"],
            layout=next_state["current_layout"],
            out_dir=next_state["out_dir"],
            model_name=next_state.get("model_name", getattr(next_state.get("model"), "name", "unknown_model")),
            benchmark_config=next_state.get("benchmark_config"),
            layout_schema=next_state.get("layout_schema"),
            iteration=iteration,
        )
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
        next_state["case_metrics_path"] = str(Path(next_state["out_dir"]) / "case_metrics.json")
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
        layout = next_state["model"].repair_layout(
            next_state["input_json"],
            next_state["current_layout"],
            next_state["current_feedback"],
            next_state["layout_schema"],
        )
        path = Path(next_state["out_dir"]) / f"repaired_layout_iter_{next_iteration}.json"
        save_intermediate = save_intermediate_artifacts(next_state.get("benchmark_config"))
        if save_intermediate:
            write_json(path, layout)
        next_state["current_layout"] = layout
        next_state["current_layout_path"] = str(path) if save_intermediate else ""
        next_state["iteration"] = next_iteration
        return next_state


class ComputeMetricsNode:
    def __call__(self, state: BenchmarkState) -> BenchmarkState:
        next_state = dict(state)
        metrics = compute_case_metrics(
            next_state["history"],
            max_repair_iterations=int(next_state.get("max_repair_iterations", 0)),
        )
        result = {
            "task_id": next_state["task_id"],
            "model_name": next_state.get("model_name", getattr(next_state.get("model"), "name", "unknown_model")),
            "max_repair_iterations": int(next_state.get("max_repair_iterations", 0)),
            "metrics": metrics,
            "case_metrics_path": next_state.get("case_metrics_path", ""),
            "history": compact_history(next_state["history"]),
            "final_evaluation": next_state["current_evaluation"],
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
            )
            viewer_scene["workflow"] = build_workflow_metadata(
                {
                    **next_state,
                    "per_case_result": result,
                    "per_case_result_path": str(result_path),
                    "viewer_scene_path": str(Path(next_state["out_dir"]) / "viewer_scene.json"),
                }
            )
            viewer_scene["feedback"] = next_state.get("current_feedback", {})
            viewer_scene["metrics"] = metrics
            viewer_scene["metrics_summary"] = next_state.get("case_metrics", metrics)
            viewer_scene["view_artifacts"] = {
                "room": next_state["current_evaluation"].get("room_consistency", {}).get("view_artifacts", []),
                "relations": next_state["current_evaluation"].get("specified_relations", {}).get("results", []),
                "attachments": next_state["current_evaluation"].get("specified_attachments", {}).get("results", []),
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
