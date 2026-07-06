from __future__ import annotations

from pathlib import Path

from benchmark.utils.io import write_json
from benchmark.workflow.artifacts import artifact_path


WORKFLOW_VERSION = "layout_workflow_v0"


def build_workflow_trace(state: dict, out_dir: str | Path | None = None) -> dict:
    case_id = state.get("task_id") or state.get("input_json", {}).get("case_id")
    current = state.get("current_evaluation", {})
    room_views = current.get("room_consistency", {}).get("view_artifacts", [])
    debug_evidence = current.get("debug_evidence", {})
    group_views = debug_evidence.get("group_view_artifacts", []) if isinstance(debug_evidence, dict) else []
    vlm_judge_artifacts = current.get("vlm_judge_artifacts", {})
    pair_views = []
    for section in ["specified_relations", "specified_attachments"]:
        for item in current.get(section, {}).get("results", []):
            pair_views.extend(item.get("view_artifacts", []))

    nodes = [
        {
            "id": "normalize_input",
            "status": "success",
            "artifacts": {
                "case": artifact_path(state.get("case_path", ""), out_dir),
                "resolved_run_config": artifact_path(state.get("resolved_run_config_path", ""), out_dir),
            },
        },
        {
            "id": "generate_layout",
            "status": "success",
            "artifacts": {
                "layout": artifact_path(state.get("current_layout_path", ""), out_dir),
                "request_metadata": artifact_path(state.get("generation_request_metadata_path", ""), out_dir),
                "raw_response": artifact_path(state.get("generation_raw_response_path", ""), out_dir),
            },
        },
        {
            "id": "evaluate_layout",
            "status": "success",
            "artifacts": {
                "evaluation_report": "evaluation_report.json",
                "case_metrics": artifact_path(state.get("case_metrics_path", "case_metrics.json"), out_dir),
                "room_views": room_views,
                "global_views": room_views,
                "group_views": group_views,
                "pair_views": pair_views,
                "vlm_judge": vlm_judge_artifacts,
            },
        },
        {
            "id": "compute_metrics",
            "status": "success",
            "artifacts": {"per_case_result": artifact_path(state.get("per_case_result_path", ""), out_dir)},
        },
    ]
    return {
        "case_id": case_id,
        "workflow_version": WORKFLOW_VERSION,
        "nodes": nodes,
        "edges": [
            ["normalize_input", "generate_layout"],
            ["generate_layout", "evaluate_layout"],
            ["evaluate_layout", "compute_metrics"],
        ],
        "attempts": [
            {
                "attempt_id": item.get("iteration", index),
                "layout": artifact_path(item.get("layout_path", ""), out_dir),
                "evaluation_report": artifact_path(item.get("evaluation_path", ""), out_dir),
                "case_metrics": f"case_metrics_iter_{item.get('iteration', index)}.json",
            }
            for index, item in enumerate(state.get("history", []))
        ],
    }


def write_workflow_trace(state: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    trace = build_workflow_trace(state, out)
    trace_path = write_json(out / "workflow_trace.json", trace)
    graph_path = out / "workflow_graph.mmd"
    graph_path.write_text(
        "\n".join(
            [
                "flowchart TD",
                "  START([START]) --> normalize_input",
                "  normalize_input --> generate_layout",
                "  generate_layout --> evaluate_layout[VLM-as-judge final validity source]",
                "  evaluate_layout -->|metrics| compute_metrics",
                "  evaluate_layout -->|repair| build_feedback",
                "  build_feedback --> repair_layout",
                "  repair_layout --> evaluate_layout",
                "  compute_metrics --> END([END])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return trace_path, graph_path
