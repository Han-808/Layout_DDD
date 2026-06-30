from __future__ import annotations

from pathlib import Path
from typing import Any


RICH_HISTORY_KEYS = {"layout", "evaluation", "feedback"}


def benchmark_settings(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("benchmark")
    return section if isinstance(section, dict) else config


def output_settings(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("outputs")
    return section if isinstance(section, dict) else {}


def save_intermediate_artifacts(benchmark_config: dict | None) -> bool:
    return bool(benchmark_settings(benchmark_config).get("save_intermediate_artifacts", True))


def save_viewer_scene(benchmark_config: dict | None) -> bool:
    return bool(benchmark_settings(benchmark_config).get("save_viewer_scene", True))


def configured_max_repair_iterations(benchmark_config: dict | None) -> int:
    return int(benchmark_settings(benchmark_config).get("max_repair_iterations", 0))


def per_case_filename(benchmark_config: dict | None) -> str:
    return str(output_settings(benchmark_config).get("per_case_filename") or "per_case_result.json")


def compact_history(history: list[dict]) -> list[dict]:
    return [{key: value for key, value in item.items() if key not in RICH_HISTORY_KEYS} for item in history]


def make_history_entry(
    *,
    iteration: int,
    layout_path: str,
    evaluation_path: str,
    report: dict,
    layout: dict | None = None,
    evaluation: dict | None = None,
) -> dict:
    summary = report["summary"]
    entry: dict[str, Any] = {
        "iteration": iteration,
        "layout_path": layout_path,
        "evaluation_path": evaluation_path,
        "feedback_path": "",
        "metrics": report.get(
            "metrics",
            {
                "schema_validity": int(bool(summary["schema_valid"])),
                "physical_validity": int(bool(summary["physical_valid"])),
                "spatial_relation_validity": int(bool(summary["spatial_relation_valid"])),
            },
        ),
        "schema_valid": summary["schema_valid"],
        "physical_valid": summary["physical_valid"],
        "spatial_relation_valid": summary["spatial_relation_valid"],
        "overall_valid": report["overall_valid"],
        "num_schema_errors": summary["num_schema_errors"],
        "num_physical_errors": summary["num_physical_errors"],
        "num_spatial_relation_errors": summary["num_spatial_relation_errors"],
    }
    if layout is not None:
        entry["layout"] = layout
    if evaluation is not None:
        entry["evaluation"] = evaluation
    return entry


def attach_feedback_to_history(history: list[dict], iteration: int, feedback_path: str, feedback: dict) -> list[dict]:
    updated = []
    for item in history:
        next_item = dict(item)
        if int(next_item.get("iteration", -1)) == iteration:
            next_item["feedback_path"] = feedback_path
            next_item["feedback"] = feedback
        updated.append(next_item)
    return updated


def artifact_path(path: str | Path | None, out_dir: str | Path | None = None) -> str:
    if not path:
        return ""
    target = Path(path)
    if out_dir is not None:
        try:
            return target.resolve().relative_to(Path(out_dir).resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return target.as_posix()


def build_workflow_metadata(
    state: dict,
    *,
    include_data: bool = False,
) -> dict:
    out_dir = state.get("out_dir")
    artifacts: list[dict] = [
        _artifact(
            step="input",
            label="Benchmark instance",
            path=artifact_path(state.get("case_path", ""), out_dir),
            status="source",
            data=state.get("input_json"),
            include_data=include_data,
        )
    ]

    for item in state.get("history", []):
        iteration = int(item.get("iteration", 0))
        artifacts.append(
            _artifact(
                step="generate" if iteration == 0 else "repair",
                label="Generate initial layout" if iteration == 0 else f"Repair layout iteration {iteration}",
                path=artifact_path(item.get("layout_path", ""), out_dir),
                iteration=iteration,
                status=evaluation_status(item.get("overall_valid")),
                data=item.get("layout"),
                include_data=include_data,
            )
        )
        artifacts.append(
            _artifact(
                step="evaluate",
                label="Evaluate layout" if iteration == 0 else f"Evaluate repair iteration {iteration}",
                path=artifact_path(item.get("evaluation_path", ""), out_dir),
                iteration=iteration,
                status=evaluation_status(item.get("overall_valid")),
                data=item.get("evaluation"),
                include_data=include_data,
            )
        )
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        for key, label in [
            ("input_manifest_path", "VLM judge input manifest"),
            ("prompt_path", "VLM judge prompt"),
            ("image_manifest_path", "VLM judge image manifest"),
            ("request_metadata_path", "VLM judge request metadata"),
            ("raw_response_path", "VLM judge raw response"),
            ("parsed_response_path", "VLM judge parsed response"),
        ]:
            path = evaluation.get("vlm_judge_artifacts", {}).get(key)
            if path:
                artifacts.append(
                    _artifact(
                        step="judge",
                        label=label,
                        path=path,
                        iteration=iteration,
                        status="evidence",
                        include_data=False,
                    )
                )
        if iteration == 0 and state.get("generation_request_metadata_path"):
            artifacts.append(
                _artifact(
                    step="generate",
                    label="Generation request metadata",
                    path=artifact_path(state.get("generation_request_metadata_path", ""), out_dir),
                    iteration=iteration,
                    status="api_metadata",
                    include_data=False,
                )
            )
        if iteration == 0 and state.get("generation_raw_response_path"):
            artifacts.append(
                _artifact(
                    step="generate",
                    label="Generation raw response",
                    path=artifact_path(state.get("generation_raw_response_path", ""), out_dir),
                    iteration=iteration,
                    status="raw_response",
                    include_data=False,
                )
            )
        if item.get("feedback_path") or item.get("feedback") is not None:
            artifacts.append(
                _artifact(
                    step="feedback",
                    label=f"Build feedback iteration {iteration}",
                    path=artifact_path(item.get("feedback_path", ""), out_dir),
                    iteration=iteration,
                    status="repair_instructions",
                    data=item.get("feedback"),
                    include_data=include_data,
                )
            )

    artifacts.extend(
        [
            _artifact(
                step="metrics",
                label="Case result",
                path=artifact_path(state.get("per_case_result_path", ""), out_dir),
                status="summary",
                data=state.get("per_case_result"),
                include_data=include_data,
            ),
            _artifact(
                step="visualize",
                label="Viewer scene",
                path=artifact_path(state.get("viewer_scene_path", "viewer_scene.json"), out_dir),
                status="visualization_only",
                include_data=False,
            ),
        ]
    )

    return {
        "case_path": artifact_path(state.get("case_path", "")),
        "model_name": state.get("model_name", getattr(state.get("model"), "name", "unknown_model")),
        "max_repair_iterations": int(state.get("max_repair_iterations", 0)),
        "artifacts": artifacts,
    }


def _artifact(
    *,
    step: str,
    label: str,
    path: str,
    status: str,
    iteration: int | None = None,
    data: Any = None,
    include_data: bool,
) -> dict:
    item: dict[str, Any] = {
        "step": step,
        "label": label,
        "path": path,
        "status": status,
    }
    if iteration is not None:
        item["iteration"] = iteration
    if include_data and data is not None:
        item["data"] = data
    return item


def evaluation_status(overall_valid: object) -> str:
    if overall_valid is True:
        return "valid"
    if overall_valid is False:
        return "invalid"
    return "unknown"
