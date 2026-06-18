from __future__ import annotations

from pathlib import Path

from benchmark.utils.io import write_json
from benchmark.visualization.view_renderer import SimpleBBoxRenderer
from benchmark.workflow.scoring import (
    build_case_metrics,
    compute_object_presence,
    compute_validity_gate,
    find_layout_object,
    get_description_text,
    infer_input_level,
    visible_attachments,
    visible_relations,
)
from benchmark.workflow.vlm_judge import create_pair_judge, create_room_judge


EVALUATOR_NAME = "layered_vlm_room_pair_evaluator_v0"


def evaluate_layout_v0(
    *,
    case: dict,
    layout: dict,
    out_dir: str | Path,
    model_name: str,
    benchmark_config: dict | None = None,
    layout_schema: dict | None = None,
    iteration: int = 0,
) -> tuple[dict, dict]:
    output_dir = Path(out_dir)
    evaluation_config = _evaluation_config(benchmark_config)
    input_level = infer_input_level(case)
    validity_gate = compute_validity_gate(case, layout, layout_schema)
    object_presence = compute_object_presence(case, layout)

    renderer = SimpleBBoxRenderer(output_dir)
    room_artifacts = renderer.render_room_views(case, layout) if evaluation_config.get("save_room_views", True) else []
    room_judge = create_room_judge(benchmark_config)
    room_result = room_judge.judge(
        description=get_description_text(case),
        input_level=input_level,
        view_artifacts=[artifact["abs_path"] for artifact in room_artifacts if artifact["id"] != "camera_policy"],
        object_summary=_object_summary(layout),
        validity_gate_passed=validity_gate.passed,
    )
    room_score = int(room_result["score"])
    room_score_norm = float(room_score) / 4.0

    relation_results, relation_pass_rate = _judge_specs(
        specs=visible_relations(case),
        kind="relation",
        case=case,
        layout=layout,
        renderer=renderer,
        benchmark_config=benchmark_config,
        save_pair_views=evaluation_config.get("save_pair_views", True),
    )
    attachment_results, attachment_pass_rate = _judge_specs(
        specs=visible_attachments(case),
        kind="attachment",
        case=case,
        layout=layout,
        renderer=renderer,
        benchmark_config=benchmark_config,
        save_pair_views=evaluation_config.get("save_pair_views", True),
    )

    case_metrics = build_case_metrics(
        case=case,
        model_name=model_name,
        validity_gate=validity_gate,
        room_consistency_score=room_score,
        object_presence_rate=object_presence.rate,
        relation_pass_rate=relation_pass_rate,
        attachment_pass_rate=attachment_pass_rate,
    )
    case_metrics_path = output_dir / "case_metrics.json"
    write_json(case_metrics_path, case_metrics)

    failed_relations = [
        {
            "type": item.get("type", "relation"),
            "objects": [item.get("subject"), item.get("object")],
            "message": f"Explicit relation {item.get('id')} did not pass pair judge.",
        }
        for item in relation_results
        if not item.get("pass")
    ]
    failed_attachments = [
        {
            "type": item.get("type", "attachment"),
            "objects": [item.get("child"), item.get("parent")],
            "message": f"Explicit attachment {item.get('id')} did not pass pair judge.",
        }
        for item in attachment_results
        if not item.get("pass")
    ]

    report = {
        "evaluator": EVALUATOR_NAME,
        "task_id": case.get("task_id") or case.get("case_id"),
        "case_id": case_metrics["case_id"],
        "iteration": iteration,
        "overall_valid": bool(validity_gate.passed),
        "input_level": input_level,
        "validity_gate": {
            "passed": bool(validity_gate.passed),
            "errors": validity_gate.errors,
        },
        "room_consistency": {
            "score": room_score,
            "score_norm": room_score_norm,
            "judge": evaluation_config.get("room_judge", "mock"),
            "short_reason": room_result.get("short_reason", ""),
            "view_artifacts": _public_artifacts(room_artifacts),
        },
        "object_presence": {
            "evaluated": object_presence.evaluated,
            "rate": object_presence.rate,
            "missing_objects": object_presence.missing_objects,
            "placed_required_objects": object_presence.placed_required_objects,
            "required_objects": object_presence.required_objects,
        },
        "specified_relations": {
            "evaluated": bool(visible_relations(case)),
            "pass_rate": relation_pass_rate,
            "results": relation_results,
        },
        "specified_attachments": {
            "evaluated": bool(visible_attachments(case)),
            "pass_rate": attachment_pass_rate,
            "results": attachment_results,
        },
        "debug_evidence": {
            "physical_flags": [],
            "spatial_evidence": [],
        },
        "case_metrics_path": "case_metrics.json",
        "metrics": case_metrics,
        "summary": {
            "schema_valid": bool(validity_gate.passed),
            "physical_valid": None,
            "spatial_relation_valid": None,
            "num_schema_errors": len(validity_gate.errors),
            "num_physical_errors": 0,
            "num_spatial_relation_errors": 0,
        },
        "schema_failures": [{"type": "validity_gate", "message": error, "objects": []} for error in validity_gate.errors],
        "physical_failures": [],
        "spatial_relation_failures": failed_relations + failed_attachments,
        "repair_targets": [],
    }
    return report, case_metrics


def _judge_specs(
    *,
    specs: list[dict],
    kind: str,
    case: dict,
    layout: dict,
    renderer: SimpleBBoxRenderer,
    benchmark_config: dict | None,
    save_pair_views: bool,
) -> tuple[list[dict], float | None]:
    if not specs:
        return [], None

    pair_judge = create_pair_judge(benchmark_config)
    results = []
    passes = 0
    for spec in specs:
        artifacts = renderer.render_pair_views(case, layout, spec) if save_pair_views else []
        subject_ref = spec.get("subject") or spec.get("child")
        object_ref = spec.get("object") or spec.get("parent")
        symbolic = {
            "subject_found": find_layout_object(layout, str(subject_ref or "")) is not None,
            "object_found": find_layout_object(layout, str(object_ref or "")) is not None,
        }
        judge_result = pair_judge.judge(
            spec=spec,
            pair_view_artifacts=[artifact["abs_path"] for artifact in artifacts if artifact["id"] != "camera_policy"],
            symbolic_evidence=symbolic,
        )
        passed = bool(judge_result["pass"])
        passes += int(passed)
        if kind == "attachment":
            item = {
                "id": spec.get("id"),
                "type": spec.get("type"),
                "child": spec.get("child"),
                "parent": spec.get("parent"),
                "pass": passed,
                "short_reason": judge_result.get("short_reason", ""),
                "view_artifacts": _public_artifacts(artifacts),
            }
        else:
            item = {
                "id": spec.get("id"),
                "type": spec.get("type"),
                "subject": spec.get("subject"),
                "object": spec.get("object"),
                "pass": passed,
                "short_reason": judge_result.get("short_reason", ""),
                "view_artifacts": _public_artifacts(artifacts),
            }
        results.append(item)
    return results, float(passes) / float(len(specs))


def _public_artifacts(artifacts: list[dict]) -> list[dict]:
    return [{"id": artifact["id"], "path": artifact["path"]} for artifact in artifacts]


def _object_summary(layout: dict) -> list[dict]:
    return [
        {"object_id": obj.get("object_id") or obj.get("id"), "category": obj.get("category")}
        for obj in layout.get("objects", [])
        if isinstance(obj, dict)
    ]


def _evaluation_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("evaluation")
    return section if isinstance(section, dict) else {}
