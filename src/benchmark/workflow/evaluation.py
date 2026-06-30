from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from benchmark.evidence_config import resolve_runtime_evidence_config
from benchmark.models.base_model import ModelResponseError
from benchmark.utils.io import write_json
from benchmark.visualization.view_renderer import SimpleBBoxRenderer
from benchmark.workflow.grouping import build_object_grouping_report
from benchmark.workflow.judge_summaries import build_layout_summary, build_scene_summary, text_budget_config
from benchmark.workflow.layout_normalization import sanitize_layout_optional_nulls
from benchmark.workflow.physical_flags import collect_physical_flags
from benchmark.workflow.scoring import (
    ValidityGateResult,
    build_case_metrics,
    compute_object_presence,
    compute_validity_gate,
    infer_input_level,
    visible_attachments,
    visible_relations,
)
from benchmark.workflow.vlm_judge import create_vlm_judge


EVALUATOR_NAME = "vlm_as_judge_v1"
CASE_METRICS_FILENAME = "case_metrics.json"
DEFAULT_VLM_JUDGE = "same_model"
JUDGE_SKIPPED_UNPARSEABLE = "model_output_unparseable"
SCORE_DENOMINATOR = 4.0
DEFAULT_EVALUATION_POLICY = {
    "evaluator_identity": EVALUATOR_NAME,
    "deterministic_flags_affect_validity": False,
    "skip_vlm_judge_only_if_unparseable": True,
    "parseable_layout_requires_vlm_judge": True,
    "overall_valid_source": "vlm_judgement.valid",
}


def evaluate_layout_vlm_as_judge_v1(
    *,
    case: dict,
    layout: dict,
    out_dir: str | Path,
    model_name: str,
    benchmark_config: dict | None = None,
    layout_schema: dict | None = None,
    iteration: int = 0,
    generator_model: Any | None = None,
    judge_model: Any | None = None,
    judge_model_name: str | None = None,
    generation_error: str | None = None,
) -> tuple[dict, dict]:
    output_dir = Path(out_dir)
    evaluation_config = _evaluation_config(benchmark_config)
    evaluation_policy = _evaluation_policy(benchmark_config)
    input_level = infer_input_level(case)
    eval_layout, layout_normalization = sanitize_layout_optional_nulls(layout)
    object_set_normalization = eval_layout.get("_layout_object_set_normalization") if isinstance(eval_layout.get("_layout_object_set_normalization"), dict) else {}
    if object_set_normalization:
        layout_normalization = {**layout_normalization, "object_set_normalization": object_set_normalization}
    repair_change_summary = eval_layout.get("_repair_change_summary") if isinstance(eval_layout.get("_repair_change_summary"), dict) else {}
    if repair_change_summary:
        layout_normalization = {**layout_normalization, "repair_change_summary": repair_change_summary}
    raw_validity_gate = compute_validity_gate(case, eval_layout, layout_schema)
    validity_gate = ValidityGateResult(False, [generation_error]) if generation_error else ValidityGateResult(True, [])
    object_presence = compute_object_presence(case, eval_layout)
    relations = visible_relations(case)
    attachments = visible_attachments(case)

    renderer = SimpleBBoxRenderer(output_dir, benchmark_config=benchmark_config)
    sanity_flags = _sanity_flags(raw_validity_gate)
    physical_flags: list[dict] = []
    view_flags: list[dict] = []
    render_skipped_objects: list[dict] = []
    object_groups: list[dict] = []
    grouping_report: dict = {}
    global_artifacts: list[dict] = []
    group_artifacts: list[dict] = []
    group_view_records: list[dict] = []
    judge_error = ""
    judge_skipped_reason = ""
    judge_artifacts: dict = {}
    judge_input_manifest: dict = {}
    scene_summary: dict = {}
    layout_summary: dict = {}
    text_budget_used: dict = {}
    renderable_layout, render_skipped_objects = _renderable_layout(eval_layout)
    if generation_error:
        sanity_flags.append(_flag(JUDGE_SKIPPED_UNPARSEABLE, generation_error, severity="critical"))
    if render_skipped_objects:
        sanity_flags.extend(render_skipped_objects)

    if not generation_error:
        physical_flags = collect_physical_flags(renderable_layout, case, benchmark_config or {})
        grouping_report = build_object_grouping_report(renderable_layout, case, benchmark_config or {})
        object_groups = grouping_report.get("object_groups", [])
        if evaluation_config.get("save_global_view", evaluation_config.get("save_room_views", True)):
            global_artifacts = renderer.render_global_top_view(case, renderable_layout)
        if evaluation_config.get("save_group_views", True):
            for group in object_groups:
                artifacts, flags = renderer.render_group_views(case, renderable_layout, group, _view_validation_config(benchmark_config))
                view_flags.extend(flags)
                group_artifacts.extend(artifacts)
                group_view_records.append(_group_view_record(group, artifacts, flags))
        if not renderable_layout.get("objects"):
            view_flags.append(
                {
                    "type": "no_renderable_objects",
                    "message": "No objects had renderable object_id, center, and positive size.",
                    "severity": "high",
                }
            )

    if generation_error:
        judge_skipped_reason = JUDGE_SKIPPED_UNPARSEABLE
        judgement = _invalid_judgement(f"Model output was unparseable: {generation_error}")
    else:
        try:
            judgement = create_vlm_judge(benchmark_config, judge_model).judge(
                case=case,
                layout=eval_layout,
                input_level=input_level,
                sanity_flags=sanity_flags,
                physical_flags=physical_flags,
                view_flags=view_flags,
                render_skipped_objects=render_skipped_objects,
                object_groups=object_groups,
                global_view_artifacts=global_artifacts,
                group_view_artifacts=group_artifacts,
                relation_specs=relations,
                attachment_specs=attachments,
                renderable_layout=renderable_layout,
                layout_normalization_summary=layout_normalization,
                validity_gate_passed=True,
                artifact_dir=_judge_artifact_dir(output_dir, iteration),
            )
            judge_artifacts = judgement.pop("_judge_artifacts", {})
            judge_input_manifest = judgement.pop("_judge_input_manifest", {})
            scene_summary = judgement.pop("_scene_summary", {})
            layout_summary = judgement.pop("_layout_summary", {})
            text_budget_used = judgement.pop("_text_budget_used", {})
        except (ModelResponseError, RuntimeError, TypeError, ValueError) as exc:
            judge_error = str(exc)
            judge_artifacts = _existing_judge_artifacts(_judge_artifact_dir(output_dir, iteration), output_dir)
            judge_input_manifest = _existing_judge_input_manifest(_judge_artifact_dir(output_dir, iteration))
            judgement = _invalid_judgement(f"VLM judge failed: {judge_error}")

    if not scene_summary or not layout_summary:
        summary_budget = text_budget_config(benchmark_config)
        scene_summary = build_scene_summary(case, input_level, summary_budget)
        layout_summary = build_layout_summary(
            layout=eval_layout,
            renderable_layout=renderable_layout,
            layout_normalization_summary=layout_normalization,
            object_groups=object_groups,
            sanity_flags=sanity_flags,
            physical_flags=physical_flags,
            view_flags=view_flags,
            render_skipped_objects=render_skipped_objects,
            judge_input_manifest=judge_input_manifest,
            text_budget=summary_budget,
        )
        text_budget_used = {
            "max_total_chars": summary_budget["max_total_chars"],
            "max_scene_summary_chars": summary_budget["max_scene_summary_chars"],
            "max_layout_summary_chars": summary_budget["max_layout_summary_chars"],
            "scene_summary_chars": len(str(scene_summary)),
            "layout_summary_chars": len(str(layout_summary)),
            "prompt_chars": 0,
            "truncated": bool(scene_summary.get("truncated") or layout_summary.get("truncated")),
        }

    if generation_error:
        judgement["judgement_status"] = "unparseable_layout"
        judgement["brief_reasoning"] = judgement.get("brief_reasoning") or judgement.get("short_reason", "")
    if judge_error:
        judgement["judgement_status"] = "judge_error"

    judgement_status = str(judgement.get("judgement_status") or "valid_judgement")

    room_score = int(judgement.get("score", 0))
    relation_results = _align_binary_results(relations, judgement.get("relation_results", []))
    attachment_results = _align_binary_results(attachments, judgement.get("attachment_results", []))
    relation_pass_rate = _pass_rate(relation_results) if relations else None
    attachment_pass_rate = _pass_rate(attachment_results) if attachments else None

    case_metrics = build_case_metrics(
        case=case,
        model_name=model_name,
        validity_gate=validity_gate,
        room_consistency_score=room_score,
        object_presence_rate=object_presence.rate,
        relation_pass_rate=relation_pass_rate,
        attachment_pass_rate=attachment_pass_rate,
    )
    if judge_error:
        case_metrics["primary_score"] = 0.0
    case_metrics_path = output_dir / CASE_METRICS_FILENAME
    write_json(case_metrics_path, case_metrics)

    failed_relations = _failed_relation_items(relation_results, relations, "relation")
    failed_attachments = _failed_relation_items(attachment_results, attachments, "attachment")
    failed_groups = [
        group
        for group in judgement.get("group_results", [])
        if isinstance(group, dict) and group.get("valid") is False
    ]
    repair_targets = sorted(
        {
            object_id
            for item in failed_relations + failed_attachments
            for object_id in item.get("objects", [])
            if isinstance(object_id, str) and object_id
        }
        | {
            object_id
            for group in failed_groups
            for object_id in group.get("object_ids", [])
            if isinstance(object_id, str) and object_id
        }
    )

    generator_metadata = _model_metadata(generator_model or judge_model, model_name)
    same_as_generator = _same_model_endpoint(judge_model, generator_model) if (judge_model is not None or generator_model is not None) else True
    evaluator_metadata = {
        **_model_metadata(judge_model, judge_model_name or model_name),
        "same_as_generator": same_as_generator,
    }
    overall_valid = bool(judgement.get("valid")) if not generation_error else False
    report = {
        "evaluator": EVALUATOR_NAME,
        "evaluator_identity": evaluation_policy["evaluator_identity"],
        "task_id": case.get("task_id") or case.get("case_id"),
        "case_id": case_metrics["case_id"],
        "iteration": iteration,
        "overall_valid": overall_valid,
        "evaluation_policy": evaluation_policy,
        "config_refs": _config_refs(benchmark_config),
        "config_hash": _config_hash(benchmark_config),
        "scene_summary": scene_summary,
        "layout_summary": layout_summary,
        "text_budget_used": text_budget_used,
        "generator_metadata": generator_metadata,
        "evaluator_metadata": evaluator_metadata,
        "input_level": input_level,
        "validity_gate": {
            "passed": bool(validity_gate.passed),
            "errors": validity_gate.errors,
        },
        "vlm_judgement": judgement,
        "vlm_judge_artifacts": judge_artifacts,
        "judgement_status": judgement_status,
        "insufficient_evidence": bool(judgement.get("insufficient_evidence", False)),
        "judge_error": judge_error,
        "judge_skipped_reason": judge_skipped_reason,
        "room_consistency": {
            "score": room_score,
            "score_norm": _score_norm(room_score),
            "judge": evaluation_config.get("vlm_judge", DEFAULT_VLM_JUDGE),
            "short_reason": judgement.get("short_reason", ""),
            "global_assessment": judgement.get("global_assessment", ""),
            "view_artifacts": _public_artifacts(global_artifacts),
        },
        "object_presence": {
            "evaluated": object_presence.evaluated,
            "rate": object_presence.rate,
            "missing_objects": object_presence.missing_objects,
            "placed_required_objects": object_presence.placed_required_objects,
            "required_objects": object_presence.required_objects,
        },
        "specified_relations": {
            "evaluated": bool(relations),
            "pass_rate": relation_pass_rate,
            "results": relation_results,
        },
        "specified_attachments": {
            "evaluated": bool(attachments),
            "pass_rate": attachment_pass_rate,
            "results": attachment_results,
        },
        "debug_evidence": {
            "runtime_evidence_config": resolve_runtime_evidence_config(benchmark_config, case, renderable_layout),
            "layout_normalization": layout_normalization,
            "sanity_flags": sanity_flags,
            "physical_flags": physical_flags,
            "view_flags": view_flags,
            "object_groups": _annotate_groups_with_manifest(object_groups, judge_input_manifest),
            "resolved_grouping_config": grouping_report.get("resolved_grouping_config", {}),
            "omitted_grouping_edges": grouping_report.get("omitted_edges", []),
            "cross_group_relations": grouping_report.get("cross_group_relations", []),
            "render_skipped_objects": render_skipped_objects,
            "group_view_artifacts": group_view_records,
            "judge_input_manifest": judge_input_manifest,
        },
        "case_metrics_path": CASE_METRICS_FILENAME,
        "metrics": case_metrics,
        "summary": {
            "schema_valid": not bool(sanity_flags),
            "physical_valid": None,
            "spatial_relation_valid": _all_passed(relation_results + attachment_results) if (relations or attachments) else None,
            "num_schema_errors": len(sanity_flags),
            "num_physical_errors": len(physical_flags),
            "num_spatial_relation_errors": len(failed_relations) + len(failed_attachments),
        },
        "schema_failures": [{"type": "sanity_flag", "message": flag.get("message", ""), "objects": flag.get("objects", [])} for flag in sanity_flags],
        "physical_failures": [],
        "spatial_relation_failures": failed_relations + failed_attachments,
        "repair_targets": repair_targets,
    }
    return report, case_metrics


def evaluate_layout_v0(**kwargs: Any) -> tuple[dict, dict]:
    return evaluate_layout_vlm_as_judge_v1(**kwargs)


def _align_binary_results(specs: list[dict], judge_results: Any) -> list[dict]:
    by_id = {}
    if isinstance(judge_results, list):
        by_id = {item.get("id"): item for item in judge_results if isinstance(item, dict)}
    results = []
    for spec in specs:
        spec_id = spec.get("id")
        item = by_id.get(spec_id)
        if item is None:
            results.append({**_spec_refs(spec), "id": spec_id, "type": spec.get("type"), "pass": False, "reason": "VLM judge did not return this item."})
        else:
            results.append({**_spec_refs(spec), "id": spec_id, "type": spec.get("type"), "pass": bool(item.get("pass")), "reason": str(item.get("reason", ""))})
    return results


def _spec_refs(spec: dict) -> dict:
    if "child" in spec or "parent" in spec:
        return {"child": spec.get("child"), "parent": spec.get("parent")}
    return {"subject": spec.get("subject"), "object": spec.get("object")}


def _failed_relation_items(results: list[dict], specs: list[dict], kind: str) -> list[dict]:
    specs_by_id = {spec.get("id"): spec for spec in specs}
    failures = []
    for result in results:
        if result.get("pass"):
            continue
        spec = specs_by_id.get(result.get("id"), {})
        objects = [spec.get("child"), spec.get("parent")] if kind == "attachment" else [spec.get("subject"), spec.get("object")]
        failures.append(
            {
                "type": spec.get("type", kind),
                "objects": [item for item in objects if isinstance(item, str)],
                "message": f"VLM judge failed {kind} {result.get('id')}: {result.get('reason', '')}",
            }
        )
    return failures


def _pass_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return float(sum(1 for item in results if item.get("pass"))) / float(len(results))


def _all_passed(results: list[dict]) -> bool:
    return bool(results) and all(bool(item.get("pass")) for item in results)


def _public_artifacts(artifacts: list[dict]) -> list[dict]:
    return [{"id": artifact["id"], "path": artifact["path"], "diagnostics": artifact.get("diagnostics")} for artifact in artifacts]


def _group_view_record(group: dict, artifacts: list[dict], flags: list[dict]) -> dict:
    views: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for artifact in artifacts:
        projection = _projection_for_artifact(artifact)
        if not projection:
            continue
        public = _public_artifacts([artifact])[0]
        views[projection] = public
        diagnostics[projection] = artifact.get("diagnostics", {})
    return {
        "group_id": group.get("group_id"),
        "object_ids": group.get("object_ids", []),
        "num_objects": group.get("num_objects"),
        "group_footprint_diameter_m": group.get("group_footprint_diameter_m"),
        "edge_reasons": group.get("edge_reasons", []),
        "group_label": group.get("group_id"),
        "formation_edges": group.get("formation_edges", []),
        "views": views,
        "diagnostics": diagnostics,
        "view_flags": flags,
        "view_artifacts": _public_artifacts(artifacts),
    }


def _projection_for_artifact(artifact: dict) -> str:
    artifact_id = str(artifact.get("id", ""))
    for suffix in ["xy", "yz", "xz"]:
        if artifact_id.endswith(f"_{suffix}") or artifact_id == suffix:
            return suffix
    return ""


def _score_norm(score: int) -> float:
    return float(score) / SCORE_DENOMINATOR


def _invalid_judgement(reason: str) -> dict:
    return {
        "valid": False,
        "score": 0,
        "score_norm": 0.0,
        "confidence": "low",
        "judgement_status": "judge_error",
        "brief_reasoning": reason,
        "issues": [
            {
                "group_id": None,
                "issue_type": "parseability",
                "severity": "critical",
                "object_ids": [],
                "evidence": reason,
                "repair_hint": "",
            }
        ],
        "insufficient_evidence": False,
        "short_reason": reason,
        "global_assessment": "",
        "group_results": [],
        "relation_results": [],
        "attachment_results": [],
    }


def _evaluation_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("evaluation")
    return section if isinstance(section, dict) else {}


def _evaluation_policy(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("evaluation_policy")
    policy = dict(DEFAULT_EVALUATION_POLICY)
    if isinstance(section, dict):
        policy.update(section)
    policy["evaluator_identity"] = str(policy.get("evaluator_identity") or EVALUATOR_NAME)
    return policy


def _config_refs(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    refs = config.get("config_refs")
    return dict(refs) if isinstance(refs, dict) else {}


def _config_hash(benchmark_config: dict | None) -> str:
    config = benchmark_config or {}
    return str(config.get("config_hash") or "")


def _view_validation_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("view_validation")
    return section if isinstance(section, dict) else {}


def _sanity_flags(validity_gate: ValidityGateResult) -> list[dict]:
    return [_flag("layout_sanity", error, severity="medium") for error in validity_gate.errors]


def _renderable_layout(layout: dict) -> tuple[dict, list[dict]]:
    if not isinstance(layout, dict):
        return {"objects": []}, [_flag("render_skipped_object", "Layout is not a JSON object.", severity="high")]

    renderable = dict(layout)
    objects = []
    skipped = []
    for index, obj in enumerate(layout.get("objects", []) if isinstance(layout.get("objects"), list) else []):
        if not isinstance(obj, dict):
            skipped.append(_skip_flag(index, "Object entry is not a JSON object.", obj))
            continue
        object_id = obj.get("object_id") or obj.get("id")
        if not isinstance(object_id, str) or not object_id:
            skipped.append(_skip_flag(index, "Object has no usable object_id/id.", obj))
            continue
        if not _valid_vector(obj.get("center"), positive=False):
            skipped.append(_skip_flag(index, f"{object_id} has invalid center.", obj))
            continue
        if not _valid_vector(obj.get("size"), positive=True):
            skipped.append(_skip_flag(index, f"{object_id} has invalid positive size.", obj))
            continue
        objects.append(obj)
    renderable["objects"] = objects
    return renderable, skipped


def _skip_flag(index: int, message: str, obj: Any) -> dict:
    object_id = obj.get("object_id") or obj.get("id") if isinstance(obj, dict) else None
    flag = _flag("render_skipped_object", message, objects=[object_id] if isinstance(object_id, str) and object_id else [], severity="high")
    flag["object_index"] = index
    flag["object_id"] = object_id
    flag["category"] = obj.get("category") if isinstance(obj, dict) else None
    flag["reason"] = message
    flag["raw_object"] = obj
    return flag


def _flag(flag_type: str, message: str, *, objects: list[str] | None = None, severity: str = "medium") -> dict:
    return {
        "type": flag_type,
        "objects": objects or [],
        "severity": severity,
        "message": message,
    }


def _valid_vector(value: object, *, positive: bool) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            return False
        if positive and float(item) <= 0:
            return False
    return True


def _model_metadata(model: Any | None, fallback_name: str) -> dict:
    return {
        "model_id": getattr(model, "model_id", None) or getattr(model, "model", None) or getattr(model, "name", fallback_name),
        "endpoint": getattr(model, "endpoint", None),
        "runtime_profile": getattr(model, "runtime_profile", None),
        "judge_evidence_budgeting": bool(getattr(model, "judge_evidence_budgeting", False)),
        "temperature": getattr(model, "temperature", None),
        "max_tokens": getattr(model, "max_tokens", None),
        "timeout_seconds": getattr(model, "timeout_seconds", None),
        "response_format_json": getattr(model, "response_format_json", None),
    }


def _judge_artifact_dir(output_dir: Path, iteration: int) -> Path:
    return output_dir / "vlm_judge" / f"iter_{iteration:03d}"


def _existing_judge_artifacts(artifact_dir: Path, output_dir: Path) -> dict:
    files = {
        "input_manifest_path": artifact_dir / "judge_input_manifest.json",
        "prompt_path": artifact_dir / "judge_prompt.json",
        "image_manifest_path": artifact_dir / "judge_image_manifest.json",
        "request_metadata_path": artifact_dir / "judge_request_metadata.json",
        "raw_response_path": artifact_dir / "judge_raw_response.txt",
        "parsed_response_path": artifact_dir / "judge_parsed_response.json",
    }
    if not any(path.exists() for path in files.values()):
        return {}
    return {key: _relative_to(path, output_dir) for key, path in files.items() if path.exists()}


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _existing_judge_input_manifest(artifact_dir: Path) -> dict:
    path = artifact_dir / "judge_input_manifest.json"
    if not path.exists():
        return {}
    try:
        import json

        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        return {}


def _annotate_groups_with_manifest(object_groups: list[dict], manifest: dict) -> list[dict]:
    if not isinstance(manifest, dict) or not manifest.get("budgeting_enabled"):
        return object_groups
    selected = {
        item.get("group_id"): item
        for item in manifest.get("selected_groups", [])
        if isinstance(item, dict)
    } if isinstance(manifest, dict) else {}
    omitted = {
        item.get("group_id"): item
        for item in manifest.get("omitted_groups", [])
        if isinstance(item, dict)
    } if isinstance(manifest, dict) else {}
    annotated = []
    for group in object_groups:
        if not isinstance(group, dict):
            continue
        group_id = group.get("group_id")
        item = dict(group)
        if group_id in selected:
            item["sent_to_judge"] = True
            item["selection_score"] = int(selected[group_id].get("selection_score", 0))
            item["selection_reasons"] = list(selected[group_id].get("selection_reasons", []))
        elif group_id in omitted:
            item["sent_to_judge"] = False
            item["selection_score"] = int(omitted[group_id].get("selection_score", 0))
            item["selection_reasons"] = list(omitted[group_id].get("selection_reasons", []))
        else:
            item.setdefault("sent_to_judge", False)
            item.setdefault("selection_score", 0)
            item.setdefault("selection_reasons", [])
        annotated.append(item)
    return annotated


def _same_model_endpoint(left: Any | None, right: Any | None) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return False
    left_id = getattr(left, "model_id", None) or getattr(left, "name", None)
    right_id = getattr(right, "model_id", None) or getattr(right, "name", None)
    return bool(left_id and right_id and left_id == right_id and getattr(left, "endpoint", None) == getattr(right, "endpoint", None))
