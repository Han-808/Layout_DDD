#!/usr/bin/env python3
"""Build a simple local, read-only viewer for persisted VLM evidence/results."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.visual_judge.l3_prompts import (  # noqa: E402
    L3_METRIC_BOUNDARY_RULES,
    L3_METRIC_PROMPT_VERSION,
    L3_METRIC_RUBRICS,
)
from benchmark.visual_judge.functional_discovery_contract import (  # noqa: E402
    normalized_functional_relation_predicates,
)

GROUP_COLORS = (
    "#d1242f",
    "#0969da",
    "#1a7f37",
    "#8250df",
    "#bf8700",
    "#cf4a00",
)

TRACE_STAGE_LABELS = {
    "evidence_gate": "Evidence gate",
    "judge": "Judge",
    "judge_evidence_request": "Judge evidence request",
    "acquisition_planner": "Acquisition plan",
    "camera_dsl": "Camera constraints",
    "trusted_candidate_bank": "Candidate views",
    "preview_render": "Candidate previews",
    "camera_selector": "Camera selection",
    "camera_escalation": "Camera escalation",
    "render": "Evidence render",
}

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

OBJECT_LEVEL_ATTRIBUTION_METRICS = (
    "functional_consistency",
    "semantic_placement_consistency",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL record at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"expected a JSON object at {path}:{line_number}"
            )
        records.append(value)
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceURLResolver:
    def __init__(
        self,
        *,
        serve_root: Path,
        bundle_dir: Path | None,
    ) -> None:
        self.serve_root = serve_root.expanduser().resolve()
        self.bundle_dir = (
            bundle_dir.expanduser().resolve()
            if bundle_dir is not None
            else None
        )
        self.bundle_records: dict[str, dict[str, Any]] = {}

    def url_for(self, path: Path) -> str | None:
        source = path.expanduser().resolve()
        if self.bundle_dir is None:
            return relative_url(source, self.serve_root)
        digest = file_sha256(source)
        suffix = source.suffix.lower() or ".bin"
        destination_name = f"{digest[:20]}{suffix}"
        destination = self.bundle_dir / "evidence" / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            not destination.is_file()
            or file_sha256(destination) != digest
        ):
            shutil.copy2(source, destination)
        copied_digest = file_sha256(destination)
        if copied_digest != digest:
            raise RuntimeError(
                f"viewer evidence copy failed SHA-256 verification: {source}"
            )
        self.bundle_records[str(source)] = {
            "source_path": str(source),
            "viewer_path": str(destination),
            "sha256": digest,
            "byte_size": source.stat().st_size,
            "copy_verified": True,
            "source_modified": False,
        }
        return "/evidence/" + quote(destination_name)

    def write_manifest(self) -> Path | None:
        if self.bundle_dir is None:
            return None
        manifest_path = self.bundle_dir / "bundle_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "vlm_evidence_viewer_bundle_v1",
            "source_files_modified": False,
            "copy_policy": "byte_for_byte_sha256_verified",
            "evidence_file_count": len(self.bundle_records),
            "evidence": [
                self.bundle_records[key]
                for key in sorted(self.bundle_records)
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def source_evidence_paths(case_manifest: dict[str, Any]) -> list[Path]:
    source_root = Path(
        str(case_manifest.get("source_case_root") or "")
    ).expanduser()
    source_manifest = optional_json(source_root / "case_manifest.json")
    paths = source_manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    evidence = paths.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return [
        (source_root / str(evidence[name])).resolve()
        for name in ("perspective", "top", "identity")
        if evidence.get(name)
    ]


def source_scene(case_manifest: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(
        str(case_manifest.get("source_case_root") or "")
    ).expanduser()
    source_manifest = optional_json(source_root / "case_manifest.json")
    paths = source_manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    scene_path = paths.get("canonical_scene")
    if not scene_path:
        return {}
    return optional_json((source_root / str(scene_path)).resolve())


def source_blend_path(case_manifest: dict[str, Any]) -> Path | None:
    source_root_text = str(case_manifest.get("source_case_root") or "")
    if not source_root_text:
        return None
    source_root = Path(source_root_text).expanduser().resolve()
    source_manifest = optional_json(source_root / "case_manifest.json")
    paths = source_manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    blend_path = paths.get("blend")
    if not blend_path:
        return None
    resolved = (source_root / str(blend_path)).resolve()
    return resolved if resolved.is_file() else None


def render_blender_command(
    *,
    case_id: str,
    case_manifest: dict[str, Any],
) -> str:
    blend_path = source_blend_path(case_manifest)
    if blend_path is None:
        command_html = (
            '<span class="blender-command-unavailable">'
            "Prepared Blender file is unavailable for this scene."
            "</span>"
        )
        copy_button = ""
    else:
        command = f"open -a Blender {shlex.quote(str(blend_path))}"
        command_html = f"<code>{html.escape(command)}</code>"
        copy_button = (
            '<button type="button" class="copy-blender-command" '
            f'data-copy-text="{html.escape(command, quote=True)}">'
            "Copy command</button>"
        )
    return f"""
      <section class="blender-launch" aria-label="Open scene in Blender">
        <div class="blender-launch-heading">
          <div>
            <div class="eyebrow">Local scene</div>
            <strong>Open {html.escape(case_id)} in Blender</strong>
          </div>
          {copy_button}
        </div>
        <div class="blender-command">{command_html}</div>
      </section>
    """


def object_floor_bounds(
    scene: dict[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    metadata = scene.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    registry = metadata.get("instance_registry")
    registry = registry if isinstance(registry, dict) else {}
    instances = registry.get("instances")
    instances = instances if isinstance(instances, list) else []
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        object_id = str(
            instance.get("evaluator_object_id")
            or instance.get("instance_id")
            or ""
        )
        aabb = instance.get("world_aabb")
        aabb = aabb if isinstance(aabb, dict) else {}
        minimum = aabb.get("min_m")
        maximum = aabb.get("max_m")
        if (
            object_id
            and isinstance(minimum, list)
            and isinstance(maximum, list)
            and len(minimum) >= 2
            and len(maximum) >= 2
        ):
            result[object_id] = (
                float(minimum[0]),
                float(minimum[1]),
                float(maximum[0]),
                float(maximum[1]),
            )
    objects = scene.get("objects")
    objects = objects if isinstance(objects, list) else []
    for item in objects:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or "")
        center = item.get("center")
        size = item.get("size")
        if (
            object_id
            and object_id not in result
            and isinstance(center, list)
            and isinstance(size, list)
            and len(center) >= 2
            and len(size) >= 2
        ):
            result[object_id] = (
                float(center[0]) - float(size[0]) / 2.0,
                float(center[1]) - float(size[1]) / 2.0,
                float(center[0]) + float(size[0]) / 2.0,
                float(center[1]) + float(size[1]) / 2.0,
            )
    return result


def request_usage(metadata: Any) -> dict[str, int] | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("usage")
    if not isinstance(raw, dict):
        return None
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field] = value
    prompt_details = raw.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            result["cached_prompt_tokens"] = cached
    completion_details = raw.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = completion_details.get("reasoning_tokens")
        if (
            isinstance(reasoning, int)
            and not isinstance(reasoning, bool)
            and reasoning >= 0
        ):
            result["reasoning_tokens"] = reasoning
    return result or None


def compact_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "status",
        "verdict",
        "score",
        "confidence",
        "reason",
        "defects",
        "object_findings",
        "object_penalty_count",
        "missing_evidence",
        "evidence_request",
    )
    return {
        key: deepcopy(value[key])
        for key in keys
        if key in value
    }


def evidence_packet_audit(
    *,
    metric: str,
    phase: str,
    images: list[str],
    run_prompt_version: str | None,
) -> dict[str, Any] | None:
    """Describe the persisted packet without rewriting historical evidence."""

    if (
        metric not in OBJECT_LEVEL_ATTRIBUTION_METRICS
        or phase
        not in {
            "global_discovery",
            "cross_group_relation_review",
            "group_local_review",
        }
    ):
        return None
    expected_roles = (
        ["angled_global"]
        if phase == "global_discovery"
        else [
            "angled_global_context",
            "cross_group_relation_local",
        ]
        if phase == "cross_group_relation_review"
        else ["angled_global_context", "group_local"]
    )
    actual_roles: list[str] = []
    for index, path in enumerate(images):
        filename = Path(path).name.lower()
        if any(
            token in filename
            for token in ("top", "overhead", "birdseye", "bird_eye")
        ):
            role = "top_down_global"
        elif phase == "global_discovery":
            role = "angled_global"
        elif index == 0:
            role = "angled_global_context"
        elif phase == "cross_group_relation_review":
            role = "cross_group_relation_local"
        else:
            role = "group_local"
        actual_roles.append(role)

    matches_current_default = bool(
        actual_roles == expected_roles
        and len(images) == len(expected_roles)
    )
    persisted_version = str(run_prompt_version or "").strip() or None
    if persisted_version is None:
        status = "unversioned"
    elif persisted_version != L3_METRIC_PROMPT_VERSION:
        status = "historical"
    else:
        status = (
            "current_default"
            if matches_current_default
            else "current_run_custom_or_mismatch"
        )
    return {
        "status": status,
        "run_prompt_version": persisted_version,
        "current_prompt_version": L3_METRIC_PROMPT_VERSION,
        "actual_image_count": len(images),
        "actual_roles": actual_roles,
        "expected_current_default_image_count": len(expected_roles),
        "expected_current_default_roles": expected_roles,
        "matches_current_default": matches_current_default,
        "historical_evidence_preserved": True,
    }


def object_level_finding_summary(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return persisted or visibly reconstructed object-level findings."""

    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    summaries: list[dict[str, Any]] = []
    for metric_name in OBJECT_LEVEL_ATTRIBUTION_METRICS:
        metric_report = metrics.get(metric_name)
        if not isinstance(metric_report, dict):
            continue
        persisted = metric_report.get("final_object_findings")
        if isinstance(persisted, list):
            findings = [
                deepcopy(item)
                for item in persisted
                if isinstance(item, dict)
            ]
            source = "persisted_runner_output"
        else:
            observations: list[tuple[str, dict[str, Any]]] = []
            global_record = metric_report.get("global_discovery")
            if (
                isinstance(global_record, dict)
                and str(global_record.get("verdict") or "") == "invalid"
            ):
                observations.extend(
                    ("global_discovery", defect)
                    for defect in global_record.get("defects") or []
                    if isinstance(defect, dict)
                )
            group_results = metric_report.get("group_results")
            group_results = (
                group_results if isinstance(group_results, list) else []
            )
            for group in group_results:
                if not isinstance(group, dict):
                    continue
                judgement = group.get("judgement")
                if not isinstance(judgement, dict):
                    continue
                if not (
                    str(judgement.get("verdict") or "") == "invalid"
                    or (
                        group.get("status") == "evaluated"
                        and group.get("score") == 0.0
                    )
                ):
                    continue
                phase = (
                    "group_local_review:"
                    + str(group.get("group_id") or "unknown")
                )
                observations.extend(
                    (phase, defect)
                    for defect in judgement.get("defects") or []
                    if isinstance(defect, dict)
                )
            relation_results = metric_report.get(
                "cross_group_relation_results"
            )
            relation_results = (
                relation_results
                if isinstance(relation_results, list)
                else []
            )
            for relation in relation_results:
                if not isinstance(relation, dict):
                    continue
                judgement = relation.get("judgement")
                if (
                    not isinstance(judgement, dict)
                    or relation.get("status") != "evaluated"
                    or relation.get("score") != 0.0
                ):
                    continue
                phase = (
                    "cross_group_relation_review:"
                    + str(relation.get("relation_id") or "unknown")
                )
                observations.extend(
                    (phase, defect)
                    for defect in judgement.get("defects") or []
                    if isinstance(defect, dict)
                )
            findings = _reconstruct_object_findings(
                metric_name,
                observations,
            )
            source = "viewer_reconstructed_from_persisted_defects"
        summaries.append(
            {
                "metric": metric_name,
                "source": source,
                "findings": findings,
                "penalty_unit_count": len(findings),
                "cross_phase_deduplication": True,
                "cross_metric_deduplication": False,
            }
        )
    return summaries


def _reconstruct_object_findings(
    metric_name: str,
    observations: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    findings: dict[str, dict[str, Any]] = {}
    for phase, defect in observations:
        target_ids = defect.get("target_ids")
        if not isinstance(target_ids, list):
            continue
        observation = {
            "source_phase": phase,
            "scope": defect.get("scope"),
            "target_ids": sorted(
                {
                    str(value)
                    for value in target_ids
                    if str(value).strip()
                }
            ),
            "relation": defect.get("relation"),
            "reason": defect.get("reason"),
        }
        for object_id in observation["target_ids"]:
            finding = findings.setdefault(
                object_id,
                {
                    "metric": metric_name,
                    "object_id": object_id,
                    "attribution_unit": "object",
                    "source_phases": [],
                    "observations": [],
                },
            )
            if phase not in finding["source_phases"]:
                finding["source_phases"].append(phase)
            if observation not in finding["observations"]:
                finding["observations"].append(deepcopy(observation))
    for finding in findings.values():
        count = len(finding["observations"])
        finding["observation_count"] = count
        finding["merged_duplicate_observation_count"] = max(0, count - 1)
        finding["observed_in_global_and_local"] = bool(
            "global_discovery" in finding["source_phases"]
            and any(
                phase.startswith("group_local_review")
                for phase in finding["source_phases"]
            )
        )
    return list(findings.values())


def _path_values(value: Any) -> list[str]:
    """Return persisted image paths from common evidence record shapes."""
    if isinstance(value, str):
        return [value] if Path(value).suffix.lower() in IMAGE_SUFFIXES else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_path_values(item))
        return list(dict.fromkeys(result))
    if not isinstance(value, dict):
        return []
    result = []
    for key in (
        "path",
        "image_path",
        "evidence_path",
        "rgb_path",
        "contour_path",
        "output_path",
        "source_path",
    ):
        if key in value:
            result.extend(_path_values(value[key]))
    return list(dict.fromkeys(result))


def _event_images(event: dict[str, Any]) -> list[str]:
    images = _path_values(event.get("images_used"))
    result = event.get("result")
    result = result if isinstance(result, dict) else {}
    for key in ("images_used", "visual_evidence", "render_evidence"):
        images.extend(_path_values(result.get(key)))
    return list(dict.fromkeys(images))


def _audit_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get("audit")
    return nested if isinstance(nested, dict) else value


def _event_status(event: dict[str, Any]) -> str:
    result = event.get("result")
    result = result if isinstance(result, dict) else {}
    value = (
        event.get("status")
        or result.get("status")
        or result.get("outcome")
    )
    if value is None and event.get("stage") == "evidence_gate":
        value = "ready" if result.get("ready") is True else "blocked"
    return str(value or "recorded")


def _event_summary(event: dict[str, Any], images: list[str]) -> str:
    stage = str(event.get("stage") or "event")
    result = event.get("result")
    result = result if isinstance(result, dict) else {}
    if stage == "evidence_gate":
        readiness = "ready" if result.get("ready") is True else "not ready"
        reasons = result.get("reason_codes")
        reasons = reasons if isinstance(reasons, list) else []
        suffix = f" ({', '.join(str(value) for value in reasons)})" if reasons else ""
        return f"{len(images)} image(s) checked; packet {readiness}{suffix}."
    if stage == "judge":
        status = str(result.get("status") or "recorded")
        reason = str(result.get("reason") or "").strip()
        if status == "need_more_evidence":
            prefix = "Judge requested additional visual evidence."
        else:
            prefix = f"Judge returned {status}."
        return f"{prefix} {reason}".strip()
    if stage == "judge_evidence_request":
        return str(
            event.get("reason")
            or "The Judge's evidence request was recorded."
        )
    if stage == "acquisition_planner":
        request = event.get("evidence_request")
        request = request if isinstance(request, dict) else {}
        targets = request.get("target_ids")
        targets = targets if isinstance(targets, list) else []
        observations = request.get("missing_observations")
        observations = observations if isinstance(observations, list) else []
        parts = []
        if targets:
            parts.append("targets " + ", ".join(str(value) for value in targets))
        if observations:
            parts.append(
                "missing " + ", ".join(str(value) for value in observations)
            )
        return "; ".join(parts) or "A camera repair plan was created."
    if stage == "trusted_candidate_bank":
        count = event.get("candidate_count")
        return (
            f"{count} trusted candidate view(s) prepared."
            if isinstance(count, int)
            else "Trusted candidate views were prepared."
        )
    if stage == "preview_render":
        count = event.get("preview_count")
        return (
            f"{count} candidate preview(s) rendered."
            if isinstance(count, int)
            else "Candidate previews were rendered."
        )
    if stage == "camera_selector":
        selected = result.get("selected_view_ids")
        selected = selected if isinstance(selected, list) else []
        selection_stage = str(event.get("selection_stage") or "camera")
        return (
            f"{selection_stage} selector chose "
            + ", ".join(str(value) for value in selected)
            + "."
            if selected
            else f"{selection_stage} selector returned {_event_status(event)}."
        )
    if stage == "camera_escalation":
        return str(
            event.get("reason")
            or event.get("escalation_reason")
            or "Camera selection escalated to the next stage."
        )
    if stage == "camera_dsl":
        return str(
            event.get("error")
            or "Metric-scoped camera constraints were evaluated."
        )
    if stage == "render":
        count = event.get("rendered_view_count")
        changed = event.get("packet_changed")
        if event.get("status") == "failed":
            return str(event.get("error") or "Evidence rendering failed.")
        count_text = f"{count} view(s)" if isinstance(count, int) else "Evidence"
        changed_text = (
            " changed the evidence packet"
            if changed is True
            else " did not change the evidence packet"
            if changed is False
            else " was added to the evidence packet"
        )
        return f"{count_text}{changed_text}."
    return str(
        event.get("reason")
        or result.get("reason")
        or f"{TRACE_STAGE_LABELS.get(stage, stage.replace('_', ' ').title())} recorded."
    )


def acquisition_timeline(
    *,
    control_audit: Any,
    fallback_images: list[str],
    final_result: dict[str, Any],
) -> dict[str, Any]:
    """Normalize control-loop audit records for reusable viewer rendering."""
    audit = _audit_payload(control_audit)
    raw_trace = audit.get("trace")
    trace = (
        [item for item in raw_trace if isinstance(item, dict)]
        if isinstance(raw_trace, list)
        else []
    )
    trace_source = "camera_control_audit.audit.trace" if trace else "reconstructed"
    if not trace:
        trace = [
            {
                "stage": "evidence_gate",
                "evidence_round": 0,
                "result": {"ready": bool(fallback_images)},
                "images_used": list(fallback_images),
                "reconstructed": True,
            },
            {
                "stage": "judge",
                "evidence_round": 0,
                "result": deepcopy(final_result),
                "images_used": list(fallback_images),
                "reconstructed": True,
            },
        ]

    seen_images: set[str] = set()
    steps: list[dict[str, Any]] = []
    request_keys: set[str] = set()
    judge_calls = 0
    selector_calls = 0
    completed_renders = 0
    packet_change_events = 0
    added_images: set[str] = set()
    maximum_round = 0

    for index, event in enumerate(trace):
        stage = str(event.get("stage") or "event")
        round_value = event.get("evidence_round")
        evidence_round = (
            round_value
            if isinstance(round_value, int) and not isinstance(round_value, bool)
            else 0
        )
        maximum_round = max(maximum_round, evidence_round)
        images = _event_images(event)
        new_images = [
            path
            for path in images
            if evidence_round > 0 and path not in seen_images
        ]
        seen_images.update(images)
        if stage == "render" and event.get("status") == "completed":
            completed_renders += 1
            added_images.update(new_images)
            if event.get("packet_changed") is True:
                packet_change_events += 1
        if stage == "judge":
            judge_calls += 1
            result = event.get("result")
            result = result if isinstance(result, dict) else {}
            request = result.get("evidence_request")
            if isinstance(request, dict):
                request_keys.add(
                    json.dumps(request, sort_keys=True, ensure_ascii=False)
                )
        if stage == "judge_evidence_request":
            request = event.get("evidence_request")
            request = request if isinstance(request, dict) else event
            request_keys.add(
                json.dumps(request, sort_keys=True, ensure_ascii=False)
            )
        if stage == "acquisition_planner":
            request = event.get("evidence_request")
            if isinstance(request, dict):
                request_keys.add(
                    json.dumps(request, sort_keys=True, ensure_ascii=False)
                )
        if stage == "camera_selector":
            selector_calls += 1
        steps.append(
            {
                "index": index + 1,
                "stage": stage,
                "label": TRACE_STAGE_LABELS.get(
                    stage,
                    stage.replace("_", " ").title(),
                ),
                "evidence_round": evidence_round,
                "status": _event_status(event),
                "summary": _event_summary(event, images),
                "images": images,
                "new_images": new_images,
                "details": deepcopy(event),
                "reconstructed": event.get("reconstructed") is True,
            }
        )

    telemetry = audit.get("experiment_telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    rounds_value = audit.get("rounds_used")
    rounds_used = (
        rounds_value
        if isinstance(rounds_value, int) and not isinstance(rounds_value, bool)
        else maximum_round
    )
    judge_calls_value = telemetry.get("judge_calls")
    if isinstance(judge_calls_value, int) and judge_calls_value > judge_calls:
        judge_calls = judge_calls_value
    selector_calls_value = audit.get("selector_calls_used")
    if isinstance(selector_calls_value, int) and selector_calls_value > selector_calls:
        selector_calls = selector_calls_value
    stop_reason = (
        control_audit.get("stop_reason")
        if isinstance(control_audit, dict)
        else None
    ) or telemetry.get("stop_reason")
    additional_evidence = bool(
        request_keys
        or rounds_used > 0
        or completed_renders > 0
        or added_images
        or packet_change_events > 0
    )
    return {
        "trace_source": trace_source,
        "steps": steps,
        "summary": {
            "judge_calls": judge_calls,
            "judge_request_count": len(request_keys),
            "selector_calls": selector_calls,
            "evidence_rounds": rounds_used,
            "completed_renders": completed_renders,
            "added_image_count": len(added_images),
            "packet_change_events": packet_change_events,
            "rejudged": judge_calls > 1,
            "additional_evidence": additional_evidence,
            "stop_reason": str(stop_reason or "not persisted"),
        },
    }


def grouping_call(
    grouping: dict[str, Any],
    *,
    images: list[Path],
) -> dict[str, Any] | None:
    provenance = grouping.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    if not provenance:
        return None
    groups = grouping.get("object_groups")
    groups = groups if isinstance(groups, list) else []
    return {
        "id": "grouping",
        "layer": "Grouping",
        "metric": "object_grouping",
        "scope": "scene",
        "members": [],
        "status": str(grouping.get("status") or "unknown"),
        "verdict": str(grouping.get("status") or "unknown"),
        "score": None,
        "confidence": None,
        "reason": str(grouping.get("reason") or ""),
        "images": [str(path) for path in images],
        "request_metadata": deepcopy(provenance.get("request_metadata") or {}),
        "usage": request_usage(provenance.get("request_metadata")),
        "prompt": (
            "Grouping prompt version: "
            f"{provenance.get('prompt_version') or 'unknown'}.\n\n"
            "The exact composed system/user messages were not persisted by "
            "this run."
        ),
        "prompt_note": "Prompt metadata only; exact messages unavailable.",
        "result": {
            "status": grouping.get("status"),
            "grouping_backend": grouping.get("grouping_backend"),
            "group_count": len(groups),
            "object_groups": deepcopy(groups),
        },
    }


def render_grouping_output(
    *,
    case_id: str,
    grouping: dict[str, Any],
    scene: dict[str, Any],
    top_image_path: Path | None,
    resolver: EvidenceURLResolver,
) -> str:
    groups = grouping.get("object_groups")
    groups = groups if isinstance(groups, list) else []
    cards: list[str] = []
    legend: list[str] = []
    regions: list[str] = []
    bounds_by_id = object_floor_bounds(scene)
    boundary = scene.get("boundary")
    boundary = boundary if isinstance(boundary, list) else []
    boundary_x = [
        float(point[0])
        for point in boundary
        if isinstance(point, list) and len(point) >= 2
    ]
    boundary_y = [
        float(point[1])
        for point in boundary
        if isinstance(point, list) and len(point) >= 2
    ]
    room_min_x = min(boundary_x) if boundary_x else 0.0
    room_max_x = max(boundary_x) if boundary_x else 1.0
    room_min_y = min(boundary_y) if boundary_y else 0.0
    room_max_y = max(boundary_y) if boundary_y else 1.0
    room_width = max(0.001, room_max_x - room_min_x)
    room_depth = max(0.001, room_max_y - room_min_y)
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
        group_id = html.escape(str(group.get("group_id") or "unknown"))
        label = html.escape(str(group.get("label") or "Unlabelled group"))
        anchor = html.escape(str(group.get("anchor_object_id") or "—"))
        reason = html.escape(str(group.get("reason") or "No reason persisted."))
        object_ids = group.get("object_ids")
        object_ids = object_ids if isinstance(object_ids, list) else []
        raw_object_ids = [str(value) for value in object_ids]
        members = "".join(
            f"<code>{html.escape(str(object_id))}</code>"
            for object_id in object_ids
        )
        member_bounds = [
            bounds_by_id[object_id]
            for object_id in raw_object_ids
            if object_id in bounds_by_id
        ]
        if member_bounds:
            minimum_x = max(
                room_min_x,
                min(item[0] for item in member_bounds) - 0.06,
            )
            minimum_y = max(
                room_min_y,
                min(item[1] for item in member_bounds) - 0.06,
            )
            maximum_x = min(
                room_max_x,
                max(item[2] for item in member_bounds) + 0.06,
            )
            maximum_y = min(
                room_max_y,
                max(item[3] for item in member_bounds) + 0.06,
            )
            left = (minimum_x - room_min_x) / room_width * 100.0
            top = (room_max_y - maximum_y) / room_depth * 100.0
            width = (maximum_x - minimum_x) / room_width * 100.0
            height = (maximum_y - minimum_y) / room_depth * 100.0
            regions.append(
                f"""
                <div class="group-region"
                  style="--group-color:{color};left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%"
                  title="{group_id}: {label}">
                  <span>{group_id}</span>
                </div>
                """
            )
        legend.append(
            f"""
            <div class="group-legend-row" style="--group-color:{color}">
              <span class="group-swatch"></span>
              <div>
                <strong>{group_id} · {label}</strong>
                <small>{html.escape(", ".join(raw_object_ids))}</small>
              </div>
            </div>
            """
        )
        cards.append(
            f"""
            <article class="group-card" style="--group-color:{color}">
              <div class="group-card-title">
                <span>{group_id}</span>
                <strong>{label}</strong>
              </div>
              <div class="group-members">{members}</div>
              <p><strong>Anchor:</strong> <code>{anchor}</code></p>
              <p class="group-reason">{reason}</p>
            </article>
            """
        )
    top_view = ""
    if top_image_path is not None and top_image_path.is_file():
        top_url = resolver.url_for(top_image_path)
        if top_url is not None:
            top_view = f"""
              <div class="grouping-visual">
                <div>
                  <div class="topdown-view">
                    <img src="{top_url}" alt="Top-down scene view with grouping overlay">
                    <div class="room-overlay">{''.join(regions)}</div>
                  </div>
                  <p class="topdown-caption">
                    Original standardized top-down image with a separate HTML
                    grouping overlay. The source image is unchanged.
                  </p>
                </div>
                <div class="group-legend">{''.join(legend)}</div>
              </div>
            """
    relations = grouping.get("cross_group_relations")
    relations = relations if isinstance(relations, list) else []
    omitted = grouping.get("omitted_edges")
    omitted = omitted if isinstance(omitted, list) else []
    details = {
        "cross_group_relations": deepcopy(relations),
        "omitted_edges": deepcopy(omitted),
    }
    return f"""
      <section class="grouping-output">
        <div class="grouping-heading">
          <div>
            <div class="eyebrow">Grouping output · {html.escape(case_id)}</div>
            <h2>Object grouping</h2>
            <p class="muted">{html.escape(str(grouping.get("reason") or ""))}</p>
          </div>
          <div class="grouping-meta">
            <span>status <strong>{html.escape(str(grouping.get("status") or "unknown"))}</strong></span>
            <span>backend <strong>{html.escape(str(grouping.get("grouping_backend") or "unknown"))}</strong></span>
            <span>groups <strong>{len(groups)}</strong></span>
          </div>
        </div>
        {top_view}
        <div class="group-grid">{''.join(cards)}</div>
        <details class="group-relations">
          <summary>Cross-group relations and omitted edges</summary>
          <pre>{json_block(details)}</pre>
        </details>
      </section>
    """


def _functional_discovery_call_records(
    api_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in api_calls:
        call_type = str(item.get("call_type") or "")
        if not (
            "functional_discovery" in call_type
            or "usable_surface" in call_type
        ):
            continue
        usage = item.get("tokens_usage")
        usage = usage if isinstance(usage, dict) else {}
        records.append(
            {
                "api_call_number": item.get("api_call_number"),
                "role": str(item.get("role") or "unknown"),
                "call_type": call_type,
                "status": str(item.get("status") or "unknown"),
                "image_count": item.get("image_count"),
                "total_tokens": usage.get("total_tokens"),
                "error_type": item.get("error_type"),
                "error": item.get("error"),
            }
        )
    return records


def functional_evidence_audit(
    *,
    grouping: dict[str, Any],
    report: dict[str, Any],
    api_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    metric = metrics.get("functional_consistency")
    if not isinstance(metric, dict):
        return None

    evidence = metric.get("functional_prejudgement_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    acquisition = metric.get("functional_probe_acquisition")
    acquisition = acquisition if isinstance(acquisition, dict) else {}
    runtime_audit = evidence.get("runtime_audit")
    runtime_audit = (
        runtime_audit if isinstance(runtime_audit, dict) else acquisition
    )
    discovery = evidence.get("functional_discovery")
    if not isinstance(discovery, dict):
        discovery = metric.get("functional_discovery")
    discovery = discovery if isinstance(discovery, dict) else {}

    groups = grouping.get("object_groups")
    groups = (
        [item for item in groups if isinstance(item, dict)]
        if isinstance(groups, list)
        else []
    )
    object_to_group: dict[str, str] = {}
    group_scopes: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group.get("group_id") or "unknown")
        object_ids = [
            str(value)
            for value in group.get("object_ids") or []
        ]
        for object_id in object_ids:
            object_to_group[object_id] = group_id
        group_scopes.append(
            {
                "group_id": group_id,
                "label": str(group.get("label") or "—"),
                "anchor_object_id": str(
                    group.get("anchor_object_id") or "—"
                ),
                "object_ids": object_ids,
                "reason": str(group.get("reason") or ""),
            }
        )

    directed_targets = discovery.get("directed_surface_targets")
    directed_targets = (
        [item for item in directed_targets if isinstance(item, dict)]
        if isinstance(directed_targets, list)
        else []
    )
    affordance_ledger = discovery.get("object_affordance_ledger")
    affordance_ledger = (
        [item for item in affordance_ledger if isinstance(item, dict)]
        if isinstance(affordance_ledger, list)
        else []
    )
    usable_surfaces = evidence.get("usable_surface_hypotheses")
    if not isinstance(usable_surfaces, list):
        usable_surfaces = acquisition.get("usable_surface_hypotheses")
    usable_surfaces = (
        [item for item in usable_surfaces if isinstance(item, dict)]
        if isinstance(usable_surfaces, list)
        else []
    )

    surface_objects: dict[str, dict[str, Any]] = {}
    for item in affordance_ledger:
        object_id = str(item.get("object_id") or "")
        if object_id:
            surface_objects.setdefault(object_id, {})["affordance"] = item
    for item in directed_targets:
        object_id = str(
            item.get("target_id") or item.get("object_id") or ""
        )
        if object_id:
            surface_objects.setdefault(object_id, {})["target"] = item
    for item in usable_surfaces:
        object_id = str(item.get("target_id") or "")
        if object_id:
            surface_objects.setdefault(object_id, {})["decoded"] = item

    surface_records: list[dict[str, Any]] = []
    for object_id in sorted(surface_objects):
        values = surface_objects[object_id]
        affordance = values.get("affordance")
        affordance = affordance if isinstance(affordance, dict) else {}
        target = values.get("target")
        target = target if isinstance(target, dict) else {}
        decoded = values.get("decoded")
        decoded = decoded if isinstance(decoded, dict) else {}
        surfaces = decoded.get("surfaces")
        surfaces = (
            [item for item in surfaces if isinstance(item, dict)]
            if isinstance(surfaces, list)
            else []
        )
        surface_roles = (
            affordance.get("surface_roles")
            or target.get("surface_roles")
            or target.get("requested_surface_roles")
            or decoded.get("requested_surface_roles")
            or []
        )
        decoded_sides = [
            " · ".join(
                part
                for part in (
                    str(surface.get("surface_role") or ""),
                    str(surface.get("side_id") or ""),
                )
                if part
            )
            for surface in surfaces
        ]
        need_clearance = affordance.get("need_clearance")
        if not isinstance(need_clearance, bool):
            legacy_clearance = str(
                affordance.get("clearance_need") or ""
            ).strip()
            need_clearance = (
                legacy_clearance != "none"
                if legacy_clearance
                else None
            )
        surface_records.append(
            {
                "object_id": object_id,
                "group_id": object_to_group.get(object_id, "—"),
                "directionality": affordance.get("directionality"),
                "surface_roles": [str(value) for value in surface_roles],
                "need_clearance": need_clearance,
                "decode_status": decoded.get("status"),
                "decoded_sides": decoded_sides,
                "reason": str(
                    decoded.get("reason")
                    or target.get("observation_goal")
                    or affordance.get("observation_goal")
                    or ""
                ),
            }
        )

    functional_relations: list[dict[str, Any]] = []
    rejected_functional_relations: list[dict[str, Any]] = []
    for key, scope in (
        ("within_group_correspondences", "within_group"),
        ("cross_group_correspondences", "cross_group"),
    ):
        values = discovery.get(key)
        values = values if isinstance(values, list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            try:
                predicates = normalized_functional_relation_predicates(item)
            except ValueError:
                predicates = (str(item.get("predicate") or ""),)
            for predicate in predicates:
                target_ids = [
                    str(value) for value in item.get("target_ids") or []
                ]
                record = {
                    "source": "functional_discovery",
                    "scope": str(item.get("scope") or scope),
                    "target_ids": target_ids,
                    "group_ids": [
                        str(value)
                        for value in item.get("group_ids") or []
                    ],
                    "predicate": predicate,
                    "observation_kinds": (
                        [predicate] if predicate else []
                    ),
                    "observation_goal": str(
                        item.get("observation_goal") or ""
                    ),
                    "atomicity": (
                        "atomic_pair"
                        if len(target_ids) == 2
                        else f"legacy_non_atomic_{len(target_ids)}_objects"
                    ),
                }
                if len(target_ids) == 2:
                    functional_relations.append(record)
                else:
                    rejected_functional_relations.append(
                        {
                            **record,
                            "rejection_reason": (
                                "current functional relation contract requires "
                                "exactly two object IDs"
                            ),
                        }
                    )
    grouping_context_relations = grouping.get("cross_group_relations")
    grouping_context_relations = (
        grouping_context_relations
        if isinstance(grouping_context_relations, list)
        else []
    )
    grouping_relation_records: list[dict[str, Any]] = []
    for item in grouping_context_relations:
        if not isinstance(item, dict):
            continue
        grouping_relation_records.append(
            {
                "source": "grouping",
                "scope": str(item.get("scope") or "cross_group"),
                "target_ids": [
                    str(value)
                    for value in (
                        item.get("target_ids")
                        or item.get("object_ids")
                        or []
                    )
                ],
                "group_ids": [
                    str(value)
                    for value in item.get("group_ids") or []
                ],
                "predicate": str(item.get("predicate") or ""),
                "observation_kinds": [
                    str(value)
                    for value in item.get("observation_kinds") or []
                ],
                "observation_goal": str(
                    item.get("observation_goal")
                    or item.get("reason")
                    or ""
                ),
                "atomicity": "grouping_context_not_judge_check",
            }
        )

    status = str(
        evidence.get("status")
        or runtime_audit.get("status")
        or (
            "complete"
            if discovery
            else "not_persisted"
        )
    )
    reason = str(
        runtime_audit.get("reason")
        or runtime_audit.get("error")
        or discovery.get("reason")
        or ""
    )
    calls = _functional_discovery_call_records(api_calls)
    provenance = discovery.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    relation_input_contract = provenance.get(
        "relation_input_contract"
    )
    relation_input_contract = (
        deepcopy(relation_input_contract)
        if isinstance(relation_input_contract, dict)
        else {}
    )
    if discovery and not relation_input_contract:
        source_identity = evidence.get("source_identity")
        source_identity = (
            source_identity if isinstance(source_identity, dict) else {}
        )
        visual_evidence = []
        for role, key in (
            ("scene_global", "global_image_path"),
            ("global_identity_overlay", "identity_image_path"),
        ):
            path = str(source_identity.get(key) or "").strip()
            if path:
                visual_evidence.append({"role": role, "path": path})
        relation_call = next(
            (
                item
                for item in calls
                if item.get("call_type")
                == "vlm_camera_pose.functional_discovery.relations"
            ),
            {},
        )
        relation_input_contract = {
            "status": "legacy_inferred_from_persisted_audit",
            "relation_schema_version": provenance.get(
                "relation_schema_version"
            ),
            "visual_evidence": visual_evidence,
            "visual_evidence_roles": [
                item["role"] for item in visual_evidence
            ],
            "image_count": relation_call.get("image_count"),
            "structured_context_fields": [
                "scene_id",
                "scene_type",
                "object_list",
                "identity_grounding",
                "affordance_prior",
            ],
            "structured_context": None,
            "structured_context_status": (
                "field_names_inferred_but_exact_json_not_persisted"
            ),
            "trusted_group_partition_status": (
                "not_delivered_by_legacy_relation_contract"
            ),
        }
    elif relation_input_contract:
        relation_input_contract.setdefault("status", "persisted_exact")
    completed_calls = sum(
        1 for item in calls if item.get("status") == "complete"
    )
    fail_closed = bool(
        status == "failed"
        and completed_calls
        and not discovery
        and not surface_records
        and not functional_relations
    )
    return {
        "status": status,
        "reason": reason,
        "error_type": runtime_audit.get("error_type"),
        "error": runtime_audit.get("error"),
        "decision_authority": str(
            evidence.get("decision_authority") or "none"
        ),
        "fail_closed": fail_closed,
        "group_scopes": group_scopes,
        "surface_records": surface_records,
        "relation_records": functional_relations,
        "rejected_relation_records": rejected_functional_relations,
        "grouping_relation_records": grouping_relation_records,
        "relation_input_contract": relation_input_contract,
        "discovery_calls": calls,
        "completed_discovery_calls": completed_calls,
        "discovery": deepcopy(discovery),
    }


def render_functional_evidence_audit(
    *,
    grouping: dict[str, Any],
    report: dict[str, Any],
    api_calls: list[dict[str, Any]],
    resolver: EvidenceURLResolver | None = None,
) -> str:
    audit = functional_evidence_audit(
        grouping=grouping,
        report=report,
        api_calls=api_calls,
    )
    if audit is None:
        return ""

    group_rows = []
    for group in audit["group_scopes"]:
        members = ", ".join(group["object_ids"]) or "—"
        group_rows.append(
            "<tr>"
            f"<td><code>{html.escape(group['group_id'])}</code></td>"
            f"<td>{html.escape(group['label'])}</td>"
            f"<td>{html.escape(members)}</td>"
            f"<td><code>{html.escape(group['anchor_object_id'])}</code></td>"
            f"<td>{html.escape(group['reason'])}</td>"
            "</tr>"
        )

    surface_rows = []
    for item in audit["surface_records"]:
        roles = ", ".join(item["surface_roles"]) or "—"
        decoded = ", ".join(item["decoded_sides"]) or "—"
        need_clearance = item["need_clearance"]
        clearance_label = (
            "yes"
            if need_clearance is True
            else "no"
            if need_clearance is False
            else "—"
        )
        surface_rows.append(
            "<tr>"
            f"<td><code>{html.escape(item['object_id'])}</code></td>"
            f"<td><code>{html.escape(item['group_id'])}</code></td>"
            f"<td>{html.escape(str(item['directionality'] or '—'))}</td>"
            f"<td>{html.escape(roles)}</td>"
            f"<td>{clearance_label}</td>"
            f"<td>{html.escape(decoded)}</td>"
            f"<td>{html.escape(str(item['decode_status'] or '—'))}</td>"
            "</tr>"
        )
    if not surface_rows:
        surface_rows.append(
            '<tr><td colspan="7" class="audit-empty">'
            "No accepted usable-side target or decoded surface was persisted."
            "</td></tr>"
        )

    relation_rows = []
    for item in audit["relation_records"]:
        targets = " ↔ ".join(item["target_ids"]) or "—"
        groups = ", ".join(item["group_ids"]) or "—"
        predicate = (
            str(item.get("predicate") or "").strip()
            or ", ".join(item["observation_kinds"])
            or "—"
        )
        relation_rows.append(
            "<tr>"
            f"<td>{html.escape(item['source'])}</td>"
            f"<td>{html.escape(item['scope'])}</td>"
            f"<td>{html.escape(targets)}</td>"
            f"<td>{html.escape(groups)}</td>"
            f"<td>{html.escape(predicate)}</td>"
            f"<td>{html.escape(str(item.get('atomicity') or '—'))}</td>"
            f"<td>{html.escape(item['observation_goal'])}</td>"
            "</tr>"
        )
    if not relation_rows:
        relation_rows.append(
            '<tr><td colspan="7" class="audit-empty">'
            "No accepted structured within-group or cross-group functional "
            "relation was persisted."
            "</td></tr>"
        )

    rejected_relation_rows = []
    for item in audit["rejected_relation_records"]:
        targets = " ↔ ".join(item["target_ids"]) or "—"
        rejected_relation_rows.append(
            "<tr>"
            f"<td>{html.escape(item['scope'])}</td>"
            f"<td>{html.escape(targets)}</td>"
            f"<td>{html.escape(str(item.get('predicate') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('atomicity') or '—'))}</td>"
            f"<td>{html.escape(item['rejection_reason'])}</td>"
            "</tr>"
        )
    rejected_relation_html = (
        f"""
        <details class="functional-audit-block functional-evidence-audit-failed">
          <summary>Rejected legacy non-atomic relation records · {len(rejected_relation_rows)}</summary>
          <p class="audit-empty">Shown for historical audit only; these records do not enter required checks, evidence acquisition, or Judge episodes.</p>
          <table>
            <thead><tr><th>Scope</th><th>Objects</th><th>Predicate</th><th>Contract</th><th>Rejection</th></tr></thead>
            <tbody>{''.join(rejected_relation_rows)}</tbody>
          </table>
        </details>
        """
        if rejected_relation_rows
        else ""
    )

    grouping_relation_rows = []
    for item in audit["grouping_relation_records"]:
        targets = " ↔ ".join(item["target_ids"]) or "—"
        groups = ", ".join(item["group_ids"]) or "—"
        predicate = (
            str(item.get("predicate") or "").strip()
            or ", ".join(item["observation_kinds"])
            or "—"
        )
        grouping_relation_rows.append(
            "<tr>"
            f"<td>{html.escape(item['scope'])}</td>"
            f"<td>{html.escape(targets)}</td>"
            f"<td>{html.escape(groups)}</td>"
            f"<td>{html.escape(predicate)}</td>"
            f"<td>{html.escape(item['observation_goal'])}</td>"
            "</tr>"
        )
    if not grouping_relation_rows:
        grouping_relation_rows.append(
            '<tr><td colspan="5" class="audit-empty">'
            "No grouping-level contextual relation was persisted."
            "</td></tr>"
        )

    relation_input = audit.get("relation_input_contract")
    relation_input = (
        relation_input if isinstance(relation_input, dict) else {}
    )
    relation_input_images = []
    for item in relation_input.get("visual_evidence") or []:
        if not isinstance(item, dict):
            continue
        role = html.escape(str(item.get("role") or "visual evidence"))
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        rendered = (
            render_image(path, resolver)
            if resolver is not None
            else f"<code>{html.escape(path)}</code>"
        )
        relation_input_images.append(
            f'<div class="relation-input-visual"><strong>{role}</strong>'
            f"{rendered}</div>"
        )
    relation_input_images_html = "".join(relation_input_images) or (
        '<p class="audit-empty">No exact relation-discovery image path was '
        "persisted.</p>"
    )
    relation_input_html = (
        f"""
        <details class="functional-audit-block" open>
          <summary>Relation discovery inputs · visual evidence + compact JSON</summary>
          <div class="relation-input-images">{relation_input_images_html}</div>
          <pre>{json_block(relation_input)}</pre>
        </details>
        """
        if relation_input
        else ""
    )

    call_rows = []
    for item in audit["discovery_calls"]:
        tokens = item.get("total_tokens")
        token_text = str(tokens) if isinstance(tokens, int) else "—"
        call_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('api_call_number') or '—'))}</td>"
            f"<td>{html.escape(item['call_type'])}</td>"
            f"<td>{html.escape(item['role'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(str(item.get('image_count') or 0))}</td>"
            f"<td>{html.escape(token_text)}</td>"
            "</tr>"
        )
    if not call_rows:
        call_rows.append(
            '<tr><td colspan="6" class="audit-empty">'
            "No functional discovery API-call record was persisted."
            "</td></tr>"
        )

    status_note = ""
    if audit["fail_closed"]:
        status_note = (
            "The affordance/relation calls completed, but strict validation "
            "rejected the composed discovery result. Therefore zero "
            "usable-side hypotheses and zero functional relations were "
            "forwarded to evidence acquisition or the Judge."
        )
    elif audit["status"] == "complete":
        status_note = (
            "The records below were accepted as audit-only evidence targets; "
            "they do not carry a metric verdict."
        )
    else:
        status_note = (
            "Only persisted, validated discovery records are shown. Missing "
            "records are not reconstructed by the viewer."
        )
    failed_class = (
        " functional-evidence-audit-failed"
        if audit["status"] == "failed"
        else ""
    )
    error_text = str(audit.get("error") or audit.get("reason") or "")
    return f"""
      <section class="functional-evidence-audit{failed_class}">
        <div class="functional-evidence-heading">
          <div>
            <div class="eyebrow">Pre-judgement evidence structure</div>
            <h3>Usable-side and object–group relationship audit</h3>
          </div>
          <span class="audit-only-badge">audit only · no decision authority</span>
        </div>
        <div class="functional-evidence-summary">
          <div><strong>Discovery status</strong><span>{html.escape(audit['status'])}</span></div>
          <div><strong>Discovery API calls</strong><span>{len(audit['discovery_calls'])}</span></div>
          <div><strong>Accepted usable sides</strong><span>{len(audit['surface_records'])}</span></div>
          <div><strong>Accepted relations</strong><span>{len(audit['relation_records'])}</span></div>
          <div><strong>Rejected legacy relations</strong><span>{len(audit['rejected_relation_records'])}</span></div>
          <div><strong>Grouping context</strong><span>{len(audit['grouping_relation_records'])}</span></div>
        </div>
        <div class="functional-evidence-status">
          <strong>{html.escape(status_note)}</strong>
          <p>{html.escape(error_text)}</p>
        </div>
        <details class="functional-audit-block" open>
          <summary>Object ↔ grouping scope</summary>
          <table>
            <thead><tr><th>Group</th><th>Label</th><th>Objects</th><th>Anchor</th><th>Scope / relationship basis</th></tr></thead>
            <tbody>{''.join(group_rows)}</tbody>
          </table>
        </details>
        <details class="functional-audit-block">
          <summary>Grouping contextual relations · not Functional required checks</summary>
          <p class="audit-empty">These rows explain grouping context only. They are not relation-discovery outputs, required checks, or Judge episodes.</p>
          <table>
            <thead><tr><th>Scope</th><th>Objects</th><th>Groups</th><th>Context type</th><th>Grouping rationale</th></tr></thead>
            <tbody>{''.join(grouping_relation_rows)}</tbody>
          </table>
        </details>
        <details class="functional-audit-block" open>
          <summary>Usable-side recognition</summary>
          <table>
            <thead><tr><th>Object</th><th>Group</th><th>Directionality</th><th>Requested roles</th><th>Needs clearance</th><th>Decoded side</th><th>Status</th></tr></thead>
            <tbody>{''.join(surface_rows)}</tbody>
          </table>
        </details>
        <details class="functional-audit-block" open>
          <summary>Functional relationship candidates</summary>
          <table>
            <thead><tr><th>Source</th><th>Scope</th><th>Objects</th><th>Groups</th><th>Predicate</th><th>Contract</th><th>Evidence goal</th></tr></thead>
            <tbody>{''.join(relation_rows)}</tbody>
          </table>
        </details>
        {rejected_relation_html}
        {relation_input_html}
        <details class="functional-audit-block">
          <summary>Discovery API calls and validation</summary>
          <table>
            <thead><tr><th>#</th><th>Call type</th><th>Role</th><th>Status</th><th>Images</th><th>Tokens</th></tr></thead>
            <tbody>{''.join(call_rows)}</tbody>
          </table>
          <pre>{json_block({
              "status": audit["status"],
              "reason": audit["reason"],
              "error_type": audit["error_type"],
              "error": audit["error"],
              "fail_closed": audit["fail_closed"],
              "decision_authority": audit["decision_authority"],
          })}</pre>
        </details>
      </section>
    """


def l1_calls(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    calls: list[dict[str, Any]] = []
    for metric, metric_report in metrics.items():
        if not isinstance(metric_report, dict):
            continue
        candidates: list[dict[str, Any]] = []
        for key in ("pairs", "objects"):
            values = metric_report.get(key)
            if isinstance(values, list):
                candidates.extend(
                    item for item in values if isinstance(item, dict)
                )
        for index, item in enumerate(candidates):
            judge_result = item.get("judge_result")
            if not isinstance(judge_result, dict):
                continue
            judgement = judge_result.get("judgement")
            judgement = judgement if isinstance(judgement, dict) else {}
            request = judge_result.get("request")
            request = request if isinstance(request, dict) else {}
            event = request.get("event")
            event = event if isinstance(event, dict) else {}
            members = event.get("object_ids")
            members = (
                [str(value) for value in members]
                if isinstance(members, list)
                else [
                    str(value)
                    for value in (
                        event.get("object_a"),
                        event.get("object_b"),
                    )
                    if value
                ]
            )
            images = judgement.get("images_used")
            images = (
                [str(value) for value in images]
                if isinstance(images, list)
                else []
            )
            result = compact_result(judgement or judge_result)
            metadata = judgement.get("request_metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            control_audit = (
                judge_result.get("camera_control_audit")
                or item.get("camera_control_audit")
                or {}
            )
            calls.append(
                {
                    "id": f"l1-{metric}-{index:03d}",
                    "layer": "L1",
                    "metric": str(metric),
                    "scope": " + ".join(members) or f"item {index + 1}",
                    "members": members,
                    "status": str(
                        judge_result.get("status")
                        or item.get("route")
                        or "unknown"
                    ),
                    "verdict": str(
                        judgement.get("verdict")
                        or judge_result.get("verdict")
                        or item.get("final_verdict")
                        or "unresolved"
                    ),
                    "score": judge_result.get("score"),
                    "confidence": (
                        judgement.get("confidence")
                        if judgement
                        else judge_result.get("confidence")
                    ),
                    "reason": str(
                        judgement.get("reason")
                        or judge_result.get("reason")
                        or item.get("adjudication_error")
                        or ""
                    ),
                    "images": images,
                    "request_metadata": deepcopy(metadata),
                    "usage": request_usage(metadata),
                    "prompt": str(
                        request.get("metric_rubric")
                        or "Metric rubric was not persisted."
                    ),
                    "prompt_note": (
                        "The metric rubric and structured request were "
                        "persisted; exact composed system/user messages were not."
                    ),
                    "result": result,
                    "acquisition": acquisition_timeline(
                        control_audit=control_audit,
                        fallback_images=images,
                        final_result=result,
                    ),
                    "request_context": {
                        "category": request.get("category"),
                        "metric": request.get("metric"),
                        "event": deepcopy(event),
                        "natural_language_prompt": request.get(
                            "natural_language_prompt"
                        ),
                    },
                }
            )
    return calls


def l3_calls(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    run_prompt_version = (
        str(report.get("metric_prompt_version") or "").strip()
        or None
    )
    calls: list[dict[str, Any]] = []
    for metric, metric_report in metrics.items():
        if not isinstance(metric_report, dict):
            continue
        metric_budget = metric_report.get("combined_evidence_budget")
        metric_budget = (
            metric_budget if isinstance(metric_budget, dict) else None
        )
        group_results = metric_report.get("group_results")
        group_results = (
            group_results if isinstance(group_results, list) else []
        )
        relation_results = metric_report.get(
            "cross_group_relation_results"
        )
        relation_stage_present = bool(
            "cross_group_relation_results" in metric_report
            or "cross_group_relations"
            in str(metric_report.get("route") or "")
        )
        relation_results = (
            relation_results if isinstance(relation_results, list) else []
        )
        candidates: list[
            tuple[str, list[str], dict[str, Any], str, int]
        ] = []
        global_discovery = metric_report.get("global_discovery")
        if (
            isinstance(global_discovery, dict)
            and isinstance(
                global_discovery.get("request_metadata"),
                dict,
            )
        ):
            global_verdict = str(
                global_discovery.get("verdict") or "ambiguous"
            )
            global_evaluated = (
                global_discovery.get("final_metric_verdict") is True
            )
            candidates.append(
                (
                    "scene global",
                    [],
                    {
                        "status": (
                            "evaluated"
                            if global_evaluated
                            else "unresolved"
                        ),
                        "score": (
                            1.0
                            if global_verdict == "valid"
                            else 0.0
                            if global_verdict == "invalid"
                            else None
                        ),
                        "judgement": global_discovery,
                        "evidence_paths": global_discovery.get(
                            "images_used"
                        ),
                        "camera_control_audit": metric_report.get(
                            "global_camera_control_audit"
                        ),
                    },
                    "global_discovery",
                    1,
                )
            )
        for index, relation in enumerate(relation_results):
            if not isinstance(relation, dict):
                continue
            members = relation.get("target_ids")
            members = (
                [str(value) for value in members]
                if isinstance(members, list)
                else []
            )
            candidates.append(
                (
                    str(
                        relation.get("relation_id")
                        or f"cross_group_relation_{index + 1:03d}"
                    ),
                    members,
                    relation,
                    "cross_group_relation_review",
                    2,
                )
            )
        for index, group in enumerate(group_results):
            if not isinstance(group, dict):
                continue
            members = group.get("member_ids")
            members = (
                [str(value) for value in members]
                if isinstance(members, list)
                else []
            )
            candidates.append(
                (
                    str(group.get("group_id") or f"group_{index + 1:03d}"),
                    members,
                    group,
                    "group_local_review",
                    (
                        3
                        if (
                            str(metric) == "functional_consistency"
                            and relation_stage_present
                        )
                        else 2
                    ),
                )
            )
        global_judgement = metric_report.get("judgement")
        if (
            not isinstance(global_discovery, dict)
            and
            isinstance(global_judgement, dict)
            and isinstance(global_judgement.get("request_metadata"), dict)
        ):
            candidates.insert(
                0,
                (
                    "scene",
                    [],
                    {
                        "status": metric_report.get("status"),
                        "score": metric_report.get("score"),
                        "judgement": global_judgement,
                        "evidence_paths": global_judgement.get(
                            "images_used"
                        ),
                        "camera_control_audit": metric_report.get(
                            "camera_control_audit"
                        ),
                    },
                    "scene_global",
                    1,
                ),
            )
        for index, (
            group_id,
            members,
            item,
            phase,
            workflow_step,
        ) in enumerate(candidates):
            judgement = item.get("judgement")
            judgement = judgement if isinstance(judgement, dict) else {}
            images = judgement.get("images_used")
            if not isinstance(images, list):
                images = item.get("evidence_paths")
            images = (
                [str(value) for value in images]
                if isinstance(images, list)
                else []
            )
            metadata = judgement.get("request_metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            control_audit = item.get("camera_control_audit") or {}
            verdict = str(
                judgement.get("verdict")
                or (
                    "valid"
                    if item.get("status") == "evaluated"
                    and item.get("score") == 1.0
                    else "invalid"
                    if item.get("status") == "evaluated"
                    and item.get("score") == 0.0
                    else "unresolved"
                )
            )
            packet_audit = evidence_packet_audit(
                metric=str(metric),
                phase=phase,
                images=images,
                run_prompt_version=run_prompt_version,
            )
            resolution = item.get("evidence_resolution")
            resolution = (
                resolution if isinstance(resolution, dict) else {}
            )
            resolution_budget = resolution.get("acquisition_budget")
            resolution_budget = (
                resolution_budget
                if isinstance(resolution_budget, dict)
                else None
            )
            episode = item.get("camera_acquisition_episode")
            episode = episode if isinstance(episode, dict) else None
            budget_details = (
                {
                    "metric_contract": deepcopy(metric_budget),
                    "episode": deepcopy(episode),
                    "resolution_budget": deepcopy(resolution_budget),
                    "judge_facing_image_count": len(images),
                }
                if any(
                    value is not None
                    for value in (
                        metric_budget,
                        episode,
                        resolution_budget,
                    )
                )
                else None
            )
            judge_invoked = (
                item.get("vlm_invoked") is not False
                if phase
                in {
                    "cross_group_relation_review",
                    "group_local_review",
                }
                else True
            )
            displayed_prompt_version = (
                run_prompt_version or "not persisted"
            )
            if judge_invoked:
                acquisition = acquisition_timeline(
                    control_audit=control_audit,
                    fallback_images=images,
                    final_result=(
                        compact_result(judgement)
                        or {
                            "status": item.get("status"),
                            "score": item.get("score"),
                            "reason": item.get("reason"),
                        }
                    ),
                )
            else:
                acquisition = {
                    "trace_source": "judge_not_invoked",
                    "steps": [],
                    "summary": {
                        "judge_calls": 0,
                        "judge_request_count": 0,
                        "selector_calls": 0,
                        "evidence_rounds": 0,
                        "completed_renders": 0,
                        "added_image_count": 0,
                        "packet_change_events": 0,
                        "rejudged": False,
                        "additional_evidence": False,
                        "stop_reason": str(
                            item.get("reason") or "judge_not_invoked"
                        ),
                    },
                }
            calls.append(
                {
                    "id": f"l3-{metric}-{index:03d}",
                    "layer": "L3",
                    "metric": str(metric),
                    "scope": group_id,
                    "members": members,
                    "phase": phase,
                    "workflow_step": workflow_step,
                    "status": str(item.get("status") or "unknown"),
                    "judge_invoked": judge_invoked,
                    "verdict": verdict,
                    "score": item.get("score"),
                    "confidence": judgement.get("confidence"),
                    "reason": str(
                        judgement.get("reason")
                        or item.get("reason")
                        or ""
                    ),
                    "images": images,
                    "request_metadata": deepcopy(metadata),
                    "usage": request_usage(metadata),
                    "prompt": (
                        "Prompt version persisted by this run: "
                        f"{displayed_prompt_version}\n"
                        "Current source prompt version: "
                        f"{L3_METRIC_PROMPT_VERSION}\n\n"
                        + str(
                            L3_METRIC_RUBRICS.get(
                                str(metric),
                                "Metric rubric unavailable.",
                            )
                        )
                        + "\n\nShared metric-boundary rules:\n- "
                        + "\n- ".join(L3_METRIC_BOUNDARY_RULES)
                    ),
                    "prompt_note": (
                        "The rubric text below comes from the current source. "
                        "If the persisted run version differs, use it as a "
                        "policy reference only; it is not the exact historical "
                        "composed message. Exact system/user messages were not "
                        "persisted."
                    ),
                    "evidence_packet_audit": packet_audit,
                    "result": compact_result(judgement)
                    or {
                        "status": item.get("status"),
                        "score": item.get("score"),
                        "reason": item.get("reason"),
                    },
                    "routing_details": (
                        {
                            "vlm_invoked": item.get("vlm_invoked"),
                            "evidence_resolution": deepcopy(
                                resolution
                            ),
                            "camera_acquisition_episode": deepcopy(
                                episode
                            ),
                            "camera_target_ids": deepcopy(
                                item.get("camera_target_ids")
                            ),
                            "relation_id": item.get("relation_id"),
                            "judge_episode": item.get("judge_episode"),
                            "pair_specific_evidence_available": item.get(
                                "pair_specific_evidence_available"
                            ),
                            "acquisition_status": item.get(
                                "acquisition_status"
                            ),
                            "acquisition_error": deepcopy(
                                item.get("acquisition_error")
                            ),
                            "target_ids": deepcopy(
                                item.get("target_ids")
                            ),
                            "group_ids": deepcopy(
                                item.get("group_ids")
                            ),
                            "observation_kinds": deepcopy(
                                item.get("observation_kinds")
                            ),
                            "relation_predicates": deepcopy(
                                item.get("relation_predicates")
                            ),
                            "observation_goals": deepcopy(
                                item.get("observation_goals")
                            ),
                            "routed_candidate_claims": deepcopy(
                                item.get("routed_candidate_claims")
                            ),
                            "claim_correspondence": deepcopy(
                                item.get("claim_correspondence")
                            ),
                            "scene_claim_correspondence": deepcopy(
                                item.get("scene_claim_correspondence")
                            ),
                            "functional_probe_evidence": deepcopy(
                                item.get("functional_probe_evidence")
                            ),
                            "placement_discovery": deepcopy(
                                item.get("placement_discovery")
                            ),
                        }
                        if phase
                        in {
                            "cross_group_relation_review",
                            "group_local_review",
                        }
                        else None
                    ),
                    "budget_details": budget_details,
                    "acquisition": acquisition,
                }
            )
    return calls


def l3_pipeline_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize persisted routing without inventing missing Judge calls."""

    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    summaries: list[dict[str, Any]] = []
    for metric, metric_report in metrics.items():
        if not isinstance(metric_report, dict):
            continue
        global_record = metric_report.get("global_discovery")
        global_record = (
            global_record if isinstance(global_record, dict) else {}
        )
        groups = metric_report.get("group_results")
        groups = (
            [item for item in groups if isinstance(item, dict)]
            if isinstance(groups, list)
            else []
        )
        relations = metric_report.get("cross_group_relation_results")
        relations = (
            [item for item in relations if isinstance(item, dict)]
            if isinstance(relations, list)
            else []
        )
        group_filter = metric_report.get("group_filter")
        group_filter = (
            group_filter if isinstance(group_filter, dict) else {}
        )
        skipped = group_filter.get("skipped_groups")
        skipped = skipped if isinstance(skipped, list) else []
        resolved = sum(
            1
            for item in groups
            if item.get("status") == "evaluated"
        )
        unresolved = sum(
            1
            for item in groups
            if item.get("status") == "unresolved"
        )
        invoked = sum(
            1
            for item in groups
            if item.get("vlm_invoked") is True
        )
        resolved_relations = sum(
            1
            for item in relations
            if item.get("status") == "evaluated"
        )
        unresolved_relations = sum(
            1
            for item in relations
            if item.get("status") != "evaluated"
        )
        route = str(metric_report.get("route") or "scene_global")
        relation_stage_present = bool(
            "cross_group_relation_results" in metric_report
            or "cross_group_relations" in route
        )
        discovery: dict[str, Any] | None = None
        if metric == "functional_consistency":
            functional = metric_report.get(
                "functional_prejudgement_evidence"
            )
            functional = functional if isinstance(functional, dict) else {}
            runtime_audit = functional.get("runtime_audit")
            runtime_audit = (
                runtime_audit if isinstance(runtime_audit, dict) else {}
            )
            budget = functional.get("budget_usage")
            budget = budget if isinstance(budget, dict) else {}
            discovery = {
                "label": "Usable-side / relation evidence discovery",
                "status": str(
                    functional.get("status")
                    or runtime_audit.get("status")
                    or "not persisted"
                ),
                "reason": str(
                    runtime_audit.get("reason")
                    or runtime_audit.get("error")
                    or ""
                ),
                "planned": budget.get("scheduled_probe_count"),
                "rendered": len(
                    functional.get("selected_judge_probe_paths") or []
                ),
                "budget": budget.get("max_probe_units"),
                "details": deepcopy(functional),
            }
        elif metric == "semantic_placement_consistency":
            placement = metric_report.get("placement_discovery")
            placement = placement if isinstance(placement, dict) else {}
            candidates = placement.get("candidates")
            candidates = candidates if isinstance(candidates, list) else []
            discovery = {
                "label": "Placement observation discovery",
                "status": (
                    "complete" if placement else "not persisted"
                ),
                "reason": str(placement.get("reason") or ""),
                "planned": len(candidates),
                "rendered": None,
                "budget": None,
                "details": deepcopy(placement),
            }
        summaries.append(
            {
                "metric": str(metric),
                "route": route,
                "status": str(metric_report.get("status") or "unknown"),
                "score": metric_report.get("score"),
                "reason": str(metric_report.get("reason") or ""),
                "global_status": str(
                    global_record.get("global_status")
                    or global_record.get("evidence_status")
                    or (
                        "recorded"
                        if global_record
                        else "not applicable"
                    )
                ),
                "global_verdict": str(
                    global_record.get("verdict")
                    or (
                        metric_report.get("judgement") or {}
                    ).get("verdict")
                    or "—"
                ),
                "eligible_groups": len(groups),
                "resolved_groups": resolved,
                "unresolved_groups": unresolved,
                "invoked_groups": invoked,
                "scheduled_relations": len(relations),
                "invoked_relation_episodes": sum(
                    1
                    for relation in relations
                    if relation.get("vlm_invoked") is not False
                ),
                "skipped_relation_episodes": sum(
                    1
                    for relation in relations
                    if relation.get("vlm_invoked") is False
                ),
                "resolved_relations": resolved_relations,
                "unresolved_relations": unresolved_relations,
                "relation_stage_present": relation_stage_present,
                "skipped_groups": len(skipped),
                "discovery": discovery,
                "budget": metric_budget_summary(metric_report),
                "check_chain": metric_check_chain_summary(
                    str(metric),
                    metric_report,
                ),
            }
        )
    return summaries


def metric_check_chain_summary(
    metric: str,
    metric_report: dict[str, Any],
) -> dict[str, Any] | None:
    """Expose persisted typed obligations without interpreting the verdict."""

    if metric == "functional_consistency":
        ledger_key = "functional_check_ledger"
        coverage_key = "functional_check_coverage"
        label = "Functional typed-check chain"
    elif metric == "semantic_placement_consistency":
        ledger_key = "placement_check_ledger"
        coverage_key = "placement_check_coverage"
        label = "Placement typed-check chain"
    else:
        return None
    ledger = metric_report.get(ledger_key)
    if not isinstance(ledger, dict):
        return None
    raw_checks = ledger.get("checks")
    raw_checks = raw_checks if isinstance(raw_checks, list) else []
    checks: list[dict[str, Any]] = []
    for raw in raw_checks:
        if not isinstance(raw, dict):
            continue
        checks.append(
            {
                "check_id": str(raw.get("check_id") or ""),
                "check_type": str(
                    raw.get("check_type")
                    or raw.get("predicate")
                    or "unknown"
                ),
                "owner_stage": str(
                    raw.get("owner_stage") or "not persisted"
                ),
                "owning_group_id": (
                    str(raw.get("owning_group_id"))
                    if raw.get("owning_group_id")
                    else None
                ),
                "subject_id": (
                    str(raw.get("subject_id"))
                    if raw.get("subject_id")
                    else None
                ),
                "target_ids": [
                    str(value)
                    for value in raw.get("target_ids") or []
                ],
                "context_ids": [
                    str(value)
                    for value in raw.get("context_ids") or []
                ],
                "origin": str(raw.get("origin") or "discovery"),
                "lifecycle_status": str(
                    raw.get("lifecycle_status") or "unknown"
                ),
                "observation_status": (
                    str(raw.get("observation_status"))
                    if raw.get("observation_status")
                    else None
                ),
                "conclusion": (
                    str(raw.get("check_conclusion"))
                    if raw.get("check_conclusion")
                    else None
                ),
                "judge_result_ref": (
                    str(raw.get("judge_result_ref"))
                    if raw.get("judge_result_ref")
                    else None
                ),
                "function_event_ref": (
                    str(raw.get("function_event_ref"))
                    if raw.get("function_event_ref")
                    else None
                ),
            }
        )
    ownership = metric_report.get("functional_ownership_ledger")
    ownership = ownership if isinstance(ownership, dict) else {}
    events = [
        deepcopy(item)
        for item in ownership.get("events") or []
        if isinstance(item, dict)
    ]
    coverage = metric_report.get(coverage_key)
    coverage = coverage if isinstance(coverage, dict) else {}
    cross_metric = metric_report.get("cross_metric_ownership_audit")
    cross_metric = (
        cross_metric if isinstance(cross_metric, dict) else None
    )
    return {
        "label": label,
        "ledger_key": ledger_key,
        "checks": checks,
        "coverage": deepcopy(coverage),
        "ownership_events": events,
        "cross_metric_ownership_audit": deepcopy(cross_metric),
        "raw_ledger": deepcopy(ledger),
    }


def render_metric_check_chain(
    value: dict[str, Any] | None,
) -> str:
    if not isinstance(value, dict):
        return ""
    checks = value.get("checks")
    checks = checks if isinstance(checks, list) else []
    coverage = value.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    events = value.get("ownership_events")
    events = events if isinstance(events, list) else []
    rows: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        subject = (
            str(check.get("subject_id"))
            if check.get("subject_id")
            else ", ".join(check.get("target_ids") or [])
        )
        context = ", ".join(check.get("context_ids") or []) or "—"
        route = str(check.get("owner_stage") or "unknown")
        if check.get("owning_group_id"):
            route += f" · {check['owning_group_id']}"
        conclusion = str(
            check.get("conclusion")
            or check.get("lifecycle_status")
            or "pending"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(check.get('check_type') or ''))}</code>"
            f"<small>{html.escape(str(check.get('check_id') or ''))}</small></td>"
            f"<td>{html.escape(subject or '—')}</td>"
            f"<td>{html.escape(context)}</td>"
            f"<td>{html.escape(route)}</td>"
            f"<td><strong>{html.escape(conclusion)}</strong>"
            f"<small>{html.escape(str(check.get('observation_status') or ''))}</small></td>"
            "</tr>"
        )
    if not rows:
        rows.append(
            '<tr><td colspan="5" class="muted">No typed check was persisted.</td></tr>'
        )
    resolved = coverage.get("resolved_check_count")
    required = coverage.get("required_check_count")
    coverage_text = (
        f"{resolved}/{required} resolved"
        if isinstance(resolved, int) and isinstance(required, int)
        else f"{len(checks)} persisted check(s)"
    )
    status = (
        "complete"
        if coverage.get("complete") is True
        else "incomplete"
        if coverage
        else "coverage not persisted"
    )
    event_rows = []
    for event in events:
        event_rows.append(
            "<li>"
            f"<code>{html.escape(str(event.get('event_id') or ''))}</code>"
            " · affected "
            f"{html.escape(', '.join(str(item) for item in event.get('affected_object_ids') or []) or '—')}"
            " · causal "
            f"{html.escape(', '.join(str(item) for item in event.get('causal_object_ids') or []) or '—')}"
            " · scored to "
            f"{html.escape(', '.join(str(item) for item in event.get('scoring_target_ids') or []) or '—')}"
            "</li>"
        )
    ownership_html = (
        "<details><summary>"
        f"Functional causal ownership · {len(events)} event(s)"
        "</summary><ul class=\"ownership-events\">"
        + "".join(event_rows)
        + "</ul></details>"
        if events
        else ""
    )
    return f"""
      <section class="metric-check-chain">
        <div class="metric-check-chain-heading">
          <div>
            <strong>{html.escape(str(value.get('label') or 'Typed-check chain'))}</strong>
            <span>{html.escape(coverage_text)} · {html.escape(status)}</span>
          </div>
          <span class="audit-only-badge">routing → evidence → Judge → result</span>
        </div>
        <div class="metric-check-table-wrap">
          <table class="metric-check-table">
            <thead>
              <tr><th>Check</th><th>Subject / targets</th><th>Context</th><th>Route</th><th>Result</th></tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        {ownership_html}
        <details>
          <summary>Typed ledger and cross-metric ownership audit</summary>
          <pre>{json_block({
              'ledger': value.get('raw_ledger') or {},
              'coverage': coverage,
              'cross_metric_ownership_audit': (
                  value.get('cross_metric_ownership_audit')
              ),
          })}</pre>
        </details>
      </section>
    """


def metric_budget_summary(
    metric_report: dict[str, Any],
) -> dict[str, Any] | None:
    budget = metric_report.get("combined_evidence_budget")
    if not isinstance(budget, dict):
        return None
    ledger = budget.get("camera_acquisition_ledger")
    ledger = ledger if isinstance(ledger, dict) else {}
    accounting = str(budget.get("accounting") or "not_persisted")
    aggregate_is_authority = budget.get(
        "metric_aggregate_is_budget_authority"
    )
    if not isinstance(aggregate_is_authority, bool):
        aggregate_is_authority = (
            accounting == "existing_shared_metric_camera_ledger"
        )
    return {
        "accounting": accounting,
        "scope": str(
            budget.get("budget_enforcement_scope")
            or (
                "metric"
                if aggregate_is_authority
                else "not_persisted"
            )
        ),
        "max_images_per_judge_episode": budget.get(
            "max_images_per_judge_episode"
        ),
        "metric_artifact_count": ledger.get("total_images_acquired"),
        "metric_aggregate_is_budget_authority": aggregate_is_authority,
    }


def render_l3_pipeline_summary(report: dict[str, Any]) -> str:
    summaries = l3_pipeline_summary(report)
    if not summaries:
        return ""
    rows: list[str] = []
    for item in summaries:
        route = str(item["route"])
        has_local = "group" in route
        local_text = (
            f"{item['resolved_groups']} resolved · "
            f"{item['unresolved_groups']} unresolved · "
            f"{item['skipped_groups']} skipped"
            if has_local
            else "not required"
        )
        discovery = item.get("discovery")
        discovery_html = ""
        if isinstance(discovery, dict):
            counts = []
            if isinstance(discovery.get("planned"), int):
                counts.append(f"{discovery['planned']} candidate(s)")
            if isinstance(discovery.get("rendered"), int):
                counts.append(f"{discovery['rendered']} rendered")
            if isinstance(discovery.get("budget"), int):
                counts.append(f"budget {discovery['budget']}")
            count_text = " · ".join(counts) or "no count persisted"
            discovery_html = f"""
              <div class="pipeline-discovery">
                <strong>{html.escape(str(discovery['label']))}</strong>
                <span class="pipeline-status">{html.escape(str(discovery['status']))}</span>
                <span>{html.escape(count_text)}</span>
                <p>{html.escape(str(discovery.get('reason') or ''))}</p>
                <details>
                  <summary>Discovery record</summary>
                  <pre>{json_block(discovery.get("details") or {})}</pre>
                </details>
              </div>
            """
        budget = item.get("budget")
        budget_html = ""
        if isinstance(budget, dict):
            max_images = budget.get("max_images_per_judge_episode")
            metric_artifacts = budget.get("metric_artifact_count")
            aggregate_is_authority = (
                budget.get("metric_aggregate_is_budget_authority") is True
            )
            if aggregate_is_authority:
                budget_label = "Historical metric-wide budget"
                budget_note = (
                    f"{metric_artifacts} acquired artifact(s)"
                    if isinstance(metric_artifacts, int)
                    else "artifact count not persisted"
                )
            else:
                budget_label = "Per-Judge episode budget"
                budget_note_parts = []
                if isinstance(max_images, int):
                    budget_note_parts.append(
                        f"up to {max_images} Judge-facing image(s)"
                    )
                if isinstance(metric_artifacts, int):
                    budget_note_parts.append(
                        f"{metric_artifacts} metric artifact(s), audit only"
                    )
                budget_note = (
                    " · ".join(budget_note_parts)
                    or "counts not persisted"
                )
            budget_html = f"""
              <div class="pipeline-budget">
                <strong>{html.escape(budget_label)}</strong>
                <span>{html.escape(budget_note)}</span>
              </div>
            """
        check_chain_html = render_metric_check_chain(
            item.get("check_chain")
        )
        score = item.get("score")
        score_text = (
            f"{float(score):.2f}"
            if isinstance(score, (int, float))
            else "—"
        )
        has_relation_stage = (
            str(item["metric"]) == "functional_consistency"
            and item.get("relation_stage_present") is True
        )
        relation_stage_html = (
            f"""
                <span class="pipeline-arrow">→</span>
                <div>
                  <strong>2 · Cross-group relations</strong>
                  <span>
                    {int(item['resolved_relations'])} resolved ·
                    {int(item['unresolved_relations'])} unresolved
                  </span>
                  <small>
                    {int(item['scheduled_relations'])} relation obligation(s) ·
                    {int(item.get('invoked_relation_episodes') or 0)} isolated Judge episode(s) ·
                    {int(item.get('skipped_relation_episodes') or 0)} not started
                  </small>
                </div>
            """
            if has_relation_stage
            else ""
        )
        local_step = 3 if has_relation_stage else 2
        rows.append(
            f"""
            <article class="pipeline-metric">
              <div class="pipeline-metric-heading">
                <div>
                  <strong>{html.escape(str(item['metric']))}</strong>
                  <code>{html.escape(route)}</code>
                </div>
                <span class="pipeline-final pipeline-final-{html.escape(str(item['status']))}">
                  {html.escape(str(item['status']))} · score {score_text}
                </span>
              </div>
              {discovery_html}
              {budget_html}
              {check_chain_html}
              <div class="pipeline-stages{' pipeline-stages-functional' if has_relation_stage else ''}">
                <div>
                  <strong>1 · Global</strong>
                  <span>{html.escape(str(item['global_verdict']))}</span>
                  <small>{html.escape(str(item['global_status']))}</small>
                </div>
                {relation_stage_html}
                <span class="pipeline-arrow">→</span>
                <div>
                  <strong>{local_step} · Group-local</strong>
                  <span>{html.escape(local_text)}</span>
                  <small>{int(item['invoked_groups'])} Judge call scope(s)</small>
                </div>
              </div>
              <p class="pipeline-reason">{html.escape(str(item['reason']))}</p>
            </article>
            """
        )
    return f"""
      <section class="pipeline-summary">
        <div class="pipeline-summary-heading">
          <div>
            <div class="eyebrow">Persisted workflow</div>
            <h3>L3 global → local routing</h3>
          </div>
          <p>
            Includes unresolved group scopes where the Judge was never invoked.
          </p>
        </div>
        <div class="pipeline-grid">{''.join(rows)}</div>
      </section>
    """


def render_audit_graphs(case_dir: Path) -> str:
    """Render optional graph artifacts without treating them as decisions."""

    graph_root = case_dir / "audit_graphs"
    manifest = optional_json(graph_root / "manifest.json")
    if not manifest:
        return ""
    status = str(manifest.get("status") or "unknown")
    if status != "complete":
        return f"""
          <section class="audit-graphs audit-graphs-failed">
            <div class="eyebrow">Optional post-hoc projection</div>
            <h3>Evaluation audit graphs</h3>
            <p>
              Export {html.escape(status)} ·
              {html.escape(str(manifest.get("error") or "no graph emitted"))}
            </p>
            <p class="muted">
              This export failure does not change the metric result.
            </p>
          </section>
        """

    relation_record = manifest.get("relation_candidate_graph")
    relation_record = (
        relation_record if isinstance(relation_record, dict) else {}
    )
    relation_graph = _graph_json(
        graph_root,
        relation_record.get("path"),
    )
    candidates = relation_graph.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    source_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        scope = str(candidate.get("scope") or "unknown")
        state = str(candidate.get("state") or "candidate")
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        state_counts[state] = state_counts.get(state, 0) + 1
        for source in candidate.get("sources") or []:
            if not isinstance(source, dict):
                continue
            kind = str(source.get("source_kind") or "unknown")
            source_counts[kind] = source_counts.get(kind, 0) + 1

    query_records = manifest.get("evaluation_query_graphs")
    query_records = (
        query_records if isinstance(query_records, list) else []
    )
    metric_counts: dict[str, int] = {}
    node_count = 0
    edge_count = 0
    typed_check_count = 0
    check_result_count = 0
    ownership_event_count = 0
    for record in query_records:
        if not isinstance(record, dict):
            continue
        metric = str(record.get("metric") or "unknown")
        metric_counts[metric] = metric_counts.get(metric, 0) + 1
        node_count += int(record.get("node_count") or 0)
        edge_count += int(record.get("edge_count") or 0)
        node_kinds = record.get("node_kind_counts")
        node_kinds = node_kinds if isinstance(node_kinds, dict) else {}
        typed_check_count += int(node_kinds.get("typed_check") or 0)
        check_result_count += int(node_kinds.get("check_result") or 0)
        ownership_event_count += int(
            node_kinds.get("ownership_event") or 0
        )

    def counts(value: dict[str, int]) -> str:
        return " · ".join(
            f"{key} {value[key]}" for key in sorted(value)
        ) or "none"

    return f"""
      <section class="audit-graphs">
        <div class="audit-graphs-heading">
          <div>
            <div class="eyebrow">Optional post-hoc projection</div>
            <h3>Evaluation audit graphs</h3>
          </div>
          <span class="audit-only-badge">audit only · no decision authority</span>
        </div>
        <div class="audit-graph-stats">
          <div><strong>{len(candidates)}</strong><span>relation candidates</span></div>
          <div><strong>{len(query_records)}</strong><span>query graphs</span></div>
          <div><strong>{node_count}</strong><span>typed nodes</span></div>
          <div><strong>{edge_count}</strong><span>typed edges</span></div>
          <div><strong>{typed_check_count}</strong><span>typed checks</span></div>
          <div><strong>{check_result_count}</strong><span>check results</span></div>
          <div><strong>{ownership_event_count}</strong><span>ownership events</span></div>
        </div>
        <div class="audit-graph-breakdown">
          <p><strong>Candidate sources</strong> {html.escape(counts(source_counts))}</p>
          <p><strong>Scopes</strong> {html.escape(counts(scope_counts))}</p>
          <p><strong>Lifecycle</strong> {html.escape(counts(state_counts))}</p>
          <p><strong>Query metrics</strong> {html.escape(counts(metric_counts))}</p>
        </div>
        <details>
          <summary>Graph export manifest</summary>
          <pre>{json_block(manifest)}</pre>
        </details>
      </section>
    """


def _graph_json(root: Path, relative_path: Any) -> dict[str, Any]:
    if not isinstance(relative_path, str) or not relative_path.strip():
        return {}
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return {}
    return optional_json(candidate)


def render_phase_routes(calls: list[dict[str, Any]]) -> str:
    routes: list[str] = []
    metrics = list(
        dict.fromkeys(str(call.get("metric") or "") for call in calls)
    )
    for metric in metrics:
        metric_calls = [
            call for call in calls if str(call.get("metric") or "") == metric
        ]
        global_calls = [
            call
            for call in metric_calls
            if call.get("phase") == "global_discovery"
        ]
        local_calls = [
            call
            for call in metric_calls
            if call.get("phase") == "group_local_review"
        ]
        relation_calls = [
            call
            for call in metric_calls
            if call.get("phase") == "cross_group_relation_review"
        ]
        if not global_calls or not local_calls:
            continue
        has_relation_stage = bool(
            metric == "functional_consistency"
            and (
                relation_calls
                or any(
                    call.get("workflow_step") == 3
                    for call in local_calls
                )
            )
        )
        relation_route = (
            f"""
              <span class="phase-arrow">→</span>
              <span class="phase-node phase-node-relation">
                2 · Cross-group relation review ·
                {len(relation_calls)} relation(s)
              </span>
            """
            if has_relation_stage
            else ""
        )
        local_step = 3 if has_relation_stage else 2
        routes.append(
            f"""
            <div class="phase-route">
              <strong>{html.escape(metric)}</strong>
              <span class="phase-node phase-node-global">
                1 · Global discovery
              </span>
              {relation_route}
              <span class="phase-arrow">→</span>
              <span class="phase-node phase-node-local">
                {local_step} · Group-local review · {len(local_calls)} group(s)
              </span>
            </div>
            """
        )
    if not routes:
        return ""
    return (
        '<section class="phase-routes">'
        '<div class="eyebrow">L3 evidence order</div>'
        + "".join(routes)
        + "</section>"
    )


def aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for role in ("judge", "camera_selector"):
        role_calls = [
            call
            for call in calls
            if call.get("judge_invoked") is not False
            if (
                role == "judge"
                and call.get("layer") in {"L1", "L3"}
            )
            or (
                role == "camera_selector"
                and str(
                    (call.get("request_metadata") or {}).get(
                        "call_type"
                    )
                ).startswith("camera_selector_")
            )
        ]
        usage_calls = [
            call["usage"]
            for call in role_calls
            if isinstance(call.get("usage"), dict)
        ]
        totals: dict[str, int] = {}
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_prompt_tokens",
            "reasoning_tokens",
        ):
            values = [
                usage[field]
                for usage in usage_calls
                if isinstance(usage.get(field), int)
            ]
            if values:
                totals[field] = sum(values)
        by_role[role] = {
            "api_calls_number": len(role_calls),
            "calls_with_response_usage": len(usage_calls),
            "tokens_usage": totals or None,
        }
    return {
        "by_role": by_role,
        "source": (
            "request_metadata.usage persisted in grouping/L1/L3 reports"
        ),
        "coverage_note": (
            "This run predates runner-level API accounting. Failed calls "
            "without persisted response metadata may be absent."
        ),
    }


def prefer_runner_usage(
    summary: dict[str, Any],
    reconstructed_usage: dict[str, Any],
) -> dict[str, Any]:
    """Use complete runner accounting when the run persisted it."""
    api_usage = summary.get("api_usage")
    api_usage = api_usage if isinstance(api_usage, dict) else {}
    by_role = api_usage.get("by_role")
    by_role = by_role if isinstance(by_role, dict) else {}
    if not all(
        isinstance(by_role.get(role), dict)
        and isinstance(by_role[role].get("api_calls_number"), int)
        for role in ("judge", "camera_selector")
    ):
        return reconstructed_usage
    return {
        "by_role": {
            role: deepcopy(by_role[role])
            for role in ("judge", "camera_selector")
        },
        "source": "summary.api_usage.by_role",
        "coverage_note": (
            "Runner-level API accounting includes successful and failed "
            "logical calls, including selector calls without decision cards."
        ),
    }


def relative_url(path: Path, serve_root: Path) -> str | None:
    try:
        relative = path.expanduser().resolve().relative_to(
            serve_root.expanduser().resolve()
        )
    except ValueError:
        return None
    return "/" + "/".join(quote(part) for part in relative.parts)


def render_image(
    path_value: str,
    resolver: EvidenceURLResolver,
) -> str:
    path = Path(path_value).expanduser()
    escaped_path = html.escape(str(path))
    if not path.is_file():
        return (
            '<div class="missing-image"><strong>Missing image</strong>'
            f"<code>{escaped_path}</code></div>"
        )
    url = resolver.url_for(path)
    if url is None:
        return (
            '<div class="missing-image"><strong>Outside server root</strong>'
            f"<code>{escaped_path}</code></div>"
        )
    label = html.escape(path.name)
    return f"""
      <figure>
        <a href="{url}" target="_blank" rel="noreferrer">
          <img src="{url}" alt="{label}" loading="lazy">
        </a>
        <figcaption>
          <strong>{label}</strong>
          <code>{escaped_path}</code>
        </figcaption>
      </figure>
    """


def json_block(value: Any) -> str:
    return html.escape(
        json.dumps(value, indent=2, ensure_ascii=False)
    )


def render_timeline_thumbnail(
    path_value: str,
    resolver: EvidenceURLResolver,
) -> str:
    path = Path(path_value).expanduser()
    escaped_path = html.escape(str(path))
    if not path.is_file():
        return (
            '<div class="timeline-image-missing">'
            f"<strong>Missing</strong><code>{escaped_path}</code></div>"
        )
    url = resolver.url_for(path)
    if url is None:
        return (
            '<div class="timeline-image-missing">'
            f"<strong>Outside server root</strong><code>{escaped_path}</code></div>"
        )
    label = html.escape(path.name)
    return f"""
      <a class="timeline-image" href="{url}" target="_blank" rel="noreferrer"
        title="{escaped_path}">
        <img src="{url}" alt="{label}" loading="lazy">
        <span>{label}</span>
      </a>
    """


def render_acquisition_timeline(
    acquisition: Any,
    resolver: EvidenceURLResolver,
) -> str:
    acquisition = acquisition if isinstance(acquisition, dict) else {}
    summary = acquisition.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    steps = acquisition.get("steps")
    steps = steps if isinstance(steps, list) else []
    additional = summary.get("additional_evidence") is True
    trace_source = str(acquisition.get("trace_source") or "unavailable")
    flow_class = "flow-extra" if additional else "flow-direct"
    flow_label = "Additional evidence acquired" if additional else "Direct decision"
    open_attribute = " open" if additional else ""
    step_rows: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        stage = html.escape(str(step.get("stage") or "event"))
        label = html.escape(str(step.get("label") or "Event"))
        status = html.escape(str(step.get("status") or "recorded"))
        evidence_round = step.get("evidence_round")
        round_text = (
            str(evidence_round)
            if isinstance(evidence_round, int)
            else "—"
        )
        description = html.escape(str(step.get("summary") or ""))
        new_images = step.get("new_images")
        new_images = new_images if isinstance(new_images, list) else []
        new_evidence = ""
        if new_images:
            thumbnails = "".join(
                render_timeline_thumbnail(str(path), resolver)
                for path in new_images
            )
            new_evidence = f"""
              <div class="new-evidence">
                <strong>New visual evidence at this step · {len(new_images)}</strong>
                <div class="timeline-images">{thumbnails}</div>
              </div>
            """
        reconstructed = (
            '<span class="trace-note">reconstructed</span>'
            if step.get("reconstructed") is True
            else ""
        )
        step_rows.append(
            f"""
            <li class="timeline-step timeline-{stage}">
              <div class="timeline-marker"></div>
              <div class="timeline-content">
                <div class="timeline-heading">
                  <div>
                    <span class="round-label">Round {html.escape(round_text)}</span>
                    <strong>{label}</strong>
                    {reconstructed}
                  </div>
                  <span class="timeline-status">{status}</span>
                </div>
                <p>{description}</p>
                {new_evidence}
                <details class="step-record">
                  <summary>Step record</summary>
                  <pre>{json_block(step.get("details") or {})}</pre>
                </details>
              </div>
            </li>
            """
        )
    if not step_rows:
        step_rows.append(
            '<li class="timeline-empty">No acquisition trace was persisted.</li>'
        )
    return f"""
      <details class="evidence-flow {flow_class}"{open_attribute}>
        <summary>
          <span>
            <strong>Evidence flow</strong>
            <span class="flow-label">{flow_label}</span>
          </span>
          <span class="flow-stats">
            {int(summary.get("judge_calls") or 0)} Judge call(s)
            · {int(summary.get("judge_request_count") or 0)} request(s)
            · {int(summary.get("evidence_rounds") or 0)} extra round(s)
            · {int(summary.get("added_image_count") or 0)} new image(s)
          </span>
        </summary>
        <div class="flow-body">
          <p class="trace-source">
            Trace source: <code>{html.escape(trace_source)}</code>.
            Stop reason: <code>{html.escape(str(summary.get("stop_reason") or "not persisted"))}</code>.
          </p>
          <ol class="timeline">{''.join(step_rows)}</ol>
        </div>
      </details>
    """


def acquisition_overview(calls: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        call
        for call in calls
        if call.get("judge_invoked") is not False
    ]
    decision_rows: list[dict[str, Any]] = []
    totals = {
        "decisions": len(calls),
        "traced_decisions": 0,
        "decisions_with_additional_evidence": 0,
        "judge_requests": 0,
        "evidence_rounds": 0,
        "new_images": 0,
        "rejudged_decisions": 0,
    }
    for call in calls:
        acquisition = call.get("acquisition")
        acquisition = acquisition if isinstance(acquisition, dict) else {}
        summary = acquisition.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        if acquisition.get("trace_source") != "reconstructed":
            totals["traced_decisions"] += 1
        if summary.get("additional_evidence") is True:
            totals["decisions_with_additional_evidence"] += 1
            decision_rows.append(
                {
                    "case_id": call.get("case_id"),
                    "layer": call.get("layer"),
                    "metric": call.get("metric"),
                    "scope": call.get("scope"),
                    **deepcopy(summary),
                }
            )
        totals["judge_requests"] += int(
            summary.get("judge_request_count") or 0
        )
        totals["evidence_rounds"] += int(summary.get("evidence_rounds") or 0)
        totals["new_images"] += int(summary.get("added_image_count") or 0)
        if summary.get("rejudged") is True:
            totals["rejudged_decisions"] += 1
    return {"totals": totals, "decision_rows": decision_rows}


def render_acquisition_overview(overview: dict[str, Any]) -> str:
    totals = overview.get("totals")
    totals = totals if isinstance(totals, dict) else {}
    rows = overview.get("decision_rows")
    rows = rows if isinstance(rows, list) else []
    if rows:
        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('case_id') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('layer') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('metric') or '—'))}</td>"
            f"<td>{html.escape(str(item.get('scope') or '—'))}</td>"
            f"<td>{int(item.get('judge_request_count') or 0)}</td>"
            f"<td>{int(item.get('evidence_rounds') or 0)}</td>"
            f"<td>{int(item.get('added_image_count') or 0)}</td>"
            f"<td>{'yes' if item.get('rejudged') is True else 'no'}</td>"
            "</tr>"
            for item in rows
            if isinstance(item, dict)
        )
        detail = f"""
          <table>
            <thead>
              <tr>
                <th>Scene</th><th>Layer</th><th>Metric</th><th>Scope</th>
                <th>Requests</th><th>Extra rounds</th><th>New images</th><th>Re-judged</th>
              </tr>
            </thead>
            <tbody>{body}</tbody>
          </table>
        """
    else:
        detail = """
          <p class="no-acquisition">
            No Judge requested or acquired additional visual evidence in this
            run. Every persisted decision was made from its initial packet.
          </p>
        """
    return f"""
      <section class="acquisition-overview">
        <div class="acquisition-heading">
          <div>
            <div class="eyebrow">Control-loop observability</div>
            <h2>Visual evidence acquisition</h2>
          </div>
          <p>
            Initial evidence is round 0. Every later Judge request, camera
            selection, render, and re-judgement is tracked per decision below.
          </p>
        </div>
        <div class="acquisition-summary">
          <div><strong>Additional-evidence decisions</strong><span>{int(totals.get("decisions_with_additional_evidence") or 0)}</span></div>
          <div><strong>Judge requests</strong><span>{int(totals.get("judge_requests") or 0)}</span></div>
          <div><strong>Extra rounds</strong><span>{int(totals.get("evidence_rounds") or 0)}</span></div>
          <div><strong>New evidence images</strong><span>{int(totals.get("new_images") or 0)}</span></div>
          <div><strong>Re-judged decisions</strong><span>{int(totals.get("rejudged_decisions") or 0)}</span></div>
          <div><strong>Trace coverage</strong><span>{int(totals.get("traced_decisions") or 0)}/{int(totals.get("decisions") or 0)}</span></div>
        </div>
        {detail}
      </section>
    """


def render_evidence_packet_audit(
    packet: dict[str, Any] | None,
) -> str:
    if not isinstance(packet, dict):
        return ""
    status = str(packet.get("status") or "unversioned")
    labels = {
        "current_default": "Matches current default",
        "current_run_custom_or_mismatch": "Current run: custom or mismatch",
        "historical": "Historical packet · preserved unchanged",
        "unversioned": "Run policy version not persisted",
    }
    actual_roles = " + ".join(
        str(role).replace("_", " ")
        for role in packet.get("actual_roles") or []
    ) or "no persisted images"
    expected_roles = " + ".join(
        str(role).replace("_", " ")
        for role in packet.get("expected_current_default_roles") or []
    )
    version_note = ""
    if status == "historical":
        version_note = (
            "This decision predates the current packet contract. The UI "
            "continues to display every image actually received by the VLM."
        )
    elif status == "current_run_custom_or_mismatch":
        version_note = (
            "The persisted packet differs from the current default; inspect "
            "the exact image paths below before treating this as an error."
        )
    return f"""
      <div class="packet-audit packet-{html.escape(status)}">
        <div>
          <strong>Persisted visual input</strong>
          <span>{int(packet.get("actual_image_count") or 0)} image(s) · {html.escape(actual_roles)}</span>
        </div>
        <div>
          <strong>Current default for this phase</strong>
          <span>{int(packet.get("expected_current_default_image_count") or 0)} image(s) · {html.escape(expected_roles)}</span>
        </div>
        <div class="packet-status">
          <strong>{html.escape(labels.get(status, status))}</strong>
          <span>{html.escape(version_note)}</span>
        </div>
      </div>
    """


def render_judge_episode_budget(call: dict[str, Any]) -> str:
    details = call.get("budget_details")
    if not isinstance(details, dict):
        return ""
    contract = details.get("metric_contract")
    contract = contract if isinstance(contract, dict) else {}
    episode = details.get("episode")
    episode = episode if isinstance(episode, dict) else {}
    resolution = details.get("resolution_budget")
    resolution = resolution if isinstance(resolution, dict) else {}
    metric_ledger = contract.get("camera_acquisition_ledger")
    metric_ledger = (
        metric_ledger if isinstance(metric_ledger, dict) else {}
    )
    before = episode.get("ledger_before_judge")
    before = before if isinstance(before, dict) else {}
    after = episode.get("ledger_after_judge")
    after = after if isinstance(after, dict) else {}

    accounting = str(contract.get("accounting") or "")
    aggregate_authority = contract.get(
        "metric_aggregate_is_budget_authority"
    )
    if not isinstance(aggregate_authority, bool):
        aggregate_authority = (
            accounting == "existing_shared_metric_camera_ledger"
        )
    max_images = resolution.get("max_total_images")
    if not isinstance(max_images, int):
        max_images = contract.get("max_images_per_judge_episode")
    initial_count = resolution.get("initial_judge_evidence_count")
    if not isinstance(initial_count, int):
        initial_count = before.get("total_images_acquired")
    if not isinstance(initial_count, int):
        initial_count = details.get("judge_facing_image_count")
    after_count = after.get("total_images_acquired")
    metric_count = resolution.get("metric_artifact_count_after")
    if not isinstance(metric_count, int):
        metric_count = metric_ledger.get("total_images_acquired")

    if aggregate_authority:
        heading = "Historical metric-wide evidence budget"
        scope_text = "shared across the metric"
        authority_note = "The metric aggregate was the budget authority."
    else:
        heading = "Judge episode evidence budget"
        scope_text = str(
            resolution.get("scope")
            or episode.get("scope")
            or contract.get("budget_enforcement_scope")
            or "judge_episode"
        ).replace("_", " ")
        authority_note = (
            "Metric artifact count is telemetry only; it does not block the "
            "next group."
        )

    def count_text(value: Any, *, fallback: str = "—") -> str:
        return str(value) if isinstance(value, int) else fallback

    initial_text = count_text(initial_count)
    if isinstance(max_images, int):
        initial_text += f" / {max_images}"
    after_text = count_text(after_count, fallback=initial_text)
    if isinstance(max_images, int):
        after_text = after_text.split(" / ", 1)[0] + f" / {max_images}"
    metric_text = count_text(metric_count)
    return f"""
      <div class="judge-budget">
        <div>
          <strong>{html.escape(heading)}</strong>
          <span>{html.escape(scope_text)}</span>
        </div>
        <div>
          <strong>Initial Judge packet</strong>
          <span>{html.escape(initial_text)}</span>
        </div>
        <div>
          <strong>After evidence loop</strong>
          <span>{html.escape(after_text)}</span>
        </div>
        <div>
          <strong>Metric artifacts</strong>
          <span>{html.escape(metric_text)}</span>
        </div>
        <p>{html.escape(authority_note)}</p>
      </div>
    """


def render_call(
    call: dict[str, Any],
    resolver: EvidenceURLResolver,
) -> str:
    layer = html.escape(str(call["layer"]))
    metric = html.escape(str(call["metric"]))
    scope = html.escape(str(call["scope"]))
    phase = str(call.get("phase") or "")
    phase_labels = {
        "global_discovery": "Global discovery",
        "cross_group_relation_review": "Cross-group relation review",
        "group_local_review": "Group-local review",
        "scene_global": "Scene-global judgement",
    }
    phase_label = phase_labels.get(
        phase,
        phase.replace("_", " ").title() if phase else "",
    )
    workflow_step = call.get("workflow_step")
    phase_text = (
        (
            f"Step {workflow_step} · {phase_label}"
            if isinstance(workflow_step, int)
            else phase_label
        )
        if phase_label
        else ""
    )
    phase_class = (
        phase
        if phase in {
            "global_discovery",
            "cross_group_relation_review",
            "group_local_review",
            "scene_global",
        }
        else "unspecified"
    )
    verdict = html.escape(str(call["verdict"]).lower())
    status = html.escape(str(call["status"]))
    reason = html.escape(str(call.get("reason") or "No reason persisted."))
    confidence = call.get("confidence")
    confidence_text = (
        f"{float(confidence):.2f}"
        if isinstance(confidence, (int, float))
        else "—"
    )
    score = call.get("score")
    score_text = (
        f"{float(score):.2f}"
        if isinstance(score, (int, float))
        else "—"
    )
    members = call.get("members")
    members_text = ", ".join(str(value) for value in members or []) or "scene"
    metadata = call.get("request_metadata")
    call_type = (
        str(metadata.get("call_type") or "not persisted")
        if isinstance(metadata, dict)
        else "not persisted"
    )
    acquisition = call.get("acquisition")
    acquisition = acquisition if isinstance(acquisition, dict) else {}
    acquisition_summary = acquisition.get("summary")
    acquisition_summary = (
        acquisition_summary if isinstance(acquisition_summary, dict) else {}
    )
    judge_invoked = call.get("judge_invoked") is not False
    if not judge_invoked:
        acquisition_kind = "not-invoked"
        acquisition_badge = "Judge not invoked"
    else:
        acquisition_kind = (
            "extra"
            if acquisition_summary.get("additional_evidence") is True
            else "direct"
            if acquisition.get("trace_source") != "reconstructed"
            else "untraced"
        )
        acquisition_badge = (
            f"+{int(acquisition_summary.get('added_image_count') or 0)} evidence"
            if acquisition_kind == "extra"
            else "direct evidence"
            if acquisition_kind == "direct"
            else "trace reconstructed"
        )
    packet_audit = render_evidence_packet_audit(
        call.get("evidence_packet_audit")
    )
    judge_budget = render_judge_episode_budget(call)
    images = "".join(
        render_image(path, resolver)
        for path in call.get("images") or []
    )
    if not images:
        images = (
            '<p class="empty">No image path was persisted for this result.</p>'
        )
    context = call.get("request_context")
    context_details = (
        f"""
        <details>
          <summary>Structured request context</summary>
          <pre>{json_block(context)}</pre>
        </details>
        """
        if isinstance(context, dict)
        else ""
    )
    routing_details = call.get("routing_details")
    routing_details_html = (
        f"""
        <details>
          <summary>Routing / unresolved scope</summary>
          <pre>{json_block(routing_details)}</pre>
        </details>
        """
        if isinstance(routing_details, dict)
        else ""
    )
    acquisition_html = (
        render_acquisition_timeline(acquisition, resolver)
        if judge_invoked
        else f"""
          <div class="judge-not-invoked">
            <strong>Judge was not invoked for this required scope.</strong>
            <span>{html.escape(str(call.get("reason") or "No reason persisted."))}</span>
          </div>
        """
    )
    search = html.escape(
        " ".join(
            [
                str(call["layer"]),
                str(call["metric"]),
                str(call["scope"]),
                members_text,
                str(call["verdict"]),
                str(call.get("reason") or ""),
            ]
        ).lower()
    )
    return f"""
      <article class="call call-phase-{phase_class}"
        data-layer="{layer.lower()}"
        data-metric="{metric}"
        data-phase="{html.escape(phase)}"
        data-acquisition="{acquisition_kind}"
        data-search="{search}">
        <header class="call-header">
          <div>
            <div class="eyebrow">
              {layer} · {metric}
              {f'<span class="phase-pill">{html.escape(phase_text)}</span>' if phase_text else ''}
            </div>
            <h3>{scope}</h3>
            <p class="members">{html.escape(members_text)}</p>
          </div>
          <div class="decision">
            <span class="verdict verdict-{verdict}">{verdict}</span>
            <span>status {status}</span>
            <span class="acquisition-badge acquisition-{acquisition_kind}">{html.escape(acquisition_badge)}</span>
          </div>
        </header>
        <div class="result-grid">
          <div><strong>Confidence</strong><span>{confidence_text}</span></div>
          <div><strong>Score</strong><span>{score_text}</span></div>
          <div><strong>Call type</strong><span>{html.escape(call_type)}</span></div>
        </div>
        <p class="reason">{reason}</p>
        {packet_audit}
        {judge_budget}
        {acquisition_html}
        <div class="image-grid">{images}</div>
        <div class="details-row">
          <details>
            <summary>Prompt / rubric</summary>
            <p class="note">{html.escape(str(call["prompt_note"]))}</p>
            <pre>{html.escape(str(call["prompt"]))}</pre>
          </details>
          <details>
            <summary>Parsed result</summary>
            <pre>{json_block(call.get("result") or {})}</pre>
          </details>
          <details>
            <summary>Request metadata</summary>
            <pre>{json_block(metadata or {})}</pre>
          </details>
          {routing_details_html}
          {context_details}
        </div>
      </article>
    """


def render_object_level_findings(report: dict[str, Any]) -> str:
    summaries = object_level_finding_summary(report)
    if not summaries:
        return ""
    rows: list[str] = []
    source_values: set[str] = set()
    total_penalties = 0
    for summary in summaries:
        metric = str(summary.get("metric") or "")
        source = str(summary.get("source") or "unknown")
        source_values.add(source)
        findings = summary.get("findings")
        findings = findings if isinstance(findings, list) else []
        total_penalties += len(findings)
        if not findings:
            rows.append(
                "<tr>"
                f"<td>{html.escape(metric)}</td>"
                '<td colspan="4" class="empty-finding">'
                "No invalid object finding in this metric."
                "</td>"
                "</tr>"
            )
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            phases = finding.get("source_phases")
            phases = phases if isinstance(phases, list) else []
            observations = finding.get("observations")
            observations = (
                observations if isinstance(observations, list) else []
            )
            observation_count = int(
                finding.get("observation_count")
                or len(observations)
            )
            merged_count = int(
                finding.get("merged_duplicate_observation_count")
                or max(0, observation_count - 1)
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(metric)}</td>"
                f"<td><code>{html.escape(str(finding.get('object_id') or 'unknown'))}</code></td>"
                f"<td>{html.escape(', '.join(str(value) for value in phases) or '—')}</td>"
                f"<td>{observation_count}</td>"
                f"<td>{merged_count}</td>"
                "</tr>"
            )
    reconstructed = (
        "viewer_reconstructed_from_persisted_defects" in source_values
    )
    source_note = (
        "This historical report predates runner-level object findings. The "
        "viewer reconstructed the table only from persisted invalid defects; "
        "older target_ids may also name relation context, so these rows are "
        "audit-only and do not retroactively change scoring. The original "
        "judgements and evidence remain unchanged."
        if reconstructed
        else "These object findings were persisted by the runner."
    )
    return f"""
      <section class="object-findings">
        <div class="object-findings-heading">
          <div>
            <div class="eyebrow">Deterministic attribution</div>
            <h3>Functional / placement object findings</h3>
          </div>
          <div class="object-findings-count">
            <strong>{total_penalties}</strong>
            <span>metric-object penalty unit(s)</span>
          </div>
        </div>
        <p>
          {html.escape(source_note)} Global and local observations of the same
          object merge only within one metric; no deduplication crosses metrics.
        </p>
        <table>
          <thead>
            <tr>
              <th>Metric</th><th>Object</th><th>Observed in</th>
              <th>Obs.</th><th>Merged</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </section>
    """


def comparison_rows(comparison: dict[str, Any]) -> str:
    metrics = comparison.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    rows: list[str] = []
    for metric, item in metrics.items():
        if not isinstance(item, dict):
            continue
        human = item.get("human")
        human = human if isinstance(human, dict) else {}
        model = item.get("model")
        model = model if isinstance(model, dict) else {}
        matches = item.get("matches")
        match_text = (
            "yes" if matches is True else "no" if matches is False else "—"
        )
        anomaly_level = item.get("anomaly_level")
        anomaly_level = (
            anomaly_level if isinstance(anomaly_level, dict) else {}
        )
        object_match = anomaly_level.get("exact_match")
        object_match_text = (
            "yes"
            if object_match is True
            else "no"
            if object_match is False
            else "—"
        )
        human_objects = ", ".join(
            str(value)
            for value in anomaly_level.get("human_object_ids") or []
        ) or "—"
        model_objects = ", ".join(
            str(value)
            for value in anomaly_level.get("model_object_ids") or []
        ) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metric))}</td>"
            f"<td>{html.escape(str(human.get('expected') or '—'))}</td>"
            f"<td>{html.escape(str(model.get('prediction') or '—'))}</td>"
            f"<td>{match_text}</td>"
            f"<td>{html.escape(human_objects)}</td>"
            f"<td>{html.escape(model_objects)}</td>"
            f"<td>{object_match_text}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_viewer(
    run_root: Path,
    *,
    serve_root: Path = PROJECT_ROOT,
    bundle_dir: Path | None = None,
) -> Path:
    run_root = run_root.expanduser().resolve()
    serve_root = serve_root.expanduser().resolve()
    run_manifest = read_json(run_root / "run_manifest.json")
    summary = optional_json(run_root / "summary.json")
    resolver = EvidenceURLResolver(
        serve_root=serve_root,
        bundle_dir=bundle_dir,
    )
    cases_root = run_root / "cases"
    case_dirs = (
        sorted(path for path in cases_root.iterdir() if path.is_dir())
        if cases_root.is_dir()
        else []
    )
    all_sections: list[str] = []
    all_calls: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for case_index, case_dir in enumerate(case_dirs):
        case_manifest = optional_json(case_dir / "case_run_manifest.json")
        grouping = optional_json(case_dir / "grouping.json")
        l1_report = optional_json(case_dir / "l1_report.json")
        l3_report = optional_json(case_dir / "scene_quality_report.json")
        comparison = optional_json(case_dir / "scene_comparison.json")
        api_calls = optional_jsonl(case_dir / "api_calls.jsonl")
        source_paths = source_evidence_paths(case_manifest)
        blender_command = render_blender_command(
            case_id=case_dir.name,
            case_manifest=case_manifest,
        )
        grouping_output = render_grouping_output(
            case_id=case_dir.name,
            grouping=grouping,
            scene=source_scene(case_manifest),
            top_image_path=(
                source_paths[1] if len(source_paths) >= 2 else None
            ),
            resolver=resolver,
        )
        functional_evidence = render_functional_evidence_audit(
            grouping=grouping,
            report=l3_report,
            api_calls=api_calls,
            resolver=resolver,
        )
        calls: list[dict[str, Any]] = []
        calls.extend(l1_calls(l1_report))
        calls.extend(l3_calls(l3_report))
        for call in calls:
            call["case_id"] = case_dir.name
        all_calls.extend(calls)
        metric_names.update(str(call["metric"]) for call in calls)
        cards = "".join(render_call(call, resolver) for call in calls)
        pipeline_summary = render_l3_pipeline_summary(l3_report)
        audit_graphs = render_audit_graphs(case_dir)
        phase_routes = render_phase_routes(calls)
        object_findings = render_object_level_findings(l3_report)
        if not cards:
            cards = '<p class="empty">No persisted VLM records found.</p>'
        initially_hidden = " hidden" if case_index else ""
        all_sections.append(
            f"""
            <div class="scene-page"
              id="{html.escape(case_dir.name)}"
              data-scene="{html.escape(case_dir.name)}"{initially_hidden}>
              {blender_command}
              {grouping_output}
              {functional_evidence}
              <section class="scene">
                <div class="scene-title">
                  <div>
                    <div class="eyebrow">Scene</div>
                    <h2>{html.escape(case_dir.name)}</h2>
                  </div>
                  <div class="scene-status">
                    <span>run {html.escape(str(case_manifest.get('status') or 'unknown'))}</span>
                    <span>L1 {html.escape(str(case_manifest.get('l1_status') or 'unknown'))}</span>
                    <span>L3 {html.escape(str(case_manifest.get('l3_status') or 'unknown'))}</span>
                    <span>final {html.escape(str(case_manifest.get('final_decision_status') or 'unknown'))}</span>
                  </div>
                </div>
                <details class="comparison">
                  <summary>Human comparison</summary>
                  <table>
                    <thead><tr><th>Metric</th><th>Human</th><th>Model</th><th>Scene match</th><th>Human anomaly objects</th><th>Model anomaly objects</th><th>Object-set match</th></tr></thead>
                    <tbody>{comparison_rows(comparison)}</tbody>
                  </table>
                </details>
                {object_findings}
                {audit_graphs}
                {pipeline_summary}
                {phase_routes}
                <div class="calls">{cards}</div>
              </section>
            </div>
            """
        )
    usage = prefer_runner_usage(summary, aggregate_usage(all_calls))
    acquisition_overview_html = render_acquisition_overview(
        acquisition_overview(all_calls)
    )
    judge_usage = usage["by_role"]["judge"]
    selector_usage = usage["by_role"]["camera_selector"]
    judge_tokens = judge_usage.get("tokens_usage")
    judge_tokens = (
        judge_tokens if isinstance(judge_tokens, dict) else {}
    )
    selector_tokens = selector_usage.get("tokens_usage")
    selector_tokens = (
        selector_tokens if isinstance(selector_tokens, dict) else {}
    )
    metric_options = "".join(
        f'<option value="{html.escape(metric)}">{html.escape(metric)}</option>'
        for metric in sorted(metric_names)
    )
    scene_buttons = "".join(
        (
            '<button type="button" class="scene-button" '
            f'data-scene-target="{html.escape(case_dir.name)}">'
            f"{html.escape(case_dir.name)}</button>"
        )
        for case_dir in case_dirs
    )
    scene_pages_html = "".join(all_sections)
    if not scene_pages_html:
        scene_pages_html = """
          <section class="empty-run">
            <h2>No scene report is available yet</h2>
            <p>
              This run may still be starting. Refresh this page after the
              runner creates a case directory; the local UI runner will
              rebuild the viewer from the latest persisted reports.
            </p>
          </section>
        """
    evidence_integrity_note = (
        "The dedicated local bundle contains byte-for-byte copies of the "
        "persisted evidence images. Every copy is SHA-256 verified, and no "
        "source evidence file was modified."
        if resolver.bundle_dir is not None
        else (
            "Images are loaded directly from their persisted source paths; "
            "this viewer does not copy, annotate, resize, or overwrite them."
        )
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM evidence audit · {html.escape(run_root.name)}</title>
  <style>
    :root {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2328;
      background: #f6f7f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .page {{ width: min(1440px, calc(100% - 32px)); margin: 28px auto 72px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 5px; font-size: 27px; }}
    h2 {{ margin-bottom: 0; font-size: 23px; }}
    h3 {{ margin: 4px 0 5px; font-size: 18px; }}
    .muted, .note, .members {{ color: #59636e; }}
    .eyebrow {{
      color: #59636e; font-size: 11px; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase;
    }}
    .summary {{
      display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr));
      margin: 20px 0; border: 1px solid #d0d7de; background: white;
    }}
    .summary div {{ padding: 12px; border-right: 1px solid #d8dee4; }}
    .summary div:last-child {{ border-right: 0; }}
    .summary strong, .summary span {{ display: block; }}
    .summary strong {{ color: #59636e; font-size: 11px; text-transform: uppercase; }}
    .summary span {{ margin-top: 4px; font-size: 17px; font-weight: 650; }}
    .notice {{
      padding: 12px 14px; border: 1px solid #d0d7de;
      border-left: 4px solid #57606a; background: white; line-height: 1.45;
    }}
    .empty-run {{
      margin-top: 20px; padding: 28px; border: 1px dashed #aeb7c0;
      background: white; text-align: center;
    }}
    .empty-run p {{
      max-width: 680px; margin: 9px auto 0; color: #59636e;
      line-height: 1.5;
    }}
    .acquisition-overview {{
      margin: 18px 0 22px; padding: 18px; border: 1px solid #aeb7c0;
      border-top: 3px solid #0969da; background: white;
    }}
    .acquisition-heading {{
      display: flex; justify-content: space-between; align-items: start; gap: 24px;
    }}
    .acquisition-heading h2 {{ margin: 4px 0 0; }}
    .acquisition-heading p {{
      max-width: 640px; margin: 0; color: #59636e; line-height: 1.45;
    }}
    .acquisition-summary {{
      display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr));
      margin-top: 16px; border: 1px solid #d8dee4; background: #fbfcfd;
    }}
    .acquisition-summary div {{
      min-width: 0; padding: 10px; border-right: 1px solid #d8dee4;
    }}
    .acquisition-summary div:last-child {{ border-right: 0; }}
    .acquisition-summary strong, .acquisition-summary span {{ display: block; }}
    .acquisition-summary strong {{
      min-height: 28px; color: #59636e; font-size: 10px;
      line-height: 1.35; text-transform: uppercase;
    }}
    .acquisition-summary span {{ margin-top: 3px; font-size: 18px; font-weight: 700; }}
    .acquisition-overview table {{ margin-top: 14px; }}
    .no-acquisition {{
      margin: 14px 0 0; padding: 11px 12px; border: 1px solid #d8dee4;
      border-left: 4px solid #1a7f37; background: #f6f8fa; line-height: 1.45;
    }}
    .grouping-output {{
      margin: 18px 0 22px; padding: 18px; border: 1px solid #aeb7c0;
      border-top: 3px solid #24292f; background: white;
    }}
    .grouping-heading {{
      display: flex; justify-content: space-between; align-items: start; gap: 24px;
    }}
    .grouping-heading h2 {{ margin: 4px 0 7px; }}
    .grouping-heading p {{ margin-bottom: 0; max-width: 900px; line-height: 1.45; }}
    .grouping-meta {{ display: flex; gap: 7px; flex-wrap: wrap; justify-content: end; }}
    .grouping-meta span {{
      padding: 5px 8px; border: 1px solid #d0d7de; border-radius: 3px;
      background: #f6f8fa; color: #59636e; font-size: 12px;
    }}
    .grouping-meta strong {{ color: #1f2328; }}
    .grouping-visual {{
      display: grid; grid-template-columns: minmax(0, 2fr) minmax(250px, 1fr);
      gap: 16px; align-items: start; margin-top: 16px;
    }}
    .topdown-view {{ position: relative; min-width: 0; }}
    .topdown-view > img {{
      display: block; width: 100%; height: auto; border: 1px solid #aeb7c0;
    }}
    .room-overlay {{
      position: absolute; left: 6.38%; top: 14.19%; width: 87.11%; height: 71.61%;
      pointer-events: none;
    }}
    .group-region {{
      position: absolute; border: 3px solid var(--group-color); border-radius: 5px;
      box-shadow: inset 0 0 0 999px color-mix(in srgb, var(--group-color) 12%, transparent);
    }}
    .group-region span {{
      position: absolute; left: -3px; top: -24px; padding: 3px 6px;
      border-radius: 3px 3px 0 0; background: var(--group-color); color: white;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px; font-weight: 750; white-space: nowrap;
    }}
    .topdown-caption {{ margin: 7px 0 0; color: #59636e; font-size: 11px; }}
    .group-legend {{
      padding: 4px 0; border-top: 1px solid #d8dee4;
    }}
    .group-legend-row {{
      display: grid; grid-template-columns: 13px 1fr; gap: 8px;
      padding: 9px 2px; border-bottom: 1px solid #d8dee4;
    }}
    .group-swatch {{
      width: 11px; height: 11px; margin-top: 2px; border-radius: 2px;
      background: var(--group-color);
    }}
    .group-legend-row strong, .group-legend-row small {{ display: block; }}
    .group-legend-row strong {{ font-size: 12px; }}
    .group-legend-row small {{
      margin-top: 3px; color: #59636e; font-size: 10px; line-height: 1.35;
    }}
    .group-grid {{
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; margin-top: 16px;
    }}
    .group-card {{
      padding: 13px; border: 1px solid #d0d7de;
      border-left: 4px solid var(--group-color); background: #fbfcfd;
    }}
    .group-card-title span, .group-card-title strong {{ display: block; }}
    .group-card-title span {{
      color: #59636e; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
    }}
    .group-card-title strong {{ margin-top: 3px; font-size: 15px; }}
    .group-members {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 11px 0; }}
    .group-members code {{
      padding: 3px 6px; border: 1px solid #d8dee4; border-radius: 3px;
      background: white; font-size: 11px;
    }}
    .group-card p {{ margin: 7px 0 0; font-size: 12px; line-height: 1.45; }}
    .group-reason {{ color: #47515c; }}
    .group-relations {{ margin-top: 11px; }}
    .functional-evidence-audit {{
      margin: 18px 0 22px; padding: 18px; border: 1px solid #aeb7c0;
      border-top: 3px solid #0969da; background: white;
    }}
    .functional-evidence-audit-failed {{ border-top-color: #d1242f; }}
    .functional-evidence-heading {{
      display: flex; justify-content: space-between; align-items: start;
      gap: 18px;
    }}
    .functional-evidence-heading h3 {{ margin: 4px 0 0; }}
    .functional-evidence-summary {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-top: 14px; border: 1px solid #d8dee4; background: #fbfcfd;
    }}
    .functional-evidence-summary div {{
      min-width: 0; padding: 10px; border-right: 1px solid #d8dee4;
    }}
    .functional-evidence-summary div:last-child {{ border-right: 0; }}
    .functional-evidence-summary strong,
    .functional-evidence-summary span {{ display: block; }}
    .functional-evidence-summary strong {{
      min-height: 27px; color: #59636e; font-size: 10px;
      line-height: 1.35; text-transform: uppercase;
    }}
    .functional-evidence-summary span {{
      margin-top: 3px; font-size: 18px; font-weight: 750;
    }}
    .functional-evidence-status {{
      margin-top: 11px; padding: 10px 12px; border-left: 4px solid #0969da;
      background: #ddf4ff; line-height: 1.45;
    }}
    .functional-evidence-audit-failed .functional-evidence-status {{
      border-left-color: #d1242f; background: #ffebe9;
    }}
    .functional-evidence-status strong {{ font-size: 12px; }}
    .functional-evidence-status p {{
      margin: 5px 0 0; color: #47515c; font-size: 11px;
    }}
    .functional-audit-block {{
      margin-top: 11px; border: 1px solid #d8dee4; background: #fbfcfd;
    }}
    .functional-audit-block > summary {{
      padding: 9px 11px; background: #f6f8fa; font-size: 12px;
      font-weight: 700; cursor: pointer;
    }}
    .functional-audit-block table {{
      margin: 0; border: 0; border-top: 1px solid #d8dee4;
      table-layout: fixed;
    }}
    .functional-audit-block th,
    .functional-audit-block td {{
      padding: 7px 8px; vertical-align: top; overflow-wrap: anywhere;
      font-size: 10px; line-height: 1.4;
    }}
    .functional-audit-block pre {{
      margin: 10px; max-height: 280px; overflow: auto;
    }}
    .relation-input-images {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px; padding: 10px;
    }}
    .relation-input-visual > strong {{
      display: block; margin-bottom: 6px; font-size: 10px;
      text-transform: uppercase; color: #59636e;
    }}
    .relation-input-visual figure {{ margin: 0; }}
    .audit-empty {{
      padding: 12px !important; color: #59636e; text-align: center;
      font-style: italic;
    }}
    .toolbar {{
      display: flex; gap: 10px;
      padding: 12px 0; background: #f6f7f8;
    }}
    .scene-controls {{
      position: sticky; top: 0; z-index: 5; margin-top: 18px;
      border-bottom: 1px solid #d0d7de; background: #f6f7f8;
    }}
    .scene-nav {{
      display: flex; align-items: center; gap: 7px; padding-top: 10px;
    }}
    .scene-buttons {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .scene-nav button {{
      min-height: 34px; padding: 6px 10px; border: 1px solid #afb8c1;
      border-radius: 4px; background: white; color: #1f2328;
      font: inherit; font-size: 12px; font-weight: 650; cursor: pointer;
    }}
    .scene-nav button:hover {{ background: #f0f2f4; }}
    .scene-nav button:disabled {{ color: #8c959f; cursor: default; }}
    .scene-button[aria-current="page"] {{
      border-color: #0969da; background: #0969da; color: white;
    }}
    .scene-counter {{
      margin-left: auto; color: #59636e; font-size: 12px;
      white-space: nowrap;
    }}
    .blender-launch {{
      margin: 18px 0 0; padding: 12px 14px; border: 1px solid #b6c2cf;
      border-left: 4px solid #0969da; background: white;
    }}
    .blender-launch-heading {{
      display: flex; justify-content: space-between; align-items: center;
      gap: 16px;
    }}
    .blender-launch-heading strong {{ display: block; margin-top: 2px; }}
    .blender-command {{
      margin-top: 9px; padding: 9px 11px; overflow-x: auto;
      border: 1px solid #d8dee4; border-radius: 4px; background: #f6f8fa;
      white-space: nowrap;
    }}
    .blender-command code {{ font-size: 12px; }}
    .blender-command-unavailable {{ color: #59636e; font-size: 12px; }}
    .copy-blender-command {{
      min-height: 32px; padding: 5px 9px; border: 1px solid #afb8c1;
      border-radius: 4px; background: white; color: #1f2328;
      font: inherit; font-size: 12px; font-weight: 650; cursor: pointer;
    }}
    .copy-blender-command:hover {{ background: #f0f2f4; }}
    .toolbar input, .toolbar select {{
      min-height: 38px; padding: 7px 10px; border: 1px solid #afb8c1;
      border-radius: 4px; background: white; color: inherit;
    }}
    .toolbar input {{ flex: 1; }}
    .scene {{ margin-top: 24px; scroll-margin-top: 200px; }}
    .scene-page[hidden] {{ display: none; }}
    .scene-title {{
      display: flex; justify-content: space-between; align-items: end;
      padding-bottom: 10px; border-bottom: 2px solid #24292f;
    }}
    .scene-status {{ display: flex; gap: 7px; flex-wrap: wrap; }}
    .scene-status span {{
      padding: 4px 7px; border: 1px solid #d0d7de; border-radius: 3px;
      background: white; font-size: 12px;
    }}
    .comparison {{ margin: 12px 0; }}
    .object-findings {{
      margin: 12px 0; padding: 14px; border: 1px solid #b6c2cf;
      border-left: 4px solid #1a7f37; background: white;
    }}
    .object-findings-heading {{
      display: flex; justify-content: space-between; gap: 18px;
      align-items: start;
    }}
    .object-findings-heading h3 {{ margin: 3px 0 0; }}
    .object-findings > p {{
      margin: 10px 0 12px; color: #47515c; line-height: 1.45;
    }}
    .object-findings-count {{ text-align: right; }}
    .object-findings-count strong, .object-findings-count span {{
      display: block;
    }}
    .object-findings-count strong {{ font-size: 22px; }}
    .object-findings-count span {{ color: #59636e; font-size: 10px; }}
    .object-findings table {{ table-layout: fixed; }}
    .object-findings th:nth-child(1) {{ width: 24%; }}
    .object-findings th:nth-child(2) {{ width: 19%; }}
    .object-findings th:nth-child(3) {{ width: 31%; }}
    .object-findings th:nth-child(4),
    .object-findings th:nth-child(5) {{ width: 13%; }}
    .object-findings td {{
      vertical-align: top; overflow-wrap: anywhere; font-size: 11px;
    }}
    .empty-finding {{ color: #59636e; }}
    .pipeline-summary {{
      margin: 12px 0; padding: 14px; border: 1px solid #b6c2cf;
      border-left: 4px solid #57606a; background: white;
    }}
    .pipeline-summary-heading {{
      display: flex; justify-content: space-between; gap: 18px;
      align-items: start;
    }}
    .pipeline-summary-heading h3 {{ margin: 3px 0 0; }}
    .pipeline-summary-heading p {{
      max-width: 480px; margin: 0; color: #59636e; font-size: 12px;
    }}
    .pipeline-grid {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 9px; margin-top: 12px;
    }}
    .pipeline-metric {{
      padding: 11px; border: 1px solid #d8dee4; background: #fbfcfd;
    }}
    .pipeline-metric-heading {{
      display: flex; justify-content: space-between; align-items: start;
      gap: 12px;
    }}
    .pipeline-metric-heading strong,
    .pipeline-metric-heading code {{ display: block; }}
    .pipeline-metric-heading code {{
      margin-top: 3px; color: #59636e; font-size: 10px;
    }}
    .pipeline-final {{
      padding: 3px 6px; border: 1px solid #d0d7de; border-radius: 3px;
      background: white; font-size: 10px; font-weight: 700;
      white-space: nowrap;
    }}
    .pipeline-final-evaluated {{ color: #1a7f37; }}
    .pipeline-final-unresolved {{ color: #9a6700; }}
    .pipeline-discovery {{
      margin-top: 9px; padding: 8px; border-left: 3px solid #bf8700;
      background: #fff8c5; font-size: 11px;
    }}
    .pipeline-discovery > strong,
    .pipeline-discovery > span {{ display: block; }}
    .pipeline-discovery .pipeline-status {{
      float: right; margin-top: -15px; font-weight: 700;
    }}
    .pipeline-discovery p {{
      clear: both; margin: 6px 0; color: #47515c; line-height: 1.4;
    }}
    .pipeline-budget {{
      display: flex; justify-content: space-between; gap: 10px;
      margin-top: 9px; padding: 7px 8px; border: 1px solid #d8dee4;
      background: white; font-size: 10px;
    }}
    .pipeline-budget span {{ color: #59636e; text-align: right; }}
    .metric-check-chain {{
      margin-top: 9px; padding: 8px; border: 1px solid #b6c2cf;
      border-left: 3px solid #0969da; background: white;
    }}
    .metric-check-chain-heading {{
      display: flex; justify-content: space-between; gap: 10px;
      align-items: start; margin-bottom: 7px;
    }}
    .metric-check-chain-heading strong,
    .metric-check-chain-heading span {{ display: block; }}
    .metric-check-chain-heading span {{
      margin-top: 2px; color: #59636e; font-size: 10px;
    }}
    .metric-check-chain-heading .audit-only-badge {{
      margin-top: 0; color: #0550ae;
    }}
    .metric-check-table-wrap {{ overflow-x: auto; }}
    .metric-check-table {{ font-size: 10px; }}
    .metric-check-table th,
    .metric-check-table td {{ padding: 5px 6px; vertical-align: top; }}
    .metric-check-table code,
    .metric-check-table small {{ display: block; }}
    .metric-check-table small {{
      max-width: 190px; margin-top: 2px; color: #59636e;
      overflow-wrap: anywhere;
    }}
    .ownership-events {{ margin: 5px 0 0; padding-left: 18px; }}
    .ownership-events li {{ margin: 4px 0; font-size: 10px; }}
    .pipeline-stages {{
      display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 7px; align-items: center; margin-top: 9px;
    }}
    .pipeline-stages-functional {{
      grid-template-columns:
        minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
    }}
    .pipeline-stages > div {{
      min-height: 58px; padding: 7px; border: 1px solid #d0d7de;
      background: white;
    }}
    .pipeline-stages strong,
    .pipeline-stages span,
    .pipeline-stages small {{ display: block; }}
    .pipeline-stages span {{ margin-top: 3px; font-size: 11px; }}
    .pipeline-stages small {{ margin-top: 2px; color: #59636e; }}
    .pipeline-arrow {{ color: #59636e; font-weight: 800; }}
    .pipeline-reason {{
      margin: 8px 0 0; color: #59636e; font-size: 11px;
    }}
    .audit-graphs {{
      margin: 12px 0; padding: 14px; border: 1px solid #b6c2cf;
      border-left: 4px solid #0969da; background: white;
    }}
    .audit-graphs-failed {{ border-left-color: #bf8700; }}
    .audit-graphs h3 {{ margin: 3px 0 0; }}
    .audit-graphs-heading {{
      display: flex; justify-content: space-between; gap: 18px;
      align-items: start;
    }}
    .audit-only-badge {{
      padding: 4px 7px; border: 1px solid #54aeff; border-radius: 3px;
      color: #0550ae; background: #ddf4ff; font-size: 10px;
      font-weight: 700;
    }}
    .audit-graph-stats {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px; margin-top: 12px;
    }}
    .audit-graph-stats > div {{
      padding: 9px; border: 1px solid #d8dee4; background: #fbfcfd;
    }}
    .audit-graph-stats strong,
    .audit-graph-stats span {{ display: block; }}
    .audit-graph-stats strong {{ font-size: 20px; }}
    .audit-graph-stats span {{ color: #59636e; font-size: 10px; }}
    .audit-graph-breakdown {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 5px 16px; margin-top: 9px;
    }}
    .audit-graph-breakdown p {{ margin: 0; font-size: 11px; }}
    .phase-routes {{
      margin: 12px 0; padding: 12px 14px; border: 1px solid #b6c2cf;
      border-left: 4px solid #8250df; background: white;
    }}
    .phase-route {{
      display: flex; align-items: center; gap: 9px; flex-wrap: wrap;
      margin-top: 8px;
    }}
    .phase-route > strong {{ min-width: 230px; font-size: 12px; }}
    .phase-node {{
      padding: 5px 8px; border: 1px solid #d0d7de; border-radius: 3px;
      font-size: 11px; font-weight: 700;
    }}
    .phase-node-global {{ background: #f4edff; color: #6639ba; }}
    .phase-node-relation {{ background: #fff8c5; color: #7d4e00; }}
    .phase-node-local {{ background: #ddf4ff; color: #0550ae; }}
    .phase-arrow {{ color: #59636e; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 8px 10px; border: 1px solid #d8dee4; text-align: left; }}
    th {{ background: #f0f2f4; }}
    .call {{
      margin: 12px 0; padding: 16px; border: 1px solid #d0d7de;
      border-radius: 5px; background: white;
    }}
    .call-phase-global_discovery {{ border-left: 4px solid #8250df; }}
    .call-phase-cross_group_relation_review {{
      border-left: 4px solid #bf8700;
    }}
    .call-phase-group_local_review {{ border-left: 4px solid #0969da; }}
    .phase-pill {{
      display: inline-block; margin-left: 7px; padding: 2px 5px;
      border: 1px solid #d0d7de; border-radius: 3px; background: #f6f8fa;
      color: #47515c; font-size: 9px; letter-spacing: .04em;
    }}
    .call[hidden] {{ display: none; }}
    .call-header {{ display: flex; justify-content: space-between; gap: 20px; }}
    .decision {{ text-align: right; font-size: 12px; color: #59636e; }}
    .decision span {{ display: block; }}
    .acquisition-badge {{
      display: inline-block !important; margin-top: 7px; padding: 3px 6px;
      border: 1px solid #d0d7de; border-radius: 3px; background: #f6f8fa;
      color: #47515c; font-size: 10px; font-weight: 700; text-transform: uppercase;
    }}
    .acquisition-extra {{
      border-color: #54aeff; background: #ddf4ff; color: #0550ae;
    }}
    .acquisition-not-invoked {{
      border-color: #d4a72c; background: #fff8c5; color: #7d4e00;
    }}
    .judge-not-invoked {{
      display: flex; gap: 8px; align-items: baseline; margin: 12px 0;
      padding: 9px 10px; border: 1px solid #d4a72c;
      border-left: 4px solid #bf8700; background: #fff8c5;
      font-size: 11px;
    }}
    .judge-not-invoked span {{ color: #6e4c00; }}
    .verdict {{
      color: #1f2328 !important; font-size: 14px !important; font-weight: 750;
      text-transform: uppercase;
    }}
    .verdict-valid, .verdict-complete {{ color: #1a7f37 !important; }}
    .verdict-invalid, .verdict-failed {{ color: #cf222e !important; }}
    .verdict-unresolved, .verdict-ambiguous {{ color: #9a6700 !important; }}
    .result-grid {{
      display: grid; grid-template-columns: 100px 100px minmax(240px, 1fr);
      margin: 12px 0; border: 1px solid #d8dee4;
    }}
    .result-grid div {{ padding: 8px 10px; border-right: 1px solid #d8dee4; min-width: 0; }}
    .result-grid div:last-child {{ border-right: 0; }}
    .result-grid strong, .result-grid span {{ display: block; }}
    .result-grid strong {{ color: #59636e; font-size: 10px; text-transform: uppercase; }}
    .result-grid span {{ overflow-wrap: anywhere; font-size: 12px; }}
    .reason {{ line-height: 1.5; }}
    .packet-audit {{
      display: grid; grid-template-columns: minmax(180px, 1fr)
        minmax(180px, 1fr) minmax(220px, 1.2fr);
      margin: 12px 0; border: 1px solid #b6c2cf; background: #fbfcfd;
    }}
    .packet-audit > div {{ padding: 9px 10px; border-right: 1px solid #d8dee4; }}
    .packet-audit > div:last-child {{ border-right: 0; }}
    .packet-audit strong, .packet-audit span {{ display: block; }}
    .packet-audit strong {{
      color: #59636e; font-size: 10px; text-transform: uppercase;
    }}
    .packet-audit span {{ margin-top: 4px; font-size: 11px; line-height: 1.4; }}
    .packet-current_default {{ border-left: 4px solid #1a7f37; }}
    .packet-historical {{ border-left: 4px solid #bf8700; }}
    .packet-current_run_custom_or_mismatch {{ border-left: 4px solid #cf222e; }}
    .packet-unversioned {{ border-left: 4px solid #8c959f; }}
    .judge-budget {{
      display: grid;
      grid-template-columns: minmax(170px, 1.3fr) repeat(3, minmax(110px, .7fr));
      margin: 12px 0; border: 1px solid #b6c2cf;
      border-left: 4px solid #57606a; background: #fbfcfd;
    }}
    .judge-budget > div {{ padding: 9px 10px; border-right: 1px solid #d8dee4; }}
    .judge-budget > div:nth-child(4) {{ border-right: 0; }}
    .judge-budget strong, .judge-budget span {{ display: block; }}
    .judge-budget strong {{
      color: #59636e; font-size: 10px; text-transform: uppercase;
    }}
    .judge-budget span {{ margin-top: 4px; font-size: 11px; }}
    .judge-budget p {{
      grid-column: 1 / -1; margin: 0; padding: 7px 10px;
      border-top: 1px solid #d8dee4; color: #59636e; font-size: 10px;
    }}
    .evidence-flow {{
      margin: 14px 0; padding: 0; border: 1px solid #d0d7de;
      border-left: 4px solid #8c959f; background: #fbfcfd;
    }}
    .evidence-flow.flow-extra {{ border-left-color: #0969da; }}
    .evidence-flow > summary {{
      display: flex; justify-content: space-between; gap: 18px; align-items: center;
      padding: 10px 12px; list-style-position: inside;
    }}
    .evidence-flow > summary > span:first-child strong,
    .evidence-flow > summary > span:first-child span {{ display: block; }}
    .flow-label {{ margin-top: 2px; color: #59636e; font-size: 10px; font-weight: 500; }}
    .flow-extra .flow-label {{ color: #0550ae; font-weight: 700; }}
    .flow-stats {{
      color: #59636e; font-size: 11px; font-weight: 500; text-align: right;
    }}
    .flow-body {{ padding: 0 12px 12px; border-top: 1px solid #d8dee4; }}
    .trace-source {{ margin: 10px 0; color: #59636e; font-size: 10px; }}
    .timeline {{
      position: relative; margin: 0; padding: 0; list-style: none;
    }}
    .timeline::before {{
      content: ""; position: absolute; left: 7px; top: 8px; bottom: 10px;
      width: 2px; background: #d8dee4;
    }}
    .timeline-step {{
      position: relative; display: grid; grid-template-columns: 16px minmax(0, 1fr);
      gap: 10px; padding: 5px 0 10px;
    }}
    .timeline-marker {{
      position: relative; z-index: 1; width: 10px; height: 10px; margin: 5px 0 0 3px;
      border: 2px solid #8c959f; border-radius: 50%; background: white;
    }}
    .timeline-judge .timeline-marker {{ border-color: #8250df; }}
    .timeline-judge_evidence_request .timeline-marker,
    .timeline-acquisition_planner .timeline-marker,
    .timeline-camera_selector .timeline-marker {{ border-color: #0969da; }}
    .timeline-render .timeline-marker {{ border-color: #1a7f37; }}
    .timeline-heading {{
      display: flex; justify-content: space-between; gap: 12px; align-items: start;
    }}
    .timeline-heading strong {{ font-size: 12px; }}
    .round-label {{
      display: inline-block; min-width: 54px; margin-right: 7px; color: #59636e;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10px;
    }}
    .trace-note {{
      margin-left: 6px; color: #8c959f; font-size: 9px; font-weight: 500;
    }}
    .timeline-status {{
      color: #59636e; font-size: 10px; text-transform: uppercase;
    }}
    .timeline-content > p {{
      margin: 4px 0 0; color: #47515c; font-size: 11px; line-height: 1.45;
    }}
    .new-evidence {{
      margin-top: 9px; padding: 9px; border: 1px solid #b6d7f7;
      background: #f0f8ff;
    }}
    .new-evidence > strong {{ color: #0550ae; font-size: 10px; text-transform: uppercase; }}
    .timeline-images {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 7px; margin-top: 7px;
    }}
    .timeline-image {{
      min-width: 0; color: inherit; text-decoration: none;
    }}
    .timeline-image img {{
      display: block; width: 100%; height: auto; border: 1px solid #8c959f;
    }}
    .timeline-image span {{
      display: block; margin-top: 3px; overflow: hidden; color: #59636e;
      font-size: 9px; text-overflow: ellipsis; white-space: nowrap;
    }}
    .timeline-image-missing {{
      min-width: 0; padding: 7px; border: 1px solid #d8dee4; background: white;
      font-size: 9px;
    }}
    .timeline-image-missing code {{ display: block; overflow-wrap: anywhere; }}
    .step-record {{ margin-top: 3px; padding: 2px 0; }}
    .step-record summary {{ color: #59636e; font-size: 9px; font-weight: 600; }}
    .step-record pre {{ max-height: 280px; margin-bottom: 0; }}
    .timeline-empty {{ padding: 10px; color: #59636e; font-size: 11px; }}
    .image-grid {{
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; margin-top: 12px;
    }}
    figure {{ margin: 0; min-width: 0; }}
    figure img {{
      display: block; width: 100%; height: auto; border: 1px solid #b6bec6;
      background: #eef0f2;
    }}
    figcaption {{ padding-top: 5px; }}
    figcaption strong, figcaption code {{ display: block; }}
    figcaption strong {{ font-size: 12px; }}
    figcaption code {{ color: #59636e; font-size: 9px; overflow-wrap: anywhere; }}
    .details-row {{ margin-top: 12px; border-top: 1px solid #d8dee4; }}
    details {{ padding: 7px 0; }}
    details + details {{ border-top: 1px solid #e5e8eb; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    pre {{
      max-height: 420px; overflow: auto; padding: 10px;
      background: #f6f8fa; border: 1px solid #d8dee4;
      white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; line-height: 1.45;
    }}
    .empty, .missing-image {{
      padding: 14px; border: 1px solid #d8dee4; background: #f6f8fa;
    }}
    .missing-image code {{ display: block; margin-top: 5px; overflow-wrap: anywhere; }}
    @media (max-width: 900px) {{
      .page {{ width: min(100% - 20px, 1440px); margin-top: 16px; }}
      .summary {{ grid-template-columns: repeat(2, 1fr); }}
      .summary div {{ border-bottom: 1px solid #d8dee4; }}
      .toolbar, .scene-title, .call-header,
      .object-findings-heading {{ display: block; }}
      .grouping-heading, .acquisition-heading {{ display: block; }}
      .grouping-meta {{ justify-content: start; margin-top: 10px; }}
      .acquisition-heading p {{ margin-top: 10px; }}
      .acquisition-summary {{ grid-template-columns: repeat(2, 1fr); }}
      .acquisition-summary div {{ border-bottom: 1px solid #d8dee4; }}
      .grouping-visual {{ grid-template-columns: 1fr; }}
      .functional-evidence-heading {{ display: block; }}
      .functional-evidence-heading .audit-only-badge {{
        display: inline-block; margin-top: 10px;
      }}
      .functional-evidence-summary {{ grid-template-columns: 1fr 1fr; }}
      .functional-evidence-summary div:nth-child(2) {{
        border-right: 0;
      }}
      .functional-evidence-summary div:nth-child(-n+2) {{
        border-bottom: 1px solid #d8dee4;
      }}
      .group-grid {{ grid-template-columns: 1fr; }}
      .scene-nav {{ align-items: start; }}
      .scene-buttons {{ flex: 1; }}
      .scene-counter {{ margin: 8px 0 0; }}
      .toolbar input, .toolbar select {{ width: 100%; margin-bottom: 7px; }}
      .scene-status, .decision {{ margin-top: 10px; text-align: left; }}
      .result-grid {{ grid-template-columns: 1fr 1fr; }}
      .result-grid div {{ border-bottom: 1px solid #d8dee4; }}
      .packet-audit {{ grid-template-columns: 1fr; }}
      .packet-audit > div {{
        border-right: 0; border-bottom: 1px solid #d8dee4;
      }}
      .judge-budget {{ grid-template-columns: 1fr 1fr; }}
      .judge-budget > div {{ border-bottom: 1px solid #d8dee4; }}
      .judge-budget p {{ grid-column: 1 / -1; }}
      .evidence-flow > summary {{ display: block; }}
      .flow-stats {{ display: block; margin-top: 5px; text-align: left; }}
      .timeline-images {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .image-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div class="eyebrow">Local read-only audit</div>
      <h1>VLM evidence and decisions</h1>
      <p class="muted">{html.escape(run_root.name)}</p>
    </header>
    <section class="summary">
      <div><strong>Run status</strong><span>{html.escape(str(run_manifest.get("status") or "unknown"))}</span></div>
      <div><strong>Scenes</strong><span>{len(case_dirs)}</span></div>
      <div><strong>Judge API calls</strong><span>{judge_usage["api_calls_number"]}</span></div>
      <div><strong>Judge tokens</strong><span>{judge_tokens.get("total_tokens", 0):,}</span></div>
      <div><strong>Selector API calls</strong><span>{selector_usage["api_calls_number"]}</span></div>
      <div><strong>Selector tokens</strong><span>{selector_tokens.get("total_tokens", 0):,}</span></div>
    </section>
    <p class="notice">
      {html.escape(evidence_integrity_note)} The source generation
      prompt was withheld from the Judge. Exact composed Judge messages and raw
      response text were not persisted by this run, so the UI shows the persisted
      metric rubric, parsed result, and request metadata. Token totals cover
      persisted response usage only.
    </p>
    {acquisition_overview_html}
    <div class="scene-controls">
      <nav class="scene-nav" aria-label="Scene navigation">
        <button id="previous-scene" type="button">Previous</button>
        <div class="scene-buttons">{scene_buttons}</div>
        <button id="next-scene" type="button">Next</button>
        <span id="scene-counter" class="scene-counter"></span>
      </nav>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Search metric, group, object, verdict, reason" aria-label="Search records">
        <select id="layer" aria-label="Filter by layer">
          <option value="">All layers</option>
          <option value="l1">L1</option>
          <option value="l3">L3</option>
        </select>
        <select id="metric" aria-label="Filter by metric">
          <option value="">All metrics</option>
          {metric_options}
        </select>
        <select id="acquisition" aria-label="Filter by evidence acquisition">
          <option value="">All evidence flows</option>
          <option value="extra">Additional evidence</option>
          <option value="direct">Direct decision</option>
          <option value="untraced">Reconstructed trace</option>
        </select>
      </div>
    </div>
    {scene_pages_html}
  </main>
  <script>
    const search = document.getElementById("search");
    const layer = document.getElementById("layer");
    const metric = document.getElementById("metric");
    const acquisition = document.getElementById("acquisition");
    const cards = Array.from(document.querySelectorAll(".call"));
    const scenePages = Array.from(
      document.querySelectorAll(".scene-page")
    );
    const sceneButtons = Array.from(
      document.querySelectorAll(".scene-button")
    );
    const previousScene = document.getElementById("previous-scene");
    const nextScene = document.getElementById("next-scene");
    const sceneCounter = document.getElementById("scene-counter");
    const sceneControls = document.querySelector(".scene-controls");
    const blenderCommandButtons = Array.from(
      document.querySelectorAll(".copy-blender-command")
    );
    let currentSceneIndex = 0;

    function scrollActiveSceneToTop() {{
      window.requestAnimationFrame(() => {{
        const page = scenePages[currentSceneIndex];
        const controlsHeight = sceneControls.getBoundingClientRect().height;
        const pageTop = window.scrollY + page.getBoundingClientRect().top;
        window.scrollTo({{
          top: Math.max(0, pageTop - controlsHeight - 8),
          behavior: "auto",
        }});
      }});
    }}

    function sceneIndexFromHash() {{
      const requested = decodeURIComponent(
        window.location.hash.replace(/^#/, "")
      );
      return scenePages.findIndex(
        page => page.dataset.scene === requested
      );
    }}

    function showScene(index, updateHash = true) {{
      if (!scenePages.length) {{
        previousScene.disabled = true;
        nextScene.disabled = true;
        sceneCounter.textContent = "0 / 0";
        return;
      }}
      currentSceneIndex = Math.max(
        0,
        Math.min(index, scenePages.length - 1)
      );
      const activeScene = scenePages[currentSceneIndex].dataset.scene;
      for (const [pageIndex, page] of scenePages.entries()) {{
        page.hidden = pageIndex !== currentSceneIndex;
      }}
      for (const button of sceneButtons) {{
        const active = button.dataset.sceneTarget === activeScene;
        if (active) {{
          button.setAttribute("aria-current", "page");
        }} else {{
          button.removeAttribute("aria-current");
        }}
      }}
      previousScene.disabled = currentSceneIndex === 0;
      nextScene.disabled = currentSceneIndex === scenePages.length - 1;
      sceneCounter.textContent =
        `${{currentSceneIndex + 1}} / ${{scenePages.length}}`;
      if (updateHash) {{
        window.history.replaceState(
          null,
          "",
          `#${{encodeURIComponent(activeScene)}}`
        );
      }}
      applyFilters();
      scrollActiveSceneToTop();
    }}

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const matchesSearch = !query || card.dataset.search.includes(query);
        const matchesLayer = !layer.value || card.dataset.layer === layer.value;
        const matchesMetric = !metric.value || card.dataset.metric === metric.value;
        const matchesAcquisition = !acquisition.value ||
          card.dataset.acquisition === acquisition.value;
        card.hidden = !(
          matchesSearch && matchesLayer && matchesMetric && matchesAcquisition
        );
      }}
    }}
    search.addEventListener("input", applyFilters);
    layer.addEventListener("change", applyFilters);
    metric.addEventListener("change", applyFilters);
    acquisition.addEventListener("change", applyFilters);
    for (const button of blenderCommandButtons) {{
      button.addEventListener("click", async () => {{
        const originalLabel = button.textContent;
        try {{
          await navigator.clipboard.writeText(button.dataset.copyText);
          button.textContent = "Copied";
        }} catch (_error) {{
          button.textContent = "Copy failed";
        }}
        window.setTimeout(() => {{
          button.textContent = originalLabel;
        }}, 1200);
      }});
    }}
    for (const button of sceneButtons) {{
      button.addEventListener("click", () => {{
        const index = scenePages.findIndex(
          page => page.dataset.scene === button.dataset.sceneTarget
        );
        showScene(index);
      }});
    }}
    previousScene.addEventListener(
      "click",
      () => showScene(currentSceneIndex - 1)
    );
    nextScene.addEventListener(
      "click",
      () => showScene(currentSceneIndex + 1)
    );
    window.addEventListener("hashchange", () => {{
      const index = sceneIndexFromHash();
      if (index >= 0) showScene(index, false);
    }});
    const initialSceneIndex = sceneIndexFromHash();
    showScene(initialSceneIndex >= 0 ? initialSceneIndex : 0, false);
  </script>
</body>
</html>
"""
    output_path = (
        resolver.bundle_dir / "index.html"
        if resolver.bundle_dir is not None
        else run_root / "vlm_evidence_viewer.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    resolver.write_manifest()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local HTML viewer for persisted VLM evidence/results."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--serve-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root directory used by the local HTTP server.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Create a dedicated localhost bundle containing only the HTML "
            "and SHA-256-verified evidence copies."
        ),
    )
    args = parser.parse_args()
    output_path = build_viewer(
        args.run_root,
        serve_root=args.serve_root,
        bundle_dir=args.bundle_dir,
    )
    print(output_path)


if __name__ == "__main__":
    main()
