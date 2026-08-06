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
        or phase not in {"global_discovery", "group_local_review"}
    ):
        return None
    expected_roles = (
        ["angled_global"]
        if phase == "global_discovery"
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
        group_results = metric_report.get("group_results")
        group_results = (
            group_results if isinstance(group_results, list) else []
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
        for index, group in enumerate(group_results):
            if not isinstance(group, dict) or group.get("vlm_invoked") is not True:
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
                    2,
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
            displayed_prompt_version = (
                run_prompt_version or "not persisted"
            )
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
                    "acquisition": acquisition_timeline(
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
                    ),
                }
            )
    return calls


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
        if not global_calls or not local_calls:
            continue
        routes.append(
            f"""
            <div class="phase-route">
              <strong>{html.escape(metric)}</strong>
              <span class="phase-node phase-node-global">
                1 · Global discovery
              </span>
              <span class="phase-arrow">→</span>
              <span class="phase-node phase-node-local">
                2 · Group-local review · {len(local_calls)} group(s)
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
        {render_acquisition_timeline(acquisition, resolver)}
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
    case_dirs = sorted(
        path for path in (run_root / "cases").iterdir() if path.is_dir()
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
        calls: list[dict[str, Any]] = []
        calls.extend(l1_calls(l1_report))
        calls.extend(l3_calls(l3_report))
        for call in calls:
            call["case_id"] = case_dir.name
        all_calls.extend(calls)
        metric_names.update(str(call["metric"]) for call in calls)
        cards = "".join(render_call(call, resolver) for call in calls)
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
    {''.join(all_sections)}
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
      if (!scenePages.length) return;
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
