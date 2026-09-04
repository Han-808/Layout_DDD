"""One resumable shared-DB Agent episode and canonical scene publication."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    build_polygon_architecture,
)
from .artifacts import AgentEpisodeArtifacts, read_json, write_json_or_verify
from .contracts import validate_agent_submission
from .external import AgentProcessResult, run_external_agent
from .normalization import materialize_agent_scene
from .profiles import AgentBackendProfile
from .provenance import agent_source_manifest
from .suite import AgentFloorPlanCase
from .tool_server import AgentToolPolicy, AgentToolServer, AgentToolSession


AGENT_GENERATION_MODE = "non_rectangular_agent_shared_db_v1"
RUN_MANIFEST_SCHEMA_VERSION = "non_rectangular_agent_run_manifest_v1"
SUMMARY_SCHEMA_VERSION = "non_rectangular_agent_summary_v1"
DEFAULT_TASK_PROMPT = Path(__file__).resolve().parent / "prompts/task_prompt_v1.txt"


def run_agent_episode(
    *,
    case: AgentFloorPlanCase,
    asset_catalog: Any,
    agent_profile: AgentBackendProfile,
    tool_policy: AgentToolPolicy,
    output_root: str | Path,
    suite_identity: Mapping[str, Any],
    resume: bool = False,
    environ: Mapping[str, str] | None = None,
    process_runner: Callable[..., AgentProcessResult] = run_external_agent,
) -> dict[str, Any]:
    """Run one Agent system on one fixed layout; never invoke the evaluator."""

    base_prompt = DEFAULT_TASK_PROMPT.read_text(encoding="utf-8")
    prompt = _case_prompt(base_prompt, case=case)
    task_payload = _task_payload(
        case=case,
        asset_catalog=asset_catalog,
        tool_policy=tool_policy,
    )
    shim_path = Path(__file__).resolve().parent / "tool_shim.py"
    shim_bytes = shim_path.read_bytes()
    agent_identity = {
        "agent_id": agent_profile.agent_id,
        "display_name": agent_profile.display_name,
        "implementation": agent_profile.implementation,
        "implementation_version": agent_profile.implementation_version,
        "model_id": agent_profile.model_id,
        "isolation_mode": agent_profile.isolation_mode,
        "command_sha256": _sha256_mapping(
            {"argv": list(agent_profile.command)}
        ),
        "pass_environment_names": list(agent_profile.pass_environment),
    }
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "generation_mode": AGENT_GENERATION_MODE,
        "track_id": "complicated_floorplan_agent_track_v1",
        "comparison_unit": "agent_system",
        "source_manifest_sha256": agent_source_manifest()["manifest_sha256"],
        "agent": agent_identity,
        "layout_id": case.scene_id,
        "room_layout_sha256": case.room_layout_sha256,
        "room_program_sha256": case.room_program_sha256,
        "suite_identity": dict(sorted(suite_identity.items())),
        "shared_asset_database": asset_catalog.public_manifest(),
        "task_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tool_client_sha256": hashlib.sha256(shim_bytes).hexdigest(),
        "tool_policy": tool_policy.public_dict(),
        "agent_contract": {
            "iterative_edits_allowed": True,
            "asset_selection_mode": "shared_database_agent_selected",
            "architecture_edits_allowed": False,
            "external_asset_sources_allowed": False,
            "evaluator_score_or_hidden_labels_available": False,
            "cross_room_relations_scored": False,
            "final_output_contract": "non_rectangular_agent_submission_v1",
        },
        "process_policy": {
            "timeout_seconds": agent_profile.timeout_seconds,
            "max_process_attempts": agent_profile.max_process_attempts,
            "retry_delay_seconds": agent_profile.retry_delay_seconds,
            "retryable_exit_codes": list(agent_profile.retryable_exit_codes),
            "ambiguous_timeout_retried": False,
        },
    }
    artifacts = AgentEpisodeArtifacts(output_root)
    if resume:
        terminal = artifacts.verify_resume(
            run_manifest=run_manifest,
            room_layout=case.room_layout,
            room_program=case.room_program,
            task_payload=task_payload,
            task_prompt=prompt,
        )
        if terminal is not None:
            return terminal
        _verify_tool_shim(artifacts, shim_bytes)
    else:
        artifacts.initialize(
            run_manifest=run_manifest,
            room_layout=case.room_layout,
            room_program=case.room_program,
            task_payload=task_payload,
            task_prompt=prompt,
        )
        _write_tool_shim(artifacts, shim_bytes)

    session = AgentToolSession(
        workspace=artifacts.workspace,
        room_layout=case.room_layout,
        room_program=case.room_program,
        asset_catalog=asset_catalog,
        task_payload=task_payload,
        policy=tool_policy,
    )
    if artifacts.finalization.exists() and not artifacts.final_submission.exists():
        raise RuntimeError("finalization exists without a sealed submission")
    process_result: AgentProcessResult
    if artifacts.final_submission.is_file():
        if not artifacts.finalization.is_file():
            session.dispatch(
                "finalize_submission",
                {"submission_path": "final_submission.json"},
            )
        process_result = AgentProcessResult(
            status="resumed_sealed_submission",
            attempts=0,
            returncode=None,
            timed_out=False,
            final_submission_sealed=True,
        )
    else:
        with AgentToolServer(session) as server:
            process_result = process_runner(
                profile=agent_profile,
                artifacts=artifacts,
                tool_server=server,
                task_prompt=prompt,
                environ=environ,
            )
    if not artifacts.final_submission.is_file() or not artifacts.finalization.is_file():
        return _failure_summary(
            artifacts,
            case=case,
            run_manifest=run_manifest,
            process_result=process_result,
            reason=process_result.status,
            tool_counts=session.counts(),
        )

    try:
        submission = read_json(artifacts.final_submission)
        validated = validate_agent_submission(
            submission,
            room_layout=case.room_layout,
            room_program=case.room_program,
            asset_catalog=asset_catalog,
        )
        scene, preflight = materialize_agent_scene(
            room_layout=case.room_layout,
            room_program=case.room_program,
            validated=validated,
            generation_mode=AGENT_GENERATION_MODE,
        )
        architecture = build_polygon_architecture(case.room_layout)
    except Exception as exc:
        return _failure_summary(
            artifacts,
            case=case,
            run_manifest=run_manifest,
            process_result=process_result,
            reason="submission_contract_invalid",
            error_type=type(exc).__name__,
            tool_counts=session.counts(),
        )

    write_json_or_verify(artifacts.object_plan, validated.object_plan)
    write_json_or_verify(artifacts.asset_selection, validated.asset_selection)
    write_json_or_verify(artifacts.global_placement, validated.global_placement)
    write_json_or_verify(artifacts.generated_scene, scene)
    write_json_or_verify(artifacts.compiled_architecture, architecture)
    write_json_or_verify(artifacts.evaluation_preflight, preflight)
    write_json_or_verify(artifacts.submission_validation, validated.public_dict())
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generation_mode": AGENT_GENERATION_MODE,
        "track_id": "complicated_floorplan_agent_track_v1",
        "agent_id": agent_profile.agent_id,
        "model_id": agent_profile.model_id,
        "layout_id": case.scene_id,
        "status": "complete",
        "reason": None,
        "planned_instance_count": int(
            validated.plan_validation["planned_instance_count"]
        ),
        "generated_object_count": sum(
            len(room["objects"]) for room in scene["rooms"]
        ),
        "room_count": len(scene["rooms"]),
        "catalog_snapshot_id": str(asset_catalog.snapshot_id),
        "process": process_result.public_dict(),
        "tool_counts": session.counts(),
        "count_compliance": dict(validated.plan_validation["count_compliance"]),
        "program_mapping": dict(validated.plan_validation["program_mapping"]),
        "artifact_sha256": _artifact_hashes(artifacts),
        "official_evaluation_connected": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_or_verify(artifacts.summary, summary)
    return summary


def _task_payload(
    *,
    case: AgentFloorPlanCase,
    asset_catalog: Any,
    tool_policy: AgentToolPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": "non_rectangular_agent_task_v1",
        "track_id": "complicated_floorplan_agent_track_v1",
        "layout_id": case.scene_id,
        "authoritative_inputs": {
            "room_layout": "room_layout.json",
            "room_program": "room_program.json",
            "coordinates": "shared_scene_global_x_width_y_depth_z_up_meters",
            "architecture_mutable": False,
        },
        "room_count": case.room_count,
        "wall_segment_count": case.wall_segment_count,
        "target_total_instances": dict(
            case.room_program["target_total_instances"]
        ),
        "asset_database": asset_catalog.public_manifest(),
        "tool_policy": tool_policy.public_dict(),
        "tools": {
            "command": "./layout-ddd-agent-tool",
            "methods": [
                "get-task",
                "search-assets",
                "inspect-asset",
                "validate-submission",
                "finalize-submission",
            ],
        },
        "final_submission": {
            "draft_path": "submission.json",
            "seal_command": "./layout-ddd-agent-tool finalize-submission submission.json",
            "schema_version": "non_rectangular_agent_submission_v1",
        },
        "evaluation_scope": {
            "unit": "single_room_projection",
            "cross_room_connections_scored": False,
            "evaluator_feedback_available_during_generation": False,
        },
    }


def _case_prompt(base: str, *, case: AgentFloorPlanCase) -> str:
    target = case.room_program["target_total_instances"]
    return (
        base.rstrip()
        + "\n\nCURRENT CASE\n\n"
        + f"- layout_id: `{case.scene_id}`\n"
        + f"- rooms: `{case.room_count}`\n"
        + f"- wall segments: `{case.wall_segment_count}`\n"
        + f"- required total instances: `{target['min']}–{target['max']}` inclusive\n"
    )


def _write_tool_shim(artifacts: AgentEpisodeArtifacts, content: bytes) -> None:
    target = artifacts.workspace / "layout-ddd-agent-tool"
    with target.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    target.chmod(0o700)


def _verify_tool_shim(artifacts: AgentEpisodeArtifacts, content: bytes) -> None:
    target = artifacts.workspace / "layout-ddd-agent-tool"
    if not target.is_file() or target.is_symlink() or target.read_bytes() != content:
        raise RuntimeError("Agent tool client identity drifted on resume")


def _failure_summary(
    artifacts: AgentEpisodeArtifacts,
    *,
    case: AgentFloorPlanCase,
    run_manifest: Mapping[str, Any],
    process_result: AgentProcessResult,
    reason: str,
    tool_counts: Mapping[str, int],
    error_type: str | None = None,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generation_mode": AGENT_GENERATION_MODE,
        "track_id": "complicated_floorplan_agent_track_v1",
        "agent_id": run_manifest["agent"]["agent_id"],
        "model_id": run_manifest["agent"]["model_id"],
        "layout_id": case.scene_id,
        "status": "failed",
        "reason": reason,
        "error_type": error_type,
        "process": process_result.public_dict(),
        "tool_counts": dict(tool_counts),
        "official_evaluation_connected": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_or_verify(artifacts.summary, summary)
    return summary


def _artifact_hashes(artifacts: AgentEpisodeArtifacts) -> dict[str, str]:
    paths = {
        "run_manifest": artifacts.run_manifest,
        "room_layout": artifacts.room_layout,
        "room_program": artifacts.room_program,
        "task_payload": artifacts.task_payload,
        "final_submission": artifacts.final_submission,
        "object_plan": artifacts.object_plan,
        "asset_selection": artifacts.asset_selection,
        "global_placement": artifacts.global_placement,
        "generated_scene": artifacts.generated_scene,
        "compiled_architecture": artifacts.compiled_architecture,
        "evaluation_preflight": artifacts.evaluation_preflight,
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if path.is_file()
    }


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AGENT_GENERATION_MODE",
    "DEFAULT_TASK_PROMPT",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "run_agent_episode",
]
