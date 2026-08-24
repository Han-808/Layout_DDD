"""Sequential room-isolated execution for the additive multi-room mode."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from benchmark.scene_generation.multi_room.artifacts import (
    MultiRoomArtifactError,
    MultiRoomArtifactLayout,
    artifact_hashes,
    gate_artifact_start,
    load_terminal_room_result,
)
from benchmark.scene_generation.multi_room.assembly import (
    ROOM_EVALUATION_INPUT_FILENAMES,
    RoomAssemblySource,
    RoomProjectionBundle,
    build_assembly_manifest,
    build_compiled_architecture,
    build_evaluation_index,
    build_global_scene,
    build_room_evaluation_inputs,
    build_room_projection,
    canonical_json_bytes,
    room_key_for_generation_index,
    sha256_bytes,
    validate_evaluation_index,
)
from benchmark.scene_generation.multi_room.contracts import (
    MultiRoomContractError,
    build_asset_selection,
    build_generation_input,
    build_retrieval_request,
    validate_retrieval_results,
    validate_room_object_plan,
    validate_room_placement,
)
from benchmark.scene_generation.multi_room.floor_plan import (
    LoadedFloorPlan,
    compile_room_brief,
)
from benchmark.scene_generation.multi_room.provenance import (
    compatibility_source_manifest,
)


SafeProgress = Callable[[Mapping[str, Any]], None]
DEFAULT_STAGE_A_PROMPT = Path(__file__).resolve().parent / "prompts/stage_a_prompt_v1.txt"
DEFAULT_STAGE_C_PROMPT = Path(__file__).resolve().parent / "prompts/stage_c_prompt_v1.txt"
_REQUIRED_CORE = (
    "call_model_stage",
    "canonical_json_bytes",
    "loads_strict",
    "build_retrieval_request",
    "validate_object_plan",
    "validate_placement",
    "write_exclusive",
    "write_json_exclusive",
    "sha256_file",
    "utc_now",
)


class MultiRoomRuntimeError(RuntimeError):
    """Raised when the additive runtime cannot safely complete a run."""


def _load_model_json_emission(
    core: Any, content: bytes
) -> tuple[Any, bytes, str]:
    """Strictly parse raw JSON or one otherwise-empty JSON code fence.

    Some OpenAI-compatible routes preserve a provider-added Markdown envelope
    even when the model was instructed to emit only JSON.  The compatibility
    layer may remove exactly one whole-response fence, but it must not extract
    JSON from prose, repair JSON, or alter the value.  The original bytes are
    always persisted separately as the first-emission artifact.
    """

    text = content.decode("utf-8", errors="strict")
    stripped = text.strip()
    lines = stripped.splitlines()
    envelope = "raw_json"
    normalized_text = text
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        normalized_text = "\n".join(lines[1:-1]).strip()
        envelope = "single_json_code_fence_v1"
    value = core.loads_strict(normalized_text)
    return value, normalized_text.encode("utf-8"), envelope


def _emit(progress: SafeProgress | None, event: str, **fields: Any) -> None:
    if progress is not None:
        progress({"event": event, **fields})


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiRoomArtifactError(
            f"cannot load room artifact {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise MultiRoomArtifactError(f"room artifact {path.name} must be an object")
    return value


def _write_json_or_verify(
    *,
    core: Any,
    path: Path,
    value: Mapping[str, Any],
    expected_sha256: str,
    label: str,
) -> None:
    """Resume deterministic finalization without overwriting any artifact."""

    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise MultiRoomArtifactError(f"{label} is not a regular artifact")
        if core.sha256_file(path) != expected_sha256:
            raise MultiRoomArtifactError(f"{label} resume hash mismatch")
        return
    core.write_json_exclusive(path, value)
    if core.sha256_file(path) != expected_sha256:
        raise MultiRoomRuntimeError(f"{label} write hash mismatch")


def _validate_runtime(core: Any, provider_route: Any, model: Any, spec: Mapping[str, Any]) -> None:
    missing = [name for name in _REQUIRED_CORE if not hasattr(core, name)]
    if missing:
        raise MultiRoomRuntimeError(f"frozen core lacks compatibility primitives: {missing}")
    if getattr(provider_route, "key", None) != spec["route_profile_id"]:
        raise MultiRoomRuntimeError("provider route identity mismatch")
    if getattr(model, "key", None) != spec["model_profile_id"]:
        raise MultiRoomRuntimeError("runtime model identity mismatch")
    if getattr(model, "wire_model", None) != spec["wire_model"]:
        raise MultiRoomRuntimeError("runtime wire-model identity mismatch")
    retry = spec["retry_policy"]
    if getattr(model, "max_infrastructure_retries", None) != retry.max_infrastructure_retries:
        raise MultiRoomRuntimeError("runtime retry count mismatch")
    if float(getattr(model, "retry_delay_seconds", -1)) != retry.retry_delay_seconds:
        raise MultiRoomRuntimeError("runtime retry delay mismatch")


def _safe_error_type(exc: BaseException) -> str:
    return type(exc).__name__


def _model_room_brief(room_brief: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only current-room local authority to the Stage A model."""

    return {
        "schema_version": "multi_room_model_room_brief_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "room_id": room_brief["room_id"],
        "room_type": room_brief["room_type"],
        "theme": room_brief["theme"],
        "instruction": room_brief["instruction"],
        "object_count_tier": room_brief["object_count_tier"],
        "target_instances": deepcopy(room_brief["target_instances"]),
        "room_dimensions_m": deepcopy(room_brief["room_dimensions_m"]),
        "local_room": deepcopy(room_brief["local_room"]),
        "architecture": deepcopy(room_brief["architecture"]),
        "wall_attachment_requirement": deepcopy(
            room_brief["wall_attachment_requirement"]
        ),
    }


def expected_room_resume_identities(
    plan: LoadedFloorPlan,
    *,
    campaign_id: str,
    workflow_profile_id: str,
    model_key: str,
    model_label: str,
    input_fingerprint_sha256: str,
    source_manifest_sha256: str,
    execution_policy_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build immutable terminal-room identities without external construction."""

    identities: dict[str, dict[str, Any]] = {}
    for generation_index, room_id in enumerate(plan.generation_order):
        room_key = room_key_for_generation_index(generation_index)
        room_brief = compile_room_brief(plan, room_id)
        identities[room_key] = {
            "campaign_id": campaign_id,
            "workflow_profile_id": workflow_profile_id,
            "generation_mode": "multi_room_with_architecture_v1",
            "layout_id": plan.layout_id,
            "floor_plan_sha256": plan.canonical_sha256,
            "room_id": room_id,
            "room_key": room_key,
            "generation_index": generation_index,
            "model_key": model_key,
            "model_label": model_label,
            "continued_after_terminal_room": generation_index + 1 < plan.room_count,
            "room_brief_sha256": sha256_bytes(canonical_json_bytes(room_brief)),
            "run_input_fingerprint_sha256": input_fingerprint_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "room_brief": room_brief,
        }
        if execution_policy_sha256 is not None:
            identities[room_key]["execution_policy_sha256"] = execution_policy_sha256
    return identities


def _finalize_room(
    *,
    core: Any,
    room_root: Path,
    artifact: Any,
    campaign_id: str,
    model: Any,
    room_brief: Mapping[str, Any],
    room_key: str,
    status: str,
    reason_code: str | None,
    error_type: str | None,
    stage_a: Any | None,
    stage_c: Any | None,
    retrieval_slot_count: int,
    retrieval_batch_calls: int,
    eligible: bool,
    has_later_declared_room: bool,
    run_identity: Mapping[str, str],
) -> dict[str, Any]:
    one_shot = {
        "schema_version": "multi_room_one_shot_audit_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "room_id": room_brief["room_id"],
        "room_key": room_key,
        "stage_a_semantic_emissions": int(
            stage_a is not None and getattr(stage_a, "status", None) == "captured"
        ),
        "stage_c_placement_emissions": int(
            stage_c is not None and getattr(stage_c, "status", None) == "captured"
        ),
        "retrieval_batch_calls": retrieval_batch_calls,
        "retrieval_slot_invocations": retrieval_slot_count,
        "semantic_retry_count": 0,
        "retrieval_retry_count": 0,
        "post_placement_edit_count": 0,
        "cross_room_context_used": False,
        "geometry_or_evaluator_feedback_used": False,
        "eligible_for_room_projection": eligible,
    }
    core.write_json_exclusive(room_root / "one_shot_audit.json", one_shot)
    hashes = artifact_hashes(room_root, sha256_file=core.sha256_file)
    result = {
        "schema_version": artifact.room_result_schema_version,
        "campaign_id": campaign_id,
        "workflow_profile_id": "frozen-two-stage-multi-room-with-architecture-v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "layout_id": room_brief["layout_id"],
        "floor_plan_sha256": room_brief["floor_plan_sha256"],
        "room_id": room_brief["room_id"],
        "room_key": room_key,
        "generation_index": room_brief["generation_index"],
        "model_key": model.key,
        "model_label": model.label,
        "status": status,
        "reason_code": reason_code,
        "error_type": error_type,
        "eligible_for_room_projection": eligible,
        "continued_after_terminal_room": has_later_declared_room,
        "room_brief_sha256": sha256_bytes(canonical_json_bytes(room_brief)),
        **dict(run_identity),
        "artifact_hashes": hashes,
        "completed_at": core.utc_now(),
    }
    core.write_json_exclusive(room_root / "room_result.json", result)
    return result


def _initialize_run(
    *,
    core: Any,
    layout: MultiRoomArtifactLayout,
    plan: LoadedFloorPlan,
    model: Any,
    retriever: Any,
    artifact: Any,
    run_spec: Mapping[str, Any],
) -> None:
    layout.require_fresh()
    layout.initialize_directories()
    source_manifest = run_spec["source_manifest"]
    manifest = {
        "schema_version": artifact.run_manifest_schema_version,
        "created_at": core.utc_now(),
        "campaign_id": run_spec["campaign_id"],
        "workflow_profile_id": run_spec["workflow_profile_id"],
        "generation_mode": "multi_room_with_architecture_v1",
        "model": model.to_public_dict(),
        "route_profile_id": run_spec["route_profile_id"],
        "retrieval_profile_id": run_spec["retrieval_profile_id"],
        "artifact_contract": artifact.public_dict(),
        "floor_plan": {
            **plan.public_dict(),
            "artifact_path": f"{plan.layout_id}/floor_plan.json",
            "validation_path": f"{plan.layout_id}/floor_plan_validation.json",
        },
        "input_fingerprint": dict(run_spec["input_fingerprint"]),
        "execution_policy_sha256": sha256_bytes(
            canonical_json_bytes(run_spec["execution_policy"])
        ),
        "source_manifest": dict(source_manifest),
        "retrieval": dict(retriever.public_provenance),
        "retriever_gate_status": retriever.gate_report["status"],
        "state_machine": (
            "floor_plan_gate -> sequential(room:stage_a_once -> "
            "top1_once_per_slot -> stage_c_once -> terminal) -> "
            "translation_only_assembly -> room_projections"
        ),
        "room_concurrency": 1,
        "evaluator_or_render_feedback_used": False,
    }
    core.write_json_exclusive(layout.run_manifest_path, manifest)
    core.write_json_exclusive(
        layout.execution_policy_path, dict(run_spec["execution_policy"])
    )
    core.write_exclusive(layout.floor_plan_path, plan.source_bytes)
    core.write_json_exclusive(
        layout.floor_plan_validation_path, dict(plan.validation_report)
    )


def _verify_resume(
    *,
    core: Any,
    layout: MultiRoomArtifactLayout,
    plan: LoadedFloorPlan,
    artifact: Any,
    run_spec: Mapping[str, Any],
    expected_room_results: Mapping[str, Mapping[str, Any]],
) -> None:
    gate_artifact_start(
        layout,
        resume=True,
        expected_run_schema=artifact.run_manifest_schema_version,
        expected_fingerprint=run_spec["input_fingerprint"],
        floor_plan_source_sha256=plan.source_sha256,
        floor_plan_validation=plan.validation_report,
        sha256_file=core.sha256_file,
        expected_room_ids=plan.generation_order,
        room_result_schema=artifact.room_result_schema_version,
        expected_route_binding=run_spec["execution_policy"]["run_provenance"][
            "route_binding"
        ],
        expected_room_results=expected_room_results,
    )
    execution = _load_json(layout.execution_policy_path)
    expected_execution = deepcopy(dict(run_spec["execution_policy"]))
    observed_execution = deepcopy(execution)
    for candidate in (expected_execution, observed_execution):
        provenance = candidate.get("run_provenance")
        if isinstance(provenance, dict):
            provenance.pop("preflight", None)
    if observed_execution != expected_execution:
        raise MultiRoomArtifactError("resume execution policy mismatch")


def _room_source(
    *,
    core: Any,
    layout: MultiRoomArtifactLayout,
    artifact: Any,
    room_key: str,
    room_brief: Mapping[str, Any],
    result: Mapping[str, Any],
) -> RoomAssemblySource:
    room_root = layout.room_root(room_key)
    object_plan = None
    asset_selection = None
    placement = None
    if result["status"] == "complete":
        fixed = _load_json(room_root / "fixed_instruction.json")
        if fixed.get("room_brief") != dict(room_brief):
            raise MultiRoomArtifactError("terminal room brief identity mismatch")
        try:
            raw_plan = core.loads_strict(
                (room_root / "object_plan.json").read_text(encoding="utf-8")
            )
            object_plan = validate_room_object_plan(
                raw_plan,
                room_brief=room_brief,
                frozen_validate_object_plan=core.validate_object_plan,
            )
            retrieval_results = validate_retrieval_results(
                _load_json(room_root / "retrieval_results.json"), plan=object_plan
            )
            asset_selection = _load_json(room_root / "asset_selection.json")
            raw_placement = core.loads_strict(
                (room_root / "catalog_placement_v1.json").read_text(encoding="utf-8")
            )
            placement = validate_room_placement(
                raw_placement,
                plan=object_plan,
                retrieval_results=retrieval_results,
                room_brief=room_brief,
                frozen_validate_placement=core.validate_placement,
            )
        except Exception as exc:
            raise MultiRoomArtifactError(
                f"terminal room artifacts failed validation: {_safe_error_type(exc)}"
            ) from exc
    relative_result = layout.room_result_path(room_key).relative_to(
        layout.layout_root
    )
    return RoomAssemblySource(
        room_key=room_key,
        room_id=str(room_brief["room_id"]),
        generation_index=int(room_brief["generation_index"]),
        status=str(result["status"]),
        room_result_path=relative_result,
        room_result_sha256=core.sha256_file(layout.room_result_path(room_key)),
        object_plan_artifact_sha256=(
            result.get("artifact_hashes", {}).get("object_plan.json")
            if result["status"] == "complete"
            else None
        ),
        room_brief=deepcopy(dict(room_brief)),
        object_plan=object_plan,
        asset_selection=asset_selection,
        placement=placement,
    )


def _run_room(
    *,
    core: Any,
    provider_route: Any,
    model: Any,
    retriever: Any,
    retry_policy: Any,
    artifact: Any,
    campaign_id: str,
    room_root: Path,
    room_key: str,
    room_brief: Mapping[str, Any],
    stage_a_prompt: str,
    stage_c_prompt: str,
    has_later_declared_room: bool,
    run_identity: Mapping[str, str],
) -> dict[str, Any]:
    room_root.mkdir(parents=False, exist_ok=False)
    fixed = {
        "schema_version": "multi_room_fixed_instruction_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "room_brief": deepcopy(dict(room_brief)),
        "prompt_hashes": {
            "stage_a": sha256_bytes(stage_a_prompt.encode("utf-8")),
            "stage_c": sha256_bytes(stage_c_prompt.encode("utf-8")),
        },
        "one_shot_contract": {
            "stage_a_semantic_emissions_allowed": 1,
            "retrieval_invocations_per_public_slot": 1,
            "stage_c_placement_emissions_allowed": 1,
            "post_placement_edits_allowed": 0,
            "cross_room_context_allowed": False,
            "geometry_render_or_evaluator_feedback_allowed": False,
        },
        "run_identity": dict(run_identity),
    }
    core.write_json_exclusive(room_root / "fixed_instruction.json", fixed)
    stage_a = core.call_model_stage(
        stage="stage_a_object_plan",
        stage_dir=room_root / "stage_a",
        model=model,
        system_prompt=stage_a_prompt,
        user_value={"brief": _model_room_brief(room_brief)},
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage_a.status != "captured" or stage_a.content is None:
        return _finalize_room(
            core=core,
            room_root=room_root,
            artifact=artifact,
            campaign_id=campaign_id,
            model=model,
            room_brief=room_brief,
            room_key=room_key,
            status="stage_a_failed",
            reason_code=stage_a.reason or stage_a.status,
            error_type=None,
            stage_a=stage_a,
            stage_c=None,
            retrieval_slot_count=0,
            retrieval_batch_calls=0,
            eligible=False,
            has_later_declared_room=has_later_declared_room,
            run_identity=run_identity,
        )
    core.write_exclusive(room_root / "object_plan_first_emission.json", stage_a.content)
    try:
        raw_plan, object_plan_bytes, stage_a_envelope = _load_model_json_emission(
            core, stage_a.content
        )
        plan = validate_room_object_plan(
            raw_plan,
            room_brief=room_brief,
            frozen_validate_object_plan=core.validate_object_plan,
        )
    except Exception as exc:
        core.write_json_exclusive(
            room_root / "object_plan_validation.json",
            {"valid": False, "error_type": _safe_error_type(exc)},
        )
        return _finalize_room(
            core=core,
            room_root=room_root,
            artifact=artifact,
            campaign_id=campaign_id,
            model=model,
            room_brief=room_brief,
            room_key=room_key,
            status="stage_a_schema_invalid",
            reason_code="stage_a_contract_invalid",
            error_type=_safe_error_type(exc),
            stage_a=stage_a,
            stage_c=None,
            retrieval_slot_count=0,
            retrieval_batch_calls=0,
            eligible=False,
            has_later_declared_room=has_later_declared_room,
            run_identity=run_identity,
        )
    core.write_exclusive(room_root / "object_plan.json", object_plan_bytes)
    if core.sha256_file(room_root / "object_plan.json") != sha256_bytes(
        object_plan_bytes
    ):
        raise MultiRoomRuntimeError("normalized object-plan write mismatch")
    core.write_json_exclusive(
        room_root / "object_plan_validation.json",
        {
            "valid": True,
            "response_envelope": stage_a_envelope,
            "syntactic_normalization": stage_a_envelope != "raw_json",
        },
    )
    retrieval_request = build_retrieval_request(
        plan, frozen_build_retrieval_request=core.build_retrieval_request
    )
    core.write_json_exclusive(room_root / "retrieval_requests.json", retrieval_request)
    try:
        raw_results = retriever.retrieve(retrieval_request)
        retrieval_results = validate_retrieval_results(raw_results, plan=plan)
    except Exception as exc:
        core.write_json_exclusive(
            room_root / "retrieval_failure.json",
            {"error_type": _safe_error_type(exc), "category": "retrieval"},
        )
        return _finalize_room(
            core=core,
            room_root=room_root,
            artifact=artifact,
            campaign_id=campaign_id,
            model=model,
            room_brief=room_brief,
            room_key=room_key,
            status="retrieval_failed",
            reason_code="retrieval_failed",
            error_type=_safe_error_type(exc),
            stage_a=stage_a,
            stage_c=None,
            retrieval_slot_count=len(retrieval_request["requests"]),
            retrieval_batch_calls=1,
            eligible=False,
            has_later_declared_room=has_later_declared_room,
            run_identity=run_identity,
        )
    core.write_json_exclusive(room_root / "retrieval_results.json", retrieval_results)
    asset_selection = build_asset_selection(plan, retrieval_results)
    core.write_json_exclusive(room_root / "asset_selection.json", asset_selection)
    generation_input = build_generation_input(
        room_brief=room_brief,
        plan=plan,
        asset_selection=asset_selection,
    )
    core.write_json_exclusive(room_root / "generation_input.json", generation_input)
    stage_c = core.call_model_stage(
        stage="stage_c_placement",
        stage_dir=room_root / "stage_c",
        model=model,
        system_prompt=stage_c_prompt,
        user_value=generation_input,
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage_c.status != "captured" or stage_c.content is None:
        return _finalize_room(
            core=core,
            room_root=room_root,
            artifact=artifact,
            campaign_id=campaign_id,
            model=model,
            room_brief=room_brief,
            room_key=room_key,
            status="stage_c_failed",
            reason_code=stage_c.reason or stage_c.status,
            error_type=None,
            stage_a=stage_a,
            stage_c=stage_c,
            retrieval_slot_count=len(retrieval_request["requests"]),
            retrieval_batch_calls=1,
            eligible=False,
            has_later_declared_room=has_later_declared_room,
            run_identity=run_identity,
        )
    core.write_exclusive(
        room_root / "catalog_placement_first_emission.json", stage_c.content
    )
    try:
        raw_placement, placement_bytes, stage_c_envelope = _load_model_json_emission(
            core, stage_c.content
        )
        validate_room_placement(
            raw_placement,
            plan=plan,
            retrieval_results=retrieval_results,
            room_brief=room_brief,
            frozen_validate_placement=core.validate_placement,
        )
    except Exception as exc:
        core.write_json_exclusive(
            room_root / "placement_validation.json",
            {"valid": False, "error_type": _safe_error_type(exc)},
        )
        return _finalize_room(
            core=core,
            room_root=room_root,
            artifact=artifact,
            campaign_id=campaign_id,
            model=model,
            room_brief=room_brief,
            room_key=room_key,
            status="placement_schema_invalid",
            reason_code="stage_c_contract_invalid",
            error_type=_safe_error_type(exc),
            stage_a=stage_a,
            stage_c=stage_c,
            retrieval_slot_count=len(retrieval_request["requests"]),
            retrieval_batch_calls=1,
            eligible=False,
            has_later_declared_room=has_later_declared_room,
            run_identity=run_identity,
        )
    core.write_json_exclusive(
        room_root / "placement_validation.json",
        {
            "valid": True,
            "response_envelope": stage_c_envelope,
            "syntactic_normalization": stage_c_envelope != "raw_json",
        },
    )
    core.write_exclusive(room_root / "catalog_placement_v1.json", placement_bytes)
    if core.sha256_file(room_root / "catalog_placement_v1.json") != sha256_bytes(
        placement_bytes
    ):
        raise MultiRoomRuntimeError("normalized placement write mismatch")
    if (
        stage_c_envelope == "raw_json"
        and core.sha256_file(room_root / "catalog_placement_first_emission.json")
        != core.sha256_file(room_root / "catalog_placement_v1.json")
    ):
        raise MultiRoomRuntimeError("placement byte-copy identity mismatch")
    return _finalize_room(
        core=core,
        room_root=room_root,
        artifact=artifact,
        campaign_id=campaign_id,
        model=model,
        room_brief=room_brief,
        room_key=room_key,
        status="complete",
        reason_code=None,
        error_type=None,
        stage_a=stage_a,
        stage_c=stage_c,
        retrieval_slot_count=len(retrieval_request["requests"]),
        retrieval_batch_calls=1,
        eligible=True,
        has_later_declared_room=has_later_declared_room,
        run_identity=run_identity,
    )


def run_multi_room_campaign(
    *,
    core: Any,
    provider_route: Any,
    model: Any,
    retriever: Any,
    plan: LoadedFloorPlan,
    output_root: str | Path,
    artifact: Any,
    run_spec: Mapping[str, Any],
    resume: bool = False,
    progress: SafeProgress | None = None,
) -> tuple[dict[str, Any], bool]:
    """Run every room sequentially, then publish deterministic artifacts."""

    _validate_runtime(core, provider_route, model, run_spec)
    layout = MultiRoomArtifactLayout(Path(output_root), plan.layout_id)
    execution_policy_sha256 = sha256_bytes(
        canonical_json_bytes(run_spec["execution_policy"])
    )
    run_identity = {
        "run_input_fingerprint_sha256": run_spec["input_fingerprint"][
            "fingerprint_sha256"
        ],
        "execution_policy_sha256": execution_policy_sha256,
        "source_manifest_sha256": run_spec["source_manifest"]["manifest_sha256"],
    }
    resume_identities = expected_room_resume_identities(
        plan,
        campaign_id=run_spec["campaign_id"],
        workflow_profile_id=run_spec["workflow_profile_id"],
        model_key=model.key,
        model_label=model.label,
        input_fingerprint_sha256=run_identity["run_input_fingerprint_sha256"],
        source_manifest_sha256=run_identity["source_manifest_sha256"],
        execution_policy_sha256=execution_policy_sha256,
    )
    if resume:
        _verify_resume(
            core=core,
            layout=layout,
            plan=plan,
            artifact=artifact,
            run_spec=run_spec,
            expected_room_results=resume_identities,
        )
    else:
        _initialize_run(
            core=core,
            layout=layout,
            plan=plan,
            model=model,
            retriever=retriever,
            artifact=artifact,
            run_spec=run_spec,
        )
    stage_a_prompt = DEFAULT_STAGE_A_PROMPT.read_text(encoding="utf-8")
    stage_c_prompt = DEFAULT_STAGE_C_PROMPT.read_text(encoding="utf-8")
    _emit(progress, "run_starting", requested_rooms=plan.room_count, resume=resume)

    sources: list[RoomAssemblySource] = []
    results: list[dict[str, Any]] = []
    for generation_index, room_id in enumerate(plan.generation_order):
        room_key = room_key_for_generation_index(generation_index)
        room_brief = compile_room_brief(plan, room_id)
        room_root = layout.room_root(room_key)
        terminal = load_terminal_room_result(
            room_root,
            expected_schema=artifact.room_result_schema_version,
            expected_room_id=room_id,
            expected_room_key=room_key,
            sha256_file=core.sha256_file,
            expected_identity=resume_identities[room_key],
        )
        if terminal is None:
            _emit(
                progress,
                "room_starting",
                room_id=room_id,
                room_key=room_key,
                generation_index=generation_index,
            )
            terminal = _run_room(
                core=core,
                provider_route=provider_route,
                model=model,
                retriever=retriever,
                retry_policy=run_spec["retry_policy"],
                artifact=artifact,
                campaign_id=run_spec["campaign_id"],
                room_root=room_root,
                room_key=room_key,
                room_brief=room_brief,
                stage_a_prompt=stage_a_prompt,
                stage_c_prompt=stage_c_prompt,
                has_later_declared_room=generation_index + 1 < plan.room_count,
                run_identity=run_identity,
            )
        else:
            _emit(
                progress,
                "room_resumed_terminal",
                room_id=room_id,
                room_key=room_key,
                status=terminal["status"],
            )
        results.append(dict(terminal))
        sources.append(
            _room_source(
                core=core,
                layout=layout,
                artifact=artifact,
                room_key=room_key,
                room_brief=room_brief,
                result=terminal,
            )
        )
        _emit(
            progress,
            "room_terminal",
            room_id=room_id,
            room_key=room_key,
            status=terminal["status"],
            eligible=terminal["eligible_for_room_projection"],
            continued_after_room=generation_index + 1 < plan.room_count,
        )

    runtime_manifest = compatibility_source_manifest()
    compiled = build_compiled_architecture(plan)
    compiled_bytes = canonical_json_bytes(compiled)
    compiled_sha = sha256_bytes(compiled_bytes)
    projection_values: dict[str, RoomProjectionBundle] = {}
    for source in sources:
        if not source.complete:
            continue
        projection = build_room_projection(
            source, compiled_architecture_sha256=compiled_sha
        )
        projection_path = layout.room_projection_path(source.room_key)
        relative = layout.relative_to_layout(projection_path)
        projection_sha = sha256_bytes(canonical_json_bytes(projection))
        evaluation_inputs = build_room_evaluation_inputs(source, projection)
        persisted_inputs: dict[str, dict[str, Any]] = {}
        for name, filename in ROOM_EVALUATION_INPUT_FILENAMES.items():
            value = evaluation_inputs[name]
            path = projection_path.parent / filename
            persisted_inputs[name] = {
                "path": layout.relative_to_layout(path),
                "sha256": sha256_bytes(canonical_json_bytes(value)),
                "value": value,
            }
        projection_values[source.room_id] = RoomProjectionBundle(
            canonical_scene_path=relative,
            canonical_scene_sha256=projection_sha,
            canonical_scene=projection,
            evaluation_inputs=persisted_inputs,
        )

    global_scene = build_global_scene(
        plan, sources, compiled_architecture_sha256=compiled_sha
    )
    global_bytes = canonical_json_bytes(global_scene)
    global_sha = sha256_bytes(global_bytes)
    evaluation_index = build_evaluation_index(
        plan,
        sources,
        compiled_architecture_sha256=compiled_sha,
        projections=projection_values,
    )
    index_bytes = canonical_json_bytes(evaluation_index)
    index_sha = sha256_bytes(index_bytes)
    manifest = build_assembly_manifest(
        plan,
        sources,
        compiled_architecture_path=layout.relative_to_layout(
            layout.compiled_architecture_path
        ),
        compiled_architecture_sha256=compiled_sha,
        global_scene_path=layout.relative_to_layout(layout.global_scene_path),
        global_scene_sha256=global_sha,
        global_scene=global_scene,
        evaluation_index_path=layout.relative_to_layout(layout.evaluation_index_path),
        evaluation_index_sha256=index_sha,
        projections=projection_values,
        runtime_source_manifest_sha256=runtime_manifest["manifest_sha256"],
    )

    _write_json_or_verify(
        core=core,
        path=layout.compiled_architecture_path,
        value=compiled,
        expected_sha256=compiled_sha,
        label="compiled architecture",
    )
    for room_id in plan.generation_order:
        if room_id not in projection_values:
            continue
        bundle = projection_values[room_id]
        _write_json_or_verify(
            core=core,
            path=layout.layout_root / bundle.canonical_scene_path,
            value=bundle.canonical_scene,
            expected_sha256=bundle.canonical_scene_sha256,
            label="room projection",
        )
        for companion in bundle.evaluation_inputs.values():
            _write_json_or_verify(
                core=core,
                path=layout.layout_root / companion["path"],
                value=companion["value"],
                expected_sha256=companion["sha256"],
                label="room evaluation input",
            )
    _write_json_or_verify(
        core=core,
        path=layout.global_scene_path,
        value=global_scene,
        expected_sha256=global_sha,
        label="global scene",
    )
    _write_json_or_verify(
        core=core,
        path=layout.evaluation_index_path,
        value=evaluation_index,
        expected_sha256=index_sha,
        label="evaluation index",
    )

    if core.sha256_file(layout.compiled_architecture_path) != compiled_sha:
        raise MultiRoomRuntimeError("compiled architecture write hash mismatch")
    if core.sha256_file(layout.global_scene_path) != global_sha:
        raise MultiRoomRuntimeError("global scene write hash mismatch")
    if core.sha256_file(layout.evaluation_index_path) != index_sha:
        raise MultiRoomRuntimeError("evaluation index write hash mismatch")
    for bundle in projection_values.values():
        if (
            core.sha256_file(layout.layout_root / bundle.canonical_scene_path)
            != bundle.canonical_scene_sha256
        ):
            raise MultiRoomRuntimeError("room projection write hash mismatch")
        for companion in bundle.evaluation_inputs.values():
            if (
                core.sha256_file(layout.layout_root / companion["path"])
                != companion["sha256"]
            ):
                raise MultiRoomRuntimeError(
                    "room evaluation input write hash mismatch"
                )
    validate_evaluation_index(evaluation_index, layout_root=layout.layout_root)
    expected_evaluation_paths = {
        bundle.canonical_scene_path
        for bundle in projection_values.values()
    } | {
        companion["path"]
        for bundle in projection_values.values()
        for companion in bundle.evaluation_inputs.values()
    }
    evaluation_tree = list(layout.evaluation_rooms_root.rglob("*"))
    if any(path.is_symlink() for path in evaluation_tree):
        raise MultiRoomArtifactError("room-evaluation artifacts may not be symlinks")
    actual_evaluation_paths = {
        path.relative_to(layout.layout_root).as_posix()
        for path in evaluation_tree
        if path.is_file()
    }
    if actual_evaluation_paths != expected_evaluation_paths:
        raise MultiRoomArtifactError(
            "room-evaluation artifact tree differs from the exact index"
        )
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
    _write_json_or_verify(
        core=core,
        path=layout.assembly_manifest_path,
        value=manifest,
        expected_sha256=manifest_sha,
        label="assembly manifest",
    )

    complete = sum(result["status"] == "complete" for result in results)
    failed = len(results) - complete
    summary = {
        "schema_version": artifact.summary_schema_version,
        "campaign_id": run_spec["campaign_id"],
        "workflow_profile_id": run_spec["workflow_profile_id"],
        "generation_mode": "multi_room_with_architecture_v1",
        "layout_id": plan.layout_id,
        "model_key": model.key,
        "model_label": model.label,
        "requested_rooms": plan.room_count,
        "processed_rooms": len(results),
        "complete_rooms": complete,
        "failed_rooms": failed,
        "projected_rooms": len(projection_values),
        "assembly_status": "complete" if failed == 0 else "incomplete",
        "results": results,
        "artifacts": {
            "assembled_multi_room_scene": layout.relative_to_layout(
                layout.global_scene_path
            ),
            "compiled_architecture": layout.relative_to_layout(
                layout.compiled_architecture_path
            ),
            "room_evaluation_index": layout.relative_to_layout(
                layout.evaluation_index_path
            ),
            "assembly_manifest": layout.relative_to_layout(
                layout.assembly_manifest_path
            ),
        },
        "completed_at": core.utc_now(),
    }
    core.write_json_exclusive(layout.summary_path, summary)
    _emit(
        progress,
        "run_terminal",
        requested_rooms=plan.room_count,
        complete_rooms=complete,
        failed_rooms=failed,
        projected_rooms=len(projection_values),
        assembly_status=summary["assembly_status"],
    )
    return summary, False
