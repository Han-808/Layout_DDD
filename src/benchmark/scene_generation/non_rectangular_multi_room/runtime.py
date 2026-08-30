"""One global Stage A and one global Stage C with write-once artifacts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    build_polygon_architecture,
)
from benchmark.scene_generation.non_rectangular_multi_room.artifacts import (
    NonRectangularGenerationArtifactError,
    NonRectangularGenerationArtifacts,
    read_json,
)
from benchmark.scene_generation.non_rectangular_multi_room.contracts import (
    GENERATION_MODE,
    GENERATION_MODE_V2,
    build_global_retrieval_plan,
    build_stage_a_user_value,
    build_stage_a_user_value_v2,
    build_stage_c_user_value,
    build_stage_c_user_value_v2,
    group_asset_selection,
    materialize_generated_scene,
    validate_global_placement,
    validate_stage_a_artifacts,
)
from benchmark.scene_generation.non_rectangular_multi_room.provenance import (
    compatibility_source_manifest,
    run_input_fingerprint,
    sha256_bytes,
    sha256_mapping,
)


DEFAULT_STAGE_A_PROMPT = (
    Path(__file__).resolve().parent / "prompts/stage_a_prompt_v1.txt"
)
DEFAULT_STAGE_C_PROMPT = (
    Path(__file__).resolve().parent / "prompts/stage_c_prompt_v1.txt"
)
DEFAULT_STAGE_A_PROMPT_V2 = (
    Path(__file__).resolve().parent / "prompts/stage_a_prompt_v3.txt"
)
DEFAULT_STAGE_C_PROMPT_V2 = (
    Path(__file__).resolve().parent / "prompts/stage_c_prompt_v3.txt"
)
RUN_MANIFEST_SCHEMA_VERSION = "non_rectangular_generation_run_manifest_v1"
SUMMARY_SCHEMA_VERSION = "non_rectangular_generation_summary_v1"
SafeProgress = Callable[[Mapping[str, Any]], None]

_REQUIRED_CORE = (
    "call_model_stage",
    "loads_strict",
    "build_retrieval_request",
    "write_exclusive",
    "write_json_exclusive",
    "sha256_file",
)


class NonRectangularGenerationRuntimeError(RuntimeError):
    """Raised for runtime identity or write-once workflow violations."""


def _runtime_contract(version: str) -> dict[str, Any]:
    if version == "v1":
        return {
            "generation_mode": GENERATION_MODE,
            "plan_contract_version": "v1",
            "stage_a_prompt_path": DEFAULT_STAGE_A_PROMPT,
            "stage_c_prompt_path": DEFAULT_STAGE_C_PROMPT,
            "stage_a_builder": build_stage_a_user_value,
            "stage_c_builder": build_stage_c_user_value,
            "stage_a_name": "stage_a_non_rectangular_global_object_plan",
            "stage_c_name": "stage_c_non_rectangular_global_placement",
        }
    if version == "v2":
        return {
            "generation_mode": GENERATION_MODE_V2,
            "plan_contract_version": "v2",
            "stage_a_prompt_path": DEFAULT_STAGE_A_PROMPT_V2,
            "stage_c_prompt_path": DEFAULT_STAGE_C_PROMPT_V2,
            "stage_a_builder": build_stage_a_user_value_v2,
            "stage_c_builder": build_stage_c_user_value_v2,
            "stage_a_name": "stage_a_non_rectangular_global_object_plan_v2",
            "stage_c_name": "stage_c_non_rectangular_global_placement_v2",
        }
    raise NonRectangularGenerationRuntimeError(
        f"unsupported generation_contract_version: {version!r}"
    )


def run_non_rectangular_generation_v2(**kwargs: Any) -> dict[str, Any]:
    """Run the simplified v2 plan pipeline without changing the v1 entrypoint."""

    if "generation_contract_version" in kwargs:
        raise TypeError(
            "run_non_rectangular_generation_v2 fixes generation_contract_version"
        )
    return run_non_rectangular_generation(
        **kwargs,
        generation_contract_version="v2",
    )


def run_non_rectangular_generation(
    *,
    core: Any,
    provider_route: Any,
    model: Any,
    retriever: Any,
    retry_policy: Any,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    output_root: str | Path,
    campaign_id: str,
    workflow_profile_id: str,
    retrieval_profile_id: str,
    configuration_identity: Mapping[str, str] | None = None,
    resume: bool = False,
    progress: SafeProgress | None = None,
    generation_contract_version: str = "v1",
) -> dict[str, Any]:
    """Run the additive global workflow; never call an evaluator or camera."""

    _validate_runtime(core, provider_route, model)
    contract = _runtime_contract(generation_contract_version)
    stage_a_prompt = contract["stage_a_prompt_path"].read_text(encoding="utf-8")
    stage_c_prompt = contract["stage_c_prompt_path"].read_text(encoding="utf-8")
    source_manifest = compatibility_source_manifest()
    fingerprint = run_input_fingerprint(
        campaign_id=campaign_id,
        workflow_profile_id=workflow_profile_id,
        model_profile_id=str(model.key),
        route_profile_id=str(provider_route.key),
        retrieval_profile_id=retrieval_profile_id,
        room_layout_sha256=sha256_mapping(room_layout),
        room_program_sha256=sha256_mapping(room_program),
        source_manifest_sha256=source_manifest["manifest_sha256"],
        stage_a_prompt_sha256=sha256_bytes(stage_a_prompt.encode("utf-8")),
        stage_c_prompt_sha256=sha256_bytes(stage_c_prompt.encode("utf-8")),
        generation_mode=str(contract["generation_mode"]),
    )
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "generation_mode": contract["generation_mode"],
        "campaign_id": campaign_id,
        "workflow_profile_id": workflow_profile_id,
        "model_profile_id": str(model.key),
        "model_label": str(getattr(model, "label", model.key)),
        "wire_model": str(model.wire_model),
        "route_profile_id": str(provider_route.key),
        "retrieval_profile_id": retrieval_profile_id,
        "configuration_identity": dict(
            sorted((configuration_identity or {}).items())
        ),
        "input_fingerprint": fingerprint,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "prompt_hashes": {
            "stage_a": sha256_bytes(stage_a_prompt.encode("utf-8")),
            "stage_c": sha256_bytes(stage_c_prompt.encode("utf-8")),
        },
        "one_shot_contract": {
            "stage_a_semantic_emissions_allowed": 1,
            "stage_c_semantic_emissions_allowed": 1,
            "retrieval_invocations_per_public_slot": 1,
            "post_placement_edits_allowed": 0,
            "camera_render_or_evaluator_feedback_allowed": False,
            "coordinates_transformed": False,
            **(
                {"object_plan_contract_version": "v2"}
                if generation_contract_version == "v2"
                else {}
            ),
        },
        "retry_policy": _retry_policy_dict(retry_policy),
    }
    artifacts = NonRectangularGenerationArtifacts(Path(output_root))
    if resume:
        terminal = artifacts.verify_resume(
            run_manifest=run_manifest,
            room_layout=room_layout,
            room_program=room_program,
        )
        if terminal is not None:
            _emit(progress, "run_resumed_terminal", status=terminal.get("status"))
            return terminal
    else:
        artifacts.initialize(
            core=core,
            run_manifest=run_manifest,
            room_layout=room_layout,
            room_program=room_program,
        )
    _emit(progress, "run_starting", layout_id=room_layout.get("layout_id"))

    plan, stage_a_validation, stage_a_called = _stage_a(
        core=core,
        provider_route=provider_route,
        model=model,
        retry_policy=retry_policy,
        artifacts=artifacts,
        prompt=stage_a_prompt,
        room_layout=room_layout,
        room_program=room_program,
        build_user_value=contract["stage_a_builder"],
        expected_plan_contract_version=str(
            contract["plan_contract_version"]
        ),
        stage_name=str(contract["stage_a_name"]),
    )
    if plan is None:
        return _finalize_failure(
            core=core,
            artifacts=artifacts,
            run_manifest=run_manifest,
            status="stage_a_failed",
            reason=stage_a_validation["reason"],
            stage_a_called=stage_a_called,
            retrieval_called=False,
            stage_c_called=False,
        )
    if stage_a_validation["terminal_status"] == "failed":
        failure_reason = str(stage_a_validation["failure_reason"])
        return _finalize_failure(
            core=core,
            artifacts=artifacts,
            run_manifest=run_manifest,
            status=failure_reason,
            reason=failure_reason,
            stage_a_called=stage_a_called,
            retrieval_called=False,
            stage_c_called=False,
            stage_a_validation=stage_a_validation,
        )

    assets = _retrieval(
        core=core,
        retriever=retriever,
        artifacts=artifacts,
        object_plan=plan,
    )
    if assets is None:
        return _finalize_failure(
            core=core,
            artifacts=artifacts,
            run_manifest=run_manifest,
            status="retrieval_failed",
            reason="retrieval_failed",
            stage_a_called=stage_a_called,
            retrieval_called=True,
            stage_c_called=False,
            stage_a_validation=stage_a_validation,
        )

    placement, stage_c_called, stage_c_reason = _stage_c(
        core=core,
        provider_route=provider_route,
        model=model,
        retry_policy=retry_policy,
        artifacts=artifacts,
        prompt=stage_c_prompt,
        room_layout=room_layout,
        room_program=room_program,
        object_plan=plan,
        asset_selection=assets,
        build_user_value=contract["stage_c_builder"],
        stage_name=str(contract["stage_c_name"]),
    )
    if placement is None:
        return _finalize_failure(
            core=core,
            artifacts=artifacts,
            run_manifest=run_manifest,
            status="stage_c_failed",
            reason=stage_c_reason or "stage_c_failed",
            stage_a_called=stage_a_called,
            retrieval_called=True,
            stage_c_called=stage_c_called,
            stage_a_validation=stage_a_validation,
        )

    scene, evaluation_preflight = materialize_generated_scene(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=plan,
        asset_selection=assets,
        placement=placement,
        generation_mode=str(contract["generation_mode"]),
    )
    architecture = build_polygon_architecture(room_layout)
    _write_json_or_verify(core, artifacts.generated_scene, scene)
    _write_json_or_verify(core, artifacts.compiled_architecture, architecture)
    _write_json_or_verify(
        core,
        artifacts.evaluation_preflight,
        evaluation_preflight,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generation_mode": contract["generation_mode"],
        "layout_id": str(room_layout["layout_id"]),
        "status": "complete",
        "reason": None,
        "planned_instance_count": int(
            stage_a_validation["planned_instance_count"]
        ),
        "generated_object_count": sum(
            len(room["objects"]) for room in scene["rooms"]
        ),
        "room_count": len(scene["rooms"]),
        "stage_calls": {
            "stage_a": int(
                stage_a_called or artifacts.stage_a_first_emission.is_file()
            ),
            "retrieval_batch": 1,
            "stage_c": int(
                stage_c_called or artifacts.stage_c_first_emission.is_file()
            ),
        },
        "count_compliance": deepcopy(
            stage_a_validation["count_compliance"]
        ),
        "program_mapping": deepcopy(
            stage_a_validation["program_mapping"]
        ),
        "input_fingerprint_sha256": fingerprint["fingerprint_sha256"],
        "artifact_sha256": _artifact_hashes(core, artifacts),
        "official_evaluation_connected": False,
        **(
            {"object_plan_contract_version": "v2"}
            if generation_contract_version == "v2"
            else {}
        ),
    }
    _write_json_or_verify(core, artifacts.summary, summary)
    _emit(progress, "run_terminal", status="complete")
    return summary


def _stage_a(
    *,
    core: Any,
    provider_route: Any,
    model: Any,
    retry_policy: Any,
    artifacts: NonRectangularGenerationArtifacts,
    prompt: str,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    build_user_value: Callable[..., Mapping[str, Any]],
    expected_plan_contract_version: str,
    stage_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    artifacts.reject_ambiguous_partial_stage(
        first_emission=artifacts.stage_a_first_emission,
        normalized_artifact=artifacts.object_plan,
        stage="Stage A",
    )
    if artifacts.object_plan.is_file():
        if not artifacts.stage_a_first_emission.is_file():
            raise NonRectangularGenerationArtifactError(
                "normalized Stage-A plan lacks its first-emission artifact"
            )
        plan = read_json(artifacts.object_plan)
        validation = validate_stage_a_artifacts(
            room_layout=room_layout,
            room_program=room_program,
            object_plan=plan,
            expected_plan_contract_version=expected_plan_contract_version,
        )
        return plan, validation, False
    stage = core.call_model_stage(
        stage=stage_name,
        stage_dir=artifacts.stage_a_dir,
        model=model,
        system_prompt=prompt,
        user_value=build_user_value(
            room_layout=room_layout,
            room_program=room_program,
        ),
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage.status != "captured" or stage.content is None:
        return None, {"reason": stage.reason or stage.status}, True
    core.write_exclusive(artifacts.stage_a_first_emission, stage.content)
    try:
        raw, _, envelope = _load_model_json_emission(core, stage.content)
        validation = validate_stage_a_artifacts(
            room_layout=room_layout,
            room_program=room_program,
            object_plan=raw,
            expected_plan_contract_version=expected_plan_contract_version,
        )
    except Exception as exc:
        core.write_json_exclusive(
            artifacts.object_plan_validation,
            {"valid": False, "error_type": type(exc).__name__},
        )
        return None, {"reason": "stage_a_contract_invalid"}, True
    core.write_json_exclusive(artifacts.object_plan, raw)
    core.write_json_exclusive(
        artifacts.object_plan_validation,
        {
            "valid": True,
            "response_envelope": envelope,
            **validation,
        },
    )
    return deepcopy(dict(raw)), validation, True


def _retrieval(
    *,
    core: Any,
    retriever: Any,
    artifacts: NonRectangularGenerationArtifacts,
    object_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    existing = [
        artifacts.retrieval_request.exists(),
        artifacts.retrieval_results.exists(),
        artifacts.asset_selection.exists(),
    ]
    if any(existing) and not all(existing):
        raise NonRectangularGenerationArtifactError(
            "retrieval stage is partial and cannot be invoked twice"
        )
    flat_plan, request, bindings = build_global_retrieval_plan(
        object_plan,
        frozen_build_retrieval_request=core.build_retrieval_request,
    )
    if all(existing):
        if read_json(artifacts.retrieval_request) != request:
            raise NonRectangularGenerationArtifactError(
                "resume retrieval request identity mismatch"
            )
        raw_results = read_json(artifacts.retrieval_results)
        _, expected_selection = group_asset_selection(
            object_plan=object_plan,
            flat_plan=flat_plan,
            raw_retrieval_results=raw_results,
            bindings=bindings,
        )
        observed = read_json(artifacts.asset_selection)
        if observed != expected_selection:
            raise NonRectangularGenerationArtifactError(
                "resume asset-selection identity mismatch"
            )
        return observed
    if retriever is None:
        raise NonRectangularGenerationRuntimeError(
            "retriever is required before asset-selection completion"
        )
    core.write_json_exclusive(artifacts.retrieval_request, request)
    try:
        raw_results = retriever.retrieve(request)
        validated, selection = group_asset_selection(
            object_plan=object_plan,
            flat_plan=flat_plan,
            raw_retrieval_results=raw_results,
            bindings=bindings,
        )
    except Exception:
        return None
    core.write_json_exclusive(artifacts.retrieval_results, validated)
    core.write_json_exclusive(artifacts.asset_selection, selection)
    return selection


def _stage_c(
    *,
    core: Any,
    provider_route: Any,
    model: Any,
    retry_policy: Any,
    artifacts: NonRectangularGenerationArtifacts,
    prompt: str,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
    build_user_value: Callable[..., Mapping[str, Any]],
    stage_name: str,
) -> tuple[dict[str, Any] | None, bool, str | None]:
    generation_input = build_user_value(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=object_plan,
        asset_selection=asset_selection,
    )
    _write_json_or_verify(core, artifacts.stage_c_input, generation_input)
    artifacts.reject_ambiguous_partial_stage(
        first_emission=artifacts.stage_c_first_emission,
        normalized_artifact=artifacts.global_placement,
        stage="Stage C",
    )
    if artifacts.global_placement.is_file():
        if not artifacts.stage_c_first_emission.is_file():
            raise NonRectangularGenerationArtifactError(
                "normalized Stage-C placement lacks its first-emission artifact"
            )
        placement = read_json(artifacts.global_placement)
        validate_global_placement(
            placement,
            object_plan=object_plan,
            asset_selection=asset_selection,
        )
        return placement, False, None
    stage = core.call_model_stage(
        stage=stage_name,
        stage_dir=artifacts.stage_c_dir,
        model=(
            model.for_stage("stage_c")
            if callable(getattr(model, "for_stage", None))
            else model
        ),
        system_prompt=prompt,
        user_value=generation_input,
        provider_route=provider_route,
        retry_policy=retry_policy,
    )
    if stage.status != "captured" or stage.content is None:
        return None, True, stage.reason or stage.status
    core.write_exclusive(artifacts.stage_c_first_emission, stage.content)
    try:
        raw, _, envelope = _load_model_json_emission(core, stage.content)
        placement = validate_global_placement(
            raw,
            object_plan=object_plan,
            asset_selection=asset_selection,
        )
    except Exception as exc:
        core.write_json_exclusive(
            artifacts.placement_validation,
            {"valid": False, "error_type": type(exc).__name__},
        )
        return None, True, "stage_c_contract_invalid"
    core.write_json_exclusive(artifacts.global_placement, placement)
    core.write_json_exclusive(
        artifacts.placement_validation,
        {"valid": True, "response_envelope": envelope},
    )
    return placement, True, None


def _finalize_failure(
    *,
    core: Any,
    artifacts: NonRectangularGenerationArtifacts,
    run_manifest: Mapping[str, Any],
    status: str,
    reason: str,
    stage_a_called: bool,
    retrieval_called: bool,
    stage_c_called: bool,
    stage_a_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generation_mode": str(run_manifest["generation_mode"]),
        "layout_id": str(read_json(artifacts.room_layout)["layout_id"]),
        "status": status,
        "reason": reason,
        "stage_calls": {
            "stage_a": int(
                stage_a_called or artifacts.stage_a_first_emission.is_file()
            ),
            "retrieval_batch": int(
                retrieval_called or artifacts.retrieval_request.is_file()
            ),
            "stage_c": int(
                stage_c_called or artifacts.stage_c_first_emission.is_file()
            ),
        },
        "count_compliance": (
            deepcopy(stage_a_validation.get("count_compliance"))
            if stage_a_validation is not None
            else None
        ),
        "program_mapping": (
            deepcopy(stage_a_validation.get("program_mapping"))
            if stage_a_validation is not None
            else None
        ),
        "input_fingerprint_sha256": run_manifest["input_fingerprint"][
            "fingerprint_sha256"
        ],
        "artifact_sha256": _artifact_hashes(core, artifacts),
        "official_evaluation_connected": False,
    }
    _write_json_or_verify(core, artifacts.summary, summary)
    return summary


def _load_model_json_emission(
    core: Any,
    content: bytes,
) -> tuple[Any, bytes, str]:
    text = content.decode("utf-8", errors="strict")
    stripped = text.strip()
    lines = stripped.splitlines()
    envelope = "raw_json"
    normalized = text
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        normalized = "\n".join(lines[1:-1]).strip()
        envelope = "single_json_code_fence_v1"
    return (
        core.loads_strict(normalized),
        normalized.encode("utf-8"),
        envelope,
    )


def _write_json_or_verify(core: Any, path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise NonRectangularGenerationArtifactError(
                f"artifact is not a regular file: {path.name}"
            )
        if read_json(path) != dict(value):
            raise NonRectangularGenerationArtifactError(
                f"artifact resume identity mismatch: {path.name}"
            )
        return
    core.write_json_exclusive(path, value)
    if read_json(path) != dict(value):
        raise NonRectangularGenerationRuntimeError(
            f"artifact write/read identity mismatch: {path.name}"
        )


def _artifact_hashes(
    core: Any,
    artifacts: NonRectangularGenerationArtifacts,
) -> dict[str, str]:
    paths = {
        "run_manifest": artifacts.run_manifest,
        "room_layout": artifacts.room_layout,
        "room_program": artifacts.room_program,
        "object_plan": artifacts.object_plan,
        "retrieval_request": artifacts.retrieval_request,
        "retrieval_results": artifacts.retrieval_results,
        "asset_selection": artifacts.asset_selection,
        "stage_c_input": artifacts.stage_c_input,
        "global_placement": artifacts.global_placement,
        "generated_scene": artifacts.generated_scene,
        "compiled_architecture": artifacts.compiled_architecture,
        "evaluation_preflight": artifacts.evaluation_preflight,
    }
    return {
        name: core.sha256_file(path)
        for name, path in paths.items()
        if path.is_file() and not path.is_symlink()
    }


def _retry_policy_dict(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_public_dict", None)
    if callable(method):
        return dict(method())
    return {
        "max_infrastructure_retries": getattr(
            value, "max_infrastructure_retries", None
        ),
        "retry_delay_seconds": getattr(value, "retry_delay_seconds", None),
    }


def _validate_runtime(core: Any, provider_route: Any, model: Any) -> None:
    missing = [name for name in _REQUIRED_CORE if not hasattr(core, name)]
    if missing:
        raise NonRectangularGenerationRuntimeError(
            f"frozen core lacks required primitives: {missing}"
        )
    for obj, field, label in (
        (provider_route, "key", "provider route"),
        (model, "key", "model"),
        (model, "wire_model", "wire model"),
    ):
        value = getattr(obj, field, None)
        if not isinstance(value, str) or not value:
            raise NonRectangularGenerationRuntimeError(
                f"{label} identity is missing"
            )


def _emit(progress: SafeProgress | None, event: str, **fields: Any) -> None:
    if progress is not None:
        progress({"event": event, **fields})


__all__ = [
    "DEFAULT_STAGE_A_PROMPT",
    "DEFAULT_STAGE_C_PROMPT",
    "DEFAULT_STAGE_A_PROMPT_V2",
    "DEFAULT_STAGE_C_PROMPT_V2",
    "NonRectangularGenerationRuntimeError",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "run_non_rectangular_generation",
    "run_non_rectangular_generation_v2",
]
