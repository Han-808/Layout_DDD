"""End-to-end reference pipeline: generate -> (optional render) -> evaluate.

Summary:
    Orchestrates the full benchmark pipeline for one case and is the main
    reference harness. It chains the generate and evaluate stages, optionally
    renders Blender evidence, and can retry generation in a self-reflexive loop.

Input:
    - ``--instruction`` / ``--scene-type`` and the generator submission
      (``--adapter`` + ``--method-output`` or a generator structure).
    - Optional assets (``--asset-csv`` / ``--asset-root``), rendered evidence
      (``--blender-bin`` + camera-pose mode), and a frozen
      ``--specification-contract`` / ``--reference-annotation``.
    - ``--out-dir`` for all artifacts.

Output:
    Under ``--out-dir``: per-attempt and final ``generated_scene.json`` and
    ``evaluation_report.json``, render manifests, ``run_manifest.json``, and
    ``self_reflexive_history.json``.

Function:
    Runs ``run_generate`` then ``run_evaluate`` per attempt, threads P0b options
    (official mode and collision camera/overlay evidence; the support switch is
    retained only for the legacy Game profile),
    enforces the scene/architecture contract, and records auditable artifacts.
    Camera-pose selection stays an injected, bounded evidence provider.
    Every non-game case follows one canonical L0--L4 evaluator. Prompt
    granularity is descriptive metadata; frozen specification claims activate
    L2 metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.api.evaluation import run_evaluate
from benchmark.api.generation import run_generate
from benchmark.api.submission import (
    evaluate_prepared_submission,
    prepare_submission,
)

from benchmark.architecture_policy import (
    DEFAULT_PHYSICAL_WALL_POLICY,
    PHYSICAL_WALL_POLICIES,
    resolve_architecture_activation,
)
from benchmark.adapters import get_adapter
from benchmark.adapters.defaults import (
    DEFAULT_GENERATION_ADAPTER,
    LEGACY_LAYOUT_REPLAY_ADAPTER,
)
from benchmark.evaluator.profile import (
    build_evaluation_plan,
    is_legacy_game_profile,
    resolve_evaluation_profile,
)
from benchmark.io_contracts import O1_OBJECT_STATE, O3_SCENE_PACKAGE
from benchmark.materialization import NativeRegistryAuthority
from benchmark.assets.generation import load_asset_generation_tool
from benchmark.assets.mode import resolve_asset_mode
from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_object_plan
from benchmark.nl_scene.converter import (
    AUTO_GRANULARITY,
    COARSE_GRAINED,
    FINE_GRAINED,
    PROMPT_GRANULARITIES,
    extract_room_dimension_claims,
)
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.reference_annotation import validate_reference_annotation
from benchmark.scene_io.validate import validate_asset_selection, validate_object_plan, validate_scene_request
from benchmark.rendering import CAMERA_POSE_MODES, CYCLES_DEVICES, RENDER_ENGINES, BlenderRenderer
from benchmark.rendering.camera_pose import (
    parse_metric_camera_modes,
    validate_camera_pose_mode,
    validate_metric_camera_modes,
)
from benchmark.task_contract import (
    require_scene_matches_architecture,
    resolve_room_contract,
)
from benchmark.utils.io import load_yaml, read_json, write_json
from benchmark.grouping import grouping_evidence_from_render_manifest
from benchmark.visual_judge import (
    CameraCandidatePreviewRenderer,
    CameraViewEvidenceRenderer,
    CameraEvidenceProvider,
    DeterministicLocalCameraSelector,
    build_conditional_active_camera_evidence_provider,
    build_openai_compatible_vlm_judge,
    build_openai_compatible_camera_selector,
)
from benchmark.vlm_assistance import budget_for_output

def run_scene_harness(
    *,
    instruction: str,
    scene_type: str,
    out_dir: str | Path,
    room: dict | None = None,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    asset_index_path: str | Path | None = None,
    retrieval_k: int = 1,
    use_vlm_asset_selector: bool = False,
    asset_selector_model_config: dict | None = None,
    asset_generation_tool: Any | None = None,
    asset_mode: str = "off",
    adapter: str | None = None,
    adapter_config: dict | None = None,
    vlm_budget_config: dict | None = None,
    # Retained only to fail old callers with an explicit boundary message.
    converter_model_config: dict | None = None,
    method_output: str | Path | None = None,
    run_generation: bool = False,
    iteration_limit: int = 0,
    case_bundle: Any | str | Path | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: Any | None = None,
    structure: bool | None = None,
    prompt_granularity: str = FINE_GRAINED,
    generator_structure: dict | None = None,
    # Deprecated API alias for generator_structure. It is public generator
    # input, never evaluator ground truth.
    object_plan: dict | None = None,
    reference_annotation: dict | None = None,
    specification_contract: dict | None = None,
    functional_semantic_config: dict | None = None,
    scene_quality_config: dict | None = None,
    object_grouping_report: dict | list | None = None,
    asset_policy: dict | None = None,
    authorized_deviations: list | None = None,
    visual_style_spec: dict | None = None,
    asset_selection: dict | None = None,
    evaluator_output_type: str | None = None,
    # Deprecated compatibility inputs. Canonical L0--L4 routing is owned by the
    # frozen profile and specification contract, not runtime booleans.
    eval_generic_validity: bool = False,
    eval_oor: bool = False,
    eval_oar: bool = False,
    enrich_assets: bool | None = None,
    render_evidence: list[str] | None = None,
    evaluator_vlm_judge: Any | None = None,
    blender_bin: str | Path | None = None,
    blender_timeout_seconds: int = 900,
    render_width: int = 768,
    render_height: int = 768,
    blender_render_engine: str = "BLENDER_EEVEE_NEXT",
    blender_cycles_device: str = "CPU",
    blender_cycles_samples: int = 16,
    blender_cycles_denoising: bool = False,
    evaluation_profile: dict | None = None,
    vlm_evaluation_control: dict[str, Any] | None = None,
    spatial_fidelity_ontology: dict | str | Path | None = None,
    support_enabled: bool = True,
    p0b_official_mode: bool = False,
    p0b_local_view_provider: object | None = None,
    camera_pose_mode: str | None = None,
    camera_pose_metric_modes: dict[str, str] | None = None,
    camera_pose_max_views: int = 2,
    camera_pose_max_steps: int = 1,
    camera_active_fallback: bool = False,
    camera_active_shadow_mode: bool = True,
    camera_active_candidate_count: int = 5,
    camera_active_selector: Any | None = None,
    l3_vlm_camera_selector: Any | None = None,
    collision_pair_overlay: bool = True,
    physical_wall_policy: str = DEFAULT_PHYSICAL_WALL_POLICY,
) -> dict:
    _reject_literal_api_key(asset_selector_model_config, "asset selector config")
    _reject_literal_api_key(adapter_config, "adapter config")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = _request_id(output_dir)
    if generator_structure is not None and object_plan is not None:
        raise ValueError("provide only one of generator_structure or the deprecated object_plan alias")
    public_object_plan = generator_structure if generator_structure is not None else object_plan
    public_structure_provided = public_object_plan is not None
    if converter_model_config:
        raise ValueError(
            "runtime NL conversion is disabled in benchmark runs; use "
            "scripts/author_reference_annotation.py offline, review the draft, and pass "
            "--reference-annotation for scoring"
        )
    requested_granularity = str(prompt_granularity)
    if requested_granularity == AUTO_GRANULARITY:
        raise ValueError(
            "prompt granularity must be frozen in the benchmark case; runtime auto classification is disabled"
        )
    if requested_granularity not in PROMPT_GRANULARITIES:
        raise ValueError(
            f"prompt_granularity must be one of {sorted(PROMPT_GRANULARITIES)}, "
            f"got {requested_granularity!r}"
        )
    resolved_granularity = requested_granularity
    resolved_profile = resolve_evaluation_profile(evaluation_profile)
    legacy_game_profile = is_legacy_game_profile(resolved_profile)
    legacy_game_plan = (
        build_evaluation_plan(
            prompt_granularity=resolved_granularity,
            render_evidence_count=0,
            profile=resolved_profile,
        )
        if legacy_game_profile
        else None
    )
    if spatial_fidelity_ontology is not None and not legacy_game_profile:
        raise ValueError(
            "spatial_fidelity_ontology belongs to the retired non-game workflow; "
            "canonical L2 accepts only specification_contract claims"
        )
    resolved_structure = public_structure_provided if structure is None else bool(structure)
    if resolved_structure and not public_structure_provided:
        raise ValueError(
            "structured generator mode requires --generator-structure; the private "
            "reference annotation is never projected into generator input"
        )
    if not resolved_structure and public_structure_provided:
        raise ValueError(
            "generator_structure is public method input and cannot be supplied with structure=false; "
            "use --reference-annotation for benchmark-private evaluation claims"
        )
    prompt_room_dimensions = extract_room_dimension_claims(instruction)
    resolved_room = resolve_room_contract(room, prompt_dimensions=prompt_room_dimensions)
    scene_request = build_scene_request(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=resolved_room,
        structure=resolved_structure,
        prompt_granularity=resolved_granularity,
    )
    validate_scene_request(scene_request)
    if reference_annotation is not None:
        validate_reference_annotation(reference_annotation)
        annotation_request_id = str(reference_annotation.get("request_id") or "")
        if annotation_request_id != request_id:
            raise ValueError(
                "reference_annotation.request_id must match the harness request_id "
                f"{request_id!r}; got {annotation_request_id!r}"
            )
        if int(iteration_limit) > 0:
            raise ValueError(
                "reference_annotation cannot be combined with self-reflective generation: "
                "the evaluation report contains benchmark-private alignment evidence. "
                "Use iteration_limit=0 until a frozen feedback sanitizer defines a separate refinement track."
            )
    artifacts: dict[str, str | None] = {}

    if public_object_plan is not None:
        public_object_plan = _canonical_object_plan(scene_request, public_object_plan)
        validate_object_plan(public_object_plan)
    resolved_architecture = resolve_architecture_activation(
        resolved_room,
        instruction=instruction,
        specification_contract=specification_contract,
        reference_annotation=reference_annotation,
        object_plan=public_object_plan,
        visual_style_spec=visual_style_spec,
        physical_wall_policy=physical_wall_policy,
        policy_source=(
            "canonical_default"
            if physical_wall_policy == DEFAULT_PHYSICAL_WALL_POLICY
            else "runtime_config"
        ),
    )

    requested_adapter = adapter or DEFAULT_GENERATION_ADAPTER
    skip_only_legacy_compatibility = (
        adapter is None
        and method_output is None
        and not run_generation
        and asset_mode == "off"
    )
    resolved_adapter = (
        LEGACY_LAYOUT_REPLAY_ADAPTER
        if skip_only_legacy_compatibility
        else requested_adapter
    )
    generation_adapter = get_adapter(resolved_adapter)
    trusted_catalog_route = (
        resolved_adapter == DEFAULT_GENERATION_ADAPTER
        and (method_output is not None or run_generation)
    )
    if int(iteration_limit) < 0:
        raise ValueError("iteration_limit must be >= 0")
    if trusted_catalog_route:
        if int(iteration_limit) != 0:
            raise ValueError(
                "catalog_placement trusted submission does not support "
                "self-reflection iterations; set iteration_limit=0"
            )
        if case_bundle is None:
            raise ValueError(
                "active catalog_placement generation requires a trusted case_bundle"
            )
        if asset_root is None or asset_csv is None or blender_bin is None:
            raise ValueError(
                "active catalog_placement generation requires asset_root, asset_csv, "
                "and blender_bin for benchmark-owned preparation"
            )
        if render_evidence:
            raise ValueError(
                "catalog_placement trusted submission does not accept submitted "
                "render_evidence; official evidence is rendered from the sanitized blend"
            )
    resolved_evaluator_output_type = (
        evaluator_output_type
        if evaluator_output_type is not None
        else O3_SCENE_PACKAGE
        if resolved_adapter == DEFAULT_GENERATION_ADAPTER
        else O1_OBJECT_STATE
    )
    if trusted_catalog_route and resolved_evaluator_output_type != O3_SCENE_PACKAGE:
        raise ValueError(
            "active catalog_placement generation requires "
            "evaluator_output_type='o3_scene_package'"
        )
    declared_asset_support = generation_adapter.capabilities.asset_support
    asset_decision = resolve_asset_mode(
        mode=asset_mode,
        adapter_support=declared_asset_support,
        structure=scene_request["structure"],
        source_available=asset_selection is not None or asset_index_path is not None,
        generation_tool_configured=asset_generation_tool is not None,
    )

    if asset_decision.retrieval_enabled:
        if asset_selection is None:
            if not asset_index_path:
                raise ValueError("asset_index_path is required when asset retrieval is enabled without --asset-selection.")
            asset_selection = retrieve_assets_for_object_plan(
                public_object_plan,
                asset_index_path=str(asset_index_path),
                retrieval_k=retrieval_k,
                use_vlm_selector=use_vlm_asset_selector,
                model_config=asset_selector_model_config,
                asset_generation_tool=(asset_generation_tool if asset_decision.generation_enabled else None),
            )
        asset_selection = _canonical_asset_selection(scene_request, public_object_plan, asset_selection)
        validate_asset_selection(asset_selection)
    else:
        asset_selection = None

    generation_input = build_generation_input(
        scene_request=scene_request,
        object_plan=public_object_plan,
        asset_selection=asset_selection,
        evaluator_output_type=resolved_evaluator_output_type,
        architecture_contract=resolved_architecture,
    )
    input_mode = generation_input["generation_contract"]["input_mode"]
    if input_mode not in generation_adapter.capabilities.input_modes:
        raise ValueError(
            f"Adapter {resolved_adapter!r} does not declare support for input mode {input_mode!r}; "
            f"supported modes: {list(generation_adapter.capabilities.input_modes)}"
        )
    io_contract = generation_adapter.resolve_io_contract(generation_input, config=adapter_config)
    resolved_enrich_assets = bool(enrich_assets) if enrich_assets is not None else bool(asset_csv or asset_root)
    resolved_adapter_config = dict(adapter_config or {})
    resolved_vlm_budget = budget_for_output(vlm_budget_config, io_contract.native_output_type)
    resolved_adapter_config["vlm_budget"] = resolved_vlm_budget.as_dict()
    vlm_assistance = generation_adapter.resolve_vlm_assistance(resolved_adapter_config)
    asset_adapter_config = {
        "asset_csv": str(asset_csv) if asset_csv else None,
        "asset_root": str(asset_root) if asset_root else None,
        "enrich_assets": resolved_enrich_assets,
    }
    resolved_adapter_config.update({key: value for key, value in asset_adapter_config.items() if value is not None})
    if legacy_game_profile and not eval_generic_validity and not eval_oor and not eval_oar:
        eval_generic_validity = True
    resolved_camera_metric_modes = validate_metric_camera_modes(camera_pose_metric_modes)
    resolved_camera_pose_mode = validate_camera_pose_mode(camera_pose_mode)
    if (
        resolved_camera_metric_modes or camera_active_fallback
    ) and resolved_camera_pose_mode is None:
        resolved_camera_pose_mode = "auto"
    if resolved_camera_pose_mode is not None and p0b_local_view_provider is not None:
        raise ValueError("camera_pose_mode cannot be combined with a custom p0b_local_view_provider")
    if resolved_camera_pose_mode is not None and blender_bin is None:
        raise ValueError("camera_pose_mode requires blender_bin so local evidence can be rendered")
    if resolved_camera_pose_mode is not None and evaluator_vlm_judge is None:
        raise ValueError(
            "camera_pose_mode requires an evaluator VLM judge; deterministic camera modes avoid only "
            "the pose-selection VLM call, not final P0b adjudication"
        )
    query_cov_requested = resolved_camera_pose_mode == "query_cov" or (
        "query_cov" in resolved_camera_metric_modes.values()
    )
    if camera_active_fallback and query_cov_requested:
        raise ValueError(
            "camera_active_fallback requires a deterministic base camera policy; "
            "do not configure query_cov as the base"
        )
    if (camera_active_fallback or query_cov_requested) and not callable(
        getattr(camera_active_selector, "select_camera_views", None)
    ):
        raise ValueError(
            "VLM-active camera selection requires a separate camera_active_selector; "
            "the final evaluator judge is not reused"
        )
    if (
        camera_active_fallback or query_cov_requested
    ) and camera_active_selector is evaluator_vlm_judge:
        raise ValueError(
            "camera_active_selector and evaluator_vlm_judge must be separate "
            "runtime objects, even when they share one model config"
        )
    if not 1 <= int(camera_pose_max_views) <= 4:
        raise ValueError("camera_pose_max_views must be between 1 and 4")
    if not 0 <= int(camera_pose_max_steps) <= 3:
        raise ValueError("camera_pose_max_steps must be between 0 and 3")
    if camera_active_fallback and not (
        int(camera_pose_max_views)
        <= int(camera_active_candidate_count)
        <= 8
    ):
        raise ValueError(
            "camera_active_candidate_count must be between "
            "camera_pose_max_views and 8"
        )
    scene_renderer = (
        BlenderRenderer(
            blender_bin=blender_bin,
            timeout_seconds=blender_timeout_seconds,
            width=render_width,
            height=render_height,
            render_engine=blender_render_engine,
            cycles_device=blender_cycles_device,
            cycles_samples=blender_cycles_samples,
            cycles_denoising=blender_cycles_denoising,
            require_asset_mesh=asset_decision.retrieval_enabled,
        )
        if blender_bin
        else None
    )

    try:
        loop_result = _run_generation_evaluation_loop(
            generation_input=generation_input,
            adapter=resolved_adapter,
            output_dir=output_dir,
            method_output=method_output,
            adapter_config=resolved_adapter_config,
            run_generation=run_generation,
            iteration_limit=int(iteration_limit),
            eval_generic_validity=(eval_generic_validity if legacy_game_profile else False),
            eval_oor=(eval_oor if legacy_game_profile else False),
            eval_oar=(eval_oar if legacy_game_profile else False),
            asset_csv=asset_csv,
            asset_root=asset_root,
            enrich_assets=resolved_enrich_assets,
            scene_request=scene_request,
            object_plan=public_object_plan,
            reference_annotation=reference_annotation,
            specification_contract=specification_contract,
            functional_semantic_config=functional_semantic_config,
            scene_quality_config=scene_quality_config,
            object_grouping_report=object_grouping_report,
            asset_policy=asset_policy,
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            render_evidence=render_evidence,
            vlm_judge=evaluator_vlm_judge,
            scene_renderer=scene_renderer,
            evaluation_profile=resolved_profile,
            vlm_evaluation_control=vlm_evaluation_control,
            spatial_fidelity_ontology=spatial_fidelity_ontology,
            support_enabled=(support_enabled if legacy_game_profile else None),
            p0b_official_mode=p0b_official_mode,
            p0b_local_view_provider=p0b_local_view_provider,
            camera_pose_mode=resolved_camera_pose_mode,
            camera_pose_metric_modes=resolved_camera_metric_modes,
            camera_pose_max_views=int(camera_pose_max_views),
            camera_pose_max_steps=int(camera_pose_max_steps),
            camera_active_fallback=bool(camera_active_fallback),
            camera_active_shadow_mode=bool(camera_active_shadow_mode),
            camera_active_candidate_count=int(camera_active_candidate_count),
            camera_active_selector=camera_active_selector,
            l3_vlm_camera_selector=l3_vlm_camera_selector,
            collision_pair_overlay=bool(collision_pair_overlay),
            architecture_contract=resolved_architecture,
            trusted_catalog_route=trusted_catalog_route,
            case_bundle=case_bundle,
            native_registry_path=native_registry_path,
            native_registry_authority=native_registry_authority,
            blender_bin=blender_bin,
            blender_timeout_seconds=int(blender_timeout_seconds),
        )
    finally:
        # Benchmark artifacts are persisted only after generator code returns.
        artifacts["scene_request"] = write_json(output_dir / "scene_request.json", scene_request).as_posix()
        artifacts["generator_structure"] = (
            write_json(output_dir / "generator_structure.json", public_object_plan).as_posix()
            if public_object_plan is not None
            else None
        )
        artifacts["reference_annotation"] = (
            write_json(output_dir / "reference_annotation.json", reference_annotation).as_posix()
            if reference_annotation is not None
            else None
        )
        artifacts["specification_contract"] = (
            write_json(
                output_dir / "specification_contract.json",
                specification_contract,
            ).as_posix()
            if specification_contract is not None
            else None
        )
        for artifact_name, payload in (
            ("functional_semantic_config", functional_semantic_config),
            ("scene_quality_config", scene_quality_config),
            ("object_grouping_report", object_grouping_report),
            ("asset_policy", asset_policy),
            ("authorized_deviations", authorized_deviations),
            ("visual_style_spec", visual_style_spec),
        ):
            artifacts[artifact_name] = (
                write_json(output_dir / f"{artifact_name}.json", payload).as_posix()
                if payload is not None
                else None
            )
        artifacts["asset_selection"] = (
            write_json(output_dir / "asset_selection.json", asset_selection).as_posix()
            if asset_selection is not None
            else None
        )
        artifacts["generation_input"] = write_json(output_dir / "generation_input.json", generation_input).as_posix()
    artifacts.update(loop_result["artifacts"])

    if loop_result.get("evaluation_report"):
        evaluation_summary = _evaluation_summary(loop_result["evaluation_report"])
    else:
        evaluation_summary = None

    manifest = {
        "request_id": request_id,
        "status": loop_result["status"],
        "artifacts": artifacts,
        "evaluation_summary": evaluation_summary,
        "self_reflexive": loop_result["self_reflexive"],
        "data_isolation": {
            "generator_workspace": "generator/",
            "benchmark_private_artifacts_written_after_generation": True,
            "adapter_process_is_trusted": True,
            "runtime_converter_called": False,
            "generator_structure_visibility": (
                "public_generator_input" if public_object_plan is not None else "not_provided"
            ),
            "reference_annotation_visibility": (
                "benchmark_private_evaluator_only" if reference_annotation is not None else "not_provided"
            ),
            "reference_annotation_used_for_asset_retrieval": False,
        },
        "asset_resolution": {
            **asset_decision.as_dict(),
            "capability_source": "adapter",
            "retrieval_k": max(1, int(retrieval_k)),
            "selector": "vlm" if use_vlm_asset_selector else "top1",
            "generation_tool_configured": asset_generation_tool is not None,
        },
        "vlm_assistance": vlm_assistance,
        "rendering": {
            "enabled": scene_renderer is not None,
            "backend": (
                "benchmark_owned_sanitized_blend"
                if trusted_catalog_route
                else "blender_canonical_scene_v1"
                if scene_renderer is not None
                else None
            ),
            "blender_bin": str(blender_bin) if blender_bin else None,
            "width": int(render_width),
            "height": int(render_height),
            "render_engine": blender_render_engine,
            "cycles_device": blender_cycles_device,
            "cycles_samples": int(blender_cycles_samples),
            "cycles_denoising": bool(blender_cycles_denoising),
            "require_asset_mesh": asset_decision.retrieval_enabled,
            "submitted_native_blend_rendered_directly": False,
        },
        "evaluation": {
            "profile": resolved_profile,
            "gate": (
                {
                    **legacy_game_plan["gate"],
                    "active_categories": list(legacy_game_plan["categories"]),
                }
                if legacy_game_plan is not None
                else {
                    "workflow": "canonical_l0_l4",
                    "prompt_granularity": resolved_granularity,
                    "prompt_granularity_role": "metadata_only",
                    "activation_source": "canonical_profile_plus_specification_contract",
                    "active_layers": [
                        "l0_structural_validity",
                        "l1_physical_plausibility",
                        "l2_specification_fidelity",
                        "l3_scene_quality",
                        "l4_downstream_task_functionality",
                    ],
                }
            ),
            **(
                {
                    "spatial_fidelity_ontology_configured": (
                        spatial_fidelity_ontology is not None
                    ),
                    "support_enabled": bool(support_enabled),
                }
                if legacy_game_profile
                else {
                    "specification_contract_configured": (
                        specification_contract is not None
                    ),
                    "canonical_configs": {
                        "functional_semantic": functional_semantic_config is not None,
                        "scene_quality": scene_quality_config is not None,
                        "object_grouping_report": object_grouping_report is not None,
                        "asset_policy": asset_policy is not None,
                        "authorized_deviations": authorized_deviations is not None,
                        "visual_style_spec": visual_style_spec is not None,
                    },
                    "l1_applicability_source": "frozen_canonical_profile",
                }
            ),
            "p0b_official_mode": bool(p0b_official_mode),
            "p0b_local_view_provider_configured": (
                p0b_local_view_provider is not None or resolved_camera_pose_mode is not None
            ),
            "camera_pose": {
                "mode": resolved_camera_pose_mode,
                "max_views": int(camera_pose_max_views),
                "max_steps": (
                    int(camera_pose_max_steps)
                    if query_cov_requested or camera_active_fallback
                    else 0
                ),
                "active": resolved_camera_pose_mode is not None,
                "active_fallback": {
                    "enabled": bool(camera_active_fallback),
                    "scope": "legacy_p0b_compatibility",
                    "trigger": (
                        "legacy_p0b_deterministic_preview_sufficiency"
                    ),
                    "selector_decoupled_from_judge": True,
                    "max_views": int(camera_pose_max_views),
                    "max_camera_actions": (
                        int(camera_pose_max_steps) if camera_active_fallback else 0
                    ),
                    "candidate_count": (
                        int(camera_active_candidate_count)
                        if camera_active_fallback
                        else 0
                    ),
                    "shadow_mode": bool(camera_active_shadow_mode),
                },
            },
        },
        "task_contract": {
            "architecture": resolved_architecture,
            "prompt_room_dimensions": prompt_room_dimensions,
        },
        "adapter": {
            "name": generation_adapter.name,
            "requested": requested_adapter,
            "default": DEFAULT_GENERATION_ADAPTER,
            "legacy_skip_only_compatibility": skip_only_legacy_compatibility,
            "trusted_submission_route": (
                "prepare_submission_then_evaluate_prepared_submission"
                if trusted_catalog_route
                else None
            ),
            "capabilities": generation_adapter.capabilities.as_dict(),
            "io_contract": io_contract.as_dict(),
            "generator_output_schema": getattr(generation_adapter, "output_schema", None),
        },
        "prompt_granularity": {
            "requested": requested_granularity,
            "resolved": resolved_granularity,
            **(
                {"evaluation_mode": legacy_game_plan["evaluation_mode"]}
                if legacy_game_plan is not None
                else {"role": "metadata_only"}
            ),
            "classifier_called": False,
            "classification": None,
        },
        "converter": {
            "called": False,
            "runtime_allowed": False,
            "role": "offline_reference_annotation_authoring_only",
            "endpoint": None,
            "model": None,
        },
    }
    artifacts["run_manifest"] = write_json(output_dir / "run_manifest.json", manifest).as_posix()
    manifest["artifacts"] = artifacts
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical adapter-based scene-construction/evaluation harness.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene-type", default="room")
    parser.add_argument(
        "--prompt-granularity",
        choices=[FINE_GRAINED, COARSE_GRAINED],
        default=FINE_GRAINED,
        help="Frozen case metadata. Runtime model-based granularity classification is disabled.",
    )
    parser.add_argument(
        "--room-json",
        default=None,
        help=(
            "Optional explicit room dimensions/boundary. Values are merged with explicit "
            "natural-language dimensions; conflicts fail and missing axes use the benchmark policy."
        ),
    )
    parser.add_argument(
        "--physical-wall-policy",
        choices=PHYSICAL_WALL_POLICIES,
        default=DEFAULT_PHYSICAL_WALL_POLICY,
        help=(
            "Versioned physical-wall policy. explicit_only is the canonical "
            "wall-free default unless benchmark-owned claims activate walls; "
            "always_enclosed is legacy replay compatibility."
        ),
    )
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-index-path", default=None)
    parser.add_argument(
        "--asset-mode",
        choices=["off", "retrieve", "retrieve-generate"],
        default="off",
        help="Explicit benchmark asset route: disabled, retrieval only, or retrieval with generation fallback.",
    )
    parser.add_argument("--retrieval-k", type=int, default=1, help="Number of database candidates sent to the asset selector.")
    parser.add_argument(
        "--asset-selection-strategy",
        choices=["vlm", "top1"],
        default="top1",
        help="Use the top retrieval result directly (default), or opt into VLM candidate decisions. Generation is permitted only in retrieve-generate mode.",
    )
    parser.add_argument("--asset-selector-config", default=None, help="Optional JSON model config for API or localhost VLM selection.")
    parser.add_argument("--asset-selector-endpoint", default=None, help="OpenAI-compatible API/localhost endpoint for the asset selector.")
    parser.add_argument("--asset-selector-model", default=None, help="Served model id for the asset selector.")
    parser.add_argument(
        "--asset-selector-api-key-env",
        default=None,
        help="Environment variable containing the remote asset-selector API key.",
    )
    parser.add_argument(
        "--asset-selector-max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
    )
    parser.add_argument(
        "--asset-selector-send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--asset-generator-plugin",
        default=None,
        help="Optional module:attribute or /path/plugin.py:attribute asset-generation tool.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            f"Generation adapter (default for active generation: "
            f"{DEFAULT_GENERATION_ADAPTER}; generation-skipped legacy harness "
            "calls retain layout_json replay compatibility)."
        ),
    )
    parser.add_argument("--adapter-config", default=None, help="JSON configuration for the selected generation adapter.")
    parser.add_argument(
        "--vlm-budget-config",
        default=str(PROJECT_ROOT / "configs" / "vlm_assistance_budget.yaml"),
        help="YAML hard limits for optional O2/O3 VLM assistance; all defaults are zero.",
    )
    parser.add_argument("--generator-endpoint", default=None, help="OpenAI-compatible endpoint override for generation adapters that call an LLM.")
    parser.add_argument("--generator-model", default=None, help="Served model id override for generation adapters that call an LLM.")
    parser.add_argument(
        "--generator-api-key-env",
        default=None,
        help="Environment variable containing the remote generator API key.",
    )
    parser.add_argument(
        "--generator-max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
        help="Chat Completions output-token field required by the selected model.",
    )
    parser.add_argument(
        "--generator-send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include or omit temperature in generator Chat Completions requests.",
    )
    parser.add_argument(
        "--converter-config",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--converter-endpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--converter-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--method-output",
        default=None,
        help="Preferred generic name for external O1, O2, or O3 native generator output.",
    )
    parser.add_argument(
        "--case-bundle",
        default=None,
        help=(
            "Hash-verified benchmark case bundle required by active "
            "catalog_placement preparation/evaluation."
        ),
    )
    parser.add_argument(
        "--native-registry",
        default=None,
        help=(
            "Benchmark-owned native placement registry required only when the "
            "catalog_placement method output is a registered .blend."
        ),
    )
    parser.add_argument(
        "--native-registry-authority-key-file",
        default=None,
        help=(
            "Benchmark-operator secret key file used to verify the signed "
            "native registry. Never generator-visible."
        ),
    )
    parser.add_argument(
        "--native-registry-authority-key-id",
        default=None,
        help="Trusted key identifier embedded in the signed native registry.",
    )
    parser.add_argument("--run-generation", action="store_true", help="Ask the adapter to run generation when no --method-output is supplied.")
    parser.add_argument("--iteration-limit", type=int, default=0, help="Maximum self-reflexive regeneration attempts after the initial evaluation.")
    parser.add_argument(
        "--structure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Expose an explicitly supplied --generator-structure to an I2 generator. "
            "When omitted, mode is I2 iff that public structure is supplied; otherwise I1."
        ),
    )
    parser.add_argument(
        "--evaluator-output-type",
        choices=[O1_OBJECT_STATE, O3_SCENE_PACKAGE],
        default=None,
        help=(
            "Canonical evaluator boundary after native output. Defaults to O3 for "
            "catalog_placement and O1 for legacy/other adapters."
        ),
    )
    parser.add_argument(
        "--generator-structure",
        "--object-plan",
        dest="generator_structure",
        default=None,
        help=(
            "Public structural input intentionally exposed to an I2 generator. "
            "It is not scoring ground truth; omit it for I1 natural-language-only runs."
        ),
    )
    parser.add_argument(
        "--reference-annotation",
        default=None,
        help="Frozen benchmark-owned reference annotation; never exposed to the generator.",
    )
    parser.add_argument(
        "--specification-contract",
        default=None,
        help=(
            "Frozen benchmark-owned L2 claim contract. Prompt granularity does "
            "not activate or suppress its metric families."
        ),
    )
    parser.add_argument(
        "--functional-semantic-config",
        default=None,
        help="Optional canonical Functional Semantic Fidelity config (JSON or YAML).",
    )
    parser.add_argument(
        "--scene-quality-config",
        default=None,
        help="Optional canonical L3 Scene Quality config (JSON or YAML).",
    )
    parser.add_argument(
        "--object-grouping-report",
        default=None,
        help=(
            "Optional frozen grouping report; otherwise the canonical "
            "VLM visual-evidence-scope grouping backend runs. Topology and "
            "anchor are deprecated replay-only backends."
        ),
    )
    parser.add_argument(
        "--asset-policy",
        default=None,
        help="Optional asset-policy JSON; orthogonal to prompt granularity.",
    )
    parser.add_argument(
        "--authorized-deviations",
        default=None,
        help="Optional JSON list of prompt-authorized deviations.",
    )
    parser.add_argument(
        "--visual-style-spec",
        default=None,
        help="Optional benchmark-owned visual_style_spec_v1 for L3 Style Consistency.",
    )
    parser.add_argument("--asset-selection", default=None)
    parser.add_argument("--eval-generic-validity", action="store_true", help="Legacy Game-profile compatibility only.")
    parser.add_argument("--eval-oor", action="store_true", help="Legacy Game-profile compatibility only.")
    parser.add_argument("--eval-oar", action="store_true", help="Legacy Game-profile compatibility only.")
    parser.add_argument(
        "--support-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy Game-profile compatibility only. Canonical L1 is profile-owned.",
    )
    parser.add_argument(
        "--p0b-official-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail when enabled Collision/OOB/Support adjudication cannot produce a binary VLM verdict. "
            "Leave disabled for detector diagnostics."
        ),
    )
    parser.add_argument(
        "--camera-pose-mode",
        choices=CAMERA_POSE_MODES,
        default=None,
        help=(
            "P0b camera policy: global_only, frozen bbox_track, deterministic visibility_ranked, "
            "Support-specific support_contact_plane, bounded selector-only query_cov, or auto for "
            "frozen per-metric defaults."
        ),
    )
    parser.add_argument(
        "--camera-pose-metric-mode",
        action="append",
        default=[],
        metavar="METRIC=MODE",
        help=(
            "Override one P0b metric camera policy; repeat as needed. Example: "
            "--camera-pose-metric-mode support=query_cov. Overrides cannot use auto."
        ),
    )
    parser.add_argument("--camera-pose-max-views", type=int, default=2)
    parser.add_argument(
        "--camera-pose-max-steps",
        type=int,
        default=1,
        help="Maximum bounded discrete camera adjustments in query_cov mode (0 disables refinement).",
    )
    parser.add_argument(
        "--camera-active-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy P0b compatibility: after its deterministic preview packet, "
            "invoke bounded query_cov when the legacy P0b sufficiency check is "
            "insufficient. Canonical L3 semantic repair is Judge-driven."
        ),
    )
    parser.add_argument(
        "--camera-active-candidate-count",
        type=int,
        default=5,
        help=(
            "Frozen candidate-bank size for bounded active-camera fallback "
            "(default: 5, matching the checked-in selector max_images)."
        ),
    )
    parser.add_argument(
        "--camera-active-shadow-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record conditional active-camera evidence counterfactually without "
            "changing the official deterministic judge packet."
        ),
    )
    parser.add_argument(
        "--camera-selector-config",
        default=None,
        help=(
            "Independent OpenAI-compatible camera-selector JSON config. Required "
            "for query_cov or --camera-active-fallback."
        ),
    )
    parser.add_argument(
        "--vlm-evaluation-control",
        default=None,
        help=(
            "Optional JSON patch selecting the canonical L3 camera-acquisition "
            "policy and bounded Controller budgets."
        ),
    )
    parser.add_argument(
        "--collision-pair-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also render the collision pair overlay in bbox_track mode (A red, B cyan, OBB "
            "wireframes, closest-point markers). visibility_ranked/support_contact_plane/query_cov use highlighted "
            "focus evidence for every P0b metric regardless of this legacy collision switch."
        ),
    )
    parser.add_argument("--enrich-assets", action="store_true", help="Resolve object metadata from --asset-csv/--asset-root before adapter output/evaluation.")
    parser.add_argument("--render-evidence", action="append", default=[], help="Standardized render path; repeat for multiple views.")
    parser.add_argument("--blender-bin", default=None, help="Headless Blender executable. When set, renders each generated canonical scene.")
    parser.add_argument("--blender-timeout-seconds", type=int, default=900)
    parser.add_argument("--render-width", type=int, default=768)
    parser.add_argument("--render-height", type=int, default=768)
    parser.add_argument(
        "--blender-render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_EEVEE_NEXT",
        help="Explicit Blender engine. Use BLENDER_WORKBENCH for proxy diagnostics.",
    )
    parser.add_argument(
        "--blender-cycles-device",
        choices=CYCLES_DEVICES,
        default="CPU",
        help="Cycles compute backend. Explicit CUDA/OPTIX requests fail if unavailable; AUTO may fall back to CPU.",
    )
    parser.add_argument("--blender-cycles-samples", type=int, default=16)
    parser.add_argument(
        "--blender-cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--vlm-judge-config", default=None, help="JSON config for an OpenAI-compatible local or remote VLM judge.")
    parser.add_argument("--vlm-judge-endpoint", default=None)
    parser.add_argument("--vlm-judge-model", default=None)
    parser.add_argument("--vlm-judge-api-key-env", default=None)
    parser.add_argument(
        "--vlm-judge-max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
    )
    parser.add_argument(
        "--vlm-judge-send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--vlm-judge-timeout-seconds", type=int, default=None)
    parser.add_argument("--vlm-judge-max-tokens", type=int, default=None)
    parser.add_argument("--vlm-judge-context-length", type=int, default=None)
    parser.add_argument("--vlm-judge-max-images", type=int, default=None)
    parser.add_argument(
        "--evaluation-profile",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "evaluation"
            / "metric_profile_canonical_v1.yaml"
        ),
    )
    parser.add_argument(
        "--spatial-fidelity-ontology",
        default=None,
        help="Legacy Game-profile compatibility only; rejected by canonical runs.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    try:
        camera_pose_metric_modes = parse_metric_camera_modes(args.camera_pose_metric_mode)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    asset_selector_config = read_json(_path_arg(args.asset_selector_config)) if args.asset_selector_config else {}
    if not isinstance(asset_selector_config, dict):
        parser.error("--asset-selector-config must point to a JSON object")
    try:
        _reject_literal_api_key(asset_selector_config, "asset selector config")
    except ValueError as exc:
        parser.error(str(exc))
    if args.asset_selector_endpoint:
        asset_selector_config["endpoint"] = args.asset_selector_endpoint
    if args.asset_selector_model:
        asset_selector_config["model"] = args.asset_selector_model
    if args.asset_selector_api_key_env:
        asset_selector_config["api_key_env"] = args.asset_selector_api_key_env
    if args.asset_selector_max_tokens_field:
        asset_selector_config["max_tokens_field"] = args.asset_selector_max_tokens_field
    if args.asset_selector_send_temperature is not None:
        asset_selector_config["send_temperature"] = args.asset_selector_send_temperature
    asset_generation_tool = load_asset_generation_tool(args.asset_generator_plugin)
    adapter_config = read_json(_path_arg(args.adapter_config)) if args.adapter_config else {}
    if not isinstance(adapter_config, dict):
        parser.error("--adapter-config must point to a JSON object")
    try:
        _reject_literal_api_key(adapter_config, "adapter config")
    except ValueError as exc:
        parser.error(str(exc))
    if args.generator_endpoint:
        adapter_config["endpoint"] = args.generator_endpoint
    if args.generator_model:
        adapter_config["model"] = args.generator_model
    if args.generator_api_key_env:
        adapter_config["api_key_env"] = args.generator_api_key_env
    if args.generator_max_tokens_field:
        adapter_config["max_tokens_field"] = args.generator_max_tokens_field
    if args.generator_send_temperature is not None:
        adapter_config["send_temperature"] = args.generator_send_temperature
    vlm_budget_config = load_yaml(_path_arg(args.vlm_budget_config), default={})
    if not isinstance(vlm_budget_config, dict):
        parser.error("--vlm-budget-config must point to a YAML object")
    evaluation_profile = load_yaml(_path_arg(args.evaluation_profile), default={})
    if not isinstance(evaluation_profile, dict):
        parser.error("--evaluation-profile must point to a YAML object")
    if args.converter_config or args.converter_endpoint or args.converter_model:
        parser.error(
            "runtime converter options were removed from benchmark execution; create and review "
            "a frozen annotation with scripts/author_reference_annotation.py"
        )
    judge_config = read_json(_path_arg(args.vlm_judge_config)) if args.vlm_judge_config else {}
    if not isinstance(judge_config, dict):
        parser.error("--vlm-judge-config must point to a JSON object")
    try:
        _reject_literal_api_key(judge_config, "VLM judge config")
    except ValueError as exc:
        parser.error(str(exc))
    for key, value in {
        "endpoint": args.vlm_judge_endpoint,
        "model": args.vlm_judge_model,
        "api_key_env": args.vlm_judge_api_key_env,
        "max_tokens_field": args.vlm_judge_max_tokens_field,
        "send_temperature": args.vlm_judge_send_temperature,
        "timeout_seconds": args.vlm_judge_timeout_seconds,
        "max_tokens": args.vlm_judge_max_tokens,
        "context_length": args.vlm_judge_context_length,
        "max_images": args.vlm_judge_max_images,
    }.items():
        if value is not None:
            judge_config[key] = value
    if bool(judge_config.get("endpoint") or judge_config.get("base_url")) != bool(judge_config.get("model") or judge_config.get("model_id")):
        parser.error("VLM judge endpoint and model must be configured together")
    evaluator_vlm_judge = build_openai_compatible_vlm_judge(judge_config) if judge_config else None
    camera_selector_config = (
        read_json(_path_arg(args.camera_selector_config))
        if args.camera_selector_config
        else {}
    )
    if not isinstance(camera_selector_config, dict):
        parser.error("--camera-selector-config must point to a JSON object")
    try:
        _reject_literal_api_key(camera_selector_config, "camera selector config")
    except ValueError as exc:
        parser.error(str(exc))
    camera_active_selector = (
        build_openai_compatible_vlm_judge(camera_selector_config)
        if camera_selector_config
        else None
    )
    l3_vlm_camera_selector = (
        build_openai_compatible_camera_selector(
            camera_selector_config
        )
        if camera_selector_config
        else None
    )
    vlm_evaluation_control = (
        read_json(_path_arg(args.vlm_evaluation_control))
        if args.vlm_evaluation_control
        else None
    )
    if (
        vlm_evaluation_control is not None
        and not isinstance(vlm_evaluation_control, dict)
    ):
        parser.error(
            "--vlm-evaluation-control must point to a JSON object"
        )

    native_registry_authority = None
    native_authority_args = (
        args.native_registry_authority_key_file,
        args.native_registry_authority_key_id,
    )
    if any(native_authority_args):
        if not all(native_authority_args):
            parser.error(
                "--native-registry-authority-key-file and "
                "--native-registry-authority-key-id must be supplied together"
            )
        try:
            native_registry_authority = (
                NativeRegistryAuthority.from_secret(
                    key_id=args.native_registry_authority_key_id,
                    secret=_path_arg(
                        args.native_registry_authority_key_file
                    ).read_bytes(),
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(f"cannot load native registry authority: {exc}")
    if args.native_registry and native_registry_authority is None:
        parser.error(
            "--native-registry requires the benchmark-owned authority key "
            "file and key id"
        )

    manifest = run_scene_harness(
        instruction=args.instruction,
        scene_type=args.scene_type,
        room=read_json(_path_arg(args.room_json)) if args.room_json else None,
        asset_csv=_path_arg(args.asset_csv) if args.asset_csv else None,
        asset_root=_path_arg(args.asset_root) if args.asset_root else None,
        asset_index_path=_path_arg(args.asset_index_path) if args.asset_index_path else None,
        retrieval_k=args.retrieval_k,
        use_vlm_asset_selector=args.asset_selection_strategy == "vlm",
        asset_selector_model_config=asset_selector_config or None,
        asset_generation_tool=asset_generation_tool,
        asset_mode=args.asset_mode,
        adapter=args.adapter,
        adapter_config=adapter_config or None,
        vlm_budget_config=vlm_budget_config,
        method_output=_path_arg(args.method_output) if args.method_output else None,
        case_bundle=_path_arg(args.case_bundle) if args.case_bundle else None,
        native_registry_path=(
            _path_arg(args.native_registry) if args.native_registry else None
        ),
        native_registry_authority=native_registry_authority,
        run_generation=args.run_generation,
        iteration_limit=args.iteration_limit,
        structure=args.structure,
        prompt_granularity=args.prompt_granularity,
        generator_structure=(
            read_json(_path_arg(args.generator_structure))
            if args.generator_structure
            else None
        ),
        reference_annotation=(
            read_json(_path_arg(args.reference_annotation))
            if args.reference_annotation
            else None
        ),
        specification_contract=(
            read_json(_path_arg(args.specification_contract))
            if args.specification_contract
            else None
        ),
        functional_semantic_config=(
            load_yaml(_path_arg(args.functional_semantic_config), default={})
            if args.functional_semantic_config
            else None
        ),
        scene_quality_config=(
            load_yaml(_path_arg(args.scene_quality_config), default={})
            if args.scene_quality_config
            else None
        ),
        object_grouping_report=(
            read_json(_path_arg(args.object_grouping_report))
            if args.object_grouping_report
            else None
        ),
        asset_policy=(
            read_json(_path_arg(args.asset_policy)) if args.asset_policy else None
        ),
        authorized_deviations=(
            read_json(_path_arg(args.authorized_deviations))
            if args.authorized_deviations
            else None
        ),
        visual_style_spec=(
            read_json(_path_arg(args.visual_style_spec))
            if args.visual_style_spec
            else None
        ),
        asset_selection=read_json(_path_arg(args.asset_selection)) if args.asset_selection else None,
        evaluator_output_type=args.evaluator_output_type,
        eval_generic_validity=args.eval_generic_validity,
        eval_oor=args.eval_oor,
        eval_oar=args.eval_oar,
        enrich_assets=args.enrich_assets if args.enrich_assets else None,
        render_evidence=[str(_path_arg(path)) for path in args.render_evidence],
        evaluator_vlm_judge=evaluator_vlm_judge,
        blender_bin=_path_arg(args.blender_bin) if args.blender_bin else None,
        blender_timeout_seconds=args.blender_timeout_seconds,
        render_width=args.render_width,
        render_height=args.render_height,
        blender_render_engine=args.blender_render_engine,
        blender_cycles_device=args.blender_cycles_device,
        blender_cycles_samples=args.blender_cycles_samples,
        blender_cycles_denoising=args.blender_cycles_denoising,
        evaluation_profile=evaluation_profile,
        vlm_evaluation_control=vlm_evaluation_control,
        spatial_fidelity_ontology=(
            _path_arg(args.spatial_fidelity_ontology)
            if args.spatial_fidelity_ontology
            else None
        ),
        support_enabled=args.support_enabled,
        p0b_official_mode=args.p0b_official_mode,
        camera_pose_mode=args.camera_pose_mode,
        camera_pose_metric_modes=camera_pose_metric_modes,
        camera_pose_max_views=args.camera_pose_max_views,
        camera_pose_max_steps=args.camera_pose_max_steps,
        camera_active_fallback=args.camera_active_fallback,
        camera_active_shadow_mode=args.camera_active_shadow_mode,
        camera_active_candidate_count=args.camera_active_candidate_count,
        camera_active_selector=camera_active_selector,
        l3_vlm_camera_selector=l3_vlm_camera_selector,
        collision_pair_overlay=args.collision_pair_overlay,
        physical_wall_policy=args.physical_wall_policy,
        out_dir=_path_arg(args.out_dir),
    )
    print(f"status: {manifest['status']}")
    print(f"run_manifest: {manifest['artifacts']['run_manifest']}")


def _run_generation_evaluation_loop(
    *,
    generation_input: dict,
    adapter: str,
    output_dir: Path,
    method_output: str | Path | None,
    adapter_config: dict,
    run_generation: bool,
    iteration_limit: int,
    eval_generic_validity: bool,
    eval_oor: bool,
    eval_oar: bool,
    asset_csv: str | Path | None,
    asset_root: str | Path | None,
    enrich_assets: bool,
    scene_request: dict,
    object_plan: dict | None,
    reference_annotation: dict | None,
    specification_contract: dict | None,
    functional_semantic_config: dict | None,
    scene_quality_config: dict | None,
    object_grouping_report: dict | list | None,
    asset_policy: dict | None,
    authorized_deviations: list | None,
    visual_style_spec: dict | None,
    render_evidence: list[str] | None,
    vlm_judge: Any | None,
    scene_renderer: Any | None,
    evaluation_profile: dict | None,
    vlm_evaluation_control: dict[str, Any] | None,
    spatial_fidelity_ontology: dict | str | Path | None,
    support_enabled: bool | None = None,
    p0b_official_mode: bool = False,
    p0b_local_view_provider: object | None = None,
    camera_pose_mode: str | None = None,
    camera_pose_metric_modes: dict[str, str] | None = None,
    camera_pose_max_views: int = 2,
    camera_pose_max_steps: int = 1,
    camera_active_fallback: bool = False,
    camera_active_shadow_mode: bool = True,
    camera_active_candidate_count: int = 5,
    camera_active_selector: Any | None = None,
    l3_vlm_camera_selector: Any | None = None,
    collision_pair_overlay: bool = True,
    architecture_contract: dict | None = None,
    trusted_catalog_route: bool = False,
    case_bundle: Any | str | Path | None = None,
    native_registry_path: str | Path | None = None,
    native_registry_authority: Any | None = None,
    blender_bin: str | Path | None = None,
    blender_timeout_seconds: int = 900,
) -> dict:
    attempts: list[dict[str, Any]] = []
    previous_report: dict | None = None
    previous_scene: dict | None = None
    final_evaluated_attempt: dict[str, Any] | None = None

    for iteration in range(iteration_limit + 1):
        attempt_dir = output_dir if iteration == 0 else output_dir / "iterations" / f"iter_{iteration:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        generate_result = run_generate(
            generation_input=generation_input,
            adapter_name=adapter,
            out_dir=attempt_dir,
            method_output=method_output if iteration == 0 else None,
            adapter_config=adapter_config,
            run_generation=run_generation,
            evaluation_report=previous_report,
            previous_generated_scene=previous_scene,
            iteration=iteration if previous_report is not None else None,
            materialize_native_output=not trusted_catalog_route,
        )
        attempt_record: dict[str, Any] = {
            "iteration": iteration,
            "out_dir": attempt_dir.as_posix(),
            "status": generate_result["status"]["status"],
            "method_input": generate_result.get("method_input"),
            "workflow_status": generate_result.get("workflow_status"),
            "adapter_metadata": generate_result.get("adapter_metadata"),
            "native_output": generate_result.get("native_output"),
            "raw_native_artifact": generate_result.get("raw_native_artifact"),
            "generated_scene": generate_result.get("generated_scene"),
            "evaluation_report": None,
            "valid": None,
        }

        if trusted_catalog_route:
            raw_native_artifact = generate_result.get("raw_native_artifact")
            if not raw_native_artifact:
                attempts.append(attempt_record)
                break
            if (
                case_bundle is None
                or asset_root is None
                or asset_csv is None
                or blender_bin is None
                or scene_renderer is None
            ):
                raise RuntimeError(
                    "trusted catalog route requires case_bundle, asset_root, "
                    "asset_csv, blender_bin, and a prepared-scene renderer"
                )
            prepared = prepare_submission(
                artifact=raw_native_artifact,
                case_bundle=case_bundle,
                out_dir=attempt_dir / "preparation",
                asset_root=asset_root,
                asset_csv=asset_csv,
                blender_bin=blender_bin,
                generation_input=generation_input,
                native_registry_path=native_registry_path,
                native_registry_authority=native_registry_authority,
                timeout_seconds=blender_timeout_seconds,
            )
            report = evaluate_prepared_submission(
                prepared_submission=prepared,
                case_bundle=case_bundle,
                out_dir=attempt_dir,
                renderer=scene_renderer,
                vlm_judge=vlm_judge,
                camera_selector=(
                    l3_vlm_camera_selector
                    if l3_vlm_camera_selector is not None
                    else camera_active_selector
                ),
                asset_root=asset_root,
                asset_csv=asset_csv,
                blender_bin=blender_bin,
                generation_input=generation_input,
                native_registry_path=native_registry_path,
                native_registry_authority=native_registry_authority,
                official_mode=True,
                vlm_evaluation_control=vlm_evaluation_control,
            )
            trusted_scene_path = (
                prepared.normalized_scene_path.as_posix()
                if prepared.normalized_scene_path.is_file()
                else None
            )
            evaluation_report_path = attempt_dir / "evaluation_report.json"
            if not evaluation_report_path.is_file():
                write_json(evaluation_report_path, report)
            submission_run_manifest_path = (
                attempt_dir / "submission_run_manifest.json"
            )
            submission_run_manifest = (
                read_json(submission_run_manifest_path)
                if submission_run_manifest_path.is_file()
                else {}
            )
            trusted_rendering = (
                submission_run_manifest.get("rendering")
                if isinstance(submission_run_manifest, dict)
                and isinstance(
                    submission_run_manifest.get("rendering"),
                    dict,
                )
                else {}
            )
            trusted_overview_views = trusted_rendering.get(
                "overview_views"
            )
            trusted_overview_views = (
                [
                    str(path)
                    for path in trusted_overview_views
                    if str(path)
                ]
                if isinstance(trusted_overview_views, list)
                else []
            )
            attempt_record.update(
                {
                    "status": _prepared_evaluation_attempt_status(report),
                    "generated_scene": trusted_scene_path,
                    "evaluation_report": evaluation_report_path.as_posix(),
                    "benchmark_score": report.get("benchmark_score"),
                    "valid": _evaluation_is_valid(report),
                    "preparation": prepared.as_dict(),
                    "submission_run_manifest": (
                        submission_run_manifest_path.as_posix()
                        if submission_run_manifest_path.is_file()
                        else None
                    ),
                    "render_input_policy": str(
                        trusted_rendering.get("input_policy")
                        or "benchmark_owned_sanitized_blend"
                    ),
                    "render_manifest": trusted_rendering.get(
                        "manifest_path"
                    ),
                    "render_manifest_sha256": trusted_rendering.get(
                        "manifest_sha256"
                    ),
                    "render_evidence": trusted_overview_views,
                    "evaluation_render_authority": (
                        dict(
                            submission_run_manifest.get(
                                "evaluation_render_authority"
                            )
                        )
                        if isinstance(
                            submission_run_manifest.get(
                                "evaluation_render_authority"
                            ),
                            dict,
                        )
                        else None
                    ),
                }
            )
            attempts.append(attempt_record)
            final_evaluated_attempt = attempt_record
            break

        if not generate_result.get("generated_scene"):
            attempts.append(attempt_record)
            break

        previous_scene = read_json(generate_result["generated_scene"])
        require_scene_matches_architecture(previous_scene, scene_request["room"])
        attempt_render_evidence = list(render_evidence or [])
        attempt_grouping_visual_evidence: list[dict[str, Any]] | None = None
        if scene_renderer is not None:
            render_dir = attempt_dir / "renders"
            render_manifest = scene_renderer.render_scene(
                scene_path=generate_result["generated_scene"],
                out_dir=render_dir,
                asset_root=asset_root,
            )
            attempt_render_evidence = [
                str(item["path"])
                for item in render_manifest.get("views", [])
                if (
                    isinstance(item, dict)
                    and item.get("path")
                    and str(item.get("name") or "") != "identity_map"
                )
            ]
            attempt_grouping_visual_evidence = (
                grouping_evidence_from_render_manifest(
                    render_manifest
                )
            )
            attempt_record["render_manifest"] = (render_dir / "render_manifest.json").as_posix()
            attempt_record["render_evidence"] = attempt_render_evidence
            collision_geometry = render_manifest.get("collision_geometry")
        else:
            collision_geometry = None
        attempt_local_view_provider = p0b_local_view_provider
        attempt_l3_initial_evidence_provider = None
        attempt_deterministic_camera_selector = None
        attempt_vlm_camera_selector = None
        attempt_evidence_renderer = None
        attempt_candidate_preview_renderer = None
        blend_file = (
            render_manifest.get("blend_file")
            if scene_renderer is not None
            else None
        )
        if camera_pose_mode is not None:
            if (
                not isinstance(blend_file, str)
                or not Path(blend_file).is_file()
            ):
                raise RuntimeError(
                    "camera pose mode requires the Blender render manifest "
                    "to contain scene.blend"
                )
            if camera_active_fallback:
                attempt_local_view_provider = build_conditional_active_camera_evidence_provider(
                    renderer=scene_renderer,
                    blend_file=blend_file,
                    out_dir=attempt_dir / "camera_evidence",
                    deterministic_mode=camera_pose_mode,
                    metric_modes=camera_pose_metric_modes,
                    selector=camera_active_selector,
                    max_views=camera_pose_max_views,
                    max_steps=camera_pose_max_steps,
                    candidate_count=camera_active_candidate_count,
                    collision_overlay=collision_pair_overlay,
                    collision_contour=collision_pair_overlay,
                    collision_geometry=(
                        collision_geometry
                        if isinstance(collision_geometry, dict)
                        else None
                    ),
                    fail_on_exhausted=True,
                    shadow_mode=camera_active_shadow_mode,
                    architecture_contract=architecture_contract,
                )
            else:
                attempt_local_view_provider = CameraEvidenceProvider(
                    renderer=scene_renderer,
                    blend_file=blend_file,
                    out_dir=attempt_dir / "camera_evidence",
                    mode=camera_pose_mode,
                    metric_modes=camera_pose_metric_modes,
                    selector=camera_active_selector,
                    max_views=camera_pose_max_views,
                    max_steps=camera_pose_max_steps,
                    collision_overlay=collision_pair_overlay,
                    collision_contour=collision_pair_overlay,
                    collision_geometry=collision_geometry if isinstance(collision_geometry, dict) else None,
                    architecture_contract=architecture_contract,
                )
            attempt_record["camera_evidence_policy"] = attempt_local_view_provider.policy_config
        l3_camera_components_requested = (
            not is_legacy_game_profile(evaluation_profile)
            and scene_renderer is not None
            and (
                camera_pose_mode is not None
                or vlm_evaluation_control is not None
                or l3_vlm_camera_selector is not None
            )
        )
        if l3_camera_components_requested:
            if (
                not isinstance(blend_file, str)
                or not Path(blend_file).is_file()
            ):
                raise RuntimeError(
                    "canonical L3 camera acquisition requires the Blender "
                    "render manifest to contain scene.blend"
                )
            # Canonical L3 receives frozen deterministic initial evidence.
            # Repair is owned independently by the Controller; the P0b
            # provider above retains its historical compatibility path.
            attempt_l3_initial_evidence_provider = CameraEvidenceProvider(
                renderer=scene_renderer,
                blend_file=blend_file,
                out_dir=(
                    attempt_dir / "camera_evidence" / "l3_initial"
                ),
                mode="visibility_ranked",
                metric_modes={},
                selector=None,
                max_views=camera_pose_max_views,
                max_steps=camera_pose_max_steps,
                candidate_count=(
                    camera_active_candidate_count
                    if camera_active_fallback
                    else max(camera_pose_max_views, 6)
                ),
                collision_overlay=False,
                collision_contour=False,
                collision_geometry=(
                    collision_geometry
                    if isinstance(collision_geometry, dict)
                    else None
                ),
                active_repair=False,
                architecture_contract=architecture_contract,
            )
            attempt_deterministic_camera_selector = (
                DeterministicLocalCameraSelector(
                    candidate_policy=(
                        attempt_l3_initial_evidence_provider.candidate_policy
                    )
                )
            )
            attempt_vlm_camera_selector = (
                l3_vlm_camera_selector
                if l3_vlm_camera_selector is not None
                else camera_active_selector
            )
            attempt_evidence_renderer = CameraViewEvidenceRenderer(
                renderer=scene_renderer,
                blend_file=blend_file,
                out_dir=(
                    attempt_dir / "camera_evidence" / "l3_controller"
                ),
            )
            attempt_candidate_preview_renderer = (
                CameraCandidatePreviewRenderer(
                    renderer=scene_renderer,
                    blend_file=blend_file,
                    out_dir=(
                        attempt_dir
                        / "camera_evidence"
                        / "l3_controller"
                    ),
                )
            )
            initial_camera_source = (
                "config"
                if (
                    isinstance(vlm_evaluation_control, dict)
                    and isinstance(
                        vlm_evaluation_control.get(
                            "initial_group_camera"
                        ),
                        dict,
                    )
                )
                else "default"
            )
            attempt_record["l3_camera_control"] = {
                "mode": "judge_driven_independent_components",
                "deterministic_selector": (
                    type(
                        attempt_deterministic_camera_selector
                    ).__name__
                ),
                "vlm_selector_configured": (
                    attempt_vlm_camera_selector is not None
                ),
                "renderer": type(
                    attempt_evidence_renderer
                ).__name__,
                "scene_access": "read_only",
                "initial_group_camera": {
                    "mode": "visibility_ranked",
                    "selector": "deterministic",
                    "source": initial_camera_source,
                },
            }
        report = run_evaluate(
            scene=previous_scene,
            out=attempt_dir / "evaluation_report.json",
            eval_generic_validity=eval_generic_validity,
            eval_oor=eval_oor,
            eval_oar=eval_oar,
            asset_csv=asset_csv,
            asset_root=asset_root,
            enrich_assets=enrich_assets,
            scene_request=scene_request,
            object_plan=object_plan,
            reference_annotation=reference_annotation,
            specification_contract=specification_contract,
            functional_semantic_config=functional_semantic_config,
            scene_quality_config=scene_quality_config,
            object_grouping_report=object_grouping_report,
            asset_policy=asset_policy,
            authorized_deviations=authorized_deviations,
            visual_style_spec=visual_style_spec,
            collision_geometry=collision_geometry if isinstance(collision_geometry, dict) else None,
            render_evidence=attempt_render_evidence,
            **(
                {
                    "grouping_visual_evidence": (
                        attempt_grouping_visual_evidence
                    ),
                    "l3_initial_evidence_provider": (
                        attempt_l3_initial_evidence_provider
                    ),
                    "deterministic_camera_selector": (
                        attempt_deterministic_camera_selector
                    ),
                    "vlm_camera_selector": (
                        attempt_vlm_camera_selector
                    ),
                    "evidence_renderer": (
                        attempt_evidence_renderer
                    ),
                    "candidate_preview_renderer": (
                        attempt_candidate_preview_renderer
                    ),
                }
                if not is_legacy_game_profile(evaluation_profile)
                else {}
            ),
            vlm_judge=vlm_judge,
            evaluation_profile=evaluation_profile,
            **(
                {
                    "vlm_evaluation_control": (
                        vlm_evaluation_control
                    )
                }
                if not is_legacy_game_profile(evaluation_profile)
                else {}
            ),
            support_enabled=support_enabled,
            p0b_official_mode=p0b_official_mode,
            p0b_local_view_provider=attempt_local_view_provider,
            spatial_fidelity_ontology=spatial_fidelity_ontology,
        )
        attempt_record["evaluation_report"] = (attempt_dir / "evaluation_report.json").as_posix()
        attempt_record["benchmark_score"] = report.get("benchmark_score")
        attempt_record["valid"] = _evaluation_is_valid(report)
        attempts.append(attempt_record)
        final_evaluated_attempt = attempt_record

        if attempt_record["valid"] is True:
            break
        if iteration >= iteration_limit:
            break
        previous_report = report

    latest_attempt = attempts[-1] if attempts else {}
    final_attempt = final_evaluated_attempt or latest_attempt
    final_scene_path = _publish_json_artifact(final_attempt.get("generated_scene"), output_dir / "generated_scene.json")
    final_report_path = _publish_json_artifact(final_attempt.get("evaluation_report"), output_dir / "evaluation_report.json")
    history = {
        "iteration_limit": iteration_limit,
        "final_iteration": final_attempt.get("iteration"),
        "valid": final_attempt.get("valid"),
        "attempts": attempts,
    }
    history_path = write_json(output_dir / "self_reflexive_history.json", history)
    return {
        "status": _loop_status(attempts, final_attempt, iteration_limit),
        "evaluation_report": read_json(final_report_path) if final_report_path else None,
        "artifacts": {
            "method_input": latest_attempt.get("method_input"),
            "native_output": latest_attempt.get("native_output"),
            "raw_native_artifact": latest_attempt.get("raw_native_artifact"),
            "generated_scene": final_scene_path,
            "workflow_status": latest_attempt.get("workflow_status"),
            "adapter_metadata": latest_attempt.get("adapter_metadata"),
            "evaluation_report": final_report_path,
            "render_manifest": final_attempt.get("render_manifest"),
            "render_evidence": final_attempt.get("render_evidence"),
            "preparation": final_attempt.get("preparation"),
            "submission_run_manifest": final_attempt.get(
                "submission_run_manifest"
            ),
            "self_reflexive_history": history_path.as_posix(),
        },
        "self_reflexive": {
            "enabled": iteration_limit > 0,
            "iteration_limit": iteration_limit,
            "iterations_run": max(0, len(attempts) - 1),
            "final_iteration": final_attempt.get("iteration"),
            "valid": final_attempt.get("valid"),
            "attempts": attempts,
        },
    }


def _publish_json_artifact(source: str | None, destination: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if source_path.resolve() != destination.resolve():
        write_json(destination, read_json(source_path))
    return destination.as_posix()


def _loop_status(attempts: list[dict[str, Any]], final_attempt: dict[str, Any], iteration_limit: int) -> str:
    if not attempts:
        return "not_started"
    if final_attempt.get("valid") is True:
        return "valid_scene_available" if iteration_limit > 0 else attempts[-1].get("status", "generated_scene_available")
    latest_attempt = attempts[-1]
    if latest_attempt.get("status") == "generation_skipped" and int(latest_attempt.get("iteration", 0)) > 0:
        return "reflection_generation_pending"
    if final_attempt.get("valid") is False and int(final_attempt.get("iteration", 0)) >= iteration_limit and iteration_limit > 0:
        return "iteration_limit_exhausted"
    return latest_attempt.get("status", "unknown")


def _evaluation_is_valid(report: dict) -> bool:
    for key in ["overall_valid", "valid"]:
        if isinstance(report.get(key), bool):
            return bool(report[key])
    try:
        score = report.get("benchmark_score")
        return isinstance(score, (int, float)) and not isinstance(score, bool) and float(score) >= 0.999
    except (TypeError, ValueError):
        return False


def _prepared_evaluation_attempt_status(report: dict[str, Any]) -> str:
    evaluation_status = str(report.get("evaluation_status") or "").strip()
    score_status = str(
        report.get("benchmark_score_status") or ""
    ).strip()
    if (
        evaluation_status == "not_evaluable"
        or score_status == "not_evaluable"
    ):
        return "not_evaluable"
    score = report.get("benchmark_score")
    if (
        evaluation_status == "complete"
        and score_status == "complete"
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
    ):
        return "prepared_evaluation_complete"
    return "prepared_evaluation_incomplete"


def _evaluation_summary(report: dict) -> dict:
    return {
        "benchmark_score": report.get("benchmark_score"),
        "benchmark_score_status": report.get("benchmark_score_status"),
        "prompt_granularity": report.get("prompt_granularity"),
        "workflow": report.get("workflow"),
        "active_layers": sorted((report.get("layer_reports") or {}).keys()),
        "coverage": report.get("coverage"),
        "valid": _evaluation_is_valid(report),
        "reports": sorted((report.get("reports") or {}).keys()),
    }


def _canonical_object_plan(scene_request: dict, object_plan: dict) -> dict:
    if not isinstance(object_plan, dict):
        raise ValueError("object_plan must be a JSON object")
    objects = []
    for index, obj in enumerate(object_plan.get("objects", []) if isinstance(object_plan.get("objects"), list) else []):
        if not isinstance(obj, dict):
            continue
        placement_intent = obj.get("placement_intent") if isinstance(obj.get("placement_intent"), dict) else {}
        record = {
            "id": str(obj.get("id") or f"obj_{index:03d}"),
            "role": str(obj.get("role") or ""),
            "category": str(obj.get("category") or "object"),
            "description": str(obj.get("description") or obj.get("category") or "object"),
            "count": int(obj.get("count") or 1),
            "placement_intent": {
                "absolute_relations": placement_intent.get("absolute_relations") if isinstance(placement_intent.get("absolute_relations"), list) else [],
                "relative_relations": placement_intent.get("relative_relations") if isinstance(placement_intent.get("relative_relations"), list) else [],
            },
            "metadata": obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
        }
        if obj.get("estimated_size") is not None:
            record["estimated_size"] = obj.get("estimated_size")
        objects.append(record)
    return {
        "request_id": scene_request["request_id"],
        "scene_type": object_plan.get("scene_type") or scene_request.get("scene_type"),
        "scene_description": object_plan.get("scene_description") or scene_request.get("instruction"),
        "prompt_granularity": object_plan.get("prompt_granularity") or scene_request.get("prompt_granularity") or FINE_GRAINED,
        "explicit_claims": object_plan.get("explicit_claims") if isinstance(object_plan.get("explicit_claims"), list) else [],
        "objects": objects,
        "global_constraints": object_plan.get("global_constraints") if isinstance(object_plan.get("global_constraints"), list) else [],
        "relations": object_plan.get("relations") if isinstance(object_plan.get("relations"), list) else [],
    }


def _canonical_asset_selection(scene_request: dict, object_plan: dict, asset_selection: dict) -> dict:
    if not isinstance(asset_selection, dict):
        raise ValueError("asset_selection must be a JSON object")
    object_specs = {str(obj.get("id")): obj for obj in object_plan.get("objects", []) if isinstance(obj, dict)}
    objects = []
    for index, item in enumerate(asset_selection.get("objects", []) if isinstance(asset_selection.get("objects"), list) else []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or item.get("id") or f"obj_{index:03d}")
        provided_spec = item.get("object_spec") if isinstance(item.get("object_spec"), dict) else {}
        object_spec = {**object_specs.get(object_id, {}), **provided_spec}
        selected = item.get("selected_asset") if isinstance(item.get("selected_asset"), dict) else {}
        selection_action = str(item.get("selection_action") or "select")
        selection_reason = str(item.get("selection_reason") or "provided or top retrieval result")
        selection_decision = item.get("selection_decision") if isinstance(item.get("selection_decision"), dict) else {}
        selection_decision = dict(selection_decision)
        selection_decision.setdefault("action", selection_action)
        selection_decision.setdefault("selected_jid", selected.get("jid") or selected.get("asset_id") or selected.get("id"))
        selection_decision.setdefault("reason", selection_reason)
        selection_decision.setdefault("generation_request", None)
        objects.append(
            {
                "object_id": object_id,
                "object_spec": {
                    "role": object_spec.get("role"),
                    "category": object_spec.get("category"),
                    "description": object_spec.get("description"),
                    "estimated_size": object_spec.get("estimated_size"),
                    "count": object_spec.get("count", 1),
                },
                "retrieval_query": item.get("retrieval_query")
                if isinstance(item.get("retrieval_query"), dict)
                else {
                    "description": object_spec.get("description"),
                    "category": object_spec.get("category"),
                    "size_constraint": object_spec.get("estimated_size"),
                },
                "selected_asset": _canonical_selected_asset(selected),
                "candidates": item.get("candidates") if isinstance(item.get("candidates"), list) else [],
                "selection_action": selection_action,
                "selection_decision": selection_decision,
                "selection_reason": selection_reason,
            }
        )
    return {"request_id": scene_request["request_id"], "objects": objects}


def _canonical_selected_asset(asset: dict[str, Any]) -> dict[str, Any]:
    jid = asset.get("jid") or asset.get("asset_id") or asset.get("id")
    size = asset.get("size") or asset.get("dimensions")
    asset_ref = asset.get("asset_ref") if isinstance(asset.get("asset_ref"), dict) else {}
    asset_ref = dict(asset_ref)
    asset_ref.setdefault("source_db", asset_ref.pop("source", "imaginarium"))
    asset_ref.setdefault("asset_key", jid)
    asset_ref.setdefault("mesh_uri", asset.get("mesh_uri"))
    asset_ref.setdefault("pointcloud_uri", asset.get("pointcloud_uri"))
    asset_ref.setdefault("metadata_uri", asset.get("metadata_uri"))
    asset_proxy = asset.get("asset_proxy") if isinstance(asset.get("asset_proxy"), dict) else {
        "type": "obb_from_metadata_or_csv",
        "bbox_center_local": [0, 0, 0],
        "bbox_size": size,
    }
    metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.setdefault("interactive", False)
    metadata.setdefault("inner_placement", False)
    metadata.setdefault("align_to_wall_normal", False)
    metadata.setdefault("scaling_strategy", None)
    return {
        "jid": jid,
        "category": asset.get("category") or "",
        "retrieval_category": asset.get("retrieval_category") or asset.get("category") or "",
        "desc": asset.get("desc") or asset.get("description") or asset.get("short_desc") or "",
        "short_desc": asset.get("short_desc") or asset.get("description") or "",
        "size": size,
        "asset_ref": asset_ref,
        "asset_proxy": asset_proxy,
        "metadata": metadata,
    }


def _request_id(out_dir: Path) -> str:
    return out_dir.name or "scene_request"


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _reject_literal_api_key(config: dict | None, label: str) -> None:
    if isinstance(config, dict) and "api_key" in config:
        raise ValueError(f"{label} must not contain literal api_key; use api_key_env instead")


if __name__ == "__main__":
    main()
