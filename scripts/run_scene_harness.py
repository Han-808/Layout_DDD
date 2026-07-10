from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import run_evaluate
from generate import run_generate

from benchmark.adapters import get_adapter
from benchmark.assets.generation import load_asset_generation_tool
from benchmark.assets.mode import resolve_asset_mode
from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_object_plan
from benchmark.nl_scene.converter import convert_nl_to_object_plan
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.scene_io.validate import validate_asset_selection, validate_object_plan, validate_scene_request
from benchmark.utils.io import read_json, write_json


# Rounded per-scene means from local Scenes: 5.269m x 5.290m x 2.865m.
DEFAULT_ROOM = {"boundary": [[0, 0], [5, 0], [5, 5], [0, 5]], "height": 2.9, "unit": "meter"}


def run_scene_harness(
    *,
    instruction: str,
    scene_type: str,
    out_dir: str | Path,
    room: dict | None = None,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    asset_index_path: str | Path | None = None,
    retrieval_k: int = 5,
    use_vlm_asset_selector: bool = False,
    asset_selector_model_config: dict | None = None,
    asset_generation_tool: Any | None = None,
    asset_mode: str = "off",
    adapter: str = "passthrough",
    adapter_config: dict | None = None,
    converter_model_config: dict | None = None,
    generated_scene: str | Path | None = None,
    run_generation: bool = False,
    iteration_limit: int = 0,
    structure: bool = True,
    object_plan: dict | None = None,
    asset_selection: dict | None = None,
    eval_generic_validity: bool = False,
    eval_oor: bool = False,
    eval_oar: bool = False,
    enrich_assets: bool | None = None,
) -> dict:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = _request_id(output_dir)
    scene_request = build_scene_request(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=room or DEFAULT_ROOM,
        structure=structure,
    )
    validate_scene_request(scene_request)
    artifacts: dict[str, str | None] = {}
    artifacts["scene_request"] = write_json(output_dir / "scene_request.json", scene_request).as_posix()

    object_plan_provided = object_plan is not None
    resolved_converter_config = _resolve_converter_model_config(converter_model_config, adapter_config)
    if object_plan is None:
        object_plan = convert_nl_to_object_plan(
            instruction,
            request_id=request_id,
            scene_type=scene_type,
            room=scene_request["room"],
            model_config=resolved_converter_config or None,
        )
    object_plan = _canonical_object_plan(scene_request, object_plan)
    validate_object_plan(object_plan)
    artifacts["object_plan"] = write_json(output_dir / "object_plan.json", object_plan).as_posix()

    generation_adapter = get_adapter(adapter)
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
                object_plan,
                asset_index_path=str(asset_index_path),
                retrieval_k=retrieval_k,
                use_vlm_selector=use_vlm_asset_selector,
                model_config=asset_selector_model_config,
                asset_generation_tool=(asset_generation_tool if asset_decision.generation_enabled else None),
            )
        asset_selection = _canonical_asset_selection(scene_request, object_plan, asset_selection)
        validate_asset_selection(asset_selection)
        artifacts["asset_selection"] = write_json(output_dir / "asset_selection.json", asset_selection).as_posix()
    else:
        asset_selection = None
        artifacts["asset_selection"] = None

    generation_input = build_generation_input(
        scene_request=scene_request,
        object_plan=object_plan,
        asset_selection=asset_selection,
    )
    input_mode = generation_input["generation_contract"]["input_mode"]
    if input_mode not in generation_adapter.capabilities.input_modes:
        raise ValueError(
            f"Adapter {adapter!r} does not declare support for input mode {input_mode!r}; "
            f"supported modes: {list(generation_adapter.capabilities.input_modes)}"
        )
    artifacts["generation_input"] = write_json(output_dir / "generation_input.json", generation_input).as_posix()
    if int(iteration_limit) < 0:
        raise ValueError("iteration_limit must be >= 0")

    resolved_enrich_assets = bool(enrich_assets) if enrich_assets is not None else bool(asset_csv or asset_root)
    resolved_adapter_config = dict(adapter_config or {})
    asset_adapter_config = {
        "asset_csv": str(asset_csv) if asset_csv else None,
        "asset_root": str(asset_root) if asset_root else None,
        "enrich_assets": resolved_enrich_assets,
    }
    resolved_adapter_config.update({key: value for key, value in asset_adapter_config.items() if value is not None})
    if not eval_generic_validity and not eval_oor and not eval_oar:
        eval_generic_validity = True

    loop_result = _run_generation_evaluation_loop(
        generation_input=generation_input,
        adapter=adapter,
        output_dir=output_dir,
        generated_scene=generated_scene,
        adapter_config=resolved_adapter_config,
        run_generation=run_generation,
        iteration_limit=int(iteration_limit),
        eval_generic_validity=eval_generic_validity,
        eval_oor=eval_oor,
        eval_oar=eval_oar,
        asset_csv=asset_csv,
        asset_root=asset_root,
        enrich_assets=resolved_enrich_assets,
    )
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
        "asset_resolution": {
            **asset_decision.as_dict(),
            "capability_source": "adapter",
            "retrieval_k": max(1, int(retrieval_k)),
            "selector": "vlm" if use_vlm_asset_selector else "top1",
            "generation_tool_configured": asset_generation_tool is not None,
        },
        "adapter": {
            "name": generation_adapter.name,
            "capabilities": generation_adapter.capabilities.as_dict(),
            "generator_output_schema": getattr(generation_adapter, "output_schema", None),
        },
        "converter": {
            "called": not object_plan_provided,
            "model_config_source": (
                "not_used_object_plan_provided"
                if object_plan_provided
                else "explicit"
                if converter_model_config
                else "generator_adapter_fallback"
            ),
            "endpoint": None if object_plan_provided else resolved_converter_config.get("endpoint") or resolved_converter_config.get("base_url"),
            "model": None if object_plan_provided else resolved_converter_config.get("model") or resolved_converter_config.get("model_id"),
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
    parser.add_argument("--room-json", default=None)
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-index-path", default=None)
    parser.add_argument(
        "--asset-mode",
        choices=["off", "retrieve", "retrieve-generate"],
        default="off",
        help="Explicit benchmark asset route: disabled, retrieval only, or retrieval with generation fallback.",
    )
    parser.add_argument("--retrieval-k", type=int, default=5, help="Number of database candidates sent to the asset selector.")
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
        "--asset-generator-plugin",
        default=None,
        help="Optional module:attribute or /path/plugin.py:attribute asset-generation tool.",
    )
    parser.add_argument("--adapter", default="passthrough")
    parser.add_argument("--adapter-config", default=None, help="JSON configuration for the selected generation adapter.")
    parser.add_argument("--generator-endpoint", default=None, help="OpenAI-compatible endpoint override for generation adapters that call an LLM.")
    parser.add_argument("--generator-model", default=None, help="Served model id override for generation adapters that call an LLM.")
    parser.add_argument("--converter-config", default=None, help="Optional JSON model config for NL-to-object-plan conversion; defaults to generator config.")
    parser.add_argument("--converter-endpoint", default=None, help="Optional OpenAI-compatible endpoint override for the benchmark converter.")
    parser.add_argument("--converter-model", default=None, help="Optional served model id override for the benchmark converter.")
    parser.add_argument("--generated-scene", default=None)
    parser.add_argument("--run-generation", action="store_true", help="Ask the adapter to run generation when no --generated-scene is supplied.")
    parser.add_argument("--iteration-limit", type=int, default=0, help="Maximum self-reflexive regeneration attempts after the initial evaluation.")
    parser.add_argument(
        "--structure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose benchmark structure to the generator. Asset handling is controlled separately by --asset-mode.",
    )
    parser.add_argument("--object-plan", default=None)
    parser.add_argument("--asset-selection", default=None)
    parser.add_argument("--eval-generic-validity", action="store_true")
    parser.add_argument("--eval-oor", action="store_true")
    parser.add_argument("--eval-oar", action="store_true")
    parser.add_argument("--enrich-assets", action="store_true", help="Resolve object metadata from --asset-csv/--asset-root before adapter output/evaluation.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    asset_selector_config = read_json(_path_arg(args.asset_selector_config)) if args.asset_selector_config else {}
    if not isinstance(asset_selector_config, dict):
        parser.error("--asset-selector-config must point to a JSON object")
    if args.asset_selector_endpoint:
        asset_selector_config["endpoint"] = args.asset_selector_endpoint
    if args.asset_selector_model:
        asset_selector_config["model"] = args.asset_selector_model
    asset_generation_tool = load_asset_generation_tool(args.asset_generator_plugin)
    adapter_config = read_json(_path_arg(args.adapter_config)) if args.adapter_config else {}
    if not isinstance(adapter_config, dict):
        parser.error("--adapter-config must point to a JSON object")
    if args.generator_endpoint:
        adapter_config["endpoint"] = args.generator_endpoint
    if args.generator_model:
        adapter_config["model"] = args.generator_model
    converter_model_config = read_json(_path_arg(args.converter_config)) if args.converter_config else {}
    if not isinstance(converter_model_config, dict):
        parser.error("--converter-config must point to a JSON object")
    if args.converter_endpoint:
        converter_model_config["endpoint"] = args.converter_endpoint
    if args.converter_model:
        converter_model_config["model"] = args.converter_model

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
        converter_model_config=converter_model_config or None,
        generated_scene=_path_arg(args.generated_scene) if args.generated_scene else None,
        run_generation=args.run_generation,
        iteration_limit=args.iteration_limit,
        structure=args.structure,
        object_plan=read_json(_path_arg(args.object_plan)) if args.object_plan else None,
        asset_selection=read_json(_path_arg(args.asset_selection)) if args.asset_selection else None,
        eval_generic_validity=args.eval_generic_validity,
        eval_oor=args.eval_oor,
        eval_oar=args.eval_oar,
        enrich_assets=args.enrich_assets if args.enrich_assets else None,
        out_dir=_path_arg(args.out_dir),
    )
    print(f"status: {manifest['status']}")
    print(f"run_manifest: {manifest['artifacts']['run_manifest']}")


def _run_generation_evaluation_loop(
    *,
    generation_input: dict,
    adapter: str,
    output_dir: Path,
    generated_scene: str | Path | None,
    adapter_config: dict,
    run_generation: bool,
    iteration_limit: int,
    eval_generic_validity: bool,
    eval_oor: bool,
    eval_oar: bool,
    asset_csv: str | Path | None,
    asset_root: str | Path | None,
    enrich_assets: bool,
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
            generated_scene=generated_scene if iteration == 0 else None,
            adapter_config=adapter_config,
            run_generation=run_generation,
            evaluation_report=previous_report,
            previous_generated_scene=previous_scene,
            iteration=iteration if previous_report is not None else None,
        )
        attempt_record: dict[str, Any] = {
            "iteration": iteration,
            "out_dir": attempt_dir.as_posix(),
            "status": generate_result["status"]["status"],
            "method_input": generate_result.get("method_input"),
            "workflow_status": generate_result.get("workflow_status"),
            "adapter_metadata": generate_result.get("adapter_metadata"),
            "generated_scene": generate_result.get("generated_scene"),
            "evaluation_report": None,
            "valid": None,
        }

        if not generate_result.get("generated_scene"):
            attempts.append(attempt_record)
            break

        previous_scene = read_json(generate_result["generated_scene"])
        report = run_evaluate(
            scene=previous_scene,
            out=attempt_dir / "evaluation_report.json",
            eval_generic_validity=eval_generic_validity,
            eval_oor=eval_oor,
            eval_oar=eval_oar,
            asset_csv=asset_csv,
            asset_root=asset_root,
            enrich_assets=enrich_assets,
        )
        attempt_record["evaluation_report"] = (attempt_dir / "evaluation_report.json").as_posix()
        attempt_record["overall_score"] = report.get("overall_score")
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
            "generated_scene": final_scene_path,
            "workflow_status": latest_attempt.get("workflow_status"),
            "adapter_metadata": latest_attempt.get("adapter_metadata"),
            "evaluation_report": final_report_path,
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
        return float(report.get("overall_score", 0.0)) >= 0.999
    except (TypeError, ValueError):
        return False


def _evaluation_summary(report: dict) -> dict:
    return {
        "overall_score": report.get("overall_score"),
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
        object_spec = item.get("object_spec") if isinstance(item.get("object_spec"), dict) else object_specs.get(object_id, {})
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
                    "category": object_spec.get("category"),
                    "description": object_spec.get("description"),
                    "estimated_size": object_spec.get("estimated_size"),
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


def _resolve_converter_model_config(converter_config: dict | None, adapter_config: dict | None) -> dict:
    source = adapter_config if isinstance(adapter_config, dict) else {}
    allowed = {
        "endpoint",
        "base_url",
        "model",
        "model_id",
        "api_key",
        "temperature",
        "max_tokens",
    }
    resolved = {key: value for key, value in source.items() if key in allowed and value is not None}
    if converter_config:
        resolved.update(converter_config)
    return resolved


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
