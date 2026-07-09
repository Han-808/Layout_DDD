from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import run_evaluate
from generate import run_generate

from benchmark.evaluator.generic_validity.asset_resolver import resolve_asset_metadata
from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_scene_spec
from benchmark.nl_scene.converter import convert_nl_to_scene_spec
from benchmark.scene_io.validate import validate_asset_selection, validate_object_plan, validate_scene_request
from benchmark.utils.io import read_json, write_json


DEFAULT_ROOM = {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "height": 2.8, "unit": "meter"}


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
    adapter: str = "passthrough",
    generated_scene: str | Path | None = None,
    object_plan: dict | None = None,
    asset_selection: dict | None = None,
) -> dict:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = _request_id(output_dir)
    scene_request = {
        "request_id": request_id,
        "instruction": instruction,
        "scene_type": scene_type,
        "room": room or DEFAULT_ROOM,
        "metadata": {},
    }
    validate_scene_request(scene_request)
    artifacts: dict[str, str | None] = {}
    artifacts["scene_request"] = write_json(output_dir / "scene_request.json", scene_request).as_posix()

    if object_plan is None:
        object_plan = convert_nl_to_scene_spec(instruction, scene_type=scene_type, room=scene_request["room"])
        object_plan = _object_plan_from_scene_spec(scene_request, object_plan)
    else:
        object_plan = _object_plan_from_scene_spec(scene_request, object_plan)
    validate_object_plan(object_plan)
    artifacts["object_plan"] = write_json(output_dir / "object_plan.json", object_plan).as_posix()

    if asset_selection is None:
        if not asset_index_path:
            raise ValueError("asset_index_path is required unless --asset-selection is provided.")
        retrieval = retrieve_assets_for_scene_spec(object_plan, asset_index_path=str(asset_index_path), retrieval_k=retrieval_k)
        asset_selection = _asset_selection_from_retrieval(scene_request, retrieval, asset_csv=asset_csv, asset_root=asset_root)
    else:
        asset_selection = _canonical_asset_selection(scene_request, asset_selection)
    validate_asset_selection(asset_selection)
    artifacts["asset_selection"] = write_json(output_dir / "asset_selection.json", asset_selection).as_posix()

    generation_input = {
        "request_id": request_id,
        "scene_request": scene_request,
        "object_plan": object_plan,
        "asset_selection": asset_selection,
        "generation_contract": {"output_format": "canonical_generated_scene_v1", "requires_pose": True},
    }
    artifacts["generation_input"] = write_json(output_dir / "generation_input.json", generation_input).as_posix()

    adapter_config = {
        "asset_csv": str(asset_csv) if asset_csv else None,
        "asset_root": str(asset_root) if asset_root else None,
        "enrich_assets": bool(asset_csv or asset_root),
    }
    generate_result = run_generate(
        generation_input=generation_input,
        adapter_name=adapter,
        out_dir=output_dir,
        generated_scene=generated_scene,
        adapter_config={key: value for key, value in adapter_config.items() if value is not None},
        run_generation=False,
    )
    artifacts.update(
        {
            "method_input": generate_result.get("method_input"),
            "generated_scene": generate_result.get("generated_scene"),
            "workflow_status": generate_result.get("workflow_status"),
            "adapter_metadata": generate_result.get("adapter_metadata"),
        }
    )

    evaluation_report = None
    if generate_result.get("generated_scene"):
        report = run_evaluate(
            scene=read_json(generate_result["generated_scene"]),
            out=output_dir / "evaluation_report.json",
            eval_generic_validity=True,
            asset_csv=asset_csv,
            asset_root=asset_root,
            enrich_assets=bool(asset_csv or asset_root),
        )
        evaluation_report = output_dir / "evaluation_report.json"
        artifacts["evaluation_report"] = evaluation_report.as_posix()
        evaluation_summary = {"overall_score": report.get("overall_score"), "reports": sorted((report.get("reports") or {}).keys())}
    else:
        artifacts["evaluation_report"] = None
        evaluation_summary = None

    manifest = {
        "request_id": request_id,
        "status": generate_result["status"]["status"],
        "artifacts": artifacts,
        "evaluation_summary": evaluation_summary,
    }
    artifacts["run_manifest"] = write_json(output_dir / "run_manifest.json", manifest).as_posix()
    manifest["artifacts"] = artifacts
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modular scene-construction/evaluation harness.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene-type", default="room")
    parser.add_argument("--room-json", default=None)
    parser.add_argument("--asset-csv", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-index-path", default=None)
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--adapter", default="passthrough")
    parser.add_argument("--generated-scene", default=None)
    parser.add_argument("--object-plan", default=None)
    parser.add_argument("--asset-selection", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = run_scene_harness(
        instruction=args.instruction,
        scene_type=args.scene_type,
        room=read_json(_path_arg(args.room_json)) if args.room_json else None,
        asset_csv=_path_arg(args.asset_csv) if args.asset_csv else None,
        asset_root=_path_arg(args.asset_root) if args.asset_root else None,
        asset_index_path=_path_arg(args.asset_index_path) if args.asset_index_path else None,
        retrieval_k=args.retrieval_k,
        adapter=args.adapter,
        generated_scene=_path_arg(args.generated_scene) if args.generated_scene else None,
        object_plan=read_json(_path_arg(args.object_plan)) if args.object_plan else None,
        asset_selection=read_json(_path_arg(args.asset_selection)) if args.asset_selection else None,
        out_dir=_path_arg(args.out_dir),
    )
    print(f"status: {manifest['status']}")
    print(f"run_manifest: {manifest['artifacts']['run_manifest']}")


def _object_plan_from_scene_spec(scene_request: dict, scene_spec: dict) -> dict:
    objects = []
    for index, obj in enumerate(scene_spec.get("objects", []) if isinstance(scene_spec, dict) else []):
        if not isinstance(obj, dict):
            continue
        objects.append(
            {
                "id": str(obj.get("id", f"obj_{index:03d}")),
                "role": str(obj.get("role") or ""),
                "category": str(obj.get("category") or "object"),
                "description": str(obj.get("description") or obj.get("category") or "object"),
                "estimated_size": obj.get("estimated_size"),
                "count": int(obj.get("count") or 1),
                "placement_intent": obj.get("placement_intent")
                if isinstance(obj.get("placement_intent"), dict)
                else {"absolute_relations": [], "relative_relations": []},
                "metadata": obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {},
            }
        )
        if objects[-1]["estimated_size"] is None:
            objects[-1].pop("estimated_size")
    return {
        "request_id": scene_request["request_id"],
        "scene_type": scene_spec.get("scene_type") or scene_request.get("scene_type"),
        "scene_description": scene_spec.get("scene_description") or scene_request.get("instruction"),
        "objects": objects,
        "global_constraints": scene_spec.get("global_constraints") if isinstance(scene_spec.get("global_constraints"), list) else [],
        "relations": scene_spec.get("relations") if isinstance(scene_spec.get("relations"), list) else [],
    }


def _asset_selection_from_retrieval(scene_request: dict, retrieval: dict, *, asset_csv: str | Path | None, asset_root: str | Path | None) -> dict:
    objects = []
    for item in retrieval.get("objects", []) if isinstance(retrieval, dict) else []:
        if not isinstance(item, dict):
            continue
        object_spec = item.get("object_spec") if isinstance(item.get("object_spec"), dict) else {}
        object_id = str(object_spec.get("id") or f"obj_{len(objects):03d}")
        selected = item.get("selected_asset") if isinstance(item.get("selected_asset"), dict) else {}
        selected = _selected_asset_record(selected, asset_csv=asset_csv, asset_root=asset_root)
        objects.append(
            {
                "object_id": object_id,
                "object_spec": {
                    "category": object_spec.get("category"),
                    "description": object_spec.get("description"),
                    "estimated_size": object_spec.get("estimated_size"),
                },
                "selected_asset": selected,
                "candidates": item.get("candidates") if isinstance(item.get("candidates"), list) else [],
                "selection_reason": item.get("selection_reason") or "top-1 retrieval result",
            }
        )
    return {"request_id": scene_request["request_id"], "objects": objects}


def _canonical_asset_selection(scene_request: dict, asset_selection: dict) -> dict:
    if asset_selection.get("request_id") == scene_request["request_id"]:
        return asset_selection
    return {**asset_selection, "request_id": scene_request["request_id"]}


def _selected_asset_record(asset: dict, *, asset_csv: str | Path | None, asset_root: str | Path | None) -> dict:
    jid = asset.get("jid") or asset.get("asset_id") or asset.get("id")
    record = {
        "jid": jid,
        "category": asset.get("category") or "",
        "retrieval_category": asset.get("retrieval_category") or asset.get("category") or "",
        "desc": asset.get("desc") or asset.get("description") or asset.get("short_desc") or "",
        "short_desc": asset.get("short_desc") or asset.get("description") or "",
        "size": asset.get("size") or asset.get("dimensions"),
        "asset_ref": asset.get("asset_ref") if isinstance(asset.get("asset_ref"), dict) else {"source_db": "imaginarium", "asset_key": jid},
        "asset_proxy": asset.get("asset_proxy") if isinstance(asset.get("asset_proxy"), dict) else {"type": "obb_from_metadata_or_csv", "bbox_center_local": [0, 0, 0], "bbox_size": asset.get("size") or asset.get("dimensions")},
        "metadata": asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {"interactive": False, "inner_placement": False, "align_to_wall_normal": False, "scaling_strategy": None},
    }
    record["metadata"].setdefault("interactive", False)
    if asset_csv or asset_root:
        try:
            record = resolve_asset_metadata(record, asset_csv_path=asset_csv, asset_root=asset_root)
        except ValueError:
            pass
    return record


def _request_id(out_dir: Path) -> str:
    return out_dir.name or "scene_request"


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
