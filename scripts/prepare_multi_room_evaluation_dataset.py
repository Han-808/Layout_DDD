#!/usr/bin/env python3
"""Materialize verified multi-room projections as ordinary evaluator cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.multi_room_evaluation import (
    build_existing_evaluation_campaign_config,
    discover_multi_room_evaluation_inventory,
    evaluation_campaign_command,
    materialize_multi_room_evaluation_dataset,
    write_campaign_config,
)
from benchmark.multi_room_evaluation.render_profile import (
    OFFICIAL_RENDER_PROFILE,
)
from benchmark.rendering import BlenderRenderer, CYCLES_DEVICES, RENDER_ENGINES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model to materialize; repeat to build separate per-model datasets.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender-bin", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    completeness = parser.add_mutually_exclusive_group()
    completeness.add_argument(
        "--require-complete",
        dest="require_complete",
        action="store_true",
        help="Fail before rendering if any expected source room failed or is missing (default).",
    )
    completeness.add_argument(
        "--allow-incomplete",
        dest="require_complete",
        action="store_false",
        help="Build succeeded rooms only and mark the dataset diagnostic-only.",
    )
    parser.set_defaults(require_complete=True)
    parser.add_argument("--dataset-id")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["timeout_seconds"],
    )
    parser.add_argument("--width", type=int, default=OFFICIAL_RENDER_PROFILE["width"])
    parser.add_argument("--height", type=int, default=OFFICIAL_RENDER_PROFILE["height"])
    parser.add_argument(
        "--render-engine",
        choices=RENDER_ENGINES,
        default=OFFICIAL_RENDER_PROFILE["render_engine"],
    )
    parser.add_argument(
        "--cycles-device",
        choices=CYCLES_DEVICES,
        default=OFFICIAL_RENDER_PROFILE["cycles_device"],
    )
    parser.add_argument(
        "--cycles-samples",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["cycles_samples"],
    )
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=OFFICIAL_RENDER_PROFILE["cycles_denoising"],
    )
    parser.add_argument(
        "--require-asset-mesh",
        action=argparse.BooleanOptionalAction,
        default=OFFICIAL_RENDER_PROFILE["require_asset_mesh"],
    )
    parser.add_argument(
        "--collision-max-vertices-per-object",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["collision_max_vertices_per_object"],
    )
    parser.add_argument(
        "--collision-max-faces-per-object",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["collision_max_faces_per_object"],
    )
    parser.add_argument(
        "--collision-max-total-vertices",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["collision_max_total_vertices"],
    )
    parser.add_argument(
        "--collision-max-total-faces",
        type=int,
        default=OFFICIAL_RENDER_PROFILE["collision_max_total_faces"],
    )

    parser.add_argument("--campaign-template", type=Path)
    parser.add_argument("--campaign-config-out", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--attempt-parent", type=Path)
    parser.add_argument("--final-selection-root", type=Path)
    parser.add_argument(
        "--evaluation-bindings",
        type=Path,
        help="Optional private binding path used only to print the existing run command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collection = discover_multi_room_evaluation_inventory(
            args.collection_root,
            models=args.models or (),
        )
        selected_models = collection.models
        if not selected_models:
            raise ValueError("no model selected")
        if args.require_complete and not collection.complete:
            raise ValueError(
                "requested model scope is incomplete; no renderer was constructed"
            )
        campaign_requested = any(
            value is not None
            for value in (
                args.campaign_template,
                args.campaign_config_out,
                args.campaign_id,
                args.attempt_parent,
                args.final_selection_root,
                args.evaluation_bindings,
            )
        )
        if campaign_requested and len(selected_models) != 1:
            raise ValueError("campaign config output requires exactly one selected model")
        if campaign_requested and any(
            value is None
            for value in (
                args.campaign_template,
                args.campaign_config_out,
                args.campaign_id,
                args.attempt_parent,
                args.final_selection_root,
            )
        ):
            raise ValueError("all campaign config arguments must be supplied together")
        if campaign_requested and not collection.complete:
            raise ValueError(
                "diagnostic incomplete data cannot enter the evaluation campaign"
            )

        renderer = BlenderRenderer(
            blender_bin=args.blender_bin,
            timeout_seconds=args.timeout_seconds,
            width=args.width,
            height=args.height,
            render_engine=args.render_engine,
            cycles_device=args.cycles_device,
            cycles_samples=args.cycles_samples,
            cycles_denoising=args.cycles_denoising,
            require_asset_mesh=args.require_asset_mesh,
            collision_max_vertices_per_object=(
                args.collision_max_vertices_per_object
            ),
            collision_max_faces_per_object=args.collision_max_faces_per_object,
            collision_max_total_vertices=args.collision_max_total_vertices,
            collision_max_total_faces=args.collision_max_total_faces,
        )
        results: list[dict[str, Any]] = []
        for model in selected_models:
            inventory = discover_multi_room_evaluation_inventory(
                args.collection_root,
                models=(model,),
            )
            output = args.output_root.resolve()
            if len(selected_models) > 1:
                output = output / model
            dataset_id = args.dataset_id
            if dataset_id is not None and len(selected_models) > 1:
                dataset_id = f"{dataset_id}-{_slug(model)}"
            result = materialize_multi_room_evaluation_dataset(
                inventory,
                output_root=output,
                renderer=renderer,
                asset_root=args.asset_root,
                require_complete=args.require_complete,
                dataset_id=dataset_id,
            )
            results.append(result.public_dict())

        campaign_output: dict[str, Any] | None = None
        if campaign_requested:
            result = results[0]
            dataset_root = Path(str(result["output_root"]))
            config = build_existing_evaluation_campaign_config(
                repo_root=REPO_ROOT,
                template_path=args.campaign_template,
                dataset_root=dataset_root,
                campaign_id=args.campaign_id,
                model_label=selected_models[0],
                attempt_parent=args.attempt_parent,
                final_selection_root=args.final_selection_root,
            )
            config_path = write_campaign_config(args.campaign_config_out, config)
            check_command = evaluation_campaign_command(
                config_path=config_path,
                python_executable=REPO_ROOT / ".venv/bin/python",
                run=False,
            )
            run_command = (
                evaluation_campaign_command(
                    config_path=config_path,
                    python_executable=REPO_ROOT / ".venv/bin/python",
                    run=True,
                    bindings_path=args.evaluation_bindings,
                )
                if args.evaluation_bindings is not None
                else None
            )
            campaign_output = {
                "config_path": str(config_path),
                "check_command": list(check_command),
                "run_command": list(run_command) if run_command else None,
                "existing_evaluator_entrypoint": True,
            }
        print(
            json.dumps(
                {
                    "schema_version": "multi_room_evaluation_materialization_cli_v1",
                    "results": results,
                    "campaign": campaign_output,
                    "network_used": False,
                    "evaluation_api_used": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            f"error: {type(exc).__name__}: multi-room evaluation materialization failed",
            file=sys.stderr,
        )
        return 3


def _slug(value: str) -> str:
    return "-".join(
        part
        for part in "".join(
            character.lower() if character.isalnum() else "-"
            for character in value
        ).split("-")
        if part
    )


if __name__ == "__main__":
    raise SystemExit(main())
