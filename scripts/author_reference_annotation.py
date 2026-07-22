"""Author a DRAFT reference annotation from an instruction (optionally model-assisted).

Summary:
    Produces a starting-point reference annotation and object-plan draft for a
    request. The output is a draft only - it is not official or scoreable until
    it is human-reviewed and approved.

Input:
    - ``--request-id`` (required), ``--instruction`` / ``--scene-type``, and an
      optional authoring model config (``--model-config`` or
      ``--endpoint``/``--model``/``--api-key-env``).

Output:
    - ``--out``: reference_annotation draft JSON plus a sibling object_plan
      draft; prints ``official_scoreable: false`` and the next review step.

Function:
    Converts the instruction into a draft annotation/object plan for downstream
    human confirmation, claim states, and validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.nl_scene.converter import (  # noqa: E402
    COARSE_GRAINED,
    FINE_GRAINED,
    convert_nl_to_object_plan,
)
from benchmark.reference_annotation import (  # noqa: E402
    ANNOTATION_SOURCES,
    build_reference_annotation_draft,
    validate_reference_annotation,
)
from benchmark.scene_io.validate import validate_object_plan  # noqa: E402
from benchmark.utils.io import read_json, write_json  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline benchmark-authoring tool. It creates an unconfirmed reference "
            "annotation draft and never participates in generator execution."
        )
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--scene-type", default="room")
    parser.add_argument(
        "--prompt-granularity",
        choices=[FINE_GRAINED, COARSE_GRAINED],
        default=FINE_GRAINED,
    )
    parser.add_argument(
        "--object-plan",
        default=None,
        help="Optional existing model/human draft. When omitted, the configured authoring model is called.",
    )
    parser.add_argument(
        "--source",
        choices=ANNOTATION_SOURCES,
        default="model_assisted",
        help="Provenance of the draft. Model-assisted drafts require later human approval.",
    )
    parser.add_argument("--model-config", default=None, help="OpenAI-compatible authoring model JSON config.")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
    )
    parser.add_argument(
        "--send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--object-plan-out",
        default=None,
        help="Optional path for the intermediate object-plan draft.",
    )
    args = parser.parse_args()

    model_config = read_json(_path(args.model_config)) if args.model_config else {}
    if not isinstance(model_config, dict):
        parser.error("--model-config must point to a JSON object")
    if "api_key" in model_config:
        parser.error("--model-config must not contain literal api_key; use api_key_env instead")
    for key, value in {
        "endpoint": args.endpoint,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "max_tokens_field": args.max_tokens_field,
        "send_temperature": args.send_temperature,
    }.items():
        if value is not None:
            model_config[key] = value
    instruction = str(args.instruction or "").strip()
    model_called = args.object_plan is None
    if model_called:
        if not instruction:
            parser.error("--instruction is required when --object-plan is omitted")
        if not str(model_config.get("endpoint") or model_config.get("base_url") or "").strip():
            parser.error("offline model-assisted authoring requires --endpoint or model_config.endpoint")
        if not str(model_config.get("model") or model_config.get("model_id") or "").strip():
            parser.error("offline model-assisted authoring requires --model or model_config.model")
        object_plan = convert_nl_to_object_plan(
            instruction,
            request_id=args.request_id,
            scene_type=args.scene_type,
            prompt_granularity=args.prompt_granularity,
            model_config=model_config,
        )
        source = "model_assisted"
    else:
        object_plan = read_json(_path(args.object_plan))
        source = args.source
        if not instruction:
            instruction = str(object_plan.get("scene_description") or "").strip()

    validate_object_plan(object_plan)
    scene_request = {
        "request_id": args.request_id,
        "instruction": instruction,
        "scene_type": args.scene_type,
        "prompt_granularity": args.prompt_granularity,
    }
    annotation = build_reference_annotation_draft(
        object_plan,
        scene_request,
        source=source,
    )
    annotation["provenance"] = {
        **annotation.get("provenance", {}),
        "authoring_tool": "scripts/author_reference_annotation.py",
        "runtime_benchmark_component": False,
        "generator_visible": False,
        "model_called": model_called,
        "model": (
            str(model_config.get("model") or model_config.get("model_id"))
            if model_called
            else None
        ),
        "endpoint": (
            str(model_config.get("endpoint") or model_config.get("base_url"))
            if model_called
            else None
        ),
    }
    validate_reference_annotation(annotation)

    out_path = write_json(_path(args.out), annotation)
    object_plan_out = (
        _path(args.object_plan_out)
        if args.object_plan_out
        else out_path.with_name(f"{out_path.stem}.object_plan.draft.json")
    )
    write_json(object_plan_out, object_plan)
    print(f"reference_annotation_draft: {out_path}")
    print(f"object_plan_draft: {object_plan_out}")
    print("official_scoreable: false")
    print("next_step: human review, claim confirmation, inventory policy, and validation")


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
