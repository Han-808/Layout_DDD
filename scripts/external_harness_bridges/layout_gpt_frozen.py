#!/usr/bin/env python3
"""Controlled LayoutGPT bridge for exact FrozenAssets experiments.

This bridge retains LayoutGPT's released CSS numerical representation while
making the custom benchmark room, inventory, asset dimensions, and model
identity explicit.  It never retrieves or replaces an asset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _common import (
    file_sha256,
    observed_model_identity,
    openai_chat_completion,
    public_object_plan,
    read_mapping,
    required_model_identity,
    required_model_deployment_id,
    require_observed_model_match,
    total_tokens,
    verify_catalog_contract,
    verify_api_endpoint_contract,
    verify_model_contract,
    write_json,
    write_runner_report,
    write_text,
)


CSS_OBJECT = re.compile(r"([^\n{}]+)\s*\{([^{}]+)\}")
CSS_FIELD = re.compile(
    r"([A-Za-z_]+)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(m|degrees)\s*;?",
    flags=re.IGNORECASE,
)
REQUIRED_FIELDS = {"length", "width", "height", "left", "top", "depth", "orientation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--method-input", required=True, type=Path)
    parser.add_argument("--comparison-input", required=True, type=Path)
    parser.add_argument("--comparison-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-ids-output", required=True, type=Path)
    parser.add_argument("--runner-report", required=True, type=Path)
    parser.add_argument("--raw-response", required=True, type=Path)
    parser.add_argument("--icl-examples", required=True, type=Path)
    parser.add_argument("--icl-snapshot-output", required=True, type=Path)
    parser.add_argument("--expected-icl-sha256", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--size-tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    repo = args.repo_path.expanduser().resolve()
    if not (repo / "run_layoutgpt_3d.py").is_file() or not (
        repo / "parse_llm_output.py"
    ).is_file():
        raise FileNotFoundError(
            "configured checkout is not the released LayoutGPT repository"
        )
    request = read_mapping(args.request, "LayoutGPT request")
    method_input = read_mapping(args.method_input, "method input")
    control = read_mapping(args.comparison_input, "comparison control")
    catalog = read_mapping(args.comparison_catalog, "LayoutGPT catalog")
    identity = required_model_identity()
    deployment_id = required_model_deployment_id()
    endpoint_sha256 = verify_api_endpoint_contract(
        os.environ.get("LAYOUT_DDD_API_ENDPOINT", ""),
        completion_endpoint=True,
    )
    verify_model_contract(control, identity)
    verify_catalog_contract(control, catalog)
    plan = public_object_plan(method_input)
    expected = _expected_rows(control, catalog)
    icl_source = args.icl_examples.expanduser().resolve()
    icl_hash_before = file_sha256(icl_source)
    expected_icl_hash = str(args.expected_icl_sha256).strip().lower()
    if (
        len(expected_icl_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_icl_hash)
        or icl_hash_before != expected_icl_hash
    ):
        raise RuntimeError(
            "LayoutGPT ICL source does not match the frozen expected SHA-256"
        )
    generation_policy = control.get("generation")
    generation_policy = (
        generation_policy if isinstance(generation_policy, dict) else {}
    )
    harness_inputs = generation_policy.get("harness_inputs")
    harness_inputs = harness_inputs if isinstance(harness_inputs, dict) else {}
    icl_contract = harness_inputs.get("layout_gpt")
    if (
        not isinstance(icl_contract, dict)
        or icl_contract.get("status") != "human_approved"
        or icl_contract.get("icl_sha256") != expected_icl_hash
        or icl_contract.get("hidden_evaluator_data_used") is not False
        or not str(icl_contract.get("provenance") or "").strip()
    ):
        raise RuntimeError(
            "LayoutGPT ICL does not match the approved public protocol contract"
        )
    args.icl_snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(icl_source, args.icl_snapshot_output)
    if file_sha256(args.icl_snapshot_output) != icl_hash_before:
        raise RuntimeError("LayoutGPT ICL snapshot differs from its configured source")
    messages = _messages(request, plan, expected, args.icl_snapshot_output)
    sys.path.insert(0, repo.as_posix())
    from parse_llm_output import parse_3D_layout as released_parse_3d_layout

    response, request_metadata = openai_chat_completion(
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    observed_identities = require_observed_model_match(
        identity,
        [
            observed_model_identity(
                {"model": request_metadata.get("response_model")},
                provider=identity["provider"],
            )
        ],
    )
    write_text(args.raw_response, response)
    if file_sha256(icl_source) != icl_hash_before:
        raise RuntimeError("LayoutGPT ICL source changed during generation")
    object_list = _parse_and_validate(
        response,
        expected,
        size_tolerance=args.size_tolerance,
        released_parser=released_parse_3d_layout,
    )
    output = {
        "query_id": request.get("query_id"),
        "unit": "m",
        "prompt": request.get("prompt"),
        "response": response,
        "object_list": object_list,
        "comparison_mode": "frozen_assets",
    }
    write_json(args.output, output)
    bindings = {
        row["native_object_id"]: row["asset_id"] for row in expected
    }
    write_json(args.asset_ids_output, {"asset_ids": bindings})
    write_runner_report(
        args.runner_report,
        adapter="layout_gpt",
        identity=identity,
        generation_calls=1,
        tokens=total_tokens(request_metadata["usage"]),
        protocol_observation={
            "exact_inventory": True,
            "exact_asset_bindings_sidecar": True,
            "fixed_native_dimensions": True,
            "retrieval_calls": 0,
        },
        observed_model_identities=observed_identities,
        model_identity_evidence="observed_response",
        model_deployment_id=deployment_id,
        model_endpoint_sha256=endpoint_sha256,
        extra={
            "request_metadata": request_metadata,
            "released_reference_checkout": repo.as_posix(),
            "released_parser": (repo / "parse_llm_output.py").as_posix(),
            "released_parser_sha256": file_sha256(repo / "parse_llm_output.py"),
            "released_parser_used": True,
            "icl_examples_source": icl_source.as_posix(),
            "icl_examples_snapshot": args.icl_snapshot_output.resolve().as_posix(),
            "icl_examples_sha256": icl_hash_before,
            "expected_icl_examples_sha256": expected_icl_hash,
            "raw_response": args.raw_response.resolve().as_posix(),
            "asset_ids": args.asset_ids_output.resolve().as_posix(),
        },
    )


def _expected_rows(control: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    assets = catalog.get("dataset_asset_index")
    assets = assets if isinstance(assets, dict) else {}
    slot_map = catalog.get("logical_to_native_slot")
    slot_map = slot_map if isinstance(slot_map, dict) else {}
    frozen = catalog.get("frozen_asset_ids")
    frozen = frozen if isinstance(frozen, dict) else {}
    rows = []
    for item in control.get("objects", []):
        slot_id = str(item["slot_id"])
        native_id = str(slot_map[slot_id])
        asset_id = str(frozen[native_id])
        asset = assets.get(asset_id)
        if not isinstance(asset, dict):
            raise RuntimeError(f"LayoutGPT catalog lacks asset {asset_id!r}")
        selector, separator, ordinal = native_id.rpartition("_")
        if not separator or not ordinal.isdigit():
            raise RuntimeError(f"invalid LayoutGPT native ID {native_id!r}")
        rows.append(
            {
                "slot_id": slot_id,
                "native_object_id": native_id,
                "selector": selector,
                "ordinal": int(ordinal),
                "asset_id": asset_id,
                "description": asset["description"],
                "canonical_front": asset.get("canonical_front"),
                "size": [float(value) for value in asset["physical_dimensions"]],
            }
        )
    return rows


def _messages(
    request: dict[str, Any],
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    icl_path: Path,
) -> list[dict[str, Any]]:
    room = request["room_dimensions_m"]
    system = (
        "You are LayoutGPT, a numerical 3D indoor-layout planner. Return only CSS "
        "layout rows. Each row must be `selector {length: ?m; width: ?m; height: "
        "?m; orientation: ? degrees; left: ?m; top: ?m; depth: ?m;}`. left/top/"
        "depth are bbox-center coordinates in a meter-scale room whose minimum floor "
        "corner is [0,0,0]. Do not add prose or Markdown."
        " Orientation is an active counterclockwise yaw about +Z applied "
        "directly to each frozen asset's canonical local frame. For canonical "
        "front [0,-1,0], 0 degrees faces world -Y and +90 faces world +X."
    )
    required = [
        {
            "row": index,
            "logical_slot_id": row["slot_id"],
            "selector": row["selector"],
            "native_object_id_after_released_category_counting": row[
                "native_object_id"
            ],
            "frozen_size_m": row["size"],
            "description": row["description"],
            "canonical_front": row.get("canonical_front"),
        }
        for index, row in enumerate(rows)
    ]
    user = {
        "instruction": request["prompt"],
        "room_dimensions_m": room,
        "public_object_plan": plan,
        "frozen_output_rows_in_required_order": required,
        "rules": [
            "Emit exactly one row for every listed row and no other row.",
            "Use each selector exactly in the supplied order.",
            "Copy every frozen_size_m exactly; choose only pose and orientation.",
            "Keep every oriented bbox inside the room.",
            "Apply orientation directly to canonical_front; do not assume a "
            "different dataset-native front. A null front is unavailable and "
            "must not be invented.",
        ],
    }
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    examples = json.loads(icl_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("LayoutGPT ICL examples must be a non-empty JSON message list")
    if len(examples) % 2:
        raise ValueError(
            "LayoutGPT ICL examples must contain complete user/assistant pairs"
        )
    for index, item in enumerate(examples):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if (
            not isinstance(item, dict)
            or item.get("role") != expected_role
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError(
                f"LayoutGPT ICL examples[{index}] must be an alternating "
                f"{expected_role} message with string content"
            )
    messages.extend(examples)
    messages.append(
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
        }
    )
    return messages


def _parse_and_validate(
    response: str,
    expected: list[dict[str, Any]],
    *,
    size_tolerance: float,
    released_parser: Any | None = None,
) -> list[list[Any]]:
    parsed = []
    counts: Counter[str] = Counter()
    native_ids = []
    for match in CSS_OBJECT.finditer(response):
        selector = match.group(1).strip()
        native_row = match.group(0)
        matches = CSS_FIELD.findall(match.group(2))
        fields = {key.lower(): float(value) for key, value, _unit in matches}
        units = {key.lower(): unit.lower() for key, _value, unit in matches}
        if set(fields) != REQUIRED_FIELDS or len(matches) != len(fields):
            raise RuntimeError(
                f"LayoutGPT selector {selector!r} fields differ from released contract"
            )
        invalid_units = {
            key: unit
            for key, unit in units.items()
            if unit != ("degrees" if key == "orientation" else "m")
        }
        if invalid_units:
            raise RuntimeError(
                f"LayoutGPT selector {selector!r} uses invalid controlled units: "
                f"{invalid_units}"
            )
        if released_parser is not None:
            upstream_selector, upstream_fields = released_parser(native_row, unit="m")
            if (
                upstream_selector != selector
                or not isinstance(upstream_fields, dict)
                or any(
                    abs(float(upstream_fields.get(key)) - value) > 1.0e-9
                    for key, value in fields.items()
                )
            ):
                raise RuntimeError(
                    f"LayoutGPT row {selector!r} disagrees with the released parser"
                )
        counts[selector] += 1
        native_ids.append(f"{_slug(selector)}_{counts[selector]}")
        parsed.append([selector, fields])
    expected_ids = [row["native_object_id"] for row in expected]
    if native_ids != expected_ids:
        raise RuntimeError(
            "LayoutGPT output inventory/order differs from frozen contract: "
            f"expected={expected_ids}, actual={native_ids}"
        )
    for index, (item, row) in enumerate(zip(parsed, expected)):
        fields = item[1]
        actual_size = [fields["length"], fields["width"], fields["height"]]
        if any(
            abs(actual_size[axis] - row["size"][axis]) > size_tolerance
            for axis in range(3)
        ):
            raise RuntimeError(
                f"LayoutGPT row {index} changed frozen physical dimensions: "
                f"expected={row['size']}, actual={actual_size}"
            )
    return parsed


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "object"


if __name__ == "__main__":
    main()
