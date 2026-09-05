#!/usr/bin/env python3
"""Thin FrozenAssets bridge around the released DirectLayout pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _common import (
    file_sha256,
    observed_model_identity,
    public_object_plan,
    read_mapping,
    redact_error_detail,
    required_model_identity,
    required_model_deployment_id,
    require_observed_model_match,
    response_total_tokens,
    verify_catalog_contract,
    verify_api_endpoint_contract,
    verify_model_contract,
    write_json,
    write_runner_report,
)


class FrozenContractViolation(RuntimeError):
    """A native state violated FrozenAssets and must never be retried/repaired."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--method-input", required=True, type=Path)
    parser.add_argument("--comparison-input", required=True, type=Path)
    parser.add_argument("--comparison-catalog", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-bindings-output", required=True, type=Path)
    parser.add_argument("--runner-report", required=True, type=Path)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--views", default="0,1,2")
    parser.add_argument("--size-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    repo = args.repo_path.expanduser().resolve()
    if not (repo / "demo.py").is_file() or not (repo / "services/pipeline.py").is_file():
        raise FileNotFoundError("configured checkout is not the released DirectLayout repo")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if not (
        isinstance(request, list)
        and len(request) == 2
        and isinstance(request[0], list)
        and len(request[0]) == 1
        and isinstance(request[1], list)
        and len(request[1]) == 1
    ):
        raise ValueError("DirectLayout request must be the released two-list batch format")
    method_input = read_mapping(args.method_input, "method input")
    control = read_mapping(args.comparison_input, "comparison control")
    catalog = read_mapping(args.comparison_catalog, "DirectLayout catalog")
    identity = required_model_identity()
    deployment_id = required_model_deployment_id()
    verify_model_contract(control, identity)
    verify_catalog_contract(control, catalog)
    plan = public_object_plan(method_input)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    native_input = args.work_dir / "directlayout_input.json"
    output_dir = args.work_dir / "output_layout"
    render_dir = args.work_dir / "output_scene"
    assets_dir = args.work_dir / "asset_library"
    room_name = "controlled_scene"
    prompt, expected = _controlled_prompt(
        original_prompt=str(request[0][0]),
        plan=plan,
        control=control,
        catalog=catalog,
    )
    write_json(native_input, [[prompt], request[1]])
    _materialize_slot_assets(
        catalog=catalog,
        catalog_path=args.comparison_catalog,
        target=assets_dir / room_name,
        expected=expected,
    )

    api_base = os.environ.get("LAYOUT_DDD_API_BASE_URL", "").strip()
    api_key = os.environ.get("LAYOUT_DDD_API_KEY", "").strip()
    if not api_base or not api_key:
        raise RuntimeError(
            "DirectLayout bridge requires LAYOUT_DDD_API_BASE_URL and "
            "LAYOUT_DDD_API_KEY"
        )
    endpoint_sha256 = verify_api_endpoint_contract(
        api_base,
        completion_endpoint=False,
    )
    dimensions = [float(value) for value in request[1][0]]
    pixels = [int(round(value * 100.0)) for value in dimensions]
    if any(abs(pixels[index] / 100.0 - dimensions[index]) > 1.0e-9 for index in range(3)):
        raise ValueError("DirectLayout v1 bridge requires room dimensions on a 1cm grid")

    sys.path.insert(0, repo.as_posix())
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        from config.settings import ProjectSettings
        import services.pipeline as pipeline_module

        settings = ProjectSettings()
        settings.paths.input_file = native_input.resolve().as_posix()
        settings.paths.output_dir = output_dir.resolve().as_posix()
        settings.paths.assets_dir = assets_dir.resolve().as_posix()
        settings.paths.render_dir = render_dir.resolve().as_posix()
        settings.runtime.max_retries = int(args.max_retries)
        settings.runtime.max_iterations = int(args.max_iterations)
        settings.runtime.views = [int(value) for value in args.views.split(",") if value]
        settings.runtime.length, settings.runtime.width, settings.runtime.height = pixels
        settings.models.bev_model = identity["model_id"]
        settings.models.threed_model = identity["model_id"]
        settings.models.reasoning_model = identity["model_id"]
        settings.api.bev_base_url = api_base
        settings.api.openai_base_url = api_base
        settings.api.reasoning_base_url = api_base
        settings.api.bev_api_key = api_key
        settings.api.openai_api_key = api_key
        settings.api.reasoning_api_key = api_key
        pipeline_module.room_name_from_prompt = lambda _prompt: room_name
        pipeline = pipeline_module.DirectLayoutPipeline(settings)
        observer = _ResponseObserver(identity)
        _install_observed_clients(pipeline, observer)
        optimization_tracking = _install_stable_room_bridge(
            pipeline,
            semantic_prompt=prompt,
            room_name=room_name,
            expected=expected,
            tolerance=args.size_tolerance,
            audit_dir=args.work_dir / "validated_layout_states",
            error_secrets=(api_key,),
        )
        pipeline.run()
    finally:
        os.chdir(old_cwd)

    selected_state = _select_completed_layout_state(optimization_tracking)
    selected = Path(selected_state["snapshot_path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, args.output)
    output_payload = json.loads(args.output.read_text(encoding="utf-8"))
    observations = _observe_output(output_payload, expected, args.size_tolerance)
    if not observations["valid"]:
        raise FrozenContractViolation(
            f"selected DirectLayout state violates frozen controls: {observations['violations']}"
        )
    observed_identities = require_observed_model_match(
        identity,
        observer.identities,
    )
    bindings = {
        native_id: {
            "asset_key": row["asset_id"],
            "source_db": row["source_db"],
            "category": row["asset_category"],
            "description": row["asset_description"],
            "mesh_uri": row["mesh_uri"],
            "mesh_sha256": row["mesh_sha256"],
            "canonical_front": row["canonical_front"],
            "canonical_front_source": row["canonical_front_source"],
            "bbox_size_local": row["bbox_size_local"],
            "bbox_center_local": row["bbox_center_local"],
        }
        for native_id, row in ((item["native_object_id"], item) for item in expected)
        if native_id in observations["actual_object_ids"]
    }
    write_json(args.asset_bindings_output, {"asset_bindings": bindings})
    write_runner_report(
        args.runner_report,
        adapter="direct_layout",
        identity=identity,
        generation_calls=observer.calls,
        tokens=observer.tokens,
        iteration_count=optimization_tracking["feedback_rounds"],
        rendering_calls=optimization_tracking["rendering_calls"],
        protocol_observation=observations,
        observed_model_identities=observed_identities,
        model_identity_evidence="observed_response",
        model_deployment_id=deployment_id,
        model_endpoint_sha256=endpoint_sha256,
        extra={
            "selected_native_layout": selected.resolve().as_posix(),
            "selected_native_layout_sha256": file_sha256(selected),
            "all_native_layouts": [
                state["snapshot_path"]
                for state in optimization_tracking["layout_states"]
            ],
            "asset_bindings": args.asset_bindings_output.resolve().as_posix(),
            "stable_room_path_bridge": True,
            "semantic_prompt_preserved_in_feedback": True,
            "optimization_tracking": optimization_tracking,
        },
    )


def _install_stable_room_bridge(
    pipeline: Any,
    *,
    semantic_prompt: str,
    room_name: str,
    expected: list[dict[str, Any]],
    tolerance: float,
    audit_dir: Path,
    error_secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Decouple upstream artifact filenames from its semantic prompt string.

    Released DirectLayout uses ``room_name_from_prompt`` for initial artifacts
    but uses the full prompt directly during refinement.  The latter is not a
    safe filename for structured benchmark requests.  Only the path token is
    replaced here; both feedback calls still receive the original prompt.
    """

    service = pipeline.optimization_service
    original_reasoning = service._request_reasoning_feedback
    original_vlm = service._request_vlm_feedback
    renderer = pipeline.rendering_service
    original_render = renderer.render_views
    audit_dir.mkdir(parents=True, exist_ok=True)
    tracking: dict[str, Any] = {
        "feedback_rounds": 0,
        "optimization_attempts": 0,
        "optimization_completed": False,
        "render_invocations": 0,
        "rendering_calls": 0,
        "layout_states": [],
        "active_optimization_attempt": 0,
        "completed_attempt": None,
        "optimization_attempt_results": [],
    }

    def semantic_call(delegate: Any, *args: Any, **kwargs: Any) -> Any:
        if "prompt" in kwargs:
            kwargs["prompt"] = semantic_prompt
            return delegate(*args, **kwargs)
        values = list(args)
        if values:
            values[0] = semantic_prompt
        else:
            kwargs["prompt"] = semantic_prompt
        return delegate(*values, **kwargs)

    def reasoning(*args: Any, **kwargs: Any) -> Any:
        tracking["feedback_rounds"] += 1
        return semantic_call(original_reasoning, *args, **kwargs)

    def vlm(*args: Any, **kwargs: Any) -> Any:
        return semantic_call(original_vlm, *args, **kwargs)

    def render_views(*args: Any, **kwargs: Any) -> Any:
        layout_path = kwargs.get("layout_path")
        if layout_path is None and len(args) >= 2:
            layout_path = args[1]
        if layout_path is None:
            raise FrozenContractViolation(
                "DirectLayout render call did not expose its native layout path"
            )
        source = Path(str(layout_path)).resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            observation = _observe_output(payload, expected, tolerance)
        except Exception as exc:
            raise FrozenContractViolation(
                f"cannot validate DirectLayout state before render: {source}"
            ) from exc
        sequence = len(tracking["layout_states"])
        snapshot = audit_dir / f"layout_state_{sequence:03d}.json"
        shutil.copyfile(source, snapshot)
        state = {
            "sequence": sequence,
            "optimization_attempt": tracking["active_optimization_attempt"],
            "source_path": source.as_posix(),
            "snapshot_path": snapshot.resolve().as_posix(),
            "sha256": file_sha256(snapshot),
            "observation": observation,
            "rendered": False,
        }
        tracking["layout_states"].append(state)
        if not observation["valid"]:
            raise FrozenContractViolation(
                "DirectLayout native state violates frozen controls before render: "
                f"{observation['violations']}"
            )
        tracking["render_invocations"] += 1
        result = original_render(*args, **kwargs)
        views = kwargs.get("views")
        if views is None and len(args) >= 6:
            views = args[5]
        state["rendered"] = True
        tracking["rendering_calls"] += len(list(views or []))
        return result

    def run_optimization(_prompt: str, generation_result: Any) -> Any:
        max_retries = int(pipeline.settings.runtime.max_retries)
        for attempt in range(1, max_retries + 1):
            tracking["optimization_attempts"] += 1
            tracking["active_optimization_attempt"] = attempt
            try:
                result = service.optimize(
                    prompt=room_name,
                    length=pipeline.settings.runtime.length,
                    width=pipeline.settings.runtime.width,
                    height=pipeline.settings.runtime.height,
                    generation_result=generation_result,
                    render_dir=pipeline.settings.paths.render_dir,
                    output_dir=pipeline.settings.paths.output_dir,
                    assets_dir=pipeline.settings.paths.assets_dir,
                    views=pipeline.settings.runtime.views,
                    max_iterations=pipeline.settings.runtime.max_iterations,
                    retry_limit=pipeline.settings.runtime.max_retries,
                )
                tracking["optimization_completed"] = True
                tracking["completed_attempt"] = attempt
                tracking["optimization_attempt_results"].append(
                    {"attempt": attempt, "status": "completed"}
                )
                return result
            except FrozenContractViolation:
                raise
            except Exception as exc:
                tracking["optimization_attempt_results"].append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": redact_error_detail(
                            str(exc),
                            secrets=error_secrets,
                            truncated=False,
                        ),
                    }
                )
        raise RuntimeError(
            "DirectLayout optimization failed after "
            f"{max_retries} attempts; no base-layout fallback is permitted"
        ) from None

    service._request_reasoning_feedback = reasoning
    service._request_vlm_feedback = vlm
    renderer.render_views = render_views
    service.rendering_service = renderer
    pipeline._run_optimization_with_retry = run_optimization
    return tracking


def _select_completed_layout_state(tracking: dict[str, Any]) -> dict[str, Any]:
    """Ignore valid-but-stale files left by a failed optimization attempt."""

    completed_attempt = tracking.get("completed_attempt")
    states = tracking.get("layout_states")
    states = states if isinstance(states, list) else []
    base_states = [
        state
        for state in states
        if isinstance(state, dict)
        and state.get("rendered") is True
        and state.get("optimization_attempt") == 0
    ]
    completed_states = [
        state
        for state in states
        if isinstance(state, dict)
        and state.get("rendered") is True
        and state.get("optimization_attempt") == completed_attempt
    ]
    candidates = completed_states or base_states
    if not candidates:
        raise FileNotFoundError(
            "DirectLayout emitted no successfully validated/rendered native layout"
        )
    return candidates[-1]


class _ResponseObserver:
    def __init__(self, expected_identity: dict[str, str]) -> None:
        self.expected_identity = dict(expected_identity)
        self.identities: list[dict[str, str]] = []
        self.calls = 0
        self._tokens = 0
        self._token_reports = 0

    @property
    def tokens(self) -> int | None:
        return self._tokens if self._token_reports == self.calls else None

    def begin_call(self) -> None:
        self.calls += 1

    def observe(self, response: Any) -> None:
        identity = observed_model_identity(
            response,
            provider=self.expected_identity["provider"],
        )
        if identity not in self.identities:
            self.identities.append(identity)
        tokens = response_total_tokens(response)
        if tokens is not None:
            self._tokens += tokens
            self._token_reports += 1


class _ObservedCompletions:
    def __init__(self, delegate: Any, observer: _ResponseObserver) -> None:
        self._delegate = delegate
        self._observer = observer

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._observer.begin_call()
        response = self._delegate.create(*args, **kwargs)
        self._observer.observe(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ObservedChat:
    def __init__(self, delegate: Any, observer: _ResponseObserver) -> None:
        self._delegate = delegate
        self.completions = _ObservedCompletions(delegate.completions, observer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ObservedClient:
    def __init__(self, delegate: Any, observer: _ResponseObserver) -> None:
        self._delegate = delegate
        self.chat = _ObservedChat(delegate.chat, observer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _install_observed_clients(
    pipeline: Any,
    observer: _ResponseObserver,
) -> None:
    bev = _ObservedClient(pipeline.bev_client, observer)
    threed = _ObservedClient(pipeline.openai_client, observer)
    reasoning = _ObservedClient(pipeline.reasoning_client, observer)
    pipeline.bev_client = bev
    pipeline.openai_client = threed
    pipeline.reasoning_client = reasoning
    pipeline.generation_service.bev_client = bev
    pipeline.generation_service.openai_client = threed
    pipeline.optimization_service.bev_client = bev
    pipeline.optimization_service.openai_client = threed
    pipeline.optimization_service.reasoning_client = reasoning


def _controlled_prompt(
    *,
    original_prompt: str,
    plan: dict[str, Any],
    control: dict[str, Any],
    catalog: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    selector_by_slot = catalog.get("native_selector_by_slot")
    bindings = catalog.get("frozen_asset_bindings")
    assets = {
        str(item["new_object_id"]): item for item in catalog.get("asset_library", [])
    }
    if not isinstance(selector_by_slot, dict) or not isinstance(bindings, dict):
        raise RuntimeError("DirectLayout materialization lacks frozen selector/binding maps")
    counts: Counter[str] = Counter()
    rows = []
    contracts = []
    for item in control.get("objects", []):
        slot_id = str(item["slot_id"])
        selector = str(selector_by_slot[slot_id])
        counts[selector] += 1
        native_id = f"{selector}_{counts[selector]}"
        if native_id != slot_id:
            raise RuntimeError(
                "DirectLayout expanded slot does not match released category-counting "
                f"identity: slot={slot_id!r}, expected_native={native_id!r}"
            )
        asset_id = str(bindings[slot_id])
        asset = assets.get(asset_id)
        if not isinstance(asset, dict):
            raise RuntimeError(f"DirectLayout catalog lacks asset {asset_id!r}")
        size = [float(value) for value in asset["physical_dimensions"]]
        rows.append(
            {
                "native_object_id": native_id,
                "selector": selector,
                "asset_id": asset_id,
                "asset_category": asset["category"],
                "asset_description": asset["description"],
                "source_db": asset.get("source_db") or "imaginarium",
                "mesh_uri": asset.get("source_mesh_uri"),
                "runtime_mesh_path": asset.get("materialized_mesh_path"),
                "mesh_sha256": asset.get("mesh_sha256"),
                "canonical_front": asset.get("canonical_front"),
                "canonical_front_source": asset.get("canonical_front_source"),
                "bbox_size_local": asset.get("bbox_size_local"),
                "bbox_center_local": asset.get("bbox_center_local"),
                "size": size,
            }
        )
        contracts.append(
            {
                "selector": selector,
                "resulting_new_object_id": native_id,
                "fixed_size_in_meters": size,
                "fixed_size_in_native_pixels": [
                    value * 100.0 for value in size
                ],
                "canonical_front": asset.get("canonical_front"),
            }
        )
    addition = {
        "frozen_directlayout_contract": contracts,
        "public_object_plan": plan,
        "rules": [
            "Emit exactly the listed selectors in the listed order and no others.",
            "DirectLayout CSS uses px with 1 px = 0.01 m. Copy each "
            "fixed_size_in_native_pixels length/width exactly into BEV CSS, then "
            "copy length/width/height exactly into 3D CSS.",
            "Choose only center positions and orientation.",
            "DirectLayout zero orientation faces rendered native +Y. For an "
            "Imaginarium asset with canonical functional front local -Y, that "
            "local -Y side is rendered as native +Y at zero by the released "
            "renderer; choose orientation in this native frame.",
        ],
    }
    return (
        original_prompt
        + "\n\nCONTROLLED FROZEN-ASSET INPUT:\n"
        + json.dumps(addition, ensure_ascii=False, separators=(",", ":")),
        rows,
    )


def _materialize_slot_assets(
    *,
    catalog: dict[str, Any],
    catalog_path: Path,
    target: Path,
    expected: list[dict[str, Any]],
) -> None:
    assets = {
        str(item["new_object_id"]): item for item in catalog.get("asset_library", [])
    }
    target.mkdir(parents=True, exist_ok=True)
    for row in expected:
        asset = assets[row["asset_id"]]
        value = asset.get("materialized_mesh_path") or asset.get("source_mesh_uri")
        if not value:
            raise FileNotFoundError(f"asset {row['asset_id']!r} has no materialized mesh")
        source = Path(str(value))
        if not source.is_absolute():
            source = catalog_path.resolve().parent / source
        source = source.resolve()
        if source.suffix.lower() != ".glb" or not source.is_file():
            raise RuntimeError(
                f"DirectLayout requires frozen GLB input for {row['asset_id']!r}: {source}"
            )
        expected_hash = row.get("mesh_sha256")
        if not expected_hash or file_sha256(source) != expected_hash:
            raise RuntimeError(
                f"DirectLayout frozen mesh hash mismatch for {row['asset_id']!r}"
            )
        link = target / f"{row['native_object_id']}.glb"
        link.symlink_to(source)


def _observe_output(
    payload: Any, expected: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    if not isinstance(payload, list):
        return {"valid": False, "violations": ["native_output_not_array"], "actual_object_ids": []}
    actual = {
        str(item.get("new_object_id")): item
        for item in payload
        if isinstance(item, dict) and item.get("new_object_id")
    }
    expected_ids = {item["native_object_id"] for item in expected}
    violations = []
    if set(actual) != expected_ids:
        violations.append("object_inventory_mismatch")
    expected_by_id = {item["native_object_id"]: item for item in expected}
    for object_id in sorted(set(actual) & expected_ids):
        size = actual[object_id].get("size_in_meters")
        if isinstance(size, dict):
            observed = [size.get("length"), size.get("width"), size.get("height")]
        else:
            observed = size
        if not (
            isinstance(observed, list)
            and len(observed) == 3
            and all(
                abs(float(observed[index]) - expected_by_id[object_id]["size"][index])
                <= tolerance
                for index in range(3)
            )
        ):
            violations.append(f"fixed_size_changed:{object_id}")
    return {
        "valid": not violations,
        "violations": violations,
        "actual_object_ids": sorted(actual),
        "expected_object_ids": sorted(expected_ids),
        "exact_asset_bindings_sidecar": True,
        "retrieval_calls": 0,
    }


if __name__ == "__main__":
    main()
