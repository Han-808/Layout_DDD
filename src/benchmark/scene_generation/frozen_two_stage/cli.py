"""Generic CLI for config-only frozen two-stage model onboarding.

Usage is intentionally separate from ``layout-ddd-generate``.  The architecture
and evaluator boundary are documented in
``docs/generation_transport_compatibility.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    ModelMetadata,
    StaticCoreMetadata,
    inspect_brief_ids,
    inspect_core_metadata,
    inspect_model_metadata,
    load_frozen_core,
    load_runtime_inputs,
    select_static_brief_ids,
    validate_runtime_consistency,
)
from benchmark.scene_generation.frozen_two_stage.config import (
    FrozenTwoStageRunConfig,
    load_run_config,
)
from benchmark.scene_generation.frozen_two_stage.orchestrator import (
    FrozenTwoStageOrchestrator,
    SafeProgress,
)
from benchmark.scene_generation.frozen_two_stage.provenance import (
    compatibility_source_manifest,
)
from benchmark.scene_generation.frozen_two_stage.spec import GenerationRunSpec
from benchmark.scene_generation.frozen_two_stage.trust import (
    TrustInventory,
    TrustReport,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _source_manifest(
    config: FrozenTwoStageRunConfig,
    static_core: StaticCoreMetadata,
    route: Any,
    trust_report: TrustReport,
) -> dict[str, Any]:
    payload = {
        "schema_version": "frozen_two_stage_configured_source_manifest_v1",
        "run_config": {
            "path": str(config.path),
            "sha256": config.sha256,
        },
        "compatibility_layer": compatibility_source_manifest(),
        "trust": trust_report.to_public_dict(),
        "frozen_core": {
            "root": str(static_core.root),
            "runner_version": static_core.runner_version,
            "generation_runner_sha256": static_core.runner_sha256,
            "bundle_declaration_sha256": (
                trust_report.core_bundle.declaration_sha256
            ),
        },
        "retriever": trust_report.retriever_bundle.to_public_dict(),
        "provider_route": route.source_manifest(),
        "preflight_contract": "post_preflight_only",
    }
    return {
        **payload,
        "manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest(),
    }


def _generation_spec(
    config: FrozenTwoStageRunConfig,
    *,
    model: Any,
    route: Any,
    retry_policy: Any,
    source_manifest: Mapping[str, Any],
    output_root: Path,
) -> GenerationRunSpec:
    """Build the same validated spec for static check and an actual run."""

    return GenerationRunSpec(
        provider_key=route.key,
        model_key=model.key,
        wire_model=model.wire_model,
        ordered_brief_ids=config.ordered_brief_ids,
        briefs_path=config.briefs_path,
        models_path=config.models_path,
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy=config.execution_policy,
        summary_schema_version=config.summary_schema_version,
        summary_extra=config.summary_extra,
        generation_parameters={
            "run_config_schema": config.to_public_dict()["schema_version"],
            "run_config_sha256": config.sha256,
            "route_kind": config.route.kind,
        },
        source_manifest=source_manifest,
    )


def _validate_route_model_alignment(
    config: FrozenTwoStageRunConfig,
    *,
    runner_version: str,
    model: Any,
) -> None:
    """Reject config/model combinations that would misstate frozen options."""

    if config.route.runner_version != runner_version:
        raise ValueError(
            "route runner_version does not match frozen core: "
            f"route={config.route.runner_version!r} core={runner_version!r}"
        )
    if (
        config.route.kind == "api3_chat"
        and config.route.chat_option_style == "adaptive_thinking"
    ):
        actual_effort = getattr(model, "reasoning_effort", None)
        if actual_effort != config.route.reasoning_effort:
            raise ValueError(
                "adaptive-thinking effort differs between route and model config: "
                f"route={config.route.reasoning_effort!r} model={actual_effort!r}"
            )
        if getattr(model, "preserved_thinking", None) is not True:
            raise ValueError(
                "adaptive-thinking route requires preserved_thinking=true in "
                "the model config"
            )


@dataclass(frozen=True, slots=True)
class _StaticPreparedRun:
    """All validation completed before credential or retriever initialization."""

    trust_report: TrustReport
    static_core: StaticCoreMetadata
    static_model: ModelMetadata
    brief_ids: tuple[str, ...]
    route: Any
    retry_policy: Any
    source_manifest: Mapping[str, Any]
    spec: GenerationRunSpec


def _prepare_static(
    config: FrozenTwoStageRunConfig,
    *,
    output_root: Path,
    trust_manifest: str | Path | None,
) -> _StaticPreparedRun:
    """Hash and parse all configurable inputs without executing core code."""

    inventory = TrustInventory.load(trust_manifest)
    trust_report = inventory.verify_run_inputs(
        core_root=config.core_root,
        models_path=config.models_path,
        briefs_path=config.briefs_path,
        retriever_root=config.retriever_root,
        run_config_path=config.path,
    )
    if trust_report.run_config_file.sha256 != config.sha256:
        raise ValueError(
            "loaded run config hash differs from the trusted file hash"
        )
    trusted_config = load_run_config(config.path)
    if trusted_config.to_public_dict() != config.to_public_dict():
        raise ValueError(
            "in-memory run config differs from its trusted on-disk declaration"
        )
    static_core = inspect_core_metadata(config.core_root)
    static_model = inspect_model_metadata(config.models_path, config.model_key)
    brief_ids = select_static_brief_ids(
        inspect_brief_ids(config.briefs_path), config.ordered_brief_ids
    )
    _validate_route_model_alignment(
        config,
        runner_version=static_core.runner_version,
        model=static_model,
    )
    route = config.route.build_route()
    retry_policy = config.retry.build_policy(
        max_infrastructure_retries=static_model.max_infrastructure_retries,
        retry_delay_seconds=static_model.retry_delay_seconds,
    )
    source_manifest = _source_manifest(config, static_core, route, trust_report)
    spec = _generation_spec(
        config,
        model=static_model,
        route=route,
        retry_policy=retry_policy,
        source_manifest=source_manifest,
        output_root=output_root,
    )
    return _StaticPreparedRun(
        trust_report=trust_report,
        static_core=static_core,
        static_model=static_model,
        brief_ids=brief_ids,
        route=route,
        retry_policy=retry_policy,
        source_manifest=source_manifest,
        spec=spec,
    )


def check_config(
    config: FrozenTwoStageRunConfig,
    *,
    trust_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Perform static checks without loading credentials, retriever, or network."""

    prepared = _prepare_static(
        config,
        output_root=config.path.parent / ".frozen_two_stage_check_not_created",
        trust_manifest=trust_manifest,
    )
    return {
        "valid": True,
        "schema_version": "frozen_two_stage_config_check_v1",
        "run_config_sha256": config.sha256,
        "model_key": prepared.spec.model_key,
        "wire_model": prepared.spec.wire_model,
        "ordered_brief_ids": list(prepared.brief_ids),
        "provider_route": prepared.route.public_dict(),
        "maximum_infrastructure_retries": (
            prepared.retry_policy.max_infrastructure_retries
        ),
        "maximum_attempts_per_stage": (
            prepared.retry_policy.maximum_attempts_per_stage
        ),
        "source_manifest_sha256": prepared.source_manifest["manifest_sha256"],
        "trust": prepared.trust_report.to_public_dict(),
        "credential_loaded": False,
        "retriever_loaded": False,
        "network_used": False,
        "preflight_contract": "post_preflight_only",
    }


def run_config(
    config: FrozenTwoStageRunConfig,
    *,
    output_root: str | Path,
    progress: SafeProgress | None = None,
    trust_manifest: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Execute an already-preflighted config through the trusted workflow."""

    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    prepared = _prepare_static(
        config,
        output_root=destination,
        trust_manifest=trust_manifest,
    )
    # Everything configurable has now been hash/static validated.  Only this
    # post-preflight section may import the core, read its credential, or build
    # the retriever runtime.
    core = load_frozen_core(config.core_root)
    runtime = load_runtime_inputs(
        core,
        models_path=config.models_path,
        model_key=config.model_key,
        briefs_path=config.briefs_path,
        ordered_brief_ids=config.ordered_brief_ids,
        retriever_root=config.retriever_root,
    )
    validate_runtime_consistency(
        core=core,
        static_core=prepared.static_core,
        model=runtime.model,
        static_model=prepared.static_model,
        briefs=runtime.briefs,
        static_brief_ids=prepared.brief_ids,
    )
    _validate_route_model_alignment(
        config,
        runner_version=str(core.RUNNER_VERSION),
        model=runtime.model,
    )
    return FrozenTwoStageOrchestrator(core, prepared.route).run(
        spec=prepared.spec,
        model=runtime.model,
        briefs=runtime.briefs,
        retriever=runtime.retriever,
        progress=progress,
    )


def _print_progress(record: Mapping[str, Any]) -> None:
    if record.get("event") not in {"case_terminal", "run_terminal"}:
        return
    print(json.dumps(dict(record), ensure_ascii=False, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate without credentials/network")
    check.add_argument("--run-config", type=Path, required=True)
    check.add_argument("--trust-manifest", type=Path)
    run = subparsers.add_parser("run", help="execute the configured frozen workflow")
    run.add_argument("--run-config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--trust-manifest", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_run_config(args.run_config)
        if args.command == "check":
            print(
                json.dumps(
                    check_config(config, trust_manifest=args.trust_manifest),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        summary, stopped = run_config(
            config,
            output_root=args.output_dir,
            progress=_print_progress,
            trust_manifest=args.trust_manifest,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 2 if stopped or summary["failed"] else 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
