#!/usr/bin/env python3
"""API3 Claude Opus 4.8 full10 runner with adaptive high thinking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
CORE_ROOT = REPO_ROOT / "tools" / "api3_anthropic_runner_v2"
MODELS_PATH = RUNNER_ROOT / "models.pod.json"
MODEL_KEY = "api3-claude-opus-4-8-high"
EXPECTED_BRIEF_IDS = tuple(f"brief_{index:02d}" for index in range(10))

sys.path.insert(0, str(SRC_ROOT))
from benchmark.scene_generation.frozen_two_stage import (  # noqa: E402
    ChatOptionPolicy,
    FrozenTwoStageOrchestrator,
    GenerationRunSpec,
    RetryPolicy,
    compatibility_source_manifest,
    make_api3_chat_route,
)
from benchmark.scene_generation.frozen_two_stage.compatibility import (  # noqa: E402
    load_frozen_core,
)

core = load_frozen_core(CORE_ROOT)
_CORE_SOURCE_MANIFEST = core._runner_source_manifest


def provider_route() -> Any:
    """Build the explicit Opus 4.8 High route from typed compatibility parts."""

    return make_api3_chat_route(
        option_policy=ChatOptionPolicy.adaptive_thinking(
            reasoning_effort="high"
        ),
        route_key="api3-opus48-high-chat-v1",
        runner_version=core.RUNNER_VERSION,
    )


def _request_value(
    *,
    model: core.ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
) -> dict[str, Any]:
    return provider_route().request_value(
        model=model,
        system_prompt=system_prompt,
        user_value=user_value,
        canonical_json_bytes=core.canonical_json_bytes,
    )


def _request_headers(model: core.ModelConfig, session_id: str) -> dict[str, str]:
    """Compatibility alias for the shared API3 gateway policy."""

    return provider_route().request_headers(model, session_id)


def _extract_api_message(
    response_body: bytes,
) -> tuple[bytes, bytes | None, bytes | None, Mapping[str, Any] | None]:
    """Compatibility alias for the shared Chat response codec."""

    return provider_route().extract_api_message(response_body).as_legacy_tuple()


def _runner_source_manifest() -> dict[str, Any]:
    files = []
    for path in (Path(__file__).resolve(), MODELS_PATH):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": core.sha256_file(path),
            }
        )
    payload = {
        "runner_version": "api3_opus48_high_runner_v1",
        "thinking": {"type": "adaptive"},
        "reasoning_effort": "high",
        "compatibility_layer_source_manifest": compatibility_source_manifest(),
        "provider_route": provider_route().public_dict(),
        "files": files,
        "core_source_manifest": _CORE_SOURCE_MANIFEST(),
    }
    return {
        **payload,
        "manifest_sha256": core.sha256_bytes(core.canonical_json_bytes(payload)),
    }


def configure_core() -> Any:
    """Return the frozen core without mutating its module-global hooks."""

    return core


def check() -> dict[str, Any]:
    runner = configure_core()
    report = runner.check_runner(
        briefs_path=CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        retriever_root=CORE_ROOT,
        source_manifest=_runner_source_manifest(),
    )
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    if model.reasoning_effort != "high" or model.preserved_thinking is not True:
        raise ValueError("thinking provenance fields are not enabled")
    request = _request_value(
        model=model,
        system_prompt="Return only the final answer.",
        user_value={"check": True},
    )
    thinking = request.get("thinking")
    if thinking != {"type": "adaptive"}:
        raise ValueError("formal request does not contain the expected thinking block")
    if request.get("reasoning_effort") != "high":
        raise ValueError("formal request does not contain reasoning_effort=high")
    return {
        **report,
        "model_key": MODEL_KEY,
        "wire_model": model.wire_model,
        "reasoning_effort": model.reasoning_effort,
        "preserved_thinking": model.preserved_thinking,
        "thinking": thinking,
        "formal_max_tokens": model.max_tokens,
    }


def run_full10(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    runner = configure_core()
    retriever = runner.RetrieverAdapter(CORE_ROOT)
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    briefs = runner._load_briefs(CORE_ROOT / "briefs.json")
    brief_ids = tuple(str(brief["brief_id"]) for brief in briefs)
    if brief_ids != EXPECTED_BRIEF_IDS:
        raise ValueError(f"unexpected frozen brief set: {brief_ids}")
    route = provider_route()
    retry_policy = RetryPolicy(
        max_infrastructure_retries=model.max_infrastructure_retries,
        retryable_transport_statuses=frozenset({"transport_failure"}),
        retryable_http_statuses=runner.RETRYABLE_HTTP_STATUSES,
        retry_delay_seconds=model.retry_delay_seconds,
        retry_ambiguous_timeouts=False,
        continue_after_case_failure=True,
    )
    execution_policy = {
        "schema_version": "api3_opus48_high_full10_policy_v1",
        "case_failure_policy": "record_and_continue_next_brief",
        "expected_brief_ids": list(EXPECTED_BRIEF_IDS),
        "request_protocol": "openai_chat_completions",
        "thinking": {"type": "adaptive"},
        "reasoning_effort": model.reasoning_effort,
        "preserved_thinking": model.preserved_thinking,
        "max_tokens": model.max_tokens,
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "preflight_requires_reasoning_signal": False,
        "preflight_records_reasoning_signal_diagnostic": True,
    }
    spec = GenerationRunSpec(
        provider_key=route.key,
        model_key=model.key,
        wire_model=model.wire_model,
        ordered_brief_ids=EXPECTED_BRIEF_IDS,
        briefs_path=CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy=execution_policy,
        summary_schema_version="api3_opus48_high_full10_summary_v1",
        summary_extra={
            "thinking": {"type": "adaptive"},
            "reasoning_effort": model.reasoning_effort,
        },
        source_manifest=_runner_source_manifest(),
    )

    def progress(record: Mapping[str, Any]) -> None:
        if record.get("event") != "case_terminal":
            return
        print(
            json.dumps(
                {
                    "brief_id": record["brief_id"],
                    "status": record["status"],
                    "eligible": record["eligible"],
                    "continued_after_case": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    summary, _ = FrozenTwoStageOrchestrator(runner, route).run(
        spec=spec,
        model=model,
        briefs=briefs,
        retriever=retriever,
        progress=progress,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            print(json.dumps(check(), ensure_ascii=False, sort_keys=True))
            return 0
        summary = run_full10(args.output_dir.expanduser().resolve())
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["failed"] == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
