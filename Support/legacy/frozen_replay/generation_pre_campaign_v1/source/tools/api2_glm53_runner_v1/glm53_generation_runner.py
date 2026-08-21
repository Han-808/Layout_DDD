#!/usr/bin/env python3
"""API2 GLM-5.3 Responses adapter for the frozen HY34 scene10 generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
FROZEN_CORE_ROOT = REPO_ROOT / "tools" / "api3_anthropic_runner_v2"
MODELS_PATH = RUNNER_ROOT / "models.pod.json"
MODEL_KEY = "api2-glm-5-3"
EXPECTED_BRIEF_IDS = tuple(f"brief_{index:02d}" for index in range(10))

sys.path.insert(0, str(SRC_ROOT))
from benchmark.scene_generation.frozen_two_stage import (  # noqa: E402
    FrozenTwoStageOrchestrator,
    GenerationRunSpec,
    RetryPolicy,
    compatibility_source_manifest,
    make_api2_responses_route,
)
from benchmark.scene_generation.frozen_two_stage.compatibility import (  # noqa: E402
    load_frozen_core,
)
from benchmark.scene_generation.frozen_two_stage.providers.gateways import (  # noqa: E402
    parse_api2_credential,
)


core = load_frozen_core(FROZEN_CORE_ROOT)
_CORE_SOURCE_MANIFEST = core._runner_source_manifest


def _credential_parts(value: str) -> tuple[str, str]:
    """Compatibility alias for the shared API2 credential parser."""

    return parse_api2_credential(value)


def provider_route() -> Any:
    """Build the instance-scoped GLM route described by the architecture doc."""

    return make_api2_responses_route(
        provider="zhipu",
        gateway_model="glm-5.3",
        user_agent_suffix="api2-glm53-v1",
        default_reasoning_effort="max",
        route_key="api2-glm53-responses-v1",
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
    return provider_route().request_headers(model, session_id)


def _extract_api_message(
    response_body: bytes,
) -> tuple[bytes, bytes | None, bytes | None, dict[str, Any] | None]:
    return provider_route().extract_api_message(response_body).as_legacy_tuple()


def _source_manifest() -> dict[str, Any]:
    files = []
    for name in ("glm53_generation_runner.py", "models.pod.json"):
        path = RUNNER_ROOT / name
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": core.sha256_file(path),
            }
        )
    payload = {
        "runner_version": core.RUNNER_VERSION,
        "adapter_version": "api2-glm53-responses-v1",
        "frozen_core_root": str(FROZEN_CORE_ROOT),
        "frozen_core_source_manifest": _CORE_SOURCE_MANIFEST(),
        "compatibility_layer_source_manifest": compatibility_source_manifest(),
        "provider_route": provider_route().public_dict(),
        "adapter_files": files,
    }
    return {
        **payload,
        "manifest_sha256": core.sha256_bytes(core.canonical_json_bytes(payload)),
    }


def configure_core() -> Any:
    """Return the frozen core without mutating its module-global hooks."""

    return core


def preflight() -> dict[str, Any]:
    runner = configure_core()
    runner.RetrieverAdapter(FROZEN_CORE_ROOT)
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    request = _request_value(
        model=model,
        system_prompt="Return one JSON object and no surrounding text.",
        user_value={"preflight": "reply with exactly {\"ok\":true}"},
    )
    request_body = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    max_attempts = model.max_infrastructure_retries + 1
    result = None
    for attempt_number in range(1, max_attempts + 1):
        result = runner.post_once(
            model.endpoint,
            request_body,
            connect_timeout=30.0,
            read_timeout=min(model.timeout_seconds, 600.0),
            request_headers=_request_headers(
                model, f"glm53-preflight-{attempt_number}"
            ),
        )
        retryable_transport = result.status in {
            "transport_failure",
            "transport_ambiguous",
        }
        retryable_http = result.http_status in {429, 500}
        if (retryable_transport or retryable_http) and attempt_number < max_attempts:
            if model.retry_delay_seconds:
                time.sleep(model.retry_delay_seconds)
            continue
        break
    assert result is not None
    if result.status != "response":
        return {
            "ok": False,
            "status": result.status,
            "stage": result.stage,
            "error_type": result.error_type,
            "attempts": attempt_number,
        }
    if result.http_status is None or not 200 <= result.http_status < 300:
        return {
            "ok": False,
            "status": "http_error",
            "http_status": result.http_status,
            "http_reason": result.http_reason,
            "attempts": attempt_number,
        }
    assert result.response_body is not None
    try:
        content, _, _, usage = _extract_api_message(result.response_body)
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "invalid_api_response",
            "error_type": type(exc).__name__,
        }
    return {
        "ok": bool(isinstance(parsed, dict) and parsed.get("ok") is True),
        "status": (
            "passed"
            if isinstance(parsed, dict) and parsed.get("ok") is True
            else "unexpected_content"
        ),
        "http_status": result.http_status,
        "model": model.wire_model,
        "reasoning_effort": model.reasoning_effort,
        "usage_recorded": isinstance(usage, dict),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "attempts": attempt_number,
    }


def run_scene10(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    runner = configure_core()
    retriever = runner.RetrieverAdapter(FROZEN_CORE_ROOT)
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    if model.max_infrastructure_retries != 3:
        raise ValueError("GLM-5.3 runner must provide exactly three retries")
    briefs = runner._load_briefs(FROZEN_CORE_ROOT / "briefs.json")
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
        "schema_version": "glm53_scene10_execution_policy_v1",
        "case_failure_policy": "record_and_continue_next_brief",
        "expected_brief_ids": list(EXPECTED_BRIEF_IDS),
        "request_protocol": "openai_responses_api",
        "reasoning_effort": model.reasoning_effort,
        "max_output_tokens": model.max_tokens,
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "required_http_retry_statuses": [429, 500],
    }
    spec = GenerationRunSpec(
        provider_key=route.key,
        model_key=model.key,
        wire_model=model.wire_model,
        ordered_brief_ids=EXPECTED_BRIEF_IDS,
        briefs_path=FROZEN_CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy=execution_policy,
        source_manifest=_source_manifest(),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("preflight")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        runner = configure_core()
        if args.command == "check":
            report = runner.check_runner(
                briefs_path=FROZEN_CORE_ROOT / "briefs.json",
                models_path=MODELS_PATH,
                retriever_root=FROZEN_CORE_ROOT,
                source_manifest=_source_manifest(),
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "preflight":
            report = preflight()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0 if report.get("ok") is True else 2
        summary = run_scene10(args.output_dir.expanduser().resolve())
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["complete"] == 10 and summary["failed"] == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
