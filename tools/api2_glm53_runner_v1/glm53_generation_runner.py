#!/usr/bin/env python3
"""API2 GLM-5.3 Responses adapter for the frozen HY34 scene10 generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
FROZEN_CORE_ROOT = REPO_ROOT / "tools" / "api3_anthropic_runner_v2"
MODELS_PATH = RUNNER_ROOT / "models.pod.json"
MODEL_KEY = "api2-glm-5-3"
EXPECTED_BRIEF_IDS = tuple(f"brief_{index:02d}" for index in range(10))

sys.path.insert(0, str(FROZEN_CORE_ROOT))
import generation_runner as core  # noqa: E402


_CORE_SOURCE_MANIFEST = core._runner_source_manifest


def _credential_parts(value: str) -> tuple[str, str]:
    credential = value.split("?", 1)[0]
    app_id, separator, app_key = credential.partition(":")
    if not separator or not app_id or not app_key:
        raise ValueError("API2_APP_CREDENTIAL must have APP_ID:APP_KEY form")
    return app_id, app_key


def _request_value(
    *,
    model: core.ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.wire_model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": core.canonical_json_bytes(user_value).decode("utf-8"),
            }
        ],
        "reasoning": {"effort": model.reasoning_effort or "max"},
        "max_output_tokens": model.max_tokens,
        "text": {"format": {"type": "json_object"}},
        "store": False,
        "stream": False,
    }


def _request_headers(model: core.ModelConfig, session_id: str) -> dict[str, str]:
    app_id, app_key = _credential_parts(model.api_key)
    cache_task_id = hashlib.md5(
        f"{time.time()}{app_id}{session_id}".encode("utf-8")
    ).hexdigest()
    authorization = (
        f"Bearer {app_id}:{app_key}"
        f"?provider=zhipu&model=glm-5.3&timeout=600"
        f"&cache_task_id={cache_task_id}"
    )
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": f"hy34-two-stage-generator/{core.RUNNER_VERSION} api2-glm53-v1",
        "Authorization": authorization,
    }


def _extract_api_message(
    response_body: bytes,
) -> tuple[bytes, bytes | None, bytes | None, dict[str, Any] | None]:
    payload = json.loads(response_body.decode("utf-8", errors="strict"))
    if not isinstance(payload, dict):
        raise ValueError("GLM-5.3 response must be a JSON object")
    if payload.get("error"):
        raise ValueError("GLM-5.3 response contains an error object")
    status = str(payload.get("status") or "")
    if status != "completed":
        incomplete = payload.get("incomplete_details")
        incomplete = incomplete if isinstance(incomplete, dict) else {}
        reason = str(incomplete.get("reason") or status or "unknown")
        raise ValueError(f"GLM-5.3 response is not complete: {reason}")

    message_texts: list[str] = []
    reasoning_texts: list[str] = []
    output = payload.get("output")
    output = output if isinstance(output, list) else []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        content = item.get("content")
        content = content if isinstance(content, list) else []
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            if item_type == "message" and part.get("type") == "output_text":
                message_texts.append(part["text"])
            elif item_type == "reasoning" and part.get("type") == "reasoning_text":
                reasoning_texts.append(part["text"])
    content_text = "\n".join(message_texts).strip()
    if not content_text:
        raise ValueError("GLM-5.3 completed response has no output_text message")
    reasoning = "\n".join(reasoning_texts).strip()
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else None
    return (
        content_text.encode("utf-8"),
        reasoning.encode("utf-8") if reasoning else None,
        None,
        usage,
    )


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
        "adapter_files": files,
    }
    return {
        **payload,
        "manifest_sha256": core.sha256_bytes(core.canonical_json_bytes(payload)),
    }


def configure_core() -> Any:
    core._request_value = _request_value
    core._request_headers = _request_headers
    core._extract_api_message = _extract_api_message
    core._runner_source_manifest = _source_manifest
    return core


def preflight() -> dict[str, Any]:
    runner = configure_core()
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    request = runner._request_value(
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
            request_headers=runner._request_headers(
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
        content, _, _, usage = runner._extract_api_message(result.response_body)
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "invalid_api_response",
            "error_type": type(exc).__name__,
        }
    return {
        "ok": bool(isinstance(parsed, dict) and parsed.get("ok") is True),
        "status": "passed" if isinstance(parsed, dict) and parsed.get("ok") is True else "unexpected_content",
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
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    if model.max_infrastructure_retries != 3:
        raise ValueError("GLM-5.3 runner must provide exactly three retries")
    briefs = runner._load_briefs(FROZEN_CORE_ROOT / "briefs.json")
    brief_ids = tuple(str(brief["brief_id"]) for brief in briefs)
    if brief_ids != EXPECTED_BRIEF_IDS:
        raise ValueError(f"unexpected frozen brief set: {brief_ids}")
    retriever = runner.RetrieverAdapter(FROZEN_CORE_ROOT)
    runner.initialize_run(
        output_root=output_root,
        model=model,
        briefs_path=FROZEN_CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        retriever=retriever,
    )
    runner.write_json_exclusive(
        output_root / "execution_policy.json",
        {
            "schema_version": "glm53_scene10_execution_policy_v1",
            "case_failure_policy": "record_and_continue_next_brief",
            "expected_brief_ids": list(EXPECTED_BRIEF_IDS),
            "request_protocol": "openai_responses_api",
            "reasoning_effort": model.reasoning_effort,
            "max_output_tokens": model.max_tokens,
            "maximum_infrastructure_retries": model.max_infrastructure_retries,
            "required_http_retry_statuses": [429, 500],
        },
    )
    stage_a_prompt = runner.DEFAULT_STAGE_A_PROMPT.read_text(encoding="utf-8")
    stage_c_prompt = runner.DEFAULT_STAGE_C_PROMPT.read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []
    for brief in briefs:
        result = runner.run_case(
            output_root=output_root,
            model=model,
            brief=brief,
            retriever=retriever,
            stage_a_prompt=stage_a_prompt,
            stage_c_prompt=stage_c_prompt,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "brief_id": brief["brief_id"],
                    "status": result.get("status"),
                    "eligible": result.get(
                        "eligible_for_strict_one_shot_evaluation"
                    ),
                    "continued_after_case": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema_version": "hy34_two_stage_run_summary_v2",
        "model_key": model.key,
        "model_label": model.label,
        "requested_briefs": len(briefs),
        "processed_briefs": len(results),
        "complete": sum(item["status"] == "complete" for item in results),
        "failed": sum(item["status"] != "complete" for item in results),
        "eligible": sum(
            item["eligible_for_strict_one_shot_evaluation"] for item in results
        ),
        "stopped_early": False,
        "results": results,
        "completed_at": runner.utc_now(),
    }
    runner.write_json_exclusive(output_root / "summary.json", summary)
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
