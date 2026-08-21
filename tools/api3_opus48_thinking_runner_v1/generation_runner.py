#!/usr/bin/env python3
"""API3 Claude Opus 4.8 full10 runner with adaptive high thinking."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
CORE_ROOT = REPO_ROOT / "tools" / "api3_anthropic_runner_v2"
MODELS_PATH = RUNNER_ROOT / "models.pod.json"
MODEL_KEY = "api3-claude-opus-4-8-high"
EXPECTED_BRIEF_IDS = tuple(f"brief_{index:02d}" for index in range(10))

sys.path.insert(0, str(CORE_ROOT))

# This adapter is also named ``generation_runner.py``.  A normal
# ``import generation_runner`` therefore resolves to this file when it is
# executed directly (and is already present under that name when imported by
# preflight.py).  Load the frozen core runner from its exact path under a
# private module name so the adapter can never import itself.
_CORE_MODULE_NAME = "_api3_anthropic_runner_v2_generation_runner"
_CORE_MODULE_PATH = CORE_ROOT / "generation_runner.py"
_CORE_SPEC = importlib.util.spec_from_file_location(
    _CORE_MODULE_NAME, _CORE_MODULE_PATH
)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise ImportError(f"cannot load core runner from {_CORE_MODULE_PATH}")
core = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_MODULE_NAME] = core
_CORE_SPEC.loader.exec_module(core)

_CORE_SOURCE_MANIFEST = core._runner_source_manifest


def _request_value(
    *,
    model: core.ModelConfig,
    system_prompt: str,
    user_value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.wire_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": core.canonical_json_bytes(user_value).decode("utf-8"),
            },
        ],
        "max_tokens": model.max_tokens,
        "stream": False,
        "thinking": {"type": "adaptive"},
        "reasoning_effort": "high",
    }


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
        "files": files,
        "core_source_manifest": _CORE_SOURCE_MANIFEST(),
    }
    return {
        **payload,
        "manifest_sha256": core.sha256_bytes(core.canonical_json_bytes(payload)),
    }


def configure_core() -> Any:
    core._request_value = _request_value
    core._runner_source_manifest = _runner_source_manifest
    return core


def check() -> dict[str, Any]:
    runner = configure_core()
    report = runner.check_runner(
        briefs_path=CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        retriever_root=CORE_ROOT,
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
    model = runner._load_model_config(MODELS_PATH, MODEL_KEY)
    briefs = runner._load_briefs(CORE_ROOT / "briefs.json")
    brief_ids = tuple(str(brief["brief_id"]) for brief in briefs)
    if brief_ids != EXPECTED_BRIEF_IDS:
        raise ValueError(f"unexpected frozen brief set: {brief_ids}")
    retriever = runner.RetrieverAdapter(CORE_ROOT)
    runner.initialize_run(
        output_root=output_root,
        model=model,
        briefs_path=CORE_ROOT / "briefs.json",
        models_path=MODELS_PATH,
        retriever=retriever,
    )
    runner.write_json_exclusive(
        output_root / "execution_policy.json",
        {
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
        "schema_version": "api3_opus48_high_full10_summary_v1",
        "model_key": model.key,
        "model_label": model.label,
        "requested_briefs": len(briefs),
        "processed_briefs": len(results),
        "complete": sum(item["status"] == "complete" for item in results),
        "failed": sum(item["status"] != "complete" for item in results),
        "eligible": sum(
            bool(item["eligible_for_strict_one_shot_evaluation"])
            for item in results
        ),
        "stopped_early": False,
        "thinking": {"type": "adaptive"},
        "reasoning_effort": model.reasoning_effort,
        "results": results,
        "completed_at": runner.utc_now(),
    }
    runner.write_json_exclusive(output_root / "summary.json", summary)
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
