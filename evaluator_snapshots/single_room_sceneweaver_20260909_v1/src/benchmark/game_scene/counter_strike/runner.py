"""Resumable one-capture runner for the frozen Counter-Strike benchmark.

The generic Game route remains unchanged.  This module is the isolated
Counter-Strike adapter that:

1. captures the original Three.js runtime exactly once;
2. verifies the benchmark-owned spawn contract against source bytes and the
   exported canonical transform;
3. builds the canonical Game case bundle;
4. runs canonical Collision/Navigability/Style plus the five CS static-spatial
   metrics over that same immutable capture.

No credential value is accepted on the command line or persisted.  Model
credentials are resolved by the shared transport from the environment variable
named in the model configuration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from benchmark.game_scene.case_bundle import build_game_case_bundle
from benchmark.game_scene.mode import GameModeConfig, load_game_mode_config
from benchmark.rendering.browser import HeadlessBrowserRenderer
from benchmark.resources import runtime_resource_path
from benchmark.utils.io import read_json, write_json
from benchmark.visual_judge import build_openai_compatible_vlm_judge

from .collision_evidence import CounterStrikeFrozenCaptureRenderer
from .evaluator import (
    CANONICAL_L1_METRICS,
    CANONICAL_L3_METRICS,
    COUNTER_STRIKE_L4_METRICS,
)
from .evidence import load_counter_strike_frozen_evidence
from .integration import evaluate_counter_strike_frozen_capture
from .judge import build_counter_strike_visual_judge
from .loader import (
    CounterStrikeBenchmarkConfig,
    CounterStrikeCaseContract,
    load_counter_strike_benchmark_config,
    load_counter_strike_case_contract,
)


COUNTER_STRIKE_RUNNER_VERSION = "counter_strike_one_capture_runner_v1"
_PHASES = frozenset({"all", "capture", "evaluate"})
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


class CounterStrikeRunnerError(RuntimeError):
    """Raised when the isolated CS execution contract cannot be satisfied."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike runner failed [{code}]: {message}")


def run_counter_strike_case(
    *,
    game_root: str | Path,
    case_contract_path: str | Path,
    out_dir: str | Path,
    model_config: dict[str, Any] | str | Path,
    game_mode_config: GameModeConfig | str | Path = (
        str(runtime_resource_path("configs/game/game_mode_canonical_v1.yaml"))
    ),
    benchmark_config: CounterStrikeBenchmarkConfig | str | Path = (
        str(
            runtime_resource_path(
                "configs/game/counter_strike/benchmark_v1.yaml"
            )
        )
    ),
    entry_html: str | Path = "index.html",
    three_replacement: str | Path | None = None,
    phase: str = "all",
    official_mode: bool = True,
) -> dict[str, Any]:
    """Run one source case with capture-level resume and complete-only scoring."""

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in _PHASES:
        raise CounterStrikeRunnerError(
            "phase_invalid",
            f"phase must be one of {sorted(_PHASES)}",
        )
    source_root = Path(game_root).expanduser().resolve()
    if not source_root.is_dir():
        raise CounterStrikeRunnerError(
            "game_root_missing",
            f"game source directory does not exist: {source_root}",
        )
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    capture_dir = destination / "renders"
    run_manifest_path = destination / "counter_strike_run_manifest.json"
    mode = (
        game_mode_config
        if isinstance(game_mode_config, GameModeConfig)
        else load_game_mode_config(game_mode_config)
    )
    config = (
        benchmark_config
        if isinstance(benchmark_config, CounterStrikeBenchmarkConfig)
        else load_counter_strike_benchmark_config(benchmark_config)
    )
    model_payload, model_path, model_sha256 = _load_model_config(model_config)
    contract_path = Path(case_contract_path).expanduser().resolve()
    if not contract_path.is_file():
        raise CounterStrikeRunnerError(
            "case_contract_missing",
            f"case contract does not exist: {contract_path}",
        )
    entry = _resolve_under(source_root, entry_html, label="entry_html")
    replacement = (
        _resolve_file(three_replacement, label="three_replacement")
        if three_replacement is not None
        else None
    )
    style_spec_path = mode.default_visual_style_spec_path
    style_spec = read_json(style_spec_path)
    if not isinstance(style_spec, dict):
        raise CounterStrikeRunnerError(
            "style_spec_invalid",
            "the configured visual style spec must be a JSON object",
        )

    run_record: dict[str, Any] = {
        "runner_version": COUNTER_STRIKE_RUNNER_VERSION,
        "status": "running",
        "phase": normalized_phase,
        "official_mode": bool(official_mode),
        "scope": "counter_strike_static_3d_environment_only",
        "source": {
            "root": source_root.as_posix(),
            "entry_html": entry.as_posix(),
            "entry_html_sha256": _sha256(entry),
            "three_replacement": (
                replacement.as_posix() if replacement is not None else None
            ),
            "three_replacement_sha256": (
                _sha256(replacement) if replacement is not None else None
            ),
        },
        "inputs": {
            "game_mode_config": {
                "path": mode.path.as_posix(),
                "sha256": _sha256(mode.path),
            },
            "benchmark_config": {
                "path": config.path.as_posix(),
                "sha256": config.sha256,
            },
            "case_contract": {
                "path": contract_path.as_posix(),
                "sha256": _sha256(contract_path),
            },
            "visual_style_spec": {
                "path": style_spec_path.as_posix(),
                "sha256": _sha256(style_spec_path),
            },
            "model_config": {
                "path": model_path,
                "sha256": model_sha256,
                "model": str(model_payload.get("model") or ""),
                "endpoint": str(model_payload.get("endpoint") or ""),
                "api_key_env": str(model_payload.get("api_key_env") or ""),
            },
        },
        "credentials": {
            "embedded": False,
            "accepted_on_cli": False,
            "source": "environment_variable_named_by_model_config",
        },
        "capture": {
            "directory": capture_dir.as_posix(),
            "performed_this_invocation": False,
            "reused": False,
        },
    }
    write_json(run_manifest_path, run_record)

    try:
        capture_performed = False
        if capture_dir.is_dir():
            frozen = load_counter_strike_frozen_evidence(
                capture_dir,
                benchmark_config=config,
            )
            capture_manifest = read_json(frozen.manifest_path)
            capture_reused = True
        else:
            if normalized_phase == "evaluate":
                raise CounterStrikeRunnerError(
                    "capture_missing",
                    "phase=evaluate requires an existing hash-valid renders/",
                )
            renderer = HeadlessBrowserRenderer(
                entry_html=entry,
                game_root=source_root,
                three_replacement=replacement,
                **mode.renderer_kwargs,
            )
            capture_manifest = renderer.capture_game_source(
                out_dir=capture_dir,
                scene_id=f"{_safe_id(contract_path.stem)}_scene",
                request_id=f"{_safe_id(contract_path.stem)}_request",
                scene_type="counter_strike_static_arena",
                require_probe=True,
            )
            frozen = load_counter_strike_frozen_evidence(
                capture_dir,
                benchmark_config=config,
            )
            capture_performed = True
            capture_reused = False

        exported_scene_path = Path(
            str(capture_manifest.get("exported_scene") or "")
        ).expanduser().resolve()
        if not exported_scene_path.is_file():
            raise CounterStrikeRunnerError(
                "exported_scene_missing",
                "the frozen capture does not contain an exported canonical scene",
            )
        exported_scene = read_json(exported_scene_path)
        if not isinstance(exported_scene, dict):
            raise CounterStrikeRunnerError(
                "exported_scene_invalid",
                "the exported canonical scene must be a JSON object",
            )
        contract = load_counter_strike_case_contract(
            contract_path,
            source_root=source_root,
            canonical_scene=exported_scene,
        )
        _assert_case_identity(contract, capture_manifest)
        run_record["case_id"] = contract.case_id
        run_record["capture"] = {
            "directory": capture_dir.as_posix(),
            "performed_this_invocation": capture_performed,
            "reused": capture_reused,
            "manifest": frozen.manifest_path.as_posix(),
            "manifest_sha256": frozen.manifest_sha256,
            "exported_scene": exported_scene_path.as_posix(),
            "exported_scene_sha256": _sha256(exported_scene_path),
            "object_count": len(exported_scene.get("objects") or []),
            "global_view_count": len(frozen.global_views),
            "regional_view_count": len(frozen.regional_views),
        }
        run_record["contract"] = _contract_record(contract)
        write_json(run_manifest_path, run_record)

        if normalized_phase == "capture":
            run_record["status"] = "captured"
            write_json(run_manifest_path, run_record)
            return {
                "status": "captured",
                "run_manifest": run_manifest_path.as_posix(),
                "capture_manifest": capture_manifest,
            }

        bundle_root = build_game_case_bundle(
            exported_scene,
            out_dir=destination / "case_bundle",
            case_id=contract.case_id,
            evaluation_profile=mode.evaluation_profile,
            instruction=mode.instruction,
            visual_style_spec=style_spec,
        )
        frozen_renderer = CounterStrikeFrozenCaptureRenderer(
            capture_dir=capture_dir,
            evidence_out_dir=destination / "collision_evidence",
            benchmark_config=config,
        )
        canonical_judge = build_openai_compatible_vlm_judge(
            deepcopy(model_payload)
        )
        cs_visual_judge = build_counter_strike_visual_judge(
            deepcopy(model_payload),
            benchmark_config=config,
            evidence_repair_dir=(
                destination
                / "counter_strike_l4"
                / "observation_repairs"
            ),
        )
        result = evaluate_counter_strike_frozen_capture(
            out_dir=destination,
            capture_dir=capture_dir,
            canonical_case_bundle=bundle_root,
            benchmark_config=config,
            case_contract=contract,
            canonical_vlm_judge=canonical_judge,
            counter_strike_visual_judge=cs_visual_judge,
            renderer=frozen_renderer,
            official_mode=official_mode,
        )
        report = result["evaluation_report"]
        run_record["status"] = (
            "complete"
            if report.get("benchmark_score_status") == "complete"
            else "incomplete"
        )
        run_record["evaluation"] = {
            "report": result["report_path"],
            "report_sha256": _sha256(Path(result["report_path"])),
            "benchmark_score": report.get("benchmark_score"),
            "benchmark_score_status": report.get("benchmark_score_status"),
            "evaluation_status": report.get("evaluation_status"),
        }
        write_json(run_manifest_path, run_record)
        result["run_manifest"] = run_record
        return result
    except Exception as exc:
        run_record["status"] = "failed"
        run_record["failure"] = {
            "error_type": type(exc).__name__,
        }
        write_json(run_manifest_path, run_record)
        raise


def aggregate_counter_strike_runs(
    *,
    runs_root: str | Path,
    expected_case_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write a compact corpus matrix without inventing values for missing runs."""

    root = Path(runs_root).expanduser().resolve()
    cases_root = root / "cases"
    expected = tuple(str(value) for value in (expected_case_ids or ()))
    report_paths = sorted(
        cases_root.glob("*/counter_strike_evaluation_report.json")
    )
    rows: list[dict[str, Any]] = []
    reports_by_case: dict[str, Path] = {}
    metric_names = (
        *CANONICAL_L1_METRICS,
        *CANONICAL_L3_METRICS,
        *COUNTER_STRIKE_L4_METRICS,
    )
    for path in report_paths:
        payload = read_json(path)
        vector = (
            payload.get("metric_vector")
            if isinstance(payload.get("metric_vector"), dict)
            else {}
        )
        case_id = str(
            payload.get("request_id")
            or path.parent.name
        )
        manifest_path = path.parent / "counter_strike_run_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            case_id = str(manifest.get("case_id") or case_id)
        reports_by_case[case_id] = path
        row: dict[str, Any] = {
            "case_id": case_id,
            "case_dir": path.parent.as_posix(),
            "evaluation_status": payload.get("evaluation_status"),
            "benchmark_score_status": payload.get("benchmark_score_status"),
            "benchmark_score": payload.get("benchmark_score"),
        }
        for metric in metric_names:
            record = vector.get(metric)
            if not isinstance(record, dict):
                row[f"{metric}_status"] = "missing"
                row[f"{metric}_score"] = None
            else:
                row[f"{metric}_status"] = record.get("status")
                row[f"{metric}_score"] = record.get("score")
        rows.append(row)

    missing = [
        case_id for case_id in expected if case_id not in reports_by_case
    ]
    complete_rows = [
        row
        for row in rows
        if row["benchmark_score_status"] == "complete"
        and isinstance(row["benchmark_score"], (int, float))
        and not isinstance(row["benchmark_score"], bool)
    ]
    summary = {
        "schema_version": "counter_strike_corpus_summary_v1",
        "runner_version": COUNTER_STRIKE_RUNNER_VERSION,
        "runs_root": root.as_posix(),
        "expected_case_ids": list(expected),
        "missing_case_ids": missing,
        "case_count": len(rows),
        "complete_case_count": len(complete_rows),
        "all_expected_complete": bool(
            expected
            and not missing
            and len(complete_rows) == len(expected)
        ),
        "rows": rows,
    }
    write_json(root / "corpus_summary.json", summary)
    fieldnames = [
        "case_id",
        "evaluation_status",
        "benchmark_score_status",
        "benchmark_score",
        *[
            field
            for metric in metric_names
            for field in (f"{metric}_status", f"{metric}_score")
        ],
        "case_dir",
    ]
    tsv_path = root / "corpus_summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _load_model_config(
    value: dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], str | None, str | None]:
    if isinstance(value, dict):
        payload = deepcopy(value)
        return payload, None, None
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise CounterStrikeRunnerError(
            "model_config_invalid",
            "model config must be a JSON object",
        )
    return payload, path.as_posix(), _sha256(path)


def _resolve_under(root: Path, value: str | Path, *, label: str) -> Path:
    supplied = Path(value).expanduser()
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CounterStrikeRunnerError(
            f"{label}_outside_source",
            f"{label} must remain under the game source root",
        ) from exc
    if not path.is_file():
        raise CounterStrikeRunnerError(
            f"{label}_missing",
            f"{label} does not exist: {path}",
        )
    return path


def _resolve_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise CounterStrikeRunnerError(
            f"{label}_missing",
            f"{label} does not exist: {path}",
        )
    return path


def _assert_case_identity(
    contract: CounterStrikeCaseContract,
    capture_manifest: dict[str, Any],
) -> None:
    if not contract.case_id.startswith("cs_fps_"):
        raise CounterStrikeRunnerError(
            "case_identity_invalid",
            "the frozen CS corpus contract must use a cs_fps_* case_id",
        )
    if capture_manifest.get("backend") != "headless_browser_game_v1":
        raise CounterStrikeRunnerError(
            "capture_backend_invalid",
            "the capture is not the original-runtime browser backend",
        )


def _contract_record(contract: CounterStrikeCaseContract) -> dict[str, Any]:
    return {
        "case_id": contract.case_id,
        "path": contract.path.as_posix(),
        "sha256": contract.sha256,
        "corpus_id": str(contract.raw["corpus_id"]),
        "source_assertions": [
            {
                "path": item.declared_path,
                "sha256": item.sha256,
                "verified": True,
            }
            for item in contract.source_assertions
        ],
        "team_spawn_counts": {
            team: len(payload["points"])
            for team, payload in contract.canonical_team_spawns.items()
        },
        "score_authority": False,
    }


def _safe_id(value: str) -> str:
    return _SAFE_ID.sub("_", str(value)).strip("._-") or "counter_strike"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_case_ids(paths: list[str]) -> list[str]:
    case_ids: list[str] = []
    for value in paths:
        payload = read_json(Path(value))
        if not isinstance(payload, dict) or not str(payload.get("case_id") or ""):
            raise CounterStrikeRunnerError(
                "case_contract_invalid",
                f"could not read case_id from {value}",
            )
        case_ids.append(str(payload["case_id"]))
    return case_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or aggregate the frozen Counter-Strike static 3D benchmark."
        )
    )
    parser.add_argument("--game-root")
    parser.add_argument("--case-contract")
    parser.add_argument("--out-dir")
    parser.add_argument("--model-config")
    parser.add_argument(
        "--game-mode-config",
        default=str(
            runtime_resource_path("configs/game/game_mode_canonical_v1.yaml")
        ),
    )
    parser.add_argument(
        "--benchmark-config",
        default=str(
            runtime_resource_path(
                "configs/game/counter_strike/benchmark_v1.yaml"
            )
        ),
    )
    parser.add_argument("--entry-html", default="index.html")
    parser.add_argument("--three-replacement", default=None)
    parser.add_argument("--phase", choices=sorted(_PHASES), default="all")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--aggregate-root", default=None)
    parser.add_argument("--expected-contract", action="append", default=[])
    args = parser.parse_args()

    if args.aggregate_root:
        summary = aggregate_counter_strike_runs(
            runs_root=args.aggregate_root,
            expected_case_ids=_expected_case_ids(args.expected_contract),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(0 if summary["all_expected_complete"] else 1)

    missing = [
        name
        for name in ("game_root", "case_contract", "out_dir", "model_config")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(
            "case execution requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    result = run_counter_strike_case(
        game_root=args.game_root,
        case_contract_path=args.case_contract,
        out_dir=args.out_dir,
        model_config=args.model_config,
        game_mode_config=args.game_mode_config,
        benchmark_config=args.benchmark_config,
        entry_html=args.entry_html,
        three_replacement=args.three_replacement,
        phase=args.phase,
        official_mode=not args.diagnostic,
    )
    report = result.get("evaluation_report")
    if isinstance(report, dict):
        print(
            json.dumps(
                {
                    "case_id": result["run_manifest"]["case_id"],
                    "evaluation_status": report.get("evaluation_status"),
                    "benchmark_score_status": report.get(
                        "benchmark_score_status"
                    ),
                    "benchmark_score": report.get("benchmark_score"),
                    "report": result["report_path"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
