"""Hashing, source provenance, and atomic campaign manifests."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

from benchmark.evaluation_campaign.config import (
    EvaluationCampaignSpec,
    JudgeProfile,
    PriorAttemptRoot,
    load_profile_registry,
)
from benchmark.evaluation_campaign.dataset_identity import (
    EvaluationDatasetIdentity,
)


CAMPAIGN_STATE_SCHEMA_VERSION = "scene_evaluation_campaign_state_v1"
ROUND_RECORD_SCHEMA_VERSION = "scene_evaluation_campaign_round_v1"
SELECTION_PROVENANCE_SCHEMA_VERSION = (
    "scene_evaluation_campaign_selection_provenance_v2"
)


def protocol_manifest(
    campaign: EvaluationCampaignSpec,
    dataset: EvaluationDatasetIdentity,
    *,
    repo_root: Path,
    profile: JudgeProfile,
    route_public_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = repo_root.resolve()
    source_manifest = evaluation_source_manifest(root)
    payload = {
        "schema_version": "scene_evaluation_protocol_fingerprint_v1",
        "dataset_fingerprint_sha256": dataset.portable_fingerprint_sha256,
        "kernel": _kernel_public_dict(campaign),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "judge_profile_fingerprint_sha256": profile.fingerprint_sha256,
        "route_fingerprint_sha256": route_public_manifest.get(
            "route_fingerprint_sha256"
        ),
        "adapter_attestation_sha256": route_public_manifest.get(
            "adapter_attestation_sha256"
        ),
        "grouping_config_sha256": file_sha256(
            campaign.kernel.grouping_config
        ),
    }
    return {
        **payload,
        "protocol_fingerprint_sha256": json_sha256(payload),
        "source_manifest": source_manifest,
    }


def execution_manifest(campaign: EvaluationCampaignSpec) -> dict[str, Any]:
    policy = campaign.attempt_policy
    payload = {
        "schema_version": "scene_evaluation_execution_fingerprint_v1",
        "max_new_attempts_per_case": policy.max_new_attempts_per_case,
        "retry_delay_seconds": policy.retry_delay_seconds,
        "max_workers": policy.max_workers,
        "round0_preflight_attempts": policy.round0_preflight_attempts,
        "retry_preflight_attempts": policy.retry_preflight_attempts,
        "preflight_timeout_seconds": policy.preflight_timeout_seconds,
    }
    return {**payload, "execution_fingerprint_sha256": json_sha256(payload)}


def evaluation_source_manifest(repo_root: Path) -> dict[str, Any]:
    # The selection entrypoint pulls in the package-owned persisted-scoring
    # projection through static import closure.  HTML viewer/rendering code is
    # deliberately excluded because it cannot affect evaluation or selection.
    roots = (
        Path("scripts/run_camera_cal_scene_level.py"),
        Path("scripts/select_first_publishable_scene_evaluations.py"),
        Path("scripts/check_model_endpoint.py"),
        Path("src/benchmark/evaluation_campaign"),
        Path("src/benchmark/camera_cal_scene_level"),
        Path("src/benchmark/api/evaluation.py"),
        Path("src/benchmark/evaluator"),
        Path("src/benchmark/models"),
        Path("src/benchmark/rendering"),
        Path("src/benchmark/visual_judge"),
        Path("src/benchmark/grouping"),
        Path("src/benchmark/scoring_profiles.py"),
    )
    python_paths: set[Path] = set()
    for relative in roots:
        path = repo_root / relative
        if not path.exists() or path.is_symlink():
            raise FileNotFoundError(f"evaluation source dependency is unavailable: {relative}")
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py"))
        if not candidates:
            raise FileNotFoundError(f"evaluation source dependency is empty: {relative}")
        for candidate in candidates:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            python_paths.add(candidate.resolve())
    python_paths = _python_dependency_closure(repo_root, python_paths)
    files: list[dict[str, Any]] = [
        {
            "path": candidate.relative_to(repo_root).as_posix(),
            "bytes": candidate.stat().st_size,
            "sha256": file_sha256(candidate),
        }
        for candidate in sorted(python_paths)
    ]
    packaging = repo_root / "pyproject.toml"
    if not packaging.is_file() or packaging.is_symlink():
        raise FileNotFoundError("evaluation packaging contract is unavailable: pyproject.toml")
    files.append(
        {
            "path": "pyproject.toml",
            "bytes": packaging.stat().st_size,
            "sha256": file_sha256(packaging),
        }
    )
    dependency_lock = repo_root / "uv.lock"
    if not dependency_lock.is_file() or dependency_lock.is_symlink():
        raise FileNotFoundError("evaluation dependency lock is unavailable: uv.lock")
    files.append(
        {
            "path": "uv.lock",
            "bytes": dependency_lock.stat().st_size,
            "sha256": file_sha256(dependency_lock),
        }
    )
    # The evaluator loads several YAML policies dynamically.  Hash every
    # packaged YAML so a new or changed runtime policy cannot silently resume
    # under an old protocol fingerprint.
    yaml_roots = (
        repo_root / "configs/evaluation",
        repo_root / "configs/grouping",
        repo_root / "src/benchmark/_resources/configs/evaluation",
        repo_root / "src/benchmark/_resources/configs/grouping",
    )
    for yaml_root in yaml_roots:
        if not yaml_root.is_dir() or yaml_root.is_symlink():
            raise FileNotFoundError(
                f"packaged evaluation YAML root is unavailable: {yaml_root}"
            )
        for pattern in ("*.yaml", "*.yml"):
            for candidate in sorted(yaml_root.rglob(pattern)):
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                files.append(
                    {
                        "path": candidate.relative_to(repo_root).as_posix(),
                        "bytes": candidate.stat().st_size,
                        "sha256": file_sha256(candidate),
                    }
                )
    files.sort(key=lambda row: str(row["path"]))
    payload = {
        "schema_version": "scene_evaluation_source_manifest_v1",
        "files": files,
    }
    return {**payload, "manifest_sha256": json_sha256(payload)}


def _python_dependency_closure(repo_root: Path, seeds: set[Path]) -> set[Path]:
    """Resolve repository-local static imports from the frozen entry points."""

    source_root = (repo_root / "src").resolve()
    scripts_root = (repo_root / "scripts").resolve()
    result = set(seeds)
    pending = list(seeds)
    while pending:
        path = pending.pop()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ValueError(f"cannot inspect evaluation dependency: {path}") from exc
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    modules.add(node.module)
                    modules.update(
                        f"{node.module}.{alias.name}"
                        for alias in node.names
                        if alias.name != "*"
                    )
                elif node.level and path.is_relative_to(source_root):
                    relative = path.relative_to(source_root).with_suffix("")
                    package_parts = list(relative.parts[:-1])
                    keep = max(0, len(package_parts) - node.level + 1)
                    prefix = package_parts[:keep]
                    if node.module:
                        prefix.extend(node.module.split("."))
                    if prefix:
                        modules.add(".".join(prefix))
                        modules.update(
                            ".".join((*prefix, alias.name))
                            for alias in node.names
                            if alias.name != "*"
                        )
        for module in modules:
            candidates: tuple[Path, ...]
            if module.startswith("benchmark"):
                base = source_root.joinpath(*module.split("."))
                candidates = (base.with_suffix(".py"), base / "__init__.py")
            elif "." not in module:
                base = scripts_root / module
                candidates = (base.with_suffix(".py"),)
            else:
                continue
            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate.is_file() and not candidate.is_symlink() and candidate not in result:
                    result.add(candidate)
                    pending.append(candidate)
    return result


def git_state(repo_root: Path) -> dict[str, Any]:
    commit = _git(repo_root, ["rev-parse", "HEAD"]).strip()
    dirty_text = _git(repo_root, ["status", "--short"])
    dirty = any(line.strip() for line in dirty_text.splitlines())
    return {
        "commit": commit,
        "dirty": dirty,
    }


def validate_prior_attempt(
    prior: PriorAttemptRoot,
    *,
    dataset: EvaluationDatasetIdentity,
    protocol_fingerprint_sha256: str,
    judge_profile_fingerprint_sha256: str | None = None,
    adoption_manifest_path: Path | None = None,
) -> dict[str, Any]:
    run_manifest = read_json(prior.root / "run_manifest.json")
    if run_manifest.get("status") not in {
        "complete",
        "failed",
        "endpoint_preflight_failed",
    }:
        raise ValueError("prior attempt run manifest is not terminal")
    if prior.adoption_mode == "legacy_experiment_plan":
        if (
            run_manifest.get("experiment_plan_sha256")
            != prior.expected_experiment_plan_sha256
        ):
            raise ValueError(f"prior attempt experiment plan drift: {prior.root}")
        plan = read_json(prior.root / "experiment_plan.json")
        case_ids = _validate_plan_dataset(plan, dataset=dataset, root=prior.root)
        if case_ids != dataset.ordered_case_ids:
            raise ValueError("legacy prior must declare the complete ordered dataset")
        route_fingerprint = judge_profile_fingerprint_sha256
        source_guard = {
            "mode": prior.adoption_mode,
            "run_manifest_sha256": file_sha256(prior.root / "run_manifest.json"),
            "experiment_plan_sha256": file_sha256(prior.root / "experiment_plan.json"),
        }
    else:
        campaign_round = read_json(prior.root / "campaign_round.json")
        if campaign_round.get("schema_version") != ROUND_RECORD_SCHEMA_VERSION:
            raise ValueError("campaign prior round schema mismatch")
        if campaign_round.get("status") not in {
            "complete",
            "kernel_failed",
            "recovered_terminal",
            "abandoned_interrupted",
        } or campaign_round.get("started") is not True:
            raise ValueError("campaign prior round is not a terminal started attempt")
        if (
            campaign_round.get("protocol_fingerprint_sha256")
            != prior.expected_protocol_fingerprint_sha256
            or campaign_round.get("protocol_fingerprint_sha256")
            != protocol_fingerprint_sha256
        ):
            raise ValueError(f"prior attempt protocol mismatch: {prior.root}")
        if (
            campaign_round.get("dataset_fingerprint_sha256")
            != dataset.portable_fingerprint_sha256
        ):
            raise ValueError(f"prior attempt dataset mismatch: {prior.root}")
        case_ids = _strict_case_ids(
            campaign_round.get("case_ids"), label="campaign prior case_ids"
        )
        route = campaign_round.get("route")
        if not isinstance(route, dict):
            raise ValueError("campaign prior has no route manifest")
        route_fingerprint = route.get("route_fingerprint_sha256")
        if not isinstance(route_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", route_fingerprint
        ):
            raise ValueError("campaign prior route fingerprint is invalid")
        source_guard = {
            "mode": prior.adoption_mode,
            "run_manifest_sha256": file_sha256(prior.root / "run_manifest.json"),
            "campaign_round_sha256": file_sha256(prior.root / "campaign_round.json"),
        }

    expected_dataset_cases = {case.case_id for case in dataset.cases}
    if not set(case_ids).issubset(expected_dataset_cases):
        raise ValueError("prior attempt contains cases outside the dataset")
    cases_root = prior.root / "cases"
    actual_case_ids = tuple(
        sorted(path.name for path in cases_root.iterdir() if path.is_dir())
    ) if cases_root.is_dir() else ()
    if actual_case_ids != tuple(sorted(case_ids)):
        raise ValueError("prior attempt case directory inventory is incomplete or extra")
    report_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_root = prior.root / "cases" / case_id
        manifest = case_root / "case_run_manifest.json"
        report = case_root / "evaluation_report.json"
        l1_report = case_root / "l1_report.json"
        l3_report = case_root / "scene_quality_report.json"
        diagnostics = case_root / "l1_diagnostics.json"
        if manifest.is_file():
            case_manifest = read_json(manifest)
            if case_manifest.get("case_id") != case_id:
                raise ValueError(f"prior attempt case identity mismatch: {case_id}")
        report_rows.append(
            {
                "case_id": case_id,
                "case_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
                "evaluation_report_sha256": file_sha256(report) if report.is_file() else None,
                "l1_report_sha256": file_sha256(l1_report) if l1_report.is_file() else None,
                "l3_report_sha256": file_sha256(l3_report) if l3_report.is_file() else None,
                "l1_diagnostics_sha256": (
                    file_sha256(diagnostics) if diagnostics.is_file() else None
                ),
            }
        )
    payload = {
        "schema_version": "evaluation_prior_adoption_v1",
        "dataset_fingerprint_sha256": dataset.portable_fingerprint_sha256,
        "ordered_case_ids": list(case_ids),
        "judge_profile_id": prior.judge_profile_id,
        "route_or_profile_fingerprint_sha256": route_fingerprint,
        "source_guard": source_guard,
        "reports": report_rows,
    }
    payload["adoption_fingerprint_sha256"] = json_sha256(payload)
    if adoption_manifest_path is not None:
        if adoption_manifest_path.is_file():
            recorded = read_json(adoption_manifest_path)
            if recorded != payload:
                raise ValueError("immutable prior adoption manifest drift")
        else:
            atomic_write_json(adoption_manifest_path, payload)
            adoption_manifest_path.chmod(0o444)
    return payload


def write_round_record(
    root: Path,
    *,
    campaign: EvaluationCampaignSpec,
    dataset: EvaluationDatasetIdentity,
    protocol_fingerprint_sha256: str,
    route_public_manifest: Mapping[str, Any],
    round_index: int,
    case_ids: Sequence[str],
    exit_code: int | None,
    status: str,
    started: bool,
) -> dict[str, Any]:
    record = {
        "schema_version": ROUND_RECORD_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_config_sha256": campaign.source_sha256,
        "dataset_fingerprint_sha256": dataset.portable_fingerprint_sha256,
        "protocol_fingerprint_sha256": protocol_fingerprint_sha256,
        "route": dict(route_public_manifest),
        "round_index": round_index,
        "case_ids": list(case_ids),
        "status": status,
        "exit_code": exit_code,
        "started": bool(started),
        "updated_at": utc_now(),
    }
    assert_public_portable(record)
    atomic_write_json(root / "campaign_round.json", record)
    return record


def write_selection_provenance(
    final_root: Path,
    *,
    campaign: EvaluationCampaignSpec,
    dataset: EvaluationDatasetIdentity,
    protocol_fingerprint_sha256: str,
    attempt_route_ids: Mapping[Path, str],
) -> dict[str, Any]:
    selection = validate_final_selection(
        final_root,
        campaign=campaign,
        attempt_route_ids=attempt_route_ids,
    )
    case_rows = selection["cases"]
    ordered_roots = tuple(attempt_route_ids)
    rows: list[dict[str, Any]] = []
    for raw in case_rows:
        if not isinstance(raw, dict):
            raise ValueError("final selection case row is invalid")
        case_id = str(raw.get("case_id") or "")
        source_case_value = raw.get("source_case")
        source_run_value = raw.get("source_run")
        if not isinstance(source_case_value, str):
            raise ValueError(f"selection source_case is unavailable: {case_id}")
        source_case = Path(source_case_value).resolve()
        snapshot_case = final_root / "cases" / case_id
        source_run = (
            Path(source_run_value).resolve()
            if isinstance(source_run_value, str)
            else source_case.parent.parent.resolve()
        )
        try:
            profile_id = attempt_route_ids[source_run]
        except KeyError as exc:
            raise ValueError(f"selection source has no route provenance: {source_run}") from exc
        source_index = ordered_roots.index(source_run)
        rows.append(
            {
                "case_id": case_id,
                "source_attempt_index": source_index,
                "source_case_relative": f"cases/{case_id}",
                "judge_profile_id": profile_id,
                "case_manifest_sha256": file_sha256(
                    snapshot_case / "case_run_manifest.json"
                ),
                "evaluation_report_sha256": file_sha256(
                    snapshot_case / "evaluation_report.json"
                ),
                "snapshot_tree_sha256": raw["snapshot_tree_sha256"],
            }
        )
    result = {
        "schema_version": SELECTION_PROVENANCE_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_config_sha256": campaign.source_sha256,
        "dataset_fingerprint_sha256": dataset.portable_fingerprint_sha256,
        "protocol_fingerprint_sha256": protocol_fingerprint_sha256,
        "selection_manifest_sha256": file_sha256(
            final_root / "selection_manifest.json"
        ),
        "run_manifest_sha256": file_sha256(final_root / "run_manifest.json"),
        "summary_sha256": file_sha256(final_root / "summary.json"),
        "attempts": [
            {
                "attempt_index": index,
                "judge_profile_id": attempt_route_ids[root],
            }
            for index, root in enumerate(ordered_roots)
        ],
        "cases": rows,
    }
    assert_public_portable(result)
    atomic_write_json(final_root / "campaign_selection_provenance.json", result)
    return result


def validate_final_selection(
    final_root: Path,
    *,
    campaign: EvaluationCampaignSpec,
    attempt_route_ids: Mapping[Path, str],
) -> dict[str, Any]:
    """Validate every selector artifact before adopting an existing final root."""

    selection_path = final_root / "selection_manifest.json"
    run_path = final_root / "run_manifest.json"
    summary_path = final_root / "summary.json"
    for path in (selection_path, run_path, summary_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"final selection artifact is missing: {path.name}")
    selection = read_json(selection_path)
    run_manifest = read_json(run_path)
    summary = read_json(summary_path)
    if selection != run_manifest:
        raise ValueError("selection and run manifests differ")
    required_selection_fields = {
        "schema_version",
        "status",
        "model_label",
        "evaluator_model",
        "provider_route",
        "case_count",
        "attempt_roots",
        "selection_policy",
        "publishability_policy",
        "cases",
    }
    if not required_selection_fields.issubset(selection):
        raise ValueError("final selection schema fields are incomplete")
    if selection.get("schema_version") != "scene_level_first_publishable_selection_v2":
        raise ValueError("final selection schema mismatch")
    expected_ids = campaign.case_plan.selection_case_ids
    if selection.get("status") != "complete" or selection.get("case_count") != len(expected_ids):
        raise ValueError("final selection is not complete")
    if selection.get("selection_policy") != "first_publishable_attempt_only_no_score_selection":
        raise ValueError("final selection policy mismatch")
    if selection.get("model_label") != campaign.model_label:
        raise ValueError("final selection model label mismatch")
    profiles = load_profile_registry(campaign.profile_registry)
    current_profile = profiles.get(campaign.judge_profile_id)
    if current_profile is None:
        raise ValueError("final selection Judge profile is unavailable")
    if selection.get("evaluator_model") != current_profile.model_alias:
        raise ValueError("final selection evaluator model mismatch")
    expected_provider_route = "profiles:" + ",".join(
        dict.fromkeys(str(value) for value in attempt_route_ids.values())
    )
    if selection.get("provider_route") != expected_provider_route:
        raise ValueError("final selection provider route mismatch")
    expected_publishability = {
        "case_status": "complete",
        "final_decision_status": "resolved",
        "l1_engineering_failure": False,
        "evaluation_status": "complete",
        "benchmark_score_status": "complete",
        "benchmark_score_100": "finite_number",
    }
    if selection.get("publishability_policy") != expected_publishability:
        raise ValueError("final selection publishability policy mismatch")
    rows = selection.get("cases")
    if not isinstance(rows, list) or [row.get("case_id") if isinstance(row, dict) else None for row in rows] != list(expected_ids):
        raise ValueError("final selection case order/inventory mismatch")
    if len({str(row["case_id"]) for row in rows}) != len(expected_ids):
        raise ValueError("final selection has duplicate cases")
    roots = {path.resolve() for path in attempt_route_ids}
    ordered_roots = tuple(path.resolve() for path in attempt_route_ids)
    raw_roots = selection.get("attempt_roots")
    if (
        not isinstance(raw_roots, list)
        or tuple(Path(str(path)).resolve() for path in raw_roots) != ordered_roots
    ):
        raise ValueError("final selection attempt root order mismatch")
    snapshots_root = final_root / "cases"
    snapshot_names = (
        sorted(path.name for path in snapshots_root.iterdir())
        if snapshots_root.is_dir()
        else []
    )
    if snapshot_names != sorted(expected_ids):
        raise ValueError("final selection snapshot inventory mismatch")
    selected_scores: list[float] = []
    selected_coverages: list[float] = []
    retry_case_count = 0
    for row in rows:
        case_id = str(row["case_id"])
        required_row_fields = {
            "case_id",
            "selected_attempt_index",
            "source_run",
            "source_case",
            "storage",
            "snapshot_case",
            "snapshot_file_count",
            "snapshot_tree_sha256",
            "status",
            "final_decision_status",
            "benchmark_score_100",
            "benchmark_score_status",
            "evaluation_status",
            "grounded_score_fraction",
            "l1_engineering_failure",
            "case_manifest_sha256",
            "evaluation_report_sha256",
            "l1_report_sha256",
            "l3_report_sha256",
        }
        if not required_row_fields.issubset(row):
            raise ValueError(f"selection case schema is incomplete: {case_id}")
        if row.get("storage") != "self_contained_directory_copy_v1":
            raise ValueError(f"selection storage policy mismatch: {case_id}")
        source_run = Path(str(row.get("source_run") or "")).resolve()
        source_case = Path(str(row.get("source_case") or "")).resolve()
        if source_run not in roots:
            raise ValueError(f"selection source root is not adopted: {case_id}")
        if source_case != source_run / "cases" / case_id:
            raise ValueError(f"selection source_case relation mismatch: {case_id}")
        selected_index = row.get("selected_attempt_index")
        if selected_index != ordered_roots.index(source_run):
            raise ValueError(f"selection attempt index mismatch: {case_id}")
        if selected_index > 0:
            retry_case_count += 1
        earliest = next(
            (
                candidate / "cases" / case_id
                for candidate in ordered_roots
                if _publishable_case(candidate / "cases" / case_id)
            ),
            None,
        )
        if earliest is None or earliest.resolve() != source_case:
            raise ValueError(f"selection violates chronological first-publishable policy: {case_id}")
        snapshot_case = snapshots_root / case_id
        if row.get("snapshot_case") != f"cases/{case_id}":
            raise ValueError(f"selection snapshot relation mismatch: {case_id}")
        if not snapshot_case.is_dir() or snapshot_case.is_symlink():
            raise ValueError(f"selection snapshot is not self-contained: {case_id}")
        snapshot_file_count, snapshot_tree_sha256 = _directory_tree_identity(
            snapshot_case
        )
        if (
            row.get("snapshot_file_count") != snapshot_file_count
            or row.get("snapshot_tree_sha256") != snapshot_tree_sha256
        ):
            raise ValueError(f"selection snapshot tree identity mismatch: {case_id}")
        source_files = {
            "case_manifest_sha256": source_case / "case_run_manifest.json",
            "evaluation_report_sha256": source_case / "evaluation_report.json",
            "l1_report_sha256": source_case / "l1_report.json",
            "l3_report_sha256": source_case / "scene_quality_report.json",
        }
        snapshot_files = {
            "case_manifest_sha256": snapshot_case / "case_run_manifest.json",
            "evaluation_report_sha256": snapshot_case / "evaluation_report.json",
            "l1_report_sha256": snapshot_case / "l1_report.json",
            "l3_report_sha256": snapshot_case / "scene_quality_report.json",
        }
        for field, path in source_files.items():
            if not path.is_file() or row.get(field) != file_sha256(path):
                raise ValueError(f"selection source hash mismatch: {case_id}:{field}")
        for field, path in snapshot_files.items():
            if not path.is_file() or row.get(field) != file_sha256(path):
                raise ValueError(f"selection snapshot hash mismatch: {case_id}:{field}")
        manifest = read_json(snapshot_files["case_manifest_sha256"])
        report = read_json(snapshot_files["evaluation_report_sha256"])
        score = report.get("benchmark_score_100")
        coverage = (
            report.get("coverage", {}).get("grounded_score_fraction")
            if isinstance(report.get("coverage"), dict)
            else None
        )
        if isinstance(coverage, (int, float)) and not isinstance(coverage, bool):
            if not math.isfinite(float(coverage)):
                raise ValueError(f"selection coverage is non-finite: {case_id}")
            selected_coverages.append(float(coverage))
        if not (
            manifest.get("case_id") == case_id
            and manifest.get("status") == "complete"
            and manifest.get("final_decision_status") == "resolved"
            and manifest.get("l1_engineering_failure") is False
            and report.get("evaluation_status") == "complete"
            and report.get("benchmark_score_status") == "complete"
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        ):
            raise ValueError(f"selection contains a nonpublishable case: {case_id}")
        selected_scores.append(float(score))
        mirrored = {
            "status": manifest.get("status"),
            "final_decision_status": manifest.get("final_decision_status"),
            "l1_engineering_failure": manifest.get("l1_engineering_failure"),
            "evaluation_status": report.get("evaluation_status"),
            "benchmark_score_status": report.get("benchmark_score_status"),
            "benchmark_score_100": score,
            "grounded_score_fraction": coverage,
        }
        for field, expected in mirrored.items():
            if row.get(field) != expected:
                raise ValueError(f"selection mirrored status mismatch: {case_id}:{field}")
    totals = summary.get("totals")
    aggregate = summary.get("aggregate")
    expected_official = sum(selected_scores) / len(selected_scores)
    expected_coverage = (
        sum(selected_coverages) / len(selected_coverages)
        if len(selected_coverages) == len(selected_scores)
        else None
    )
    aggregate_valid = (
        isinstance(aggregate, dict)
        and aggregate.get("case_count") == len(expected_ids)
        and aggregate.get("published_case_count") == len(expected_ids)
        and aggregate.get("infrastructure_failure_case_count") == 0
        and isinstance(aggregate.get("metrics"), list)
        and _same_number(aggregate.get("official_score_100"), expected_official)
        and (
            aggregate.get("mean_combined_coverage_fraction") is None
            if expected_coverage is None
            else _same_number(
                aggregate.get("mean_combined_coverage_fraction"),
                expected_coverage,
            )
        )
    )
    if (
        summary.get("schema_version") != "selected_scene_level_summary_v2"
        or summary.get("status") != "complete"
        or summary.get("model_label") != campaign.model_label
        or summary.get("evaluator_model") != selection.get("evaluator_model")
        or summary.get("provider_route") != selection.get("provider_route")
        or not isinstance(totals, dict)
        or totals.get("cases") != len(expected_ids)
        or totals.get("successful") != len(expected_ids)
        or totals.get("failed") != 0
        or totals.get("final_unresolved") != 0
        or totals.get("final_infrastructure_failure") != 0
        or totals.get("l1_engineering_failure_cases") != 0
        or totals.get("retry_cases") != retry_case_count
        or totals.get("baseline_cases") != len(expected_ids) - retry_case_count
        or totals.get("attempt_rounds") != len(ordered_roots)
        or not aggregate_valid
    ):
        raise ValueError("final summary status/totals mismatch")
    return selection


def _same_number(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-9)
    )


def _publishable_case(case_root: Path) -> bool:
    manifest_path = case_root / "case_run_manifest.json"
    report_path = case_root / "evaluation_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        report = read_json(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    score = report.get("benchmark_score_100")
    return (
        manifest.get("case_id") == case_root.name
        and manifest.get("status") == "complete"
        and manifest.get("final_decision_status") == "resolved"
        and manifest.get("l1_engineering_failure") is False
        and report.get("evaluation_status") == "complete"
        and report.get("benchmark_score_status") == "complete"
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    )


def _directory_tree_identity(root: Path) -> tuple[int, str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"selection snapshot contains a symlink: {path}")
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return len(rows), json_sha256(rows)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_public_portable(value: Any) -> None:
    """Reject machine/deployment provenance from public campaign manifests."""

    forbidden_keys = {
        "pid",
        "runner_pid",
        "dirty_paths",
        "endpoint",
        "credential_env",
        "api_key_env",
        "authorization",
    }

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in forbidden_keys:
                    raise ValueError(f"public provenance contains private field: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            if (
                item.startswith(("/", "file://", "http://", "https://"))
                or "://" in item
                or re.match(r"^[A-Za-z]:[\\/]", item)
                or re.search(r"(?:^|\s)/(?:Users|home|private|tmp|var)/", item)
            ):
                raise ValueError(f"public provenance contains local/deployment path: {path}")

    visit(value, "public")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kernel_public_dict(campaign: EvaluationCampaignSpec) -> dict[str, Any]:
    kernel = asdict(campaign.kernel)
    kernel["grouping_config"] = campaign.kernel.grouping_config.name
    kernel["metrics"] = list(campaign.kernel.metrics)
    return kernel


def _validate_plan_dataset(
    plan: Mapping[str, Any],
    *,
    dataset: EvaluationDatasetIdentity,
    root: Path,
) -> tuple[str, ...]:
    rows = plan.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"legacy attempt plan has no cases: {root}")
    expected = {
        case.case_id: case.semantic_content_fingerprint for case in dataset.cases
    }
    case_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"legacy attempt plan case is invalid: {root}")
        case_id = str(row.get("case_id") or "")
        case_ids.append(case_id)
        if case_id not in expected:
            raise ValueError(f"legacy attempt has unknown dataset case: {case_id}")
        if row.get("semantic_content_fingerprint") != expected[case_id]:
            raise ValueError(f"legacy attempt dataset identity mismatch: {case_id}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"legacy attempt plan has duplicate cases: {root}")
    return tuple(case_ids)


def _strict_case_ids(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be non-empty")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique non-empty strings")
    return result


def _git(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout
