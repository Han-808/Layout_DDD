"""Build configs that hand materialized rooms to the existing campaign CLI."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from benchmark.evaluation_campaign.dataset_identity import (
    inspect_evaluation_dataset,
)
from benchmark.evaluation_campaign.config import load_campaign


_CAMPAIGN_ID = re.compile(r"[a-z][a-z0-9_.-]{2,127}")


def build_existing_evaluation_campaign_config(
    *,
    repo_root: str | Path,
    template_path: str | Path,
    dataset_root: str | Path,
    campaign_id: str,
    model_label: str,
    attempt_parent: str | Path,
    final_selection_root: str | Path,
) -> dict[str, Any]:
    """Clone a reviewed campaign and replace only dataset/run identities.

    The template's kernel, metric selection, deduction multiplier, retry
    behavior, selection policy, and Judge profile remain byte-for-byte equal as
    JSON values.  No evaluator command or ``run_evaluate`` argument is rebuilt
    here.
    """

    root = Path(repo_root).expanduser().resolve()
    resolved_template = Path(template_path).expanduser().resolve()
    load_campaign(resolved_template, repo_root=root)
    template = _read_json(resolved_template)
    dataset = Path(dataset_root).expanduser().resolve()
    manifest = _read_json(dataset / "dataset_manifest.json")
    if manifest.get("all_cases_ready") is not True:
        raise ValueError("materialized evaluator cases are not all ready")
    official = manifest.get("official_full_model_score_eligible") is True
    if not official:
        raise ValueError(
            "incomplete source collection is diagnostic-only and cannot enter "
            "the finalizing evaluation campaign"
        )
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError("materialized dataset has no ordered cases")
    identity = inspect_evaluation_dataset(dataset, expected_case_ids=case_ids)
    if identity.portable_fingerprint_sha256 != manifest.get(
        "portable_fingerprint_sha256"
    ):
        raise ValueError("materialized dataset portable identity drift")
    if not isinstance(campaign_id, str) or not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError("campaign_id is not a safe campaign identifier")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be non-empty")
    models = manifest.get("models")
    if not isinstance(models, list) or models != [model_label]:
        raise ValueError("campaign model label differs from materialized model")

    value = deepcopy(template)
    required = {
        "schema_version",
        "campaign_id",
        "model_label",
        "profile_registry",
        "judge_profile_id",
        "dataset",
        "case_plan",
        "kernel",
        "attempt_policy",
        "outputs",
        "selection",
    }
    if set(value) != required or value.get("schema_version") != (
        "scene_evaluation_campaign_v1"
    ):
        raise ValueError("campaign template contract is not supported")
    value["campaign_id"] = campaign_id
    value["model_label"] = model_label.strip()
    value["dataset"] = {
        "root": _repo_relative(root, dataset, field="dataset_root"),
        "expected_dataset_id": identity.dataset_id,
        "expected_fingerprint_sha256": identity.portable_fingerprint_sha256,
        "expected_case_ids": list(case_ids),
        "smoke_case_id": case_ids[0],
    }
    value["case_plan"] = {
        "run_case_ids": list(case_ids),
        "selection_case_ids": list(case_ids),
        "prior_attempt_roots": [],
    }
    value["outputs"] = {
        "attempt_parent": _repo_relative(
            root,
            Path(attempt_parent).expanduser().resolve(),
            field="attempt_parent",
        ),
        "final_selection_root": _repo_relative(
            root,
            Path(final_selection_root).expanduser().resolve(),
            field="final_selection_root",
        ),
    }
    _reject_overlapping_roots(
        dataset,
        Path(attempt_parent).expanduser().resolve(),
        Path(final_selection_root).expanduser().resolve(),
    )
    return value


def write_campaign_config(
    path: str | Path, value: Mapping[str, Any]
) -> Path:
    """Write once, or accept an exactly matching existing config."""

    target = Path(path).expanduser().resolve()
    encoded = _canonical_pretty(value)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise ValueError("campaign config target must be a regular file")
        if target.read_bytes() != encoded:
            raise FileExistsError("existing campaign config differs")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
    return target


def evaluation_campaign_command(
    *,
    config_path: str | Path,
    python_executable: str | Path,
    run: bool,
    bindings_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the existing campaign entrypoint; never fork evaluator logic."""

    command = [
        str(Path(python_executable).expanduser()),
        "-m",
        "benchmark.evaluation_campaign",
        "run" if run else "check",
        "--config",
        str(Path(config_path).expanduser().resolve()),
    ]
    if run:
        if bindings_path is None:
            raise ValueError("evaluation run command requires a private binding")
        command.extend(
            ["--bindings", str(Path(bindings_path).expanduser().resolve())]
        )
    elif bindings_path is not None:
        raise ValueError("static campaign check does not load private bindings")
    return tuple(command)


def campaign_config_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_pretty(value)).hexdigest()


def _repo_relative(root: Path, path: Path, *, field: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must remain inside the repository") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"{field} is not a safe repository-relative path")
    return relative.as_posix()


def _reject_overlapping_roots(dataset: Path, attempt: Path, final: Path) -> None:
    roots = (dataset, attempt, final)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("dataset and evaluation output roots must be disjoint")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_pretty(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "build_existing_evaluation_campaign_config",
    "campaign_config_sha256",
    "evaluation_campaign_command",
    "write_campaign_config",
]
