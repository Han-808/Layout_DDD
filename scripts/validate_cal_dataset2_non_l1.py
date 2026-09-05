#!/usr/bin/env python3
"""Read-only validation for ``cal_dataset2_non_l1_evidence``.

This validator deliberately does not repair, rewrite, freeze, or approve any
dataset artifact.  It validates the construction-time contract before the
benchmark owner performs the human review:

* exactly 108 opaque cases and 108 single-metric events;
* canonical scene/request/plan/asset artifacts and cross-artifact IDs;
* all eight non-L1 metric families and the registered sample distribution;
* prompt-source spans (including exact character offsets);
* target-scoped prompt-authorized L3 deviations linked to real L2 claims;
* pending-human labels that are not accuracy eligible;
* object-grouping partitions, asset closure, file hashes, and inventories;
* prompt-only / scene-only counterfactual invariants;
* outbound judge-context allowlists that contain no GT or construction truth.

The command prints a JSON report to stdout and exits non-zero on any error.  It
never writes into the dataset directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.evaluator.asset_policy import validate_asset_policy  # noqa: E402
from benchmark.evaluator.scene_quality.authorized_deviations import (  # noqa: E402
    validate_authorized_deviations,
)
from benchmark.evaluator.specification_fidelity.contract import (  # noqa: E402
    FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES,
    validate_frozen_cal_dataset2_v0_specification_contract,
)
from benchmark.reference_annotation import validate_reference_annotation  # noqa: E402
from benchmark.scene_io.validate import (  # noqa: E402
    validate_asset_selection,
    validate_generated_scene,
    validate_object_plan,
    validate_scene_request,
)


DATASET_ID = "cal_dataset2_non_l1_evidence"
EXPECTED_CASE_COUNT = 108
CANONICAL_METRICS = (
    "oor",
    "oar",
    "room_scene_type",
    "broad_semantic_intent",
    "required_functional_areas",
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
L2_METRICS = CANONICAL_METRICS[:5]
L3_METRICS = CANONICAL_METRICS[5:]

EXPECTED_METRIC_COUNTS = {
    "oor": 12,
    "oar": 12,
    "room_scene_type": 12,
    "broad_semantic_intent": 12,
    "required_functional_areas": 12,
    "scale_consistency": 16,
    "object_pairing_consistency": 16,
    "style_consistency": 16,
}
EXPECTED_BASE_COUNTS = {metric: 12 for metric in CANONICAL_METRICS}
EXPECTED_AUTHORIZATION_CELLS = {
    ("absent", "absent"),
    ("absent", "present"),
    ("present", "absent"),
    ("present", "present"),
}
BLIND_REVIEW_FORBIDDEN_COLUMNS = {
    "scenario_id",
    "design_role",
    "counterfactual_group_id",
    "prompt_authorization",
    "scene_deviation",
    "visual_source_case_id",
    "proposed_semantic_label",
    "declared_delta",
}
BLIND_REVIEW_DECISION_COLUMNS = {
    "human_semantic_label",
    "prompt_compatible",
    "target_mapping_correct",
    "needs_render_check",
    "notes",
}

MANDATORY_CASE_FILES = (
    "generated_scene.json",
    "scene_request.json",
    "object_plan.json",
    "asset_selection.json",
    "reference_annotation.json",
    "specification_contract.json",
    "authorized_deviations.json",
    "object_grouping_report.json",
    "metric_events.json",
    "metric_gt.json",
    "evidence_expectations.json",
    "construction_manifest.json",
    "provenance.json",
    "review.json",
)
MANDATORY_DATASET_FILES = (
    "README.md",
    "dataset_manifest.json",
    "cases.json",
    "configs/evaluation_profile_claim_driven_v2.json",
    "configs/evidence_arms.json",
    "review/REVIEW_PROTOCOL.md",
    "review/review_queue.tsv",
    "review/index.html",
    "validation/file_inventory.json",
    "validation/case_inventory.tsv",
    "validation/metric_inventory.tsv",
    "validation/prompt_claim_audit.tsv",
    "validation/counterfactual_audit.tsv",
)

OPAQUE_CASE_ID_RE = re.compile(r"^(?:case|n2|c)[_-]?\d{3,6}$", re.IGNORECASE)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
LEAKAGE_TOKEN_RE = re.compile(
    r"(?:^|[_\-\s])(?:valid|invalid|ambiguous|wrong|obvious|subtle|outlier|"
    r"giant|normal|deviation|authorized|present|absent)(?:$|[_\-\s])",
    re.IGNORECASE,
)
PUBLIC_TEXT_LEAK_RE = re.compile(
    r"\b(?:invalid|wrong|obvious(?:ly)?|subtle|ground[ -]?truth|test case|"
    r"counterfactual|outlier)\b",
    re.IGNORECASE,
)
FORBIDDEN_OUTBOUND_TOKENS = {
    "accuracy_eligible",
    "answer",
    "authorization_applied",
    "construction_manifest",
    "counterfactual_condition",
    "evidence_expectations",
    "event_gt",
    "ground_truth",
    "gt",
    "human_label",
    "label",
    "metric_gt",
    "proposed_semantic_label",
    "provenance",
    "raw_coherence_label",
    "review",
    "scene_deviation",
    "semantic_label",
    "verdict",
}
ALLOWED_PROPOSED_LABELS = {"valid", "invalid", "ambiguous"}
PENDING_REVIEW_STATES = {"pending", "pending_human", "pending_review"}
NEUTRAL_SCENE_TYPES = {"room", "unspecified", "unspecified_room", "generic_room"}


class DatasetValidationFailure(ValueError):
    """Raised by a validation helper when an invariant is violated."""


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    path: str
    message: str


class Report:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.issues: list[Issue] = []
        self.summary: dict[str, Any] = {}

    def error(self, code: str, path: str | Path, message: str) -> None:
        self.issues.append(Issue("error", code, _display_path(path, self.dataset_root), message))

    def warning(self, code: str, path: str | Path, message: str) -> None:
        self.issues.append(Issue("warning", code, _display_path(path, self.dataset_root), message))

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cal_dataset2_validation_report_v1",
            "dataset_root": str(self.dataset_root),
            "read_only": True,
            "ok": not self.errors,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "summary": self.summary,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class CaseBundle:
    case_id: str
    root: Path
    registry: dict[str, Any]
    generated_scene: dict[str, Any]
    scene_request: dict[str, Any]
    object_plan: dict[str, Any]
    asset_selection: dict[str, Any]
    reference_annotation: dict[str, Any]
    specification_contract: dict[str, Any]
    authorized_deviations_raw: Any
    authorized_deviations: list[dict[str, Any]]
    grouping_report: dict[str, Any]
    metric_events_raw: Any
    metric_events: list[dict[str, Any]]
    metric_gt_raw: Any
    metric_gt: list[dict[str, Any]]
    evidence_expectations: dict[str, Any]
    construction_manifest: dict[str, Any]
    provenance: dict[str, Any]
    review: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Support" / "datasets" / DATASET_ID,
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=500,
        help="Maximum issues included in stdout JSON (validation still completes).",
    )
    args = parser.parse_args(argv)

    report = validate_dataset(args.dataset_root.resolve())
    payload = report.payload()
    if len(payload["issues"]) > args.max_issues:
        payload["issues_truncated"] = len(payload["issues"]) - args.max_issues
        payload["issues"] = payload["issues"][: args.max_issues]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def validate_dataset(dataset_root: Path) -> Report:
    """Validate a dataset without modifying it."""

    report = Report(dataset_root)
    if not dataset_root.is_dir():
        report.error("dataset.missing", dataset_root, "dataset root does not exist")
        return report

    _validate_no_symlinks(dataset_root, report)
    for relative in MANDATORY_DATASET_FILES:
        path = dataset_root / relative
        if not path.is_file():
            report.error("dataset.required_file_missing", path, "required dataset file is missing")

    manifest = _read_json_or_report(dataset_root / "dataset_manifest.json", report)
    cases_raw = _read_json_or_report(dataset_root / "cases.json", report)
    if not isinstance(manifest, dict) or cases_raw is None:
        return report

    case_records = _extract_records(cases_raw, ("cases",), where="cases.json", report=report)
    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(case_records):
        path = f"cases.json/cases/{index}"
        if not isinstance(record, dict):
            report.error("registry.case_not_object", path, "case record must be a JSON object")
            continue
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            report.error("registry.case_id_missing", path, "case_id must be a non-empty string")
            continue
        if case_id in registry_by_id:
            report.error("registry.case_id_duplicate", path, f"duplicate case_id {case_id!r}")
            continue
        registry_by_id[case_id] = record
        _validate_opaque_identifier(case_id, path, report)

    fixture_root = dataset_root / "fixtures"
    fixture_ids = (
        sorted(path.name for path in fixture_root.iterdir() if path.is_dir())
        if fixture_root.is_dir()
        else []
    )
    if not fixture_root.is_dir():
        report.error("dataset.fixtures_missing", fixture_root, "fixtures directory is missing")
    if set(fixture_ids) != set(registry_by_id):
        report.error(
            "registry.fixture_mismatch",
            fixture_root,
            "fixture directories and cases.json IDs differ: "
            f"missing={sorted(set(registry_by_id) - set(fixture_ids))}, "
            f"unregistered={sorted(set(fixture_ids) - set(registry_by_id))}",
        )

    declared_count = manifest.get("case_count")
    if declared_count != EXPECTED_CASE_COUNT:
        report.error(
            "manifest.case_count",
            dataset_root / "dataset_manifest.json",
            f"case_count must be {EXPECTED_CASE_COUNT}, got {declared_count!r}",
        )
    if len(registry_by_id) != EXPECTED_CASE_COUNT:
        report.error(
            "registry.case_count",
            dataset_root / "cases.json",
            f"must contain exactly {EXPECTED_CASE_COUNT} unique cases, got {len(registry_by_id)}",
        )
    _validate_manifest_lifecycle(manifest, dataset_root / "dataset_manifest.json", report)

    bundles: dict[str, CaseBundle] = {}
    for case_id in sorted(set(fixture_ids) & set(registry_by_id)):
        bundle = _load_case_bundle(
            fixture_root / case_id,
            case_id=case_id,
            registry=registry_by_id[case_id],
            report=report,
        )
        if bundle is None:
            continue
        bundles[case_id] = bundle
        _validate_case(bundle, report)

    _validate_distribution(bundles, report)
    _validate_counterfactuals(bundles, report)
    _validate_inventories(dataset_root, bundles, report)
    _validate_file_inventory(dataset_root, report)
    _validate_global_hash_fields(dataset_root, report)

    metric_counts = Counter(
        event.get("metric")
        for bundle in bundles.values()
        for event in bundle.metric_events
        if isinstance(event, dict)
    )
    report.summary.update(
        {
            "expected_case_count": EXPECTED_CASE_COUNT,
            "discovered_case_count": len(registry_by_id),
            "loaded_case_count": len(bundles),
            "event_count": sum(metric_counts.values()),
            "metric_counts": dict(sorted(metric_counts.items())),
            "expected_metric_counts": EXPECTED_METRIC_COUNTS,
            "pending_human_case_count": sum(
                _is_pending_review(bundle.review) for bundle in bundles.values()
            ),
        }
    )
    return report


def _load_case_bundle(
    case_root: Path,
    *,
    case_id: str,
    registry: dict[str, Any],
    report: Report,
) -> CaseBundle | None:
    values: dict[str, Any] = {}
    missing = False
    for filename in MANDATORY_CASE_FILES:
        path = case_root / filename
        if not path.is_file():
            report.error("case.required_file_missing", path, "required case file is missing")
            missing = True
            continue
        values[filename] = _read_json_or_report(path, report)
    if missing or any(values.get(filename) is None for filename in MANDATORY_CASE_FILES):
        return None

    deviations = _extract_deviations(
        values["authorized_deviations.json"],
        path=case_root / "authorized_deviations.json",
        report=report,
    )
    metric_events = _extract_records(
        values["metric_events.json"],
        ("events", "metric_events"),
        where=str(case_root / "metric_events.json"),
        report=report,
    )
    metric_gt = _extract_records(
        values["metric_gt.json"],
        ("events", "metric_gt"),
        where=str(case_root / "metric_gt.json"),
        report=report,
    )
    mappings = (
        "generated_scene.json",
        "scene_request.json",
        "object_plan.json",
        "asset_selection.json",
        "reference_annotation.json",
        "specification_contract.json",
        "object_grouping_report.json",
        "evidence_expectations.json",
        "construction_manifest.json",
        "provenance.json",
        "review.json",
    )
    for filename in mappings:
        if not isinstance(values[filename], dict):
            report.error(
                "case.artifact_not_object",
                case_root / filename,
                "artifact must be a JSON object",
            )
            return None

    return CaseBundle(
        case_id=case_id,
        root=case_root,
        registry=registry,
        generated_scene=values["generated_scene.json"],
        scene_request=values["scene_request.json"],
        object_plan=values["object_plan.json"],
        asset_selection=values["asset_selection.json"],
        reference_annotation=values["reference_annotation.json"],
        specification_contract=values["specification_contract.json"],
        authorized_deviations_raw=values["authorized_deviations.json"],
        authorized_deviations=deviations,
        grouping_report=values["object_grouping_report.json"],
        metric_events_raw=values["metric_events.json"],
        metric_events=metric_events,
        metric_gt_raw=values["metric_gt.json"],
        metric_gt=metric_gt,
        evidence_expectations=values["evidence_expectations.json"],
        construction_manifest=values["construction_manifest.json"],
        provenance=values["provenance.json"],
        review=values["review.json"],
    )


def _validate_case(bundle: CaseBundle, report: Report) -> None:
    _validate_canonical_artifacts(bundle, report)
    _validate_ids(bundle, report)
    _validate_prompt_and_claims(bundle, report)
    _validate_deviations(bundle, report)
    _validate_events_and_gt(bundle, report)
    _validate_pending_lifecycle(bundle, report)
    _validate_grouping(bundle, report)
    _validate_asset_closure(bundle, report)
    _validate_public_leakage(bundle, report)
    _validate_evidence_context(bundle, report)
    _validate_case_hashes(bundle, report)


def _validate_canonical_artifacts(bundle: CaseBundle, report: Report) -> None:
    validators = (
        ("generated_scene.json", validate_generated_scene, bundle.generated_scene),
        ("scene_request.json", validate_scene_request, bundle.scene_request),
        ("object_plan.json", validate_object_plan, bundle.object_plan),
        ("asset_selection.json", validate_asset_selection, bundle.asset_selection),
    )
    for filename, validator, value in validators:
        try:
            validator(value)
        except Exception as exc:
            report.error("canonical.validation", bundle.root / filename, str(exc))

    try:
        validate_reference_annotation(bundle.reference_annotation)
    except Exception as exc:
        report.error(
            "reference_annotation.validation",
            bundle.root / "reference_annotation.json",
            str(exc),
        )

    object_ids = {
        str(item.get("id"))
        for item in bundle.generated_scene.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    try:
        validate_frozen_cal_dataset2_v0_specification_contract(
            bundle.specification_contract,
            valid_object_ids=object_ids,
        )
    except Exception as exc:
        report.error(
            "specification_contract.validation",
            bundle.root / "specification_contract.json",
            str(exc),
        )

    asset_policy = (
        bundle.scene_request.get("asset_policy")
        or bundle.object_plan.get("asset_policy")
        or bundle.specification_contract.get("asset_policy")
    )
    if asset_policy is None:
        report.error(
            "asset_policy.missing",
            bundle.root,
            "case must declare asset_policy independently of prompt granularity",
        )
    else:
        try:
            normalized_policy = validate_asset_policy(asset_policy)
            expected = {
                "mode": "fixed_catalog_selection",
                "identity_owner": "benchmark",
                "category_selection_owner": "generator",
                "scale_owner": "generator",
                "appearance_owner": "generator",
                "arrangement_owner": "generator",
            }
            if any(normalized_policy.get(key) != value for key, value in expected.items()):
                report.error(
                    "asset_policy.unexpected",
                    bundle.root,
                    f"asset policy must match frozen calibration policy {expected!r}",
                )
        except Exception as exc:
            report.error("asset_policy.validation", bundle.root, str(exc))


def _validate_ids(bundle: CaseBundle, report: Report) -> None:
    expected_request_id = bundle.registry.get("request_id") or bundle.case_id
    request_fields = (
        ("generated_scene.json", bundle.generated_scene.get("request_id")),
        ("scene_request.json", bundle.scene_request.get("request_id")),
        ("object_plan.json", bundle.object_plan.get("request_id")),
        ("asset_selection.json", bundle.asset_selection.get("request_id")),
        ("reference_annotation.json", bundle.reference_annotation.get("request_id")),
        ("specification_contract.json", bundle.specification_contract.get("request_id")),
    )
    for filename, value in request_fields:
        if value != expected_request_id:
            report.error(
                "ids.request_id_mismatch",
                bundle.root / filename,
                f"request_id must be {expected_request_id!r}, got {value!r}",
            )

    for filename, artifact in (
        ("metric_events.json", bundle.metric_events_raw),
        ("metric_gt.json", bundle.metric_gt_raw),
        ("evidence_expectations.json", bundle.evidence_expectations),
        ("construction_manifest.json", bundle.construction_manifest),
        ("provenance.json", bundle.provenance),
        ("review.json", bundle.review),
    ):
        if isinstance(artifact, dict):
            case_id = artifact.get("case_id")
            if case_id != bundle.case_id:
                report.error(
                    "ids.case_id_mismatch",
                    bundle.root / filename,
                    f"case_id must be {bundle.case_id!r}, got {case_id!r}",
                )

    scene_ids = _scene_object_ids(bundle.generated_scene)
    plan_ids = {
        str(item.get("id"))
        for item in bundle.object_plan.get("objects", [])
        if isinstance(item, dict)
    }
    asset_ids = {
        str(item.get("object_id"))
        for item in bundle.asset_selection.get("objects", [])
        if isinstance(item, dict)
    }
    if plan_ids != scene_ids:
        report.error(
            "ids.object_plan_partition",
            bundle.root / "object_plan.json",
            f"object_plan IDs differ from generated_scene: missing={sorted(scene_ids-plan_ids)}, "
            f"extra={sorted(plan_ids-scene_ids)}",
        )
    if asset_ids != scene_ids:
        report.error(
            "ids.asset_selection_partition",
            bundle.root / "asset_selection.json",
            f"asset_selection IDs differ from generated_scene: missing={sorted(scene_ids-asset_ids)}, "
            f"extra={sorted(asset_ids-scene_ids)}",
        )

    selected_by_object = {
        str(item.get("object_id")): item.get("selected_asset", {})
        for item in bundle.asset_selection.get("objects", [])
        if isinstance(item, dict)
    }
    for scene_object in bundle.generated_scene.get("objects", []):
        if not isinstance(scene_object, dict):
            continue
        object_id = str(scene_object.get("id"))
        selected = selected_by_object.get(object_id)
        if not isinstance(selected, dict):
            continue
        if scene_object.get("jid") != selected.get("jid"):
            report.error(
                "ids.asset_jid_mismatch",
                bundle.root / "generated_scene.json",
                f"{object_id}: scene jid {scene_object.get('jid')!r} != "
                f"selected jid {selected.get('jid')!r}",
            )
        scene_key = (scene_object.get("asset_ref") or {}).get("asset_key")
        selected_key = (selected.get("asset_ref") or {}).get("asset_key")
        if scene_key != selected_key:
            report.error(
                "ids.asset_key_mismatch",
                bundle.root / "generated_scene.json",
                f"{object_id}: scene asset_key {scene_key!r} != selected asset_key {selected_key!r}",
            )


def _validate_prompt_and_claims(bundle: CaseBundle, report: Report) -> None:
    prompt = bundle.scene_request.get("instruction")
    if not isinstance(prompt, str) or not prompt:
        return
    claims = bundle.specification_contract.get("claims")
    if not isinstance(claims, dict):
        return
    actual_families = set(claims)
    frozen_dataset_v0_family_set = set(FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES)
    if actual_families != frozen_dataset_v0_family_set:
        report.error(
            "claims.family_set",
            bundle.root / "specification_contract.json",
            "the read-only cal_dataset2 v0 validator requires its exact frozen "
            "claim-family inventory; "
            f"actual={sorted(actual_families)}",
        )

    claim_ids: set[str] = set()
    for family in FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES:
        family_claims = claims.get(family, [])
        if not isinstance(family_claims, list):
            continue
        for index, claim in enumerate(family_claims):
            path = (
                bundle.root
                / f"specification_contract.json#claims/{family}/{index}"
            )
            if not isinstance(claim, dict):
                continue
            claim_id = claim.get("claim_id")
            if isinstance(claim_id, str):
                claim_ids.add(claim_id)
            _validate_exact_span(
                prompt,
                claim,
                text_key="source_span",
                offset_prefix="source",
                path=path,
                report=report,
            )
            expected = claim.get("expected")
            if family in {
                "room_scene_type",
                "broad_semantic_intent",
                "required_functional_areas",
            } and (not isinstance(expected, dict) or not expected):
                report.error(
                    "claims.high_level_expected",
                    path,
                    f"{family} claim must have a non-empty typed expected payload",
                )
            if family == "oor":
                if not isinstance(claim.get("relation_id"), str):
                    report.error("claims.oor_relation_id", path, "OOR claim requires relation_id")
                if not isinstance(
                    claim.get("relation_type") or claim.get("type"), str
                ):
                    report.error(
                        "claims.oor_relation_type", path, "OOR claim requires relation_type"
                    )
                if len(claim.get("target_ids") or []) < 2:
                    report.error(
                        "claims.oor_targets", path, "OOR claim requires at least two target_ids"
                    )
            if family == "oar":
                if not isinstance(claim.get("subject_id"), str):
                    report.error("claims.oar_subject", path, "OAR claim requires subject_id")
                architecture = (
                    claim.get("architectural_element")
                    or claim.get("architecture_target")
                    or claim.get("target_architecture")
                )
                if not isinstance(architecture, str) or not architecture:
                    report.error(
                        "claims.oar_architecture",
                        path,
                        "OAR claim requires a canonical architectural target",
                    )

    active_metric = _single_metric(bundle, report)
    granularity = bundle.scene_request.get("prompt_granularity")
    if active_metric in {"oor", "oar"} and granularity != "fine_grained":
        report.error(
            "prompt_granularity.runtime_compatibility",
            bundle.root / "scene_request.json",
            f"{active_metric} cases must currently be fine_grained because runtime OOR/OAR "
            "execution is still gated there",
        )
    if (
        active_metric
        in {"room_scene_type", "broad_semantic_intent", "required_functional_areas"}
        and granularity != "coarse_grained"
    ):
        report.error(
            "prompt_granularity.high_level",
            bundle.root / "scene_request.json",
            f"{active_metric} cases must be coarse_grained",
        )
    if active_metric in L2_METRICS:
        family_claims = claims.get(active_metric, [])
        if not family_claims:
            report.error(
                "claims.active_family_empty",
                bundle.root / "specification_contract.json",
                f"metric {active_metric} requires at least one claim in its family",
            )

    if active_metric == "room_scene_type":
        scene_type = str(bundle.generated_scene.get("scene_type") or "").lower()
        if scene_type not in NEUTRAL_SCENE_TYPES:
            report.error(
                "leakage.room_scene_type_metadata",
                bundle.root / "generated_scene.json",
                f"room_scene_type cases require neutral generated_scene.scene_type, got {scene_type!r}",
            )


def _validate_deviations(bundle: CaseBundle, report: Report) -> None:
    object_ids = _scene_object_ids(bundle.generated_scene)
    try:
        normalized = validate_authorized_deviations(
            bundle.authorized_deviations,
            allowed_metrics=L3_METRICS,
        )
    except Exception as exc:
        report.error(
            "authorized_deviations.validation",
            bundle.root / "authorized_deviations.json",
            str(exc),
        )
        return
    if normalized != bundle.authorized_deviations:
        report.error(
            "authorized_deviations.noncanonical",
            bundle.root / "authorized_deviations.json",
            "authorized deviations are not in canonical normalized form",
        )

    prompt = str(bundle.scene_request.get("instruction") or "")
    all_claims = {
        str(claim.get("claim_id")): claim
        for family_claims in (bundle.specification_contract.get("claims") or {}).values()
        if isinstance(family_claims, list)
        for claim in family_claims
        if isinstance(claim, dict) and claim.get("claim_id") is not None
    }
    for index, deviation in enumerate(bundle.authorized_deviations):
        path = bundle.root / f"authorized_deviations.json#{index}"
        targets = set(map(str, deviation.get("target_ids") or []))
        if not targets.issubset(object_ids):
            report.error(
                "authorized_deviations.unknown_target",
                path,
                f"unknown targets {sorted(targets-object_ids)}",
            )
        source_claim_id = deviation.get("source_claim_id")
        if source_claim_id not in all_claims:
            report.error(
                "authorized_deviations.unknown_claim",
                path,
                f"source_claim_id {source_claim_id!r} does not name a contract claim",
            )
        else:
            claim_targets = set(map(str, all_claims[source_claim_id].get("target_ids") or []))
            if not targets.issubset(claim_targets):
                report.error(
                    "authorized_deviations.claim_scope",
                    path,
                    "deviation targets must be a subset of linked claim targets",
                )
        _validate_exact_span(
            prompt,
            deviation,
            text_key="prompt_span",
            offset_prefix="prompt",
            path=path,
            report=report,
        )

    embedded_lists: list[tuple[str, Any]] = []
    for name, value in (
        (
            "specification_contract.authorized_deviations",
            bundle.specification_contract.get("authorized_deviations"),
        ),
        ("scene_request.authorized_deviations", bundle.scene_request.get("authorized_deviations")),
        (
            "scene_request.evaluation_context.authorized_deviations",
            (bundle.scene_request.get("evaluation_context") or {}).get(
                "authorized_deviations"
            )
            if isinstance(bundle.scene_request.get("evaluation_context"), dict)
            else None,
        ),
    ):
        if value is not None:
            embedded_lists.append((name, value))
    if not embedded_lists:
        report.error(
            "authorized_deviations.embedding_missing",
            bundle.root,
            "standalone authorized deviations must be embedded in the specification "
            "contract or scene-request evaluation context",
        )
    for name, value in embedded_lists:
        if value != bundle.authorized_deviations:
            report.error(
                "authorized_deviations.embedding_mismatch",
                bundle.root,
                f"{name} differs from authorized_deviations.json",
            )


def _validate_events_and_gt(bundle: CaseBundle, report: Report) -> None:
    if len(bundle.metric_events) != 1:
        report.error(
            "events.per_case_count",
            bundle.root / "metric_events.json",
            f"each case must contain exactly one metric event, got {len(bundle.metric_events)}",
        )
    if len(bundle.metric_gt) != len(bundle.metric_events):
        report.error(
            "gt.event_count",
            bundle.root / "metric_gt.json",
            "metric_gt event count must match metric_events",
        )

    event_by_id: dict[str, dict[str, Any]] = {}
    object_ids = _scene_object_ids(bundle.generated_scene)
    for index, event in enumerate(bundle.metric_events):
        path = bundle.root / f"metric_events.json#{index}"
        if not isinstance(event, dict):
            report.error("events.not_object", path, "metric event must be an object")
            continue
        event_id = _event_id(event)
        if not event_id:
            report.error("events.id_missing", path, "event_id must be non-empty")
            continue
        _validate_opaque_identifier(event_id, path, report, allow_event_prefix=True)
        if event_id in event_by_id:
            report.error("events.id_duplicate", path, f"duplicate event_id {event_id!r}")
        event_by_id[event_id] = event
        metric = event.get("metric")
        if metric not in CANONICAL_METRICS:
            report.error(
                "events.metric",
                path,
                f"metric must be one of {list(CANONICAL_METRICS)}, got {metric!r}",
            )
        expected_level = "L2" if metric in L2_METRICS else "L3"
        if str(event.get("level") or "").upper() != expected_level:
            report.error(
                "events.level",
                path,
                f"metric {metric!r} must declare level {expected_level}",
            )
        targets = event.get("target_ids")
        if not isinstance(targets, list) or not targets:
            report.error("events.targets", path, "target_ids must be a non-empty list")
        elif not set(map(str, targets)).issubset(object_ids):
            report.error(
                "events.unknown_target",
                path,
                f"unknown targets {sorted(set(map(str, targets))-object_ids)}",
            )

    gt_by_id: dict[str, dict[str, Any]] = {}
    for index, gt in enumerate(bundle.metric_gt):
        path = bundle.root / f"metric_gt.json#{index}"
        if not isinstance(gt, dict):
            report.error("gt.not_object", path, "GT event must be an object")
            continue
        event_id = _event_id(gt)
        if not event_id:
            report.error("gt.event_id_missing", path, "event_id must be non-empty")
            continue
        gt_by_id[event_id] = gt
        event = event_by_id.get(event_id)
        if event is None:
            report.error("gt.event_unknown", path, f"unknown event_id {event_id!r}")
            continue
        if gt.get("metric") != event.get("metric"):
            report.error("gt.metric_mismatch", path, "GT metric must match metric event")
        if list(gt.get("target_ids") or []) != list(event.get("target_ids") or []):
            report.error("gt.targets_mismatch", path, "GT target_ids must match metric event")
        semantic = gt.get("semantic_label")
        if semantic not in {"pending_review", None}:
            report.error(
                "gt.premature_semantic_label",
                path,
                "semantic_label must remain pending_review/null until human approval; "
                "use proposed_semantic_label for construction-time hypotheses",
            )
        proposed = gt.get("proposed_semantic_label")
        if proposed not in ALLOWED_PROPOSED_LABELS:
            report.error(
                "gt.proposed_label",
                path,
                f"proposed_semantic_label must be one of {sorted(ALLOWED_PROPOSED_LABELS)}",
            )
        if gt.get("accuracy_eligible") is not False:
            report.error(
                "gt.accuracy_eligible",
                path,
                "pending-human GT must set accuracy_eligible=false",
            )
        review_state = gt.get("review_status") or gt.get("label_status")
        if review_state not in PENDING_REVIEW_STATES:
            report.error(
                "gt.review_status",
                path,
                f"GT review status must be pending-human, got {review_state!r}",
            )
        if not isinstance(gt.get("gt_basis"), str) or not gt.get("gt_basis"):
            report.error("gt.basis", path, "gt_basis must be a non-empty string")

        if gt.get("metric") in L3_METRICS:
            raw = gt.get("raw_coherence_label") or gt.get("raw_consistency_label")
            if raw not in ALLOWED_PROPOSED_LABELS:
                report.error(
                    "gt.raw_l3_label",
                    path,
                    "L3 GT requires raw_coherence_label/raw_consistency_label",
                )
            authorization_applied = gt.get("authorization_applied")
            if not isinstance(authorization_applied, bool):
                report.error(
                    "gt.authorization_applied",
                    path,
                    "L3 GT requires boolean authorization_applied",
                )
            if authorization_applied and not bundle.authorized_deviations:
                report.error(
                    "gt.authorization_without_deviation",
                    path,
                    "authorization_applied=true requires a scoped authorized deviation",
                )

    if set(gt_by_id) != set(event_by_id):
        report.error(
            "gt.event_partition",
            bundle.root / "metric_gt.json",
            f"GT/event IDs differ: gt_only={sorted(set(gt_by_id)-set(event_by_id))}, "
            f"event_only={sorted(set(event_by_id)-set(gt_by_id))}",
        )


def _validate_pending_lifecycle(bundle: CaseBundle, report: Report) -> None:
    if bundle.specification_contract.get("frozen") is not False:
        report.error(
            "lifecycle.contract_frozen",
            bundle.root / "specification_contract.json",
            "pre-review specification contract must set frozen=false",
        )
    if bundle.specification_contract.get("source") not in {
        "benchmark_owned",
        "diagnostic",
    }:
        report.error(
            "lifecycle.contract_source",
            bundle.root / "specification_contract.json",
            "pre-review contract source must be benchmark_owned or diagnostic",
        )
    if bundle.reference_annotation.get("validation_status") != "draft":
        report.error(
            "lifecycle.annotation_status",
            bundle.root / "reference_annotation.json",
            "reference annotation must remain draft before human review",
        )
    if not _is_pending_review(bundle.review):
        report.error(
            "lifecycle.review_status",
            bundle.root / "review.json",
            "review status must remain pending_human/pending",
        )
    for field in ("reviewer", "reviewed_at", "approved_by", "approved_at"):
        if bundle.review.get(field) not in (None, ""):
            report.error(
                "lifecycle.premature_approval",
                bundle.root / "review.json",
                f"{field} must be null before human review",
            )
    if bundle.registry.get("review_status") not in PENDING_REVIEW_STATES:
        report.error(
            "lifecycle.registry_review_status",
            "cases.json",
            f"{bundle.case_id}: registry review_status must be pending-human",
        )


def _validate_grouping(bundle: CaseBundle, report: Report) -> None:
    groups = bundle.grouping_report.get("object_groups")
    if not isinstance(groups, list) or not groups:
        report.error(
            "grouping.object_groups",
            bundle.root / "object_grouping_report.json",
            "object_groups must be a non-empty list",
        )
        return
    seen_group_ids: set[str] = set()
    grouped_ids: list[str] = []
    for index, group in enumerate(groups):
        path = bundle.root / f"object_grouping_report.json#object_groups/{index}"
        if not isinstance(group, dict):
            report.error("grouping.group_not_object", path, "group must be a JSON object")
            continue
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            report.error("grouping.group_id", path, "group_id must be non-empty")
        elif group_id in seen_group_ids:
            report.error("grouping.group_id_duplicate", path, f"duplicate {group_id!r}")
        else:
            seen_group_ids.add(group_id)
        object_ids = group.get("object_ids")
        if not isinstance(object_ids, list) or not object_ids:
            report.error(
                "grouping.object_ids",
                path,
                "every group must contain a non-empty object_ids list",
            )
        else:
            grouped_ids.extend(map(str, object_ids))
    expected = _scene_object_ids(bundle.generated_scene)
    counts = Counter(grouped_ids)
    duplicates = sorted(object_id for object_id, count in counts.items() if count != 1)
    if set(grouped_ids) != expected or duplicates:
        report.error(
            "grouping.partition",
            bundle.root / "object_grouping_report.json",
            f"groups must partition every scene object exactly once; "
            f"missing={sorted(expected-set(grouped_ids))}, "
            f"extra={sorted(set(grouped_ids)-expected)}, duplicates={duplicates}",
        )
    provenance = bundle.grouping_report.get("grouping_policy") or bundle.grouping_report.get(
        "provenance"
    )
    policy_id = (
        provenance.get("policy_id") if isinstance(provenance, dict) else None
    ) or bundle.grouping_report.get("policy_id")
    if policy_id != "deterministic_metadata_geometry":
        report.error(
            "grouping.policy",
            bundle.root / "object_grouping_report.json",
            "grouping report must identify deterministic_metadata_geometry as the "
            "evidence-partition policy",
        )


def _validate_asset_closure(bundle: CaseBundle, report: Report) -> None:
    asset_root = REPO_ROOT / "Support" / "Assets" / "imaginarium_assets"
    for index, item in enumerate(bundle.asset_selection.get("objects", [])):
        if not isinstance(item, dict):
            continue
        selected = item.get("selected_asset")
        if not isinstance(selected, dict):
            continue
        jid = selected.get("jid")
        if not isinstance(jid, str) or not jid:
            continue
        if Path(jid).name != jid or jid in {".", ".."}:
            report.error(
                "assets.unsafe_jid",
                bundle.root / "asset_selection.json",
                f"unsafe jid {jid!r}",
            )
            continue
        asset_dir = asset_root / jid
        fbx = asset_dir / f"{jid}.fbx"
        metadata = asset_dir / f"{jid}_metadata.json"
        if not asset_dir.is_dir() or not fbx.is_file() or not metadata.is_file():
            report.error(
                "assets.closure",
                bundle.root / f"asset_selection.json#objects/{index}",
                f"jid {jid!r} must resolve to {fbx.relative_to(REPO_ROOT)} and "
                f"{metadata.relative_to(REPO_ROOT)}",
            )
        asset_ref = selected.get("asset_ref")
        if not isinstance(asset_ref, dict) or asset_ref.get("asset_key") != jid:
            report.error(
                "assets.asset_ref",
                bundle.root / f"asset_selection.json#objects/{index}",
                "selected_asset.asset_ref.asset_key must equal selected jid",
            )


def _validate_public_leakage(bundle: CaseBundle, report: Report) -> None:
    _validate_opaque_identifier(bundle.case_id, bundle.root, report)
    public_artifacts = {
        "generated_scene.json": bundle.generated_scene,
        "scene_request.json": bundle.scene_request,
        "object_plan.json": bundle.object_plan,
        "asset_selection.json": bundle.asset_selection,
        "specification_contract.json": bundle.specification_contract,
        "authorized_deviations.json": bundle.authorized_deviations_raw,
        "object_grouping_report.json": bundle.grouping_report,
        "metric_events.json": bundle.metric_events_raw,
    }
    forbidden_keys = {
        "accuracy_eligible",
        "authorization_applied",
        "counterfactual_condition",
        "ground_truth",
        "gt_label",
        "human_label",
        "proposed_semantic_label",
        "raw_coherence_label",
        "raw_consistency_label",
        "scene_deviation",
        "semantic_label",
    }
    for filename, artifact in public_artifacts.items():
        for pointer, key, value in _walk_json(artifact):
            if key in forbidden_keys:
                report.error(
                    "leakage.public_gt_field",
                    bundle.root / filename,
                    f"public artifact exposes forbidden GT field at {pointer}/{key}",
                )
            if key in {"id", "case_id", "scene_id", "request_id", "event_id", "metric_event_id"}:
                if isinstance(value, str) and LEAKAGE_TOKEN_RE.search(value):
                    report.error(
                        "leakage.identifier",
                        bundle.root / filename,
                        f"identifier {value!r} at {pointer}/{key} leaks a condition/label",
                    )
        if filename in {"generated_scene.json", "object_plan.json"}:
            for pointer, key, value in _walk_json(artifact):
                if key in {"description", "desc", "name", "label"} and isinstance(value, str):
                    if PUBLIC_TEXT_LEAK_RE.search(value):
                        report.error(
                            "leakage.description",
                            bundle.root / filename,
                            f"public description at {pointer}/{key} leaks construction truth: {value!r}",
                        )


def _validate_evidence_context(bundle: CaseBundle, report: Report) -> None:
    lists = list(_outbound_allowlists(bundle.evidence_expectations))
    if not lists:
        report.error(
            "evidence.outbound_allowlist_missing",
            bundle.root / "evidence_expectations.json",
            "evidence expectations must declare an outbound judge-context allowlist",
        )
    for pointer, values in lists:
        for value in values:
            normalized = str(value).strip().lower()
            tokens = {
                token
                for token in re.split(r"[^a-z0-9_]+", normalized)
                if token
            }
            if normalized in FORBIDDEN_OUTBOUND_TOKENS or tokens & FORBIDDEN_OUTBOUND_TOKENS:
                report.error(
                    "evidence.gt_in_outbound_allowlist",
                    bundle.root / "evidence_expectations.json",
                    f"{pointer} exposes forbidden field {value!r}",
                )
        if "original_prompt" not in values:
            report.error(
                "evidence.original_prompt_missing",
                bundle.root / "evidence_expectations.json",
                f"{pointer} must include original_prompt",
            )
        if "target_ids" not in values and "group_ids" not in values:
            report.error(
                "evidence.identity_context_missing",
                bundle.root / "evidence_expectations.json",
                f"{pointer} must include target_ids or group_ids",
            )

    serialized = json.dumps(bundle.evidence_expectations, sort_keys=True)
    for private_name in (
        "metric_gt.json",
        "construction_manifest.json",
        "provenance.json",
        "review.json",
    ):
        if private_name in serialized:
            report.error(
                "evidence.private_artifact_reference",
                bundle.root / "evidence_expectations.json",
                f"judge context must not include private artifact {private_name}",
            )


def _validate_case_hashes(bundle: CaseBundle, report: Report) -> None:
    manifest = bundle.construction_manifest
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        report.error(
            "hashes.artifact_hashes_missing",
            bundle.root / "construction_manifest.json",
            "construction_manifest.artifact_hashes must be a non-empty mapping",
        )
    else:
        required_hashed = set(MANDATORY_CASE_FILES) - {"construction_manifest.json"}
        missing = required_hashed - set(artifact_hashes)
        if missing:
            report.error(
                "hashes.artifact_hashes_incomplete",
                bundle.root / "construction_manifest.json",
                f"artifact_hashes missing {sorted(missing)}",
            )
        for relative, declared in artifact_hashes.items():
            path = bundle.root / str(relative)
            _validate_declared_file_hash(
                path,
                declared,
                bundle.root / "construction_manifest.json",
                report,
            )

    computed_invariants = {
        "scene_geometry_sha256": scene_geometry_sha256(bundle.generated_scene),
        "prompt_sha256": _sha256_text(str(bundle.scene_request.get("instruction") or "")),
        "asset_selection_sha256": canonical_json_sha256(bundle.asset_selection),
        "grouping_partition_sha256": grouping_partition_sha256(bundle.grouping_report),
    }
    invariant_hashes = manifest.get("invariant_hashes") or manifest.get("hashes")
    if not isinstance(invariant_hashes, dict):
        report.error(
            "hashes.invariant_hashes_missing",
            bundle.root / "construction_manifest.json",
            "construction_manifest must declare invariant_hashes",
        )
    else:
        for key, computed in computed_invariants.items():
            declared = invariant_hashes.get(key)
            if declared != computed:
                report.error(
                    "hashes.invariant_mismatch",
                    bundle.root / "construction_manifest.json",
                    f"{key} must be {computed}, got {declared!r}",
                )

    for pointer, key, value in _walk_json(manifest):
        if key.endswith("sha256") and (
            not isinstance(value, str) or not HEX64_RE.fullmatch(value)
        ):
            report.error(
                "hashes.format",
                bundle.root / "construction_manifest.json",
                f"{pointer}/{key} must be a lowercase 64-character SHA-256",
            )


def _validate_distribution(bundles: Mapping[str, CaseBundle], report: Report) -> None:
    metric_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    cells_by_metric: dict[str, list[tuple[str, str, str, str | None]]] = defaultdict(list)
    event_ids: set[str] = set()

    for bundle in bundles.values():
        if len(bundle.metric_events) != 1:
            continue
        event = bundle.metric_events[0]
        if not isinstance(event, dict):
            continue
        metric = str(event.get("metric"))
        metric_counts[metric] += 1
        event_id = _event_id(event)
        if event_id in event_ids:
            report.error(
                "distribution.event_id_duplicate_global",
                bundle.root / "metric_events.json",
                f"event_id {event_id!r} duplicates another case",
            )
        event_ids.add(event_id)

        design_role = (
            bundle.construction_manifest.get("design_role")
            or bundle.registry.get("design_role")
        )
        if design_role == "base_metric_case":
            base_counts[metric] += 1
        elif design_role == "prompt_authorization_2x2":
            prompt_condition = bundle.construction_manifest.get("prompt_authorization")
            scene_condition = bundle.construction_manifest.get("scene_deviation")
            group_id = bundle.construction_manifest.get("counterfactual_group_id")
            visual_source = bundle.construction_manifest.get("visual_source_case_id")
            cells_by_metric[metric].append(
                (str(prompt_condition), str(scene_condition), str(group_id), _string_or_none(visual_source))
            )
        else:
            report.error(
                "distribution.design_role",
                bundle.root / "construction_manifest.json",
                "design_role must be base_metric_case or prompt_authorization_2x2",
            )

    if dict(metric_counts) != EXPECTED_METRIC_COUNTS:
        report.error(
            "distribution.metric_counts",
            "fixtures",
            f"metric counts must be {EXPECTED_METRIC_COUNTS}, got {dict(metric_counts)}",
        )
    if dict(base_counts) != EXPECTED_BASE_COUNTS:
        report.error(
            "distribution.base_counts",
            "fixtures",
            f"base case counts must be {EXPECTED_BASE_COUNTS}, got {dict(base_counts)}",
        )
    for metric in L3_METRICS:
        cells = cells_by_metric.get(metric, [])
        actual_cells = {(prompt, scene) for prompt, scene, _, _ in cells}
        if len(cells) != 4 or actual_cells != EXPECTED_AUTHORIZATION_CELLS:
            report.error(
                "distribution.authorization_2x2",
                "fixtures",
                f"{metric} must have exactly the four prompt-authorization x scene-deviation "
                f"cells; got {cells!r}",
            )
        group_ids = {group_id for _, _, group_id, _ in cells}
        if len(group_ids) != 1 or "None" in group_ids or "" in group_ids:
            report.error(
                "distribution.authorization_group",
                "fixtures",
                f"{metric} 2x2 cells must share one counterfactual_group_id",
            )
    for metric in L2_METRICS:
        if cells_by_metric.get(metric):
            report.error(
                "distribution.unexpected_l2_2x2",
                "fixtures",
                f"{metric} must not contain prompt-authorization 2x2 cases",
            )


def _validate_counterfactuals(
    bundles: Mapping[str, CaseBundle],
    report: Report,
) -> None:
    groups: dict[str, list[CaseBundle]] = defaultdict(list)
    for bundle in bundles.values():
        manifest = bundle.construction_manifest
        group_id = manifest.get("counterfactual_group_id")
        if not isinstance(group_id, str) or not group_id:
            report.error(
                "counterfactual.group_id_missing",
                bundle.root / "construction_manifest.json",
                "every case must declare a counterfactual_group_id",
            )
            continue
        _validate_opaque_identifier(group_id, bundle.root, report, allow_group_prefix=True)
        groups[group_id].append(bundle)

        comparison_id = manifest.get("comparison_case_id")
        allowed_diffs = manifest.get("allowed_changed_paths")
        if comparison_id is not None:
            if comparison_id not in bundles:
                report.error(
                    "counterfactual.comparison_case_missing",
                    bundle.root / "construction_manifest.json",
                    f"comparison_case_id {comparison_id!r} is not a dataset case",
                )
            elif not isinstance(allowed_diffs, dict):
                report.error(
                    "counterfactual.allowed_paths_missing",
                    bundle.root / "construction_manifest.json",
                    "comparison cases require allowed_changed_paths by artifact",
                )
            else:
                _validate_declared_differences(
                    bundle,
                    bundles[str(comparison_id)],
                    allowed_diffs,
                    report,
                )

    for group_id, members in groups.items():
        roles = {
            member.construction_manifest.get("design_role") for member in members
        }
        if roles == {"prompt_authorization_2x2"}:
            _validate_prompt_authorization_group(group_id, members, report)
            continue
        # Base metric groups are scene counterfactuals: the semantic request and
        # active specification claims must remain exactly the same.
        prompts = {str(member.scene_request.get("instruction") or "") for member in members}
        if len(prompts) != 1:
            report.error(
                "counterfactual.base_prompt_drift",
                "fixtures",
                f"base group {group_id!r} has non-identical prompts",
            )
        semantic_contract_hashes = {
            canonical_json_sha256(_normalized_specification_contract(member.specification_contract))
            for member in members
        }
        if len(semantic_contract_hashes) != 1:
            report.error(
                "counterfactual.base_contract_drift",
                "fixtures",
                f"base group {group_id!r} changes semantic specification claims",
            )


def _validate_prompt_authorization_group(
    group_id: str,
    members: Sequence[CaseBundle],
    report: Report,
) -> None:
    by_cell: dict[tuple[str, str], CaseBundle] = {}
    for member in members:
        key = (
            str(member.construction_manifest.get("prompt_authorization")),
            str(member.construction_manifest.get("scene_deviation")),
        )
        if key in by_cell:
            report.error(
                "counterfactual.duplicate_cell",
                member.root / "construction_manifest.json",
                f"group {group_id!r} duplicates cell {key!r}",
            )
        by_cell[key] = member
    if set(by_cell) != EXPECTED_AUTHORIZATION_CELLS:
        return

    # Same scene-deviation state, different prompts: geometry/assets/grouping
    # must be identical and both cases must point to the same visual source.
    for scene_condition in ("absent", "present"):
        neutral = by_cell[("absent", scene_condition)]
        authorized = by_cell[("present", scene_condition)]
        if scene_geometry_sha256(neutral.generated_scene) != scene_geometry_sha256(
            authorized.generated_scene
        ):
            report.error(
                "counterfactual.prompt_only_geometry_drift",
                authorized.root,
                f"group {group_id!r}, scene={scene_condition}: prompt-only siblings "
                "have different normalized scene geometry",
            )
        if canonical_json_sha256(
            _normalized_asset_selection(neutral.asset_selection)
        ) != canonical_json_sha256(
            _normalized_asset_selection(authorized.asset_selection)
        ):
            report.error(
                "counterfactual.prompt_only_asset_drift",
                authorized.root,
                f"group {group_id!r}, scene={scene_condition}: prompt-only siblings "
                "have different asset selections",
            )
        if grouping_partition_sha256(
            neutral.grouping_report
        ) != grouping_partition_sha256(authorized.grouping_report):
            report.error(
                "counterfactual.prompt_only_grouping_drift",
                authorized.root,
                f"group {group_id!r}, scene={scene_condition}: prompt-only siblings "
                "have different grouping partitions",
            )
        neutral_visual = neutral.construction_manifest.get("visual_source_case_id")
        authorized_visual = authorized.construction_manifest.get("visual_source_case_id")
        if not neutral_visual or neutral_visual != authorized_visual:
            report.error(
                "counterfactual.visual_source_mismatch",
                authorized.root / "construction_manifest.json",
                f"group {group_id!r}, scene={scene_condition}: prompt-only siblings "
                "must share visual_source_case_id",
            )

    # Same prompt state, changed scene: prompt text and semantic contract must be
    # identical. Geometry must change, otherwise the 2x2 cell is not real.
    for prompt_condition in ("absent", "present"):
        normal = by_cell[(prompt_condition, "absent")]
        deviated = by_cell[(prompt_condition, "present")]
        if normal.scene_request.get("instruction") != deviated.scene_request.get("instruction"):
            report.error(
                "counterfactual.scene_only_prompt_drift",
                deviated.root,
                f"group {group_id!r}, prompt={prompt_condition}: scene-only siblings "
                "have different prompts",
            )
        if canonical_json_sha256(
            _normalized_specification_contract(normal.specification_contract)
        ) != canonical_json_sha256(
            _normalized_specification_contract(deviated.specification_contract)
        ):
            report.error(
                "counterfactual.scene_only_contract_drift",
                deviated.root,
                f"group {group_id!r}, prompt={prompt_condition}: scene-only siblings "
                "have different specification claims",
            )
        if scene_geometry_sha256(normal.generated_scene) == scene_geometry_sha256(
            deviated.generated_scene
        ):
            report.error(
                "counterfactual.scene_delta_missing",
                deviated.root,
                f"group {group_id!r}, prompt={prompt_condition}: deviation-present "
                "scene has unchanged geometry",
            )

    # The authorized prompt must be observably different from the neutral one.
    if (
        by_cell[("absent", "absent")].scene_request.get("instruction")
        == by_cell[("present", "absent")].scene_request.get("instruction")
    ):
        report.error(
            "counterfactual.prompt_delta_missing",
            by_cell[("present", "absent")].root,
            f"group {group_id!r}: authorized and neutral prompt are identical",
        )


def _validate_declared_differences(
    left: CaseBundle,
    right: CaseBundle,
    allowed_diffs: Mapping[str, Any],
    report: Report,
) -> None:
    artifact_values = {
        "generated_scene.json": (left.generated_scene, right.generated_scene),
        "scene_request.json": (left.scene_request, right.scene_request),
        "object_plan.json": (left.object_plan, right.object_plan),
        "asset_selection.json": (left.asset_selection, right.asset_selection),
        "specification_contract.json": (
            left.specification_contract,
            right.specification_contract,
        ),
        "authorized_deviations.json": (
            left.authorized_deviations_raw,
            right.authorized_deviations_raw,
        ),
    }
    for filename, declared in allowed_diffs.items():
        if filename not in artifact_values:
            report.error(
                "counterfactual.unknown_artifact",
                left.root / "construction_manifest.json",
                f"allowed_changed_paths names unsupported artifact {filename!r}",
            )
            continue
        if not isinstance(declared, list) or not all(
            isinstance(path, str) for path in declared
        ):
            report.error(
                "counterfactual.allowed_paths_format",
                left.root / "construction_manifest.json",
                f"{filename} allowed paths must be a string list",
            )
            continue
        actual = json_diff_paths(*artifact_values[filename])
        unexpected = {
            path
            for path in actual
            if not any(_pointer_is_allowed(path, allowed) for allowed in declared)
        }
        if unexpected:
            report.error(
                "counterfactual.undeclared_change",
                left.root / "construction_manifest.json",
                f"{filename} changes outside declared paths: {sorted(unexpected)}",
            )


def _validate_inventories(
    dataset_root: Path,
    bundles: Mapping[str, CaseBundle],
    report: Report,
) -> None:
    expected_cases = set(bundles)
    expected_events = {
        (bundle.case_id, _event_id(event), str(event.get("metric")))
        for bundle in bundles.values()
        for event in bundle.metric_events
        if isinstance(event, dict)
    }
    case_rows = _read_tsv(dataset_root / "validation/case_inventory.tsv", report)
    metric_rows = _read_tsv(dataset_root / "validation/metric_inventory.tsv", report)
    review_rows = _read_tsv(dataset_root / "review/review_queue.tsv", report)
    prompt_rows = _read_tsv(
        dataset_root / "validation/prompt_claim_audit.tsv", report
    )
    counterfactual_rows = _read_tsv(
        dataset_root / "validation/counterfactual_audit.tsv", report
    )

    if {row.get("case_id") for row in case_rows} != expected_cases:
        report.error(
            "inventory.case_ids",
            dataset_root / "validation/case_inventory.tsv",
            "case inventory must contain every case exactly once",
        )
    if len(case_rows) != len(expected_cases):
        report.error(
            "inventory.case_row_count",
            dataset_root / "validation/case_inventory.tsv",
            f"expected {len(expected_cases)} rows, got {len(case_rows)}",
        )

    actual_metric_rows = {
        (
            str(row.get("case_id") or ""),
            str(row.get("event_id") or row.get("metric_event_id") or ""),
            str(row.get("metric") or ""),
        )
        for row in metric_rows
    }
    if actual_metric_rows != expected_events or len(metric_rows) != len(expected_events):
        report.error(
            "inventory.metric_events",
            dataset_root / "validation/metric_inventory.tsv",
            "metric inventory must match every case/event/metric exactly once",
        )
    review_event_rows = {
        (
            str(row.get("case_id") or ""),
            str(row.get("event_id") or row.get("metric_event_id") or ""),
        )
        for row in review_rows
    }
    expected_review_rows = {(case_id, event_id) for case_id, event_id, _ in expected_events}
    if review_event_rows != expected_review_rows or len(review_rows) != len(
        expected_review_rows
    ):
        report.error(
            "inventory.review_queue",
            dataset_root / "review/review_queue.tsv",
            "review queue must contain every pending metric event exactly once",
        )
    if review_rows:
        review_columns = set(review_rows[0])
        leaked_columns = sorted(review_columns & BLIND_REVIEW_FORBIDDEN_COLUMNS)
        if leaked_columns:
            report.error(
                "review.blind_columns",
                dataset_root / "review/review_queue.tsv",
                f"blind queue exposes construction fields: {leaked_columns}",
            )
        missing_decision_columns = sorted(
            BLIND_REVIEW_DECISION_COLUMNS - review_columns
        )
        if missing_decision_columns:
            report.error(
                "review.decision_columns",
                dataset_root / "review/review_queue.tsv",
                f"blind queue lacks human decision fields: {missing_decision_columns}",
            )
        expected_order = sorted(
            (str(row.get("case_id") or "") for row in review_rows),
            key=_blind_review_sort_key,
        )
        actual_order = [str(row.get("case_id") or "") for row in review_rows]
        if actual_order != expected_order:
            report.error(
                "review.blind_order",
                dataset_root / "review/review_queue.tsv",
                "blind queue must use the deterministic hash permutation",
            )
        for row_index, row in enumerate(review_rows):
            populated = sorted(
                field
                for field in BLIND_REVIEW_DECISION_COLUMNS
                if str(row.get(field) or "").strip()
            )
            if populated:
                report.error(
                    "review.premature_decision",
                    f"review/review_queue.tsv:{row_index + 2}",
                    f"human decision fields must be blank at construction: {populated}",
                )

    review_html_path = dataset_root / "review/index.html"
    try:
        review_html = review_html_path.read_text(encoding="utf-8")
    except OSError:
        review_html = ""
    if review_html:
        forbidden_fragments = (
            "Declared controlled delta",
            "scenario=",
            "role=",
            "proposed_semantic_label",
        )
        exposed = [
            fragment for fragment in forbidden_fragments if fragment in review_html
        ]
        if exposed:
            report.error(
                "review.html_construction_leakage",
                review_html_path,
                f"blind HTML exposes construction metadata: {exposed}",
            )
        html_order = re.findall(r'<span class="case">([^<]+)</span>', review_html)
        queue_order = [str(row.get("case_id") or "") for row in review_rows]
        if html_order != queue_order:
            report.error(
                "review.html_order",
                review_html_path,
                "HTML card order must exactly match the shuffled review queue",
            )

    expected_claims = {
        (bundle.case_id, str(claim.get("claim_id")))
        for bundle in bundles.values()
        for claims in (bundle.specification_contract.get("claims") or {}).values()
        if isinstance(claims, list)
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id") is not None
    }
    actual_claims = {
        (str(row.get("case_id") or ""), str(row.get("claim_id") or ""))
        for row in prompt_rows
    }
    if actual_claims != expected_claims or len(prompt_rows) != len(expected_claims):
        report.error(
            "inventory.prompt_claim_audit",
            dataset_root / "validation/prompt_claim_audit.tsv",
            "prompt claim audit must contain every specification claim exactly once",
        )

    expected_groups = {
        str(bundle.construction_manifest.get("counterfactual_group_id"))
        for bundle in bundles.values()
    }
    actual_groups = {
        str(row.get("counterfactual_group_id") or row.get("group_id") or "")
        for row in counterfactual_rows
    }
    if actual_groups != expected_groups:
        report.error(
            "inventory.counterfactual_audit",
            dataset_root / "validation/counterfactual_audit.tsv",
            "counterfactual audit must cover every counterfactual group",
        )


def _validate_file_inventory(dataset_root: Path, report: Report) -> None:
    path = dataset_root / "validation/file_inventory.json"
    payload = _read_json_or_report(path, report)
    if payload is None:
        return
    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("files"), list):
        entries = [entry for entry in payload["files"] if isinstance(entry, dict)]
    elif isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif isinstance(payload, dict) and all(
        isinstance(value, str) for value in payload.values()
    ):
        entries = [
            {"path": relative, "sha256": digest} for relative, digest in payload.items()
        ]
    else:
        report.error("inventory.file_format", path, "unsupported file inventory format")
        return

    by_relative: dict[str, dict[str, Any]] = {}
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            report.error("inventory.file_path", path, "file inventory path must be non-empty")
            continue
        if relative in by_relative:
            report.error(
                "inventory.file_duplicate", path, f"duplicate inventory path {relative!r}"
            )
        by_relative[relative] = entry
        candidate = (dataset_root / relative).resolve()
        try:
            candidate.relative_to(dataset_root.resolve())
        except ValueError:
            report.error(
                "inventory.file_escape", path, f"inventory path escapes dataset: {relative!r}"
            )
            continue
        _validate_declared_file_hash(candidate, entry.get("sha256"), path, report)
        if "bytes" in entry and candidate.is_file() and entry["bytes"] != candidate.stat().st_size:
            report.error(
                "inventory.file_size",
                path,
                f"{relative!r} byte size mismatch: {entry['bytes']!r} != "
                f"{candidate.stat().st_size}",
            )

    required_inventory_paths = {
        *MANDATORY_DATASET_FILES[:3],
        "configs/evaluation_profile_claim_driven_v2.json",
        "configs/evidence_arms.json",
    }
    fixture_root = dataset_root / "fixtures"
    if fixture_root.is_dir():
        for case_root in fixture_root.iterdir():
            if case_root.is_dir():
                required_inventory_paths.update(
                    f"fixtures/{case_root.name}/{filename}"
                    for filename in MANDATORY_CASE_FILES
                )
    # An inventory cannot contain its own stable hash.  Validation reports may
    # also be regenerated, so neither is required in the inventory.
    required_inventory_paths.discard("validation/file_inventory.json")
    missing = required_inventory_paths - set(by_relative)
    if missing:
        report.error(
            "inventory.file_incomplete",
            path,
            f"file inventory is missing required artifacts: {sorted(missing)}",
        )


def _validate_global_hash_fields(dataset_root: Path, report: Report) -> None:
    for path in sorted(dataset_root.rglob("*.json")):
        if path.name == "file_inventory.json":
            continue
        value = _read_json_or_report(path, report)
        if value is None:
            continue
        for pointer, key, field_value in _walk_json(value):
            if key.endswith("sha256") and (
                not isinstance(field_value, str)
                or not HEX64_RE.fullmatch(field_value)
            ):
                report.error(
                    "hashes.format_global",
                    path,
                    f"{pointer}/{key} must be a lowercase 64-character SHA-256",
                )


def _validate_manifest_lifecycle(
    manifest: Mapping[str, Any], path: Path, report: Report
) -> None:
    dataset_id = manifest.get("dataset_id")
    if dataset_id != DATASET_ID:
        report.error(
            "manifest.dataset_id",
            path,
            f"dataset_id must be {DATASET_ID!r}, got {dataset_id!r}",
        )
    audit = manifest.get("human_audit")
    if not isinstance(audit, dict) or audit.get("status") not in PENDING_REVIEW_STATES:
        report.error(
            "manifest.human_audit",
            path,
            "human_audit.status must remain pending_human/pending",
        )
    if isinstance(audit, dict):
        for field in ("reviewer", "reviewed_at", "approved_by", "approved_at"):
            if audit.get(field) not in (None, ""):
                report.error(
                    "manifest.premature_approval",
                    path,
                    f"human_audit.{field} must be null before review",
                )


def _extract_deviations(value: Any, *, path: Path, report: Report) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(value.get("authorized_deviations"), list):
        records = value["authorized_deviations"]
    else:
        report.error(
            "authorized_deviations.format",
            path,
            "must be a list or an object containing authorized_deviations",
        )
        return []
    if not all(isinstance(record, dict) for record in records):
        report.error(
            "authorized_deviations.record_format",
            path,
            "every authorized deviation must be a JSON object",
        )
    return [record for record in records if isinstance(record, dict)]


def _extract_records(
    value: Any,
    keys: Sequence[str],
    *,
    where: str,
    report: Report,
) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), list):
                return value[key]
    report.error(
        "records.format",
        where,
        f"must be a JSON list or object containing one of {list(keys)}",
    )
    return []


def _single_metric(bundle: CaseBundle, report: Report) -> str | None:
    metrics = {
        str(event.get("metric"))
        for event in bundle.metric_events
        if isinstance(event, dict) and event.get("metric") is not None
    }
    if len(metrics) != 1:
        report.error(
            "events.single_metric",
            bundle.root / "metric_events.json",
            f"case must isolate one metric, got {sorted(metrics)}",
        )
        return None
    metric = next(iter(metrics))
    registry_metric = bundle.registry.get("metric")
    if registry_metric != metric:
        report.error(
            "events.registry_metric",
            "cases.json",
            f"{bundle.case_id}: registry metric {registry_metric!r} != event metric {metric!r}",
        )
    return metric


def _validate_exact_span(
    prompt: str,
    record: Mapping[str, Any],
    *,
    text_key: str,
    offset_prefix: str,
    path: str | Path,
    report: Report,
) -> None:
    span = record.get(text_key)
    if not isinstance(span, str) or not span:
        report.error("prompt_span.missing", path, f"{text_key} must be a non-empty string")
        return
    start, end = _extract_span_offsets(record, offset_prefix)
    if start is None or end is None:
        report.error(
            "prompt_span.offsets_missing",
            path,
            f"{text_key} requires explicit character start/end offsets",
        )
        return
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(
        start, int
    ) or not isinstance(end, int):
        report.error(
            "prompt_span.offsets_type",
            path,
            f"{text_key} character offsets must be integers",
        )
        return
    if start < 0 or end <= start or end > len(prompt):
        report.error(
            "prompt_span.offsets_range",
            path,
            f"{text_key} offsets [{start}, {end}) are outside prompt length {len(prompt)}",
        )
        return
    if prompt[start:end] != span:
        report.error(
            "prompt_span.offset_mismatch",
            path,
            f"prompt[{start}:{end}]={prompt[start:end]!r}, expected exact span {span!r}",
        )
    if span not in prompt:
        report.error(
            "prompt_span.not_substring",
            path,
            f"{text_key} is not an exact prompt substring",
        )


def _extract_span_offsets(
    record: Mapping[str, Any], prefix: str
) -> tuple[Any, Any]:
    candidates = (
        (f"{prefix}_char_start", f"{prefix}_char_end"),
        (f"{prefix}_span_start", f"{prefix}_span_end"),
        ("char_start", "char_end"),
        ("span_start", "span_end"),
    )
    for start_key, end_key in candidates:
        if start_key in record or end_key in record:
            return record.get(start_key), record.get(end_key)
    offsets = record.get(f"{prefix}_span_offsets") or record.get("span_offsets")
    if isinstance(offsets, dict):
        return offsets.get("start"), offsets.get("end")
    return None, None


def _validate_opaque_identifier(
    value: str,
    path: str | Path,
    report: Report,
    *,
    allow_event_prefix: bool = False,
    allow_group_prefix: bool = False,
) -> None:
    if LEAKAGE_TOKEN_RE.search(value):
        report.error(
            "leakage.nonopaque_identifier",
            path,
            f"identifier {value!r} contains a metric condition or outcome label",
        )
        return
    patterns = [OPAQUE_CASE_ID_RE]
    if allow_event_prefix:
        patterns.append(re.compile(r"^(?:event|evt|e)[_-]?\d{3,8}$", re.IGNORECASE))
    if allow_group_prefix:
        patterns.append(re.compile(r"^(?:group|grp|g)[_-]?\d{2,8}$", re.IGNORECASE))
    if not any(pattern.fullmatch(value) for pattern in patterns):
        report.error(
            "leakage.identifier_not_opaque",
            path,
            f"identifier {value!r} must be an opaque numeric case/event/group ID",
        )


def _validate_no_symlinks(dataset_root: Path, report: Report) -> None:
    for path in dataset_root.rglob("*"):
        if path.is_symlink():
            report.error(
                "dataset.symlink",
                path,
                "dataset must be self-contained; symlinks are not allowed",
            )


def _read_json_or_report(path: Path, report: Report) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error("json.parse", path, str(exc))
        return None


def _blind_review_sort_key(case_id: str) -> str:
    return hashlib.sha256(
        f"{DATASET_ID}:blind-semantic-pass-v1:{case_id}".encode("utf-8")
    ).hexdigest()


def _read_tsv(path: Path, report: Report) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except Exception as exc:
        report.error("tsv.parse", path, str(exc))
        return []


def _outbound_allowlists(value: Any, pointer: str = "") -> Iterator[tuple[str, list[str]]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = f"{pointer}/{key}"
            lowered = key.lower()
            if (
                isinstance(child, list)
                and "allowlist" in lowered
                and ("outbound" in lowered or "judge" in lowered or "context" in lowered)
                and all(isinstance(item, str) for item in child)
            ):
                yield child_pointer, child
            yield from _outbound_allowlists(child, child_pointer)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _outbound_allowlists(child, f"{pointer}/{index}")


def _walk_json(value: Any, pointer: str = "") -> Iterator[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield pointer, str(key), child
            yield from _walk_json(child, f"{pointer}/{_escape_pointer(str(key))}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{pointer}/{index}")


def _event_id(event: Mapping[str, Any]) -> str:
    value = event.get("event_id") or event.get("metric_event_id")
    return str(value) if value is not None else ""


def _scene_object_ids(scene: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("id"))
        for item in scene.get("objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    }


def scene_geometry_sha256(scene: Mapping[str, Any]) -> str:
    """Hash visual/geometry state while ignoring per-case/request metadata."""

    objects = []
    for item in scene.get("objects", []):
        if not isinstance(item, dict):
            continue
        objects.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "jid": item.get("jid"),
                "asset_ref": item.get("asset_ref"),
                "asset_proxy": item.get("asset_proxy"),
                "size": item.get("size"),
                "center": item.get("center"),
                "rotation": item.get("rotation"),
                "geometry_provenance": item.get("geometry_provenance"),
                "support_parent": item.get("support_parent"),
            }
        )
    payload = {
        "boundary": scene.get("boundary"),
        "scene_height": scene.get("scene_height"),
        "objects": objects,
        "relations": scene.get("relations", []),
        "oor_relations": scene.get("oor_relations", []),
        "oar_relations": scene.get("oar_relations", []),
    }
    return canonical_json_sha256(payload)


def grouping_partition_sha256(grouping_report: Mapping[str, Any]) -> str:
    groups = []
    for group in grouping_report.get("object_groups", []):
        if not isinstance(group, dict):
            continue
        groups.append(sorted(map(str, group.get("object_ids") or [])))
    groups.sort()
    return canonical_json_sha256(groups)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_diff_paths(left: Any, right: Any, pointer: str = "") -> set[str]:
    """Return leaf JSON pointers that differ between two JSON-compatible values."""

    if type(left) is not type(right):
        return {pointer or "/"}
    if isinstance(left, dict):
        output: set[str] = set()
        for key in sorted(set(left) | set(right)):
            child = f"{pointer}/{_escape_pointer(str(key))}"
            if key not in left or key not in right:
                output.add(child)
            else:
                output.update(json_diff_paths(left[key], right[key], child))
        return output
    if isinstance(left, list):
        output = set()
        for index in range(max(len(left), len(right))):
            child = f"{pointer}/{index}"
            if index >= len(left) or index >= len(right):
                output.add(child)
            else:
                output.update(json_diff_paths(left[index], right[index], child))
        return output
    return set() if left == right else {pointer or "/"}


def _pointer_is_allowed(pointer: str, allowed: str) -> bool:
    normalized = allowed.rstrip("/") or "/"
    return pointer == normalized or pointer.startswith(normalized + "/")


def _normalized_specification_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "claims": contract.get("claims", {}),
        "authorized_deviations": contract.get("authorized_deviations", []),
        "asset_policy": contract.get("asset_policy"),
    }


def _normalized_asset_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    normalized.pop("request_id", None)
    return normalized


def _validate_declared_file_hash(
    path: Path,
    declared: Any,
    source_path: Path,
    report: Report,
) -> None:
    if not isinstance(declared, str) or not HEX64_RE.fullmatch(declared):
        report.error(
            "hashes.declared_format",
            source_path,
            f"declared SHA-256 for {path.name!r} must be lowercase 64-hex",
        )
        return
    if not path.is_file():
        report.error(
            "hashes.file_missing",
            source_path,
            f"hashed file does not exist: {path}",
        )
        return
    actual = _sha256_file(path)
    if actual != declared:
        report.error(
            "hashes.file_mismatch",
            source_path,
            f"{path.name}: declared {declared}, actual {actual}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_pending_review(value: Mapping[str, Any]) -> bool:
    state = value.get("status") or value.get("review_status")
    return state in PENDING_REVIEW_STATES


def _display_path(path: str | Path, dataset_root: Path) -> str:
    if isinstance(path, str):
        return path
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
