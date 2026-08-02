#!/usr/bin/env python3
"""Judge frozen cal_dataset2 non-L1 evidence against human GT.

This experiment evaluates the combined observation-and-judge path, not scene
generation.  Every model request receives only the original prompt, the active
metric claim, target identities, exact prompt-authorized deviations, an
arm-specific frozen image packet, and a metric rubric.  Human GT, construction
proposals, controlled deltas, and private fixture labels are applied only after
the call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.models import parse_json_object  # noqa: E402
from benchmark.visual_judge.openai_compatible import (  # noqa: E402
    _image_data_url,
    build_openai_compatible_vlm_judge,
)


DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
RENDER_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_review_renders"
)
CONTOUR_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_contours"
)
GT_PATH = DATASET_ROOT / "human_review" / "human_gt_20260725.tsv"
JUDGE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "gpt5_6_sol_litellm_local_non_l1_judge.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "exp2_non_l1_visual_evidence_gpt56"
)

METRICS = (
    "oor",
    "oar",
    "room_scene_type",
    "broad_semantic_intent",
    "required_functional_areas",
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
BASE_ARMS = (
    "production_default",
    "global_only",
    "local_raw_only",
    "full_raw",
)
CONTOUR_ARMS = (
    "production_raw_swap",
    "local_contour_only",
)
ARMS = BASE_ARMS + CONTOUR_ARMS
THREE_WAY_LABELS = ("valid", "invalid", "ambiguous")
EVIDENCE_STATUSES = ("sufficient", "insufficient")
SCHEMA_VERSION = "cal_dataset2_non_l1_judgement_v1"
CONTRACT_VERSION = "cal_dataset2_non_l1_judgement_contract_v1"
CANONICAL_METRIC_BY_EXPERIMENT_METRIC = {
    "room_scene_type": "functional_semantic_fidelity",
    "broad_semantic_intent": "functional_semantic_fidelity",
    "required_functional_areas": "functional_semantic_fidelity",
}

SYSTEM_PROMPT = """You adjudicate exactly one non-L1 metric event in a 3D
scene-generation benchmark. Inspect the supplied rendered views as the primary
evidence. The prompt, active claim, target identities, and exact authorized
deviations are semantic context; they are not permission to invent hidden
visual facts.

Judge only the named metric. Ignore unrelated layout oddities, physical issues,
or other metric failures. For L3, apply only an explicit authorization linked
to this metric, relation, and target set. A prompt-authorized surreal scale,
pairing, or style contrast must not be penalized by the corresponding L3
metric. A missing requested deviation can still be a separate L2 prompt
fidelity issue, but do not convert it into an unrelated L3 penalty.

For functional-semantic components and L3 Scene Quality, invalid requires one
or more significant, clearly visible, metric-scoped defects that you explicitly
identify. Minor variation, subjective preference, ordinary diversity, or a
merely non-ideal arrangement is valid. When the packet is sufficient and no
significant in-scope defect can be identified, return valid. This default-valid
rule never overrides evidence sufficiency and does not apply to OOR/OAR:
explicit relations require positive verification.

Object pairing is category/role compatibility over the supplied object group.
Do not use position, distance, angle, orientation, access, or functional
arrangement as an object-pairing defect. Prompt-specified local function belongs
to functional semantic fidelity; an explicit spatial relation belongs to
OOR/OAR.

First decide whether the supplied image packet is sufficient to adjudicate the
metric without hidden geometry. Then return exactly one JSON object:
{"evidence_status":"sufficient","verdict":"valid","confidence":0.0,
"reason":"...","missing_evidence":[]}.

evidence_status must be sufficient or insufficient. verdict must be valid,
invalid, or ambiguous. If evidence_status is insufficient, verdict must be
ambiguous and missing_evidence must state what additional visual fact or view
is required. Use ambiguous with sufficient evidence only for genuine semantic
borderline cases where reasonable reviewers can disagree. confidence must be
between 0 and 1. Do not mention dataset IDs, ground truth, construction
variants, or hidden labels."""

METRIC_RUBRICS = {
    "oor": (
        "Judge only whether the designated object-to-object relation stated by "
        "the prompt is visibly satisfied."
    ),
    "oar": (
        "Judge only whether the designated object-to-architecture attachment "
        "or relation stated by the prompt is visibly satisfied."
    ),
    "room_scene_type": (
        "Legacy dataset component of functional_semantic_fidelity. Judge whether "
        "the dominant visible room or scene type has a significant mismatch with "
        "the requested type. Ignore unrelated layout awkwardness."
    ),
    "broad_semantic_intent": (
        "Legacy dataset component of functional_semantic_fidelity. Judge whether "
        "the global visible organization has a significant contradiction with "
        "the requested visual-functional intent."
    ),
    "required_functional_areas": (
        "Legacy dataset component of functional_semantic_fidelity. Judge whether "
        "the prompt-required areas are visibly present and distinct; do not judge "
        "downstream ergonomic or gameplay success."
    ),
    "scale_consistency": (
        "Judge perceptual scale coherence of the designated targets relative to "
        "their relevant context after exact prompt-authorized exceptions."
    ),
    "object_pairing_consistency": (
        "After grouping, judge only whether the designated members have "
        "significantly incompatible object categories or semantic roles. Ignore "
        "their position, distance, angle, orientation, access, and functional "
        "arrangement. Apply exact prompt-authorized exceptions."
    ),
    "style_consistency": (
        "Judge perceptual style coherence of the designated object and visible "
        "comparators after exact prompt-authorized contrast."
    ),
    "functional_semantic_fidelity": (
        "Judge the active prompt-owned functional-semantic component. Global "
        "evidence establishes room type, intent, and required areas. Judge local "
        "group functionality only when the prompt explicitly specifies it."
    ),
}


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    render_root = args.render_root.expanduser().resolve()
    contour_root = args.contour_root.expanduser().resolve()
    gt_path = args.ground_truth.expanduser().resolve()
    judge_config_path = args.judge_config.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    judge_config = _read_json(judge_config_path)
    if int(judge_config.get("max_images") or 0) < 5:
        raise ValueError("cal_dataset2 full_raw requires judge max_images >= 5")

    cards = _review_cards(render_root)
    gt_by_case = _ground_truth(gt_path)
    contour_by_observation = _contour_index(contour_root)
    jobs = _build_jobs(
        args=args,
        cards=cards,
        gt_by_case=gt_by_case,
        dataset_root=dataset_root,
        render_root=render_root,
        contour_by_observation=contour_by_observation,
        judge_config=judge_config,
        judge_config_path=judge_config_path,
        out_dir=out_dir,
    )
    plan = _plan(
        args=args,
        dataset_root=dataset_root,
        render_root=render_root,
        contour_root=contour_root,
        gt_path=gt_path,
        judge_config_path=judge_config_path,
        jobs=jobs,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan["counts"], indent=2), flush=True)
        return

    api_key_env = str(judge_config.get("api_key_env") or "")
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"judge credential environment variable is not set: {api_key_env}"
        )

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if args.resume and _result_ready(job["result_path"], job["contract"]):
            result = _read_json(job["result_path"])
            results.append(result)
            print(
                f"[cached] {job['case_id']} {job['metric']} {job['arm']}",
                flush=True,
            )
        else:
            pending.append(job)
    started = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_judge_one, job, judge_config): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    result = _failed_result(job, exc)
                    _write_json(job["result_path"], result)
                results.append(result)
                status = "ok" if not result.get("error") else "failed"
                print(
                    f"[{status}] {job['case_id']} {job['metric']} {job['arm']}",
                    flush=True,
                )

    results.sort(key=lambda item: (item["case_id"], item["metric"], item["arm"]))
    summary = _summary_rows(results)
    paired = _paired_rows(results)
    _write_tsv(out_dir / "per_event.tsv", [_flat_result(item) for item in results])
    _write_tsv(out_dir / "summary.tsv", summary)
    _write_tsv(out_dir / "paired_arm_comparisons.tsv", paired)
    _write_json(
        out_dir / "summary.json",
        {"summary": summary, "paired_arm_comparisons": paired},
    )
    run_manifest = {
        **plan,
        "schema_version": "cal_dataset2_non_l1_judgement_run_v1",
        "elapsed_seconds": time.time() - started,
        "result_count": len(results),
        "resolved_call_count": sum(not item.get("error") for item in results),
        "failure_count": sum(bool(item.get("error")) for item in results),
        "outputs": {
            "per_event": str((out_dir / "per_event.tsv").resolve()),
            "summary": str((out_dir / "summary.tsv").resolve()),
            "paired_arm_comparisons": str(
                (out_dir / "paired_arm_comparisons.tsv").resolve()
            ),
        },
    }
    _write_json(out_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "result_count": run_manifest["result_count"],
                "resolved_call_count": run_manifest["resolved_call_count"],
                "failure_count": run_manifest["failure_count"],
                "summary": run_manifest["outputs"]["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


def _build_jobs(
    *,
    args: argparse.Namespace,
    cards: list[dict[str, Any]],
    gt_by_case: dict[str, dict[str, str]],
    dataset_root: Path,
    render_root: Path,
    contour_by_observation: dict[str, dict[str, Any]],
    judge_config: dict[str, Any],
    judge_config_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    selected_metrics = set(args.metric or METRICS)
    selected_arms = set(args.arm or ARMS)
    selected_cases = set(args.case_id)
    implementation_sha256 = {
        "runner": _file_sha256(Path(__file__).resolve()),
        "openai_compatible": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "benchmark"
            / "visual_judge"
            / "openai_compatible.py"
        ),
    }
    jobs: list[dict[str, Any]] = []
    for card in cards:
        case_id = str(card["case_id"])
        metric = str(card["metric"])
        if metric not in selected_metrics:
            continue
        if selected_cases and case_id not in selected_cases:
            continue
        gt = gt_by_case.get(case_id)
        if gt is None:
            raise ValueError(f"human GT is missing case {case_id}")
        if gt["event_id"] != str(card["event_id"]) or gt["metric"] != metric:
            raise ValueError(f"human GT identity mismatch for {case_id}")
        fixture = dataset_root / "fixtures" / case_id
        context = _judge_context(fixture, card)
        paths = _render_paths(render_root, card)
        contour = contour_by_observation.get(str(card["observation_id"]))
        selections = _arm_selections(
            metric=metric,
            fixture=fixture,
            paths=paths,
            contour_manifest=contour,
        )
        for arm, evidence in selections.items():
            if arm not in selected_arms:
                continue
            if len(evidence) > int(judge_config["max_images"]):
                raise ValueError(
                    f"{case_id} {arm} exceeds max_images: {len(evidence)}"
                )
            evidence_records = [
                {
                    **item,
                    "path": str(Path(str(item["path"])).resolve()),
                    "sha256": _file_sha256(Path(str(item["path"]))),
                }
                for item in evidence
            ]
            request_context = deepcopy(context)
            request_context["image_packet"] = {
                **request_context["image_packet"],
                "arm": arm,
                "ordered_views": [
                    {
                        "image_index": index + 1,
                        "role": item["role"],
                        "view_name": item["view_name"],
                    }
                    for index, item in enumerate(evidence_records)
                ],
            }
            contract = {
                "schema_version": CONTRACT_VERSION,
                "repeat_id": args.repeat_id,
                "case_id": case_id,
                "event_id": str(card["event_id"]),
                "metric": metric,
                "canonical_metric": request_context["metric"],
                "arm": arm,
                "observation_id": str(card["observation_id"]),
                "prompt_granularity": str(card["prompt_granularity"]),
                "evidence": evidence_records,
                "evidence_packet_sha256": _json_sha256(evidence_records),
                "judge_context_sha256": _json_sha256(request_context),
                "judge_config_sha256": _file_sha256(judge_config_path),
                "judge_identity": {
                    "name": judge_config.get("name"),
                    "model": judge_config.get("model"),
                    "endpoint": judge_config.get("endpoint"),
                    "temperature": judge_config.get("temperature"),
                    "max_tokens": judge_config.get("max_tokens"),
                },
                "implementation_sha256": implementation_sha256,
                "ground_truth_visibility": "scoring_only_after_model_call",
            }
            result_path = (
                out_dir / "events" / case_id / f"{arm}.json"
            )
            jobs.append(
                {
                    "case_id": case_id,
                    "event_id": str(card["event_id"]),
                    "metric": metric,
                    "arm": arm,
                    "gt_label": gt["human_semantic_label"],
                    "gt_notes": gt.get("notes") or "",
                    "context": request_context,
                    "evidence": evidence_records,
                    "contract": contract,
                    "result_path": result_path,
                }
            )
    if selected_cases:
        found = {job["case_id"] for job in jobs}
        missing = selected_cases - found
        if missing:
            raise ValueError(f"selected cases produced no jobs: {sorted(missing)}")
    if not jobs:
        raise ValueError("no cal_dataset2 judgement jobs matched the filters")
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    return jobs


def _judge_one(
    job: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    judge = build_openai_compatible_vlm_judge(deepcopy(judge_config))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Adjudicate this metric event.\n"
                + json.dumps(
                    job["context"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(Path(str(item["path"])))},
        }
        for item in job["evidence"]
    )
    raw = judge.model.chat_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format_json=bool(judge.response_format_json),
        call_type=f"vlm_judge.non_l1.{job['metric']}.{job['arm']}",
    )
    response = parse_json_object(raw)
    _validate_response(response)
    predicted = str(response["verdict"])
    gt_label = str(job["gt_label"])
    binary_gt = gt_label in {"valid", "invalid"}
    result = {
        **deepcopy(job["contract"]),
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "gt_label": gt_label,
        "gt_notes": job["gt_notes"],
        "predicted_label": predicted,
        "evidence_status": str(response["evidence_status"]),
        "confidence": float(response["confidence"]),
        "reason": str(response["reason"]),
        "missing_evidence": list(response.get("missing_evidence") or []),
        "three_way_match": predicted == gt_label,
        "binary_scoreable": binary_gt,
        "binary_match": predicted == gt_label if binary_gt else None,
        "elapsed_seconds": time.time() - started,
        "request_metadata": deepcopy(judge.model.last_request_metadata),
        "error": None,
    }
    _write_json(job["result_path"], result)
    return result


def _failed_result(job: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **deepcopy(job["contract"]),
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "gt_label": job["gt_label"],
        "gt_notes": job["gt_notes"],
        "predicted_label": None,
        "evidence_status": None,
        "confidence": None,
        "reason": None,
        "missing_evidence": [],
        "three_way_match": None,
        "binary_scoreable": job["gt_label"] in {"valid", "invalid"},
        "binary_match": None,
        "elapsed_seconds": None,
        "request_metadata": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _judge_context(fixture: Path, card: dict[str, Any]) -> dict[str, Any]:
    scene_request = _read_json(fixture / "scene_request.json")
    contract = _read_json(fixture / "specification_contract.json")
    event_bundle = _read_json(fixture / "metric_events.json")
    deviations = _read_json(fixture / "authorized_deviations.json")
    expectations = _read_json(fixture / "evidence_expectations.json")
    legacy_metric = str(card["metric"])
    claims = contract.get("claims") or {}
    active_claims = (
        claims.get(legacy_metric)
        if isinstance(claims, dict) and isinstance(claims.get(legacy_metric), list)
        else []
    )
    events = [
        {
            key: event.get(key)
            for key in (
                "metric",
                "relation",
                "target_ids",
                "source_claim_ids",
                "group_ids",
            )
        }
        for event in event_bundle.get("events") or []
        if isinstance(event, dict)
    ]
    metric = _canonical_experiment_metric(legacy_metric, events)
    active_deviations = [
        {
            key: deviation.get(key)
            for key in (
                "metric",
                "target_ids",
                "relation",
                "source",
                "prompt_span",
                "source_claim_id",
            )
            if key in deviation
        }
        for deviation in deviations.get("authorized_deviations") or []
        if isinstance(deviation, dict)
    ]
    return {
        "metric": metric,
        "legacy_dataset_metric": legacy_metric,
        "metric_rubric": METRIC_RUBRICS[metric],
        "evaluation_question": str(card["review_question"]),
        "original_prompt": scene_request.get("instruction"),
        "prompt_granularity": card.get("prompt_granularity"),
        "active_prompt_claims": active_claims,
        "metric_event": events,
        "target_objects": card.get("target_objects") or [],
        "prompt_authorized_deviations": active_deviations,
        "required_visible_facts": expectations.get("required_visible_facts") or [],
        "image_packet": {
            "arm": None,
            "legend": (
                "Images are ordered as listed in the request contract. Contour "
                "images preserve raw target interiors and add only an exterior "
                "colored band and outline from visible 2D segmentation."
            ),
        },
        "metric_boundary": (
            "Judge only this metric; unrelated scene defects are out of scope."
        ),
    }


def _canonical_experiment_metric(
    legacy_metric: str,
    events: list[dict[str, Any]],
) -> str:
    """Route historical cal_dataset2 labels through the canonical ownership.

    The frozen dataset predates the simplified boundary. Category-coexistence
    cases remain L3 pairing; prompt-owned function/group cases move to L2
    functional semantics; explicit relation cases move to OOR. Artifact metric
    names remain unchanged outside the outbound judge context for reproducible
    historical analysis.
    """

    if legacy_metric != "object_pairing_consistency":
        return CANONICAL_METRIC_BY_EXPERIMENT_METRIC.get(
            legacy_metric, legacy_metric
        )
    relations = {
        str(event.get("relation") or "")
        for event in events
        if event.get("metric") == legacy_metric
    }
    source_claim_ids = {
        str(claim_id)
        for event in events
        if event.get("metric") == legacy_metric
        for claim_id in (event.get("source_claim_ids") or [])
    }
    if any(claim_id.startswith("oor::") for claim_id in source_claim_ids):
        return "oor"
    if relations and relations <= {"category_coexistence"}:
        return "object_pairing_consistency"
    return "functional_semantic_fidelity"


def _arm_selections(
    *,
    metric: str,
    fixture: Path,
    paths: dict[str, list[dict[str, Any]]],
    contour_manifest: dict[str, Any] | None,
) -> OrderedDict[str, list[dict[str, Any]]]:
    global_views = paths["global"]
    local_raw = paths["local_raw"]
    if len(global_views) != 2 or len(local_raw) != 3:
        raise ValueError(
            f"frozen evidence shape must be global=2/local=3, got "
            f"{len(global_views)}/{len(local_raw)}"
        )
    local_contour = _contour_paths(contour_manifest) if metric in {"oor", "oar"} else []
    expectations = _read_json(fixture / "evidence_expectations.json")
    production = expectations.get("production_default_under_test") or {}
    global_policy = production.get("global_policy") or {}
    local_policy = production.get("local_policy") or {}
    global_budget = int(global_policy.get("image_budget") or 0)
    local_budget = int(local_policy.get("image_budget") or 0)
    presentation = str(local_policy.get("presentation") or "raw")
    production_local = (
        local_contour if presentation == "contour" else local_raw
    )
    if local_budget and len(production_local) < local_budget:
        raise ValueError(
            f"{metric} production evidence requires {local_budget} "
            f"{presentation} local views"
        )
    result: OrderedDict[str, list[dict[str, Any]]] = OrderedDict(
        [
            (
                "production_default",
                global_views[:global_budget] + production_local[:local_budget],
            ),
            ("global_only", global_views),
            ("local_raw_only", local_raw),
            ("full_raw", global_views + local_raw),
        ]
    )
    if metric in {"oor", "oar"}:
        result["production_raw_swap"] = (
            global_views[:global_budget] + local_raw[:local_budget]
        )
        result["local_contour_only"] = local_contour
    if not all(result.values()):
        raise ValueError(f"{metric} produced an empty evidence arm")
    return result


def _render_paths(
    render_root: Path,
    card: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output = {"global": [], "local_raw": []}
    for view in card.get("rendered_views") or []:
        family = str(view.get("family") or "")
        source = (
            render_root
            / "review"
            / str(view.get("src") or "")
        ).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        record = {
            "role": "global_raw" if family == "global" else "local_raw",
            "view_name": str(view.get("name") or source.stem),
            "path": str(source),
        }
        output["global" if family == "global" else "local_raw"].append(record)
    return output


def _contour_paths(
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("OOR/OAR contour evidence is not prepared")
    output = []
    for view in manifest.get("contour_views") or []:
        path = Path(str(view.get("path") or "")).expanduser().resolve()
        if not path.is_file() or _file_sha256(path) != view.get("sha256"):
            raise ValueError(f"contour evidence hash mismatch: {path}")
        output.append(
            {
                "role": "local_contour",
                "view_name": str(view.get("view_id") or path.stem),
                "path": str(path),
            }
        )
    if len(output) != 3:
        raise ValueError(f"expected three contour views, got {len(output)}")
    return output


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[(str(item["metric"]), str(item["arm"]))].append(item)
        groups[("ALL", str(item["arm"]))].append(item)
    rows = []
    for (metric, arm), items in sorted(groups.items()):
        completed = [item for item in items if not item.get("error")]
        binary = [item for item in completed if item.get("binary_scoreable")]
        human_ambiguous = [
            item for item in completed if item.get("gt_label") == "ambiguous"
        ]
        confidence = [
            float(item["confidence"])
            for item in completed
            if isinstance(item.get("confidence"), (int, float))
        ]
        rows.append(
            {
                "metric": metric,
                "arm": arm,
                "total": len(items),
                "completed": len(completed),
                "failures": len(items) - len(completed),
                "human_binary_n": len(binary),
                "human_ambiguous_n": len(human_ambiguous),
                "binary_accuracy": _mean_bool(
                    [item.get("binary_match") for item in binary]
                ),
                "three_way_agreement": _mean_bool(
                    [item.get("three_way_match") for item in completed]
                ),
                "valid_recall": _label_recall(completed, "valid"),
                "invalid_recall": _label_recall(completed, "invalid"),
                "ambiguous_recall": _label_recall(completed, "ambiguous"),
                "evidence_sufficient_rate": _mean_bool(
                    [
                        item.get("evidence_status") == "sufficient"
                        for item in completed
                    ]
                ),
                "predicted_ambiguous_rate": _mean_bool(
                    [
                        item.get("predicted_label") == "ambiguous"
                        for item in completed
                    ]
                ),
                "mean_confidence": (
                    statistics.fmean(confidence) if confidence else None
                ),
                "mean_latency_seconds": (
                    statistics.fmean(
                        float(item["elapsed_seconds"])
                        for item in completed
                        if isinstance(item.get("elapsed_seconds"), (int, float))
                    )
                    if completed
                    else None
                ),
            }
        )
    return rows


def _paired_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(item["case_id"]), str(item["metric"]), str(item["arm"])): item
        for item in results
        if not item.get("error")
    }
    pairs = (
        ("production_default", "global_only"),
        ("production_default", "local_raw_only"),
        ("production_default", "full_raw"),
        ("production_default", "production_raw_swap"),
        ("production_raw_swap", "local_contour_only"),
    )
    rows = []
    for metric in ("ALL",) + METRICS:
        case_metrics = {
            (case_id, item_metric)
            for case_id, item_metric, _arm in by_key
            if metric == "ALL" or item_metric == metric
        }
        for left_arm, right_arm in pairs:
            matched = []
            for case_id, item_metric in case_metrics:
                left = by_key.get((case_id, item_metric, left_arm))
                right = by_key.get((case_id, item_metric, right_arm))
                if left is not None and right is not None:
                    matched.append((left, right))
            if not matched:
                continue
            binary = [
                pair
                for pair in matched
                if pair[0].get("binary_scoreable")
                and pair[1].get("binary_scoreable")
            ]
            rows.append(
                {
                    "metric": metric,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "paired_n": len(matched),
                    "binary_paired_n": len(binary),
                    "left_binary_accuracy": _mean_bool(
                        [left.get("binary_match") for left, _right in binary]
                    ),
                    "right_binary_accuracy": _mean_bool(
                        [right.get("binary_match") for _left, right in binary]
                    ),
                    "right_improves": sum(
                        left.get("binary_match") is False
                        and right.get("binary_match") is True
                        for left, right in binary
                    ),
                    "right_worsens": sum(
                        left.get("binary_match") is True
                        and right.get("binary_match") is False
                        for left, right in binary
                    ),
                    "same_prediction": sum(
                        left.get("predicted_label") == right.get("predicted_label")
                        for left, right in matched
                    ),
                    "right_sufficiency_gain": sum(
                        left.get("evidence_status") == "insufficient"
                        and right.get("evidence_status") == "sufficient"
                        for left, right in matched
                    ),
                    "right_sufficiency_loss": sum(
                        left.get("evidence_status") == "sufficient"
                        and right.get("evidence_status") == "insufficient"
                        for left, right in matched
                    ),
                }
            )
    return rows


def _flat_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": item.get("case_id"),
        "event_id": item.get("event_id"),
        "metric": item.get("metric"),
        "arm": item.get("arm"),
        "gt_label": item.get("gt_label"),
        "predicted_label": item.get("predicted_label"),
        "evidence_status": item.get("evidence_status"),
        "binary_match": item.get("binary_match"),
        "three_way_match": item.get("three_way_match"),
        "confidence": item.get("confidence"),
        "elapsed_seconds": item.get("elapsed_seconds"),
        "reason": item.get("reason"),
        "missing_evidence": " | ".join(item.get("missing_evidence") or []),
        "error": item.get("error"),
        "evidence_packet_sha256": item.get("evidence_packet_sha256"),
    }


def _plan(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    render_root: Path,
    contour_root: Path,
    gt_path: Path,
    judge_config_path: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_cases = {
        job["case_id"]: job
        for job in jobs
    }
    return {
        "schema_version": "cal_dataset2_non_l1_judgement_plan_v1",
        "experiment": "all non-L1 VLM visual-evidence sufficiency and accuracy",
        "repeat_id": args.repeat_id,
        "inputs": {
            "dataset_root": str(dataset_root),
            "render_root": str(render_root),
            "contour_root": str(contour_root),
            "ground_truth": str(gt_path),
            "judge_config": str(judge_config_path),
        },
        "frozen": [
            "108 human-reviewed cases",
            "original prompt and active metric claim",
            "real-mesh Blender Workbench renders",
            "camera poses and image bytes",
            "metric-specific prompt",
            "model parameters",
        ],
        "changed_variable": "visual evidence arm",
        "gt_visibility": "never included in model request",
        "counts": {
            "jobs": len(jobs),
            "cases": len({job["case_id"] for job in jobs}),
            "metrics": dict(Counter(job["metric"] for job in jobs)),
            "arms": dict(Counter(job["arm"] for job in jobs)),
            "gt_labels": dict(
                Counter(job["gt_label"] for job in unique_cases.values())
            ),
        },
    }


def _review_cards(render_root: Path) -> list[dict[str, Any]]:
    payload = _read_json(render_root / "review" / "review_cases.json")
    cards = [
        card
        for card in payload.get("cases") or []
        if isinstance(card, dict)
    ]
    if len(cards) != 108 or any(card.get("render_status") != "ready" for card in cards):
        raise ValueError("review render bank must contain 108 ready cases")
    return cards


def _ground_truth(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_tsv(path)
    output = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        label = str(row.get("human_semantic_label") or "")
        if not case_id or case_id in output:
            raise ValueError(f"duplicate or missing human GT case ID {case_id!r}")
        if label not in THREE_WAY_LABELS:
            raise ValueError(f"invalid human GT label for {case_id}: {label!r}")
        output[case_id] = row
    if len(output) != 108:
        raise ValueError(f"human GT must contain 108 cases, got {len(output)}")
    return output


def _contour_index(root: Path) -> dict[str, dict[str, Any]]:
    run_path = root / "run_manifest.json"
    if not run_path.is_file():
        return {}
    run = _read_json(run_path)
    output = {}
    for record in run.get("records") or []:
        if not isinstance(record, dict) or record.get("status") != "ready":
            continue
        manifest = _read_json(Path(str(record["manifest_path"])))
        output[str(record["observation_id"])] = manifest
    return output


def _validate_response(value: dict[str, Any]) -> None:
    if value.get("evidence_status") not in EVIDENCE_STATUSES:
        raise ValueError("evidence_status must be sufficient or insufficient")
    if value.get("verdict") not in THREE_WAY_LABELS:
        raise ValueError("verdict must be valid, invalid, or ambiguous")
    if (
        value["evidence_status"] == "insufficient"
        and value["verdict"] != "ambiguous"
    ):
        raise ValueError("insufficient evidence requires ambiguous verdict")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    if not isinstance(value.get("missing_evidence", []), list):
        raise ValueError("missing_evidence must be a list")


def _result_ready(path: Path, contract: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        return bool(
            result.get("schema_version") == SCHEMA_VERSION
            and result.get("contract_sha256") == _json_sha256(contract)
            and result.get("predicted_label") in THREE_WAY_LABELS
            and result.get("evidence_status") in EVIDENCE_STATUSES
            and not result.get("error")
        )
    except Exception:
        return False


def _label_recall(items: list[dict[str, Any]], label: str) -> float | None:
    selected = [item for item in items if item.get("gt_label") == label]
    if not selected:
        return None
    return sum(item.get("predicted_label") == label for item in selected) / len(selected)


def _mean_bool(values: list[Any]) -> float | None:
    selected = [value for value in values if isinstance(value, bool)]
    return sum(selected) / len(selected) if selected else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    parser.add_argument("--contour-root", type=Path, default=CONTOUR_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=GT_PATH)
    parser.add_argument("--judge-config", type=Path, default=JUDGE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "repeat_1")
    parser.add_argument("--repeat-id", default="repeat_1")
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--arm", action="append", choices=ARMS, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    return args


if __name__ == "__main__":
    main()
