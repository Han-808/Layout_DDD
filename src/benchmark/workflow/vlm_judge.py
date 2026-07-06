from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.models.base_model import ModelResponseError, parse_json_object
from benchmark.workflow.judge_evidence_selector import evidence_budgeting_config, select_judge_evidence
from benchmark.workflow.judge_summaries import build_judge_prompt_payload


DEFAULT_VLM_JUDGE = "same_model"
MOCK_JUDGE_NAME = "mock"
DEFAULT_MOCK_SCORE = 3
SCORE_MIN = 0
SCORE_MAX = 4
SCORE_DENOMINATOR = 4.0


class RoomConsistencyJudge(Protocol):
    def judge(
        self,
        *,
        description: str,
        input_level: str,
        view_artifacts: list[str],
        object_summary: list[dict] | None = None,
        validity_gate_passed: bool = True,
    ) -> dict:
        ...


class PairRelationJudge(Protocol):
    def judge(
        self,
        *,
        spec: dict,
        pair_view_artifacts: list[str],
        symbolic_evidence: dict | None = None,
    ) -> dict:
        ...


class VLMSceneJudge(Protocol):
    def judge(
        self,
        *,
        case: dict,
        layout: dict,
        input_level: str,
        sanity_flags: list[dict],
        physical_flags: list[dict],
        view_flags: list[dict],
        render_skipped_objects: list[dict],
        object_groups: list[dict],
        global_view_artifacts: list[dict],
        group_view_artifacts: list[dict],
        relation_specs: list[dict],
        attachment_specs: list[dict],
        renderable_layout: dict | None = None,
        layout_normalization_summary: dict | None = None,
        validity_gate_passed: bool = True,
        artifact_dir: str | Path | None = None,
    ) -> dict:
        ...


@dataclass
class MockRoomConsistencyJudge:
    default_score: int = 3

    def judge(
        self,
        *,
        description: str,
        input_level: str,
        view_artifacts: list[str],
        object_summary: list[dict] | None = None,
        validity_gate_passed: bool = True,
    ) -> dict:
        if not validity_gate_passed:
            return {"score": 0, "short_reason": "Validity gate failed before room-level judging."}
        if not view_artifacts:
            return {"score": 0, "short_reason": "No room view artifacts were available."}
        score = max(0, min(4, int(self.default_score)))
        return {"score": score, "short_reason": "Mock room judge returned deterministic v0 score."}


@dataclass
class MockPairRelationJudge:
    default_pass: bool = True

    def judge(
        self,
        *,
        spec: dict,
        pair_view_artifacts: list[str],
        symbolic_evidence: dict | None = None,
    ) -> dict:
        if not pair_view_artifacts or any(not Path(path).exists() for path in pair_view_artifacts):
            return {"pass": False, "short_reason": "One or more pair view artifacts are missing."}
        return {"pass": bool(self.default_pass), "short_reason": "Mock pair judge returned deterministic v0 result."}


@dataclass
class MockVLMSceneJudge:
    default_score: int = DEFAULT_MOCK_SCORE
    benchmark_config: dict | None = None

    def judge(
        self,
        *,
        case: dict,
        layout: dict,
        input_level: str,
        sanity_flags: list[dict],
        physical_flags: list[dict],
        view_flags: list[dict],
        render_skipped_objects: list[dict],
        object_groups: list[dict],
        global_view_artifacts: list[dict],
        group_view_artifacts: list[dict],
        relation_specs: list[dict],
        attachment_specs: list[dict],
        renderable_layout: dict | None = None,
        layout_normalization_summary: dict | None = None,
        validity_gate_passed: bool = True,
        artifact_dir: str | Path | None = None,
    ) -> dict:
        if not validity_gate_passed:
            return {
                "valid": False,
                "score": 0,
                "score_norm": 0.0,
                "short_reason": "Validity gate failed before VLM-as-judge.",
                "global_assessment": "",
                "group_results": [],
                "relation_results": [],
                "attachment_results": [],
            }
        score = _clamp_score(self.default_score)
        image_manifest = _image_manifest(global_view_artifacts, group_view_artifacts)
        evidence_selection = select_judge_evidence(
            global_view_artifacts=global_view_artifacts,
            group_view_artifacts=group_view_artifacts,
            object_groups=object_groups,
            physical_flags=physical_flags,
            view_flags=view_flags,
            render_skipped_objects=render_skipped_objects,
            config={"enabled": False},
        )
        prompt_bundle = build_judge_prompt_payload(
            case=case,
            layout=layout,
            renderable_layout=renderable_layout or layout,
            input_level=input_level,
            layout_normalization_summary=layout_normalization_summary,
            object_groups=object_groups,
            sanity_flags=sanity_flags,
            physical_flags=physical_flags,
            view_flags=view_flags,
            render_skipped_objects=render_skipped_objects,
            relation_specs=relation_specs,
            attachment_specs=attachment_specs,
            evidence_selection=evidence_selection,
            image_manifest=image_manifest,
            benchmark_config=self.benchmark_config,
        )
        prompt = json.dumps(prompt_bundle["prompt_payload"], indent=2)
        judgement = {
            "valid": True,
            "score": score,
            "score_norm": _score_norm(score),
            "confidence": "medium",
            "judgement_status": "valid_judgement",
            "brief_reasoning": "Mock VLM judge returned deterministic v1 score.",
            "issues": [],
            "insufficient_evidence": False,
            "short_reason": "Mock VLM judge returned deterministic v1 score.",
            "global_assessment": "Mock assessment.",
            "group_results": [
                {
                    "group_id": group.get("group_id"),
                    "object_ids": group.get("object_ids", []),
                    "valid": True,
                    "score": score,
                    "issues": [],
                }
                for group in object_groups
            ],
            "relation_results": [
                {"id": spec.get("id"), "pass": True, "reason": "Mock relation pass."}
                for spec in relation_specs
            ],
            "attachment_results": [
                {"id": spec.get("id"), "pass": True, "reason": "Mock attachment pass."}
                for spec in attachment_specs
            ],
        }
        judgement["_scene_summary"] = prompt_bundle["scene_summary"]
        judgement["_layout_summary"] = prompt_bundle["layout_summary"]
        judgement["_text_budget_used"] = prompt_bundle["text_budget_used"]
        artifacts = _save_judge_artifacts(
            artifact_dir,
            system_prompt=_judge_system_prompt(),
            user_prompt=prompt,
            image_manifest=image_manifest,
            request_metadata=None,
            raw_response=json.dumps(judgement, indent=2),
            parsed_response=judgement,
        )
        if artifacts:
            judgement["_judge_artifacts"] = artifacts
        return judgement


@dataclass
class OpenAICompatibleVLMJudge:
    chat_model: Any
    benchmark_config: dict | None = None

    def judge(
        self,
        *,
        case: dict,
        layout: dict,
        input_level: str,
        sanity_flags: list[dict],
        physical_flags: list[dict],
        view_flags: list[dict],
        render_skipped_objects: list[dict],
        object_groups: list[dict],
        global_view_artifacts: list[dict],
        group_view_artifacts: list[dict],
        relation_specs: list[dict],
        attachment_specs: list[dict],
        renderable_layout: dict | None = None,
        layout_normalization_summary: dict | None = None,
        validity_gate_passed: bool = True,
        artifact_dir: str | Path | None = None,
    ) -> dict:
        if not validity_gate_passed:
            return _invalid_judgement("Validity gate failed before VLM-as-judge.")
        if not hasattr(self.chat_model, "chat_messages"):
            raise TypeError("OpenAICompatibleVLMJudge requires a model with chat_messages().")

        runtime_profile = getattr(self.chat_model, "runtime_profile", None)
        judge_evidence_budgeting = bool(getattr(self.chat_model, "judge_evidence_budgeting", False))
        budget_config = evidence_budgeting_config(
            self.benchmark_config,
            judge_evidence_budgeting=judge_evidence_budgeting,
            runtime_profile=runtime_profile,
        )
        evidence_selection = select_judge_evidence(
            global_view_artifacts=global_view_artifacts,
            group_view_artifacts=group_view_artifacts,
            object_groups=object_groups,
            physical_flags=physical_flags,
            view_flags=view_flags,
            render_skipped_objects=render_skipped_objects,
            config=budget_config,
            runtime_profile=runtime_profile,
        )
        selected_global_artifacts = evidence_selection["selected_global_artifacts"]
        selected_group_artifacts = evidence_selection["selected_group_artifacts"]
        budgeting_enabled = bool(evidence_selection.get("budgeting_enabled"))
        system_prompt = _judge_system_prompt()
        image_manifest = _image_manifest(selected_global_artifacts, selected_group_artifacts)
        prompt_bundle = build_judge_prompt_payload(
            case=case,
            layout=layout,
            renderable_layout=renderable_layout or layout,
            input_level=input_level,
            layout_normalization_summary=layout_normalization_summary,
            object_groups=object_groups,
            sanity_flags=sanity_flags,
            physical_flags=physical_flags,
            view_flags=view_flags,
            render_skipped_objects=render_skipped_objects,
            relation_specs=relation_specs,
            attachment_specs=attachment_specs,
            evidence_selection=evidence_selection,
            image_manifest=image_manifest,
            benchmark_config=self.benchmark_config,
        )
        user_prompt = json.dumps(prompt_bundle["prompt_payload"], indent=2)
        content: list[dict] = [
            {
                "type": "text",
                "text": user_prompt,
            }
        ]
        for artifact in selected_global_artifacts + selected_group_artifacts:
            if artifact.get("id") == "camera_policy":
                continue
            path = artifact.get("abs_path")
            if isinstance(path, str) and Path(path).exists():
                content.append({"type": "text", "text": f"View artifact: {artifact.get('id')} path={artifact.get('path')}"})
                content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(path))}})

        input_manifest_path = _save_judge_input_manifest(
            artifact_dir,
            _judge_input_manifest(evidence_selection),
        ) if budgeting_enabled else None
        _save_judge_artifacts(
            artifact_dir,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_manifest=image_manifest,
            request_metadata=None,
            raw_response="",
            parsed_response=None,
            input_manifest_path=input_manifest_path,
        )
        raw = self.chat_model.chat_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format_json=True,
        )
        request_metadata = getattr(self.chat_model, "last_request_metadata", None)
        try:
            judgement = normalize_vlm_judgement(parse_json_object(raw))
        except ModelResponseError:
            _save_judge_artifacts(
                artifact_dir,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_manifest=image_manifest,
                request_metadata=request_metadata if isinstance(request_metadata, dict) else None,
                raw_response=raw,
                parsed_response=None,
                input_manifest_path=input_manifest_path,
            )
            raise
        artifacts = _save_judge_artifacts(
            artifact_dir,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_manifest=image_manifest,
            request_metadata=request_metadata if isinstance(request_metadata, dict) else None,
            raw_response=raw,
            parsed_response=judgement,
            input_manifest_path=input_manifest_path,
        )
        if artifacts:
            judgement["_judge_artifacts"] = artifacts
        if budgeting_enabled:
            judgement["_judge_input_manifest"] = _judge_input_manifest(evidence_selection)
        judgement["_scene_summary"] = prompt_bundle["scene_summary"]
        judgement["_layout_summary"] = prompt_bundle["layout_summary"]
        judgement["_text_budget_used"] = prompt_bundle["text_budget_used"]
        return judgement


def create_room_judge(benchmark_config: dict | None) -> RoomConsistencyJudge:
    evaluation = _evaluation_config(benchmark_config)
    name = evaluation.get("room_judge", MOCK_JUDGE_NAME)
    if name == MOCK_JUDGE_NAME:
        return MockRoomConsistencyJudge(default_score=int(evaluation.get("mock_room_score", DEFAULT_MOCK_SCORE)))
    raise ValueError(f"Unsupported room judge '{name}'.")


def create_pair_judge(benchmark_config: dict | None) -> PairRelationJudge:
    evaluation = _evaluation_config(benchmark_config)
    name = evaluation.get("pair_judge", MOCK_JUDGE_NAME)
    if name == MOCK_JUDGE_NAME:
        return MockPairRelationJudge(default_pass=bool(evaluation.get("mock_pair_pass", True)))
    raise ValueError(f"Unsupported pair judge '{name}'.")


def create_vlm_judge(benchmark_config: dict | None, judge_model: Any | None = None) -> VLMSceneJudge:
    evaluation = _evaluation_config(benchmark_config)
    name = evaluation.get("vlm_judge", DEFAULT_VLM_JUDGE)
    if name == MOCK_JUDGE_NAME:
        return MockVLMSceneJudge(
            default_score=int(evaluation.get("mock_vlm_score", evaluation.get("mock_room_score", DEFAULT_MOCK_SCORE))),
            benchmark_config=benchmark_config,
        )
    if name == DEFAULT_VLM_JUDGE and judge_model is not None and hasattr(judge_model, "chat_messages"):
        return OpenAICompatibleVLMJudge(chat_model=judge_model, benchmark_config=benchmark_config)
    if name == DEFAULT_VLM_JUDGE and judge_model is not None and judge_model.__class__.__name__ == "MockModel":
        return MockVLMSceneJudge(default_score=int(evaluation.get("mock_vlm_score", DEFAULT_MOCK_SCORE)), benchmark_config=benchmark_config)
    if name == DEFAULT_VLM_JUDGE and judge_model is None:
        return MockVLMSceneJudge(default_score=int(evaluation.get("mock_vlm_score", DEFAULT_MOCK_SCORE)), benchmark_config=benchmark_config)
    raise ValueError(f"Unsupported VLM judge '{name}' for the supplied model.")


def normalize_vlm_judgement(raw: dict) -> dict:
    try:
        score = int(raw.get("score"))
    except (TypeError, ValueError) as exc:
        raise ModelResponseError("VLM judgement must contain integer score.") from exc
    score = _clamp_score(score)
    valid = bool(raw.get("valid"))
    relation_results = _normalize_binary_results(raw.get("relation_results", []))
    attachment_results = _normalize_binary_results(raw.get("attachment_results", []))
    insufficient = bool(raw.get("insufficient_evidence", False))
    status = _normalize_judgement_status(raw.get("judgement_status"), insufficient)
    brief_reasoning = str(raw.get("brief_reasoning") or raw.get("short_reason") or "")
    return {
        "valid": valid,
        "score": score,
        "score_norm": float(raw.get("score_norm", _score_norm(score))),
        "confidence": _normalize_confidence(raw.get("confidence")),
        "judgement_status": status,
        "brief_reasoning": brief_reasoning,
        "issues": _normalize_issues(raw.get("issues", [])),
        "insufficient_evidence": insufficient or status == "insufficient_evidence",
        "short_reason": brief_reasoning,
        "global_assessment": str(raw.get("global_assessment", "")),
        "group_results": _normalize_group_results(raw.get("group_results", [])),
        "relation_results": relation_results,
        "attachment_results": attachment_results,
    }


def _evaluation_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config or {}
    section = config.get("evaluation")
    return section if isinstance(section, dict) else {}


def _judge_system_prompt() -> str:
    return (
        "You are an independent evaluator of explicit 3D bbox-proxy layouts. Judge bbox "
        "layout quality from rendered bbox views and deterministic structured summaries. "
        "For parseable layouts, your judgement determines overall_valid. Deterministic "
        "schema, physical, view, render, and grouping flags are evidence only, not automatic "
        "verdicts. Treat high-confidence serious collisions as strong evidence; treat "
        "fallback-derived room boundary or wall-height evidence as lower-confidence and "
        "approximate. Treat floating or unsupported vertical placement as plausibility "
        "evidence, not hard invalidity by itself. The target is explicit bbox layout quality, not photorealistic scene "
        "reconstruction. Do not penalize missing meshes, textures, real wall geometry, "
        "doors, windows, or lack of photorealism. If evidence is insufficient, set "
        "insufficient_evidence=true and judgement_status='insufficient_evidence'. Return "
        "exactly one JSON object with no Markdown and no chain-of-thought."
    )


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_manifest(global_view_artifacts: list[dict], group_view_artifacts: list[dict]) -> list[dict]:
    manifest = []
    for scope, artifacts in [("global", global_view_artifacts), ("group", group_view_artifacts)]:
        for artifact in artifacts:
            if artifact.get("id") == "camera_policy":
                continue
            manifest.append(
                {
                    "scope": scope,
                    "id": artifact.get("id"),
                    "path": artifact.get("path"),
                    "abs_path": artifact.get("abs_path"),
                    "diagnostics": artifact.get("diagnostics"),
                    "included_in_prompt": bool(artifact.get("abs_path") and Path(str(artifact.get("abs_path"))).exists()),
                }
            )
    return manifest


def _judge_input_manifest(evidence_selection: dict) -> dict:
    selected_images = [
        artifact.get("path")
        for artifact in evidence_selection.get("selected_global_artifacts", []) + evidence_selection.get("selected_group_artifacts", [])
        if isinstance(artifact, dict) and artifact.get("id") != "camera_policy"
    ]
    budget = evidence_selection.get("budget", {}) if isinstance(evidence_selection.get("budget"), dict) else {}
    return {
        "judge_evidence_budgeting": bool(evidence_selection.get("judge_evidence_budgeting")),
        "budgeting_enabled": bool(evidence_selection.get("budgeting_enabled")),
        "mode": evidence_selection.get("mode", "budgeted"),
        "runtime_profile": evidence_selection.get("runtime_profile"),
        "budget": budget,
        "base_max_groups_for_judge": budget.get("base_max_groups_for_judge"),
        "budget_raise_ratio": budget.get("budget_raise_ratio"),
        "effective_max_groups_for_judge": budget.get("effective_max_groups_for_judge"),
        "max_groups_for_judge_cap": budget.get("max_groups_for_judge_cap"),
        "effective_max_images": budget.get("effective_max_images"),
        "budget_config": evidence_selection.get("budget_config", {}),
        "selected_images": selected_images,
        "global_views_sent": evidence_selection.get("global_views_sent", []),
        "selected_groups": evidence_selection.get("selected_groups", []),
        "omitted_groups": evidence_selection.get("omitted_groups", []),
    }


def _save_judge_input_manifest(artifact_dir: str | Path | None, manifest: dict) -> Path | None:
    if artifact_dir is None:
        return None
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "judge_input_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _save_judge_artifacts(
    artifact_dir: str | Path | None,
    *,
    system_prompt: str,
    user_prompt: str,
    image_manifest: list[dict],
    request_metadata: dict | None,
    raw_response: str,
    parsed_response: dict | None,
    input_manifest_path: Path | None = None,
) -> dict:
    if artifact_dir is None:
        return {}
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "judge_prompt.json"
    manifest_path = out / "judge_image_manifest.json"
    request_metadata_path = out / "judge_request_metadata.json"
    raw_path = out / "judge_raw_response.txt"
    parsed_path = out / "judge_parsed_response.json"
    prompt_path.write_text(
        json.dumps({"system": system_prompt, "user": user_prompt}, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(image_manifest, indent=2), encoding="utf-8")
    if request_metadata is not None:
        request_metadata_path.write_text(json.dumps(request_metadata, indent=2), encoding="utf-8")
    raw_path.write_text(raw_response, encoding="utf-8")
    parsed_path.write_text(json.dumps(parsed_response or {}, indent=2), encoding="utf-8")
    root = _artifact_root(out)
    artifacts = {
        "prompt_path": _relative_artifact(prompt_path, root),
        "image_manifest_path": _relative_artifact(manifest_path, root),
        "raw_response_path": _relative_artifact(raw_path, root),
        "parsed_response_path": _relative_artifact(parsed_path, root),
    }
    if request_metadata is not None:
        artifacts["request_metadata_path"] = _relative_artifact(request_metadata_path, root)
    if input_manifest_path is not None:
        artifacts["input_manifest_path"] = _relative_artifact(input_manifest_path, root)
    return artifacts


def _artifact_root(artifact_dir: Path) -> Path:
    if artifact_dir.parent.name == "vlm_judge":
        return artifact_dir.parent.parent
    return artifact_dir.parent


def _relative_artifact(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _invalid_judgement(reason: str) -> dict:
    return {
        "valid": False,
        "score": 0,
        "score_norm": 0.0,
        "confidence": "low",
        "judgement_status": "judge_error",
        "brief_reasoning": reason,
        "issues": [
            {
                "group_id": None,
                "issue_type": "evidence",
                "severity": "critical",
                "object_ids": [],
                "evidence": reason,
                "repair_hint": "",
            }
        ],
        "insufficient_evidence": False,
        "short_reason": reason,
        "global_assessment": "",
        "group_results": [],
        "relation_results": [],
        "attachment_results": [],
    }


def _normalize_group_results(raw_results: Any) -> list[dict]:
    if not isinstance(raw_results, list):
        return []
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "group_id": item.get("group_id"),
                "object_ids": list(item.get("object_ids", [])) if isinstance(item.get("object_ids"), list) else [],
                "valid": bool(item.get("valid")),
                "score": _clamp_score(item.get("score", SCORE_MIN)),
                "issues": list(item.get("issues", [])) if isinstance(item.get("issues"), list) else [],
            }
        )
    return results


def _clamp_score(value: object) -> int:
    return max(SCORE_MIN, min(SCORE_MAX, int(value)))


def _score_norm(score: int) -> float:
    return float(score) / SCORE_DENOMINATOR


def _normalize_binary_results(raw_results: Any) -> list[dict]:
    if not isinstance(raw_results, list):
        return []
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "id": item.get("id"),
                "pass": bool(item.get("pass")),
                "reason": str(item.get("reason", "")),
            }
        )
    return results


def _normalize_confidence(value: object) -> str:
    text = str(value or "medium").lower()
    return text if text in {"low", "medium", "high"} else "medium"


def _normalize_judgement_status(value: object, insufficient_evidence: bool) -> str:
    text = str(value or "").lower()
    if text in {"valid_judgement", "insufficient_evidence", "unparseable_layout", "judge_error"}:
        return text
    return "insufficient_evidence" if insufficient_evidence else "valid_judgement"


def _normalize_issues(raw_issues: Any) -> list[dict]:
    if not isinstance(raw_issues, list):
        return []
    issues = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("issue_type") or "evidence")
        if issue_type not in {"parseability", "completeness", "boundary", "height", "collision", "support", "spatial_relation", "evidence"}:
            issue_type = "evidence"
        severity = str(item.get("severity") or "minor")
        if severity not in {"minor", "major", "critical"}:
            severity = "minor"
        issues.append(
            {
                "group_id": item.get("group_id"),
                "issue_type": issue_type,
                "severity": severity,
                "object_ids": list(item.get("object_ids", [])) if isinstance(item.get("object_ids"), list) else [],
                "evidence": str(item.get("evidence", "")),
                "repair_hint": str(item.get("repair_hint", "")),
            }
        )
    return issues
