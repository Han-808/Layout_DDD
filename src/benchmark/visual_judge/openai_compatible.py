from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any

from benchmark.models import OpenAICompatibleModel, parse_json_object
from benchmark.visual_judge.active_policy import selector_safe_proposals
from benchmark.visual_judge.contracts import (
    validate_binary_judge_response,
    validate_camera_selection_response,
    validate_canonical_metric_response,
    validate_generic_visual_response,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)
from benchmark.visual_judge.interfaces import JudgeResult
from benchmark.visual_judge.camera_dsl import (
    CAMERA_OBSERVATIONS,
    METRIC_CAMERA_REQUIREMENTS,
    canonical_camera_metric,
)


SYSTEM_PROMPT = """You are the visual judge for a 3D scene-generation benchmark.
Use the rendered views as primary visual evidence. The canonical scene summary and deterministic
findings are navigation aids, not a replacement for inspecting the images. Do not invent hidden
objects or relations. Return exactly one JSON object with:
{"applicable":true,"score":0.0,"confidence":0.0,"summary":"...","issues":["..."],"evidence":["..."]}.
score and confidence must be between 0 and 1. If the supplied views cannot support this category,
return applicable=false, score=null, and explain why in summary."""

CANONICAL_METRIC_SYSTEM_PROMPT = """You judge one narrowly scoped canonical metric for a
3D scene benchmark. Use the original prompt, prompt-authorized deviations, structured context,
and supplied images. Do not judge neighboring metrics. Evidence insufficiency is not validity.
Return exactly one JSON object:
{"evidence_status":"sufficient","verdict":"valid","confidence":0.0,"reason":"...",
"missing_evidence":[],"defects":[],"evidence_request":null}.
evidence_status must be sufficient or insufficient. verdict must be valid, invalid, or ambiguous.
If evidence is insufficient, verdict must be ambiguous, defects must be empty, and
evidence_request must be:
{"target_ids":["object_id"],"missing_observations":["exact_token"],
"view_goal":"concrete visual goal","metadata":{}}.
Use only target IDs in allowed_evidence_request_target_ids and exact observation tokens in
allowed_missing_observations from the user context. missing_evidence may repeat those exact tokens
for legacy compatibility or remain empty when evidence_request is present. Do not place prose in
missing_evidence or missing_observations. Do not define camera constraints, poses, repair plans,
metric scope, or rubric in evidence_request. Invalid requires one or more explicit significant
defects in defects; otherwise return valid when evidence is sufficient. Each defect should contain
scope, target_ids, relation, and reason when available. confidence must be between 0 and 1."""

P0B_SYSTEM_PROMPT = """You adjudicate one ambiguous geometry event in a 3D scene benchmark.
Use the natural-language prompt and extracted relationships only to understand intended semantics.
Use detector evidence for measured geometry and inspect supplied images when present. Generator
relationships are claims, not automatic exemptions. Highlighted diagnostic views use the supplied
color legend; gray geometry is non-target context, not missing scene content. Decide whether this event constitutes a
structural error. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

RELATION_SYSTEM_PROMPT = """You adjudicate one explicit spatial relationship in a 3D scene benchmark.
The relation was routed because it is outside the frozen deterministic registry or because deterministic
proxy evidence could not safely resolve its semantic realization. Routing carries no valid or invalid
prior. Judge only whether that exact relationship is satisfied by the generated scene. Use the original
prompt, structured relation, detector evidence, and supplied rendered views. Canonical object geometry
is supporting context, not permission to invent hidden evidence. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

_EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT = """Return exactly one JSON object:
{"status":"valid","confidence":0.0,"reason":"...","defects":[],
"evidence_request":null}.
status must be valid, invalid, or need_more_evidence. valid and invalid must use
evidence_request=null. If the supplied views cannot support a safe binary conclusion, do not guess:
return need_more_evidence with defects=[] and a structured evidence_request:
{"target_ids":["object_id"],"missing_observations":["..."],"view_goal":"...","metadata":{}}.
The request must identify the targets, missing technical observation, and a concrete evidence view
goal. missing_observations must contain only exact Camera DSL tokens from the
allowed_missing_observations supplied in the user context. The finite vocabulary is:
target_visible, joint_visibility, contact_surface_visible, support_chain_visible,
architecture_plane_visible, front_back_disambiguated, depth_baseline_available,
group_context_visible, interaction_side_visible, limited_local_context,
global_context_preserved, occluder_avoided.
Do not put prose in missing_observations. confidence must be between 0 and 1."""

P0B_CONTROL_SYSTEM_PROMPT = (
    P0B_SYSTEM_PROMPT.split("Return exactly one JSON object:", 1)[0]
    + _EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT
)
RELATION_CONTROL_SYSTEM_PROMPT = (
    RELATION_SYSTEM_PROMPT.split("Return exactly one JSON object:", 1)[0]
    + _EVIDENCE_AWARE_BINARY_OUTPUT_CONTRACT
)

SPATIAL_FIDELITY_SYSTEM_PROMPT = """You adjudicate one candidate Spatial Fidelity issue in a
coarse-grained 3D scene benchmark. A statistical detector routed this event; routing is evidence to
inspect, not a valid/invalid prior. For scale, decide whether the object's visible real-world size is
semantically implausible, allowing legitimate variants and unusual but coherent designs. For
co-occurrence, decide whether the object-category combination is semantically incoherent in this
particular requested scene; dataset rarity or absence alone is never an error. Use the canonical
measurements and statistical packet together with the rendered views. Return exactly one JSON object:
{"verdict":"valid","confidence":0.0,"reason":"..."}.
verdict must be exactly valid or invalid. No abstention, not-applicable, insufficient-evidence,
continuous score, or third verdict is allowed. confidence must be between 0 and 1."""

CAMERA_SELECTION_SYSTEM_PROMPT = """You select camera evidence for a 3D scene benchmark.
Do not judge whether the metric event is valid or invalid. Select the candidate views that make the
specified event easiest to inspect. Prefer views where all target objects and the relevant contact,
gap, overlap, or room plane are visible and well framed. In highlighted previews, use the supplied
color legend and treat gray geometry as non-target context. A preview_warning_class means only
that deterministic highlight-pixel coverage was incomplete; do not infer target absence from it.
When selection_phase is active_fallback, evidence_deficiency states why the deterministic packet was
not sufficient and corrective_proposals lists the only permitted metric-specific repairs. Use it only
to choose views or one proposal that repairs the named evidence gap. Never invent pose coordinates or
an unlisted action.
Candidates marked render_status=blank are unusable camera evidence. Do not select them when any
render_status=ok candidate exists.
You may request at most one listed discrete
camera action when adjustment is allowed. Return exactly one JSON object:
{"selected_view_ids":["candidate_id"],"action":null,"reason":"..."}.
selected_view_ids must contain between one and max_views candidate IDs in evidence-priority order,
best view first. action must be null when
allow_adjustment is false. In active_fallback it may be null or
{"proposal_id":"one listed proposal ID"}; otherwise it may be null or
{"view_id":"candidate_id","type":"one allowed action"}. Do not return a metric verdict or score."""

CATEGORY_RUBRICS = {
    "visual_quality": (
        "Judge holistic visual coherence, commonsense plausibility, style/appearance consistency, "
        "proportions, and whether the scene looks intentionally composed. Do not deduct twice for "
        "a deterministic geometry issue unless its visual consequence is visible."
    ),
    "prompt_fidelity": (
        "Judge only whether the visible scene follows the supplied natural-language request. "
        "Check requested objects, attributes, rough counts, and explicitly requested spatial relations. "
        "Treat ordinary synonyms and subtypes as the same broad object identity. Internal object-mapping "
        "confidence, unmatched candidates, resolver requests, and retrieval metadata are non-scoring alignment "
        "diagnostics: do not reward or penalize the generator for them. Judge visible prompt adherence directly, "
        "including color, material, style, count, placement, and relations."
    ),
    "structural_validity": (
        "Use the images to resolve only ambiguous structural findings called out by deterministic evidence. "
        "Do not override exact schema, boundary, or collision calculations without visible contradictory evidence."
    ),
    "functional_semantic_fidelity": (
        "Judge prompt-conditioned functional semantics only. Global evidence checks the requested "
        "room/scene type, broad functional intent, and required functional areas. Check local "
        "functionality only when the frozen prompt contract explicitly requests that function. "
        "Do not judge generic object pairing, scale, style, physical plausibility, or unspecified "
        "local functionality. Invalid requires at least one explicit significant unmet requirement; "
        "otherwise return valid when evidence is sufficient."
    ),
    "scale_consistency": (
        "Judge generic visible relative scale coherence only. Apply exact prompt-authorized "
        "surreal or unusual scale deviations before judging. Do not judge object pairing, layout, "
        "orientation, function, style, or physical plausibility. Invalid requires an explicit "
        "significant scale defect; otherwise return valid when evidence is sufficient."
    ),
    "object_pairing_consistency": (
        "Judge only whether members of each supplied object group have broadly compatible object "
        "categories and roles. Do not judge position, distance, direction, angle, orientation, "
        "access, local functional arrangement, style, scale, or whether the prompt was followed. "
        "Apply exact prompt-authorized pair deviations. Invalid requires an explicit significant "
        "category/role incompatibility; otherwise return valid when evidence is sufficient."
    ),
    "style_consistency": (
        "Judge visible style consistency only, after applying prompt-authorized style deviations. "
        "Do not judge scale, pairing, layout, orientation, function, prompt fidelity, or physical "
        "plausibility. Invalid requires one or more explicit significant style inconsistencies; "
        "otherwise return valid when evidence is sufficient."
    ),
    "functional_consistency": (
        "Judge whether each supplied local group is visibly usable in a "
        "real-world sense. Check interaction-side access, opening clearance, "
        "orientation for use, and ensemble operability. Do not judge prompt "
        "fidelity, object-category pairing, style, scale, or unrelated exact "
        "relations. Invalid requires an explicit significant visible "
        "functional defect; otherwise return valid when evidence is sufficient."
    ),
}


class OpenAICompatibleVLMJudge:
    """Multimodal judge shared by MNET localhost and remote OpenAI-style APIs."""

    def __init__(
        self,
        model: OpenAICompatibleModel,
        *,
        max_images: int = 6,
        max_context_chars: int = 30000,
        response_format_json: bool | None = None,
    ) -> None:
        self.model = model
        self.max_images = max(1, int(max_images))
        self.max_context_chars = max(1000, int(max_context_chars))
        self.response_format_json = (
            bool(getattr(model, "response_format_json", True))
            if response_format_json is None
            else bool(response_format_json)
        )

    def _audit_result(
        self,
        result: dict[str, Any],
        *,
        role: VLMRole,
        decision_contract: DecisionContract,
        judge_method: str,
        images_used: list[str],
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result.update(
            vlm_audit_metadata(
                role,
                decision_contract=decision_contract,
                judge_method=judge_method,
            )
        )
        result["model"] = self.model.model_id
        result["endpoint"] = self.model.endpoint
        result["images_used"] = list(images_used)
        result["request_metadata"] = (
            dict(self.model.last_request_metadata)
            if request_metadata is None
            else dict(request_metadata)
        )
        return result

    def evaluate(self, request: dict) -> dict:
        """Compatibility score wrapper with a deterministic evidence preflight."""

        if not isinstance(request, dict):
            raise TypeError("VLM judge request must be a JSON object")
        gate = _check_standalone_visual_evidence(request)
        if gate is not None:
            return self._audit_result(
                {
                    "applicable": False,
                    "score": None,
                    "confidence": 0.0,
                    "summary": gate,
                    "issues": [],
                    "evidence": [],
                },
                role=VLMRole.JUDGE,
                decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
                judge_method="evaluate",
                images_used=[],
                request_metadata={},
            )
        return self._evaluate_raw(request)

    def _evaluate_raw(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("VLM judge request must be a JSON object")
        category = str(request.get("category") or "visual_quality")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"VLM render evidence does not exist: {missing}")

        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
            judge_method="evaluate",
        )
        context = {
            **audit,
            "category": category,
            "rubric": CATEGORY_RUBRICS.get(category, "Judge this category from the supplied evidence."),
            "natural_language_request": request.get("prompt"),
            "canonical_scene": request.get("scene_summary"),
            "deterministic_evidence": request.get("deterministic_evidence"),
            "view_names": _generic_view_names(selected),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "category",
                "deterministic_evidence",
                "rubric",
                "natural_language_request",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Evaluate this scene under the specified category.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.{category}",
        )
        result = parse_json_object(raw)
        validate_generic_visual_response(result)
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
            judge_method="evaluate",
            images_used=[str(path.resolve()) for path in selected],
        )

    def adjudicate_scene_quality(self, request: dict) -> dict:
        """Return the strict canonical L3 verdict contract."""

        return self._controlled_adjudicate(
            "adjudicate_scene_quality",
            request,
        )

    def _adjudicate_scene_quality_raw(self, request: dict) -> dict:
        return self._adjudicate_canonical_metric(
            request,
            family="scene_quality",
            judge_method="adjudicate_scene_quality",
        )

    def adjudicate_functional_semantic(self, request: dict) -> dict:
        """Return the strict canonical L2 functional-semantic verdict contract."""

        return self._controlled_adjudicate(
            "adjudicate_functional_semantic",
            request,
        )

    def _adjudicate_functional_semantic_raw(
        self,
        request: dict,
    ) -> dict:
        result = self._adjudicate_canonical_metric(
            request,
            family="specification_fidelity",
            judge_method="adjudicate_functional_semantic",
        )
        if result.get("verdict") == "ambiguous":
            result["canonical_verdict"] = "ambiguous"
            result["response_adapter"] = (
                "functional_semantic_insufficient_evidence_compat_v1"
            )
            result["verdict"] = "insufficient_evidence"
        result.setdefault(
            "router_state",
            "not_suspicious"
            if result.get("verdict") == "valid"
            else "suspicious"
            if result.get("verdict") == "invalid"
            else "insufficient_evidence",
        )
        return result

    def _adjudicate_canonical_metric(
        self,
        request: dict,
        *,
        family: str,
        judge_method: str,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("canonical metric judge request must be a JSON object")
        metric = str(request.get("metric") or request.get("category") or "")
        if metric not in {
            "functional_semantic_fidelity",
            "scale_consistency",
            "object_pairing_consistency",
            "style_consistency",
            "functional_consistency",
        }:
            raise ValueError(f"unsupported canonical metric {metric!r}")
        allowed_missing_observations = (
            _allowed_canonical_camera_observations(metric)
        )
        allowed_evidence_request_target_ids = (
            _canonical_evidence_request_target_ids(request)
        )
        paths = [
            Path(str(value)).expanduser()
            for value in request.get("render_evidence", [])
        ]
        selected = paths[: self.max_images]
        missing_paths = [str(path) for path in selected if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"canonical metric render evidence does not exist: {missing_paths}"
            )
        if not selected:
            return self._audit_result(
                {
                    "evidence_status": "insufficient",
                    "verdict": "ambiguous",
                    "confidence": 0.0,
                    "reason": "no rendered evidence was supplied",
                    "missing_evidence": ["target_visible"],
                    "defects": [],
                    "evidence_request": {
                        "target_ids": list(
                            allowed_evidence_request_target_ids[:1]
                            or ("scene",)
                        ),
                        "missing_observations": ["target_visible"],
                        "view_goal": (
                            "show the requested metric scope in a "
                            "decodable visual view"
                        ),
                        "metadata": {},
                    },
                },
                role=VLMRole.JUDGE,
                decision_contract=DecisionContract.CANONICAL_METRIC,
                judge_method=judge_method,
                images_used=[],
                request_metadata={},
            )
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method=judge_method,
        )
        context = {
            **audit,
            "family": family,
            "metric": metric,
            "rubric": CATEGORY_RUBRICS[metric],
            "metric_rubric": request.get("metric_rubric"),
            "natural_language_request": request.get("prompt"),
            "authorized_deviations": request.get("authorized_deviations"),
            "metric_scope": request.get("judgment_scope"),
            "claims": request.get("claims"),
            "components": request.get("components"),
            "object_groups": request.get("object_groups"),
            "visual_style_spec": request.get("visual_style_spec"),
            "canonical_scene": request.get("scene_summary"),
            "deterministic_evidence": request.get("deterministic_evidence"),
            "allowed_missing_observations": list(
                allowed_missing_observations
            ),
            "allowed_evidence_request_target_ids": list(
                allowed_evidence_request_target_ids
            ),
            "evidence_phase": request.get("evidence_phase"),
            "decision_mode": request.get("decision_mode"),
            "phase_instruction": (
                "This is the global screening pass. Return valid when the global "
                "views are sufficient and show no significant in-scope style "
                "defect. Return invalid only with explicit target IDs for a "
                "significant visible defect. Return ambiguous/insufficient when "
                "closer local evidence is needed; do not guess."
                if metric == "style_consistency"
                and request.get("evidence_phase") == "global_screen"
                else
                "This is the final local-confirmation pass. Use the local views "
                "first and the global views as context; independently confirm or "
                "reject the suspected style defect."
                if metric == "style_consistency"
                and request.get("evidence_phase") == "local_confirmation"
                else None
            ),
            "view_names": _generic_view_names(selected),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "rubric",
                "metric_rubric",
                "natural_language_request",
                "authorized_deviations",
                "metric_scope",
                "claims",
                "components",
                "object_groups",
                "visual_style_spec",
                "deterministic_evidence",
                "allowed_missing_observations",
                "allowed_evidence_request_target_ids",
                "evidence_phase",
                "decision_mode",
                "phase_instruction",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this canonical metric only.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": CANONICAL_METRIC_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.canonical.{metric}",
        )
        result = parse_json_object(raw)
        allowed_scopes = (
            (
                request.get("judgment_scope")
                if isinstance(request.get("judgment_scope"), dict)
                else {}
            ).get("included")
            or []
        )
        validate_canonical_metric_response(
            result,
            allowed_scopes=allowed_scopes,
            allowed_missing_observations=(
                allowed_missing_observations
            ),
            allowed_target_ids=(
                allowed_evidence_request_target_ids
            ),
        )
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.CANONICAL_METRIC,
            judge_method=judge_method,
            images_used=[str(path.resolve()) for path in selected],
        )

    def adjudicate_p0b(self, request: dict) -> dict:
        return self._controlled_adjudicate("adjudicate_p0b", request)

    def _adjudicate_p0b_control(self, request: dict) -> dict:
        return self._adjudicate_p0b_raw(
            request,
            _allow_need_more_evidence=True,
        )

    def _adjudicate_p0b_raw(
        self,
        request: dict,
        *,
        _allow_need_more_evidence: bool = False,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("P0b judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"P0b render evidence does not exist: {missing}")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.P0B_BINARY,
            judge_method="adjudicate_p0b",
        )
        context = {
            **audit,
            "metric": request.get("metric"),
            "metric_rubric": request.get("metric_rubric"),
            "candidate_selection_policy": request.get("candidate_selection_policy"),
            "collision_evidence_style_guide": request.get("collision_evidence_style_guide"),
            "visual_evidence_policy": request.get("visual_evidence_policy"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "objects": request.get("objects"),
            "architecture": request.get("architecture"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "extracted_relationships": request.get("extracted_relationships"),
            "view_names": _generic_view_names(selected),
            "view_evidence": _sanitize_outbound_view_evidence(
                request.get("local_render_evidence_metadata")
            ),
            "allowed_missing_observations": (
                _allowed_binary_camera_observations(
                    request.get("metric")
                )
            ),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "detector_evidence",
                "event",
                "natural_language_prompt",
                "metric_rubric",
                "candidate_selection_policy",
                "collision_evidence_style_guide",
                "visual_evidence_policy",
                "view_names",
                "view_evidence",
                "allowed_missing_observations",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this P0b event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {
                    "role": "system",
                    "content": (
                        P0B_CONTROL_SYSTEM_PROMPT
                        if _allow_need_more_evidence
                        else P0B_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.p0b.{request.get('metric') or 'event'}",
        )
        result = parse_json_object(raw)
        if _allow_need_more_evidence:
            result = _normalize_evidence_aware_binary_response(
                result,
                judge_label="P0b judge",
            )
        else:
            validate_binary_judge_response(
                result,
                judge_label="P0b judge",
                confidence_label="VLM judge",
            )
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.P0B_BINARY,
            judge_method="adjudicate_p0b",
            images_used=[str(path.resolve()) for path in selected],
        )

    def adjudicate_relation(self, request: dict) -> dict:
        return self._controlled_adjudicate(
            "adjudicate_relation",
            request,
        )

    def _adjudicate_relation_control(self, request: dict) -> dict:
        return self._adjudicate_relation_raw(
            request,
            _allow_need_more_evidence=True,
        )

    def _adjudicate_relation_raw(
        self,
        request: dict,
        *,
        _allow_need_more_evidence: bool = False,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("relationship judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"relationship render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("relationship adjudication requires at least one rendered view")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.RELATION_BINARY,
            judge_method="adjudicate_relation",
        )
        context = {
            **audit,
            "family": request.get("family"),
            "explicit_relation_claim": request.get("relation"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "view_names": _generic_view_names(selected),
            "allowed_missing_observations": (
                _allowed_binary_camera_observations(
                    "relation",
                    relation_type=_relation_type_from_request(request),
                )
            ),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "family",
                "detector_evidence",
                "explicit_relation_claim",
                "natural_language_prompt",
                "involved_objects",
                "view_names",
                "allowed_missing_observations",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this explicit spatial relationship.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {
                    "role": "system",
                    "content": (
                        RELATION_CONTROL_SYSTEM_PROMPT
                        if _allow_need_more_evidence
                        else RELATION_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.relationship.{request.get('family') or 'unknown'}",
        )
        result = parse_json_object(raw)
        if _allow_need_more_evidence:
            result = _normalize_evidence_aware_binary_response(
                result,
                judge_label="relationship judge",
            )
        else:
            validate_binary_judge_response(
                result,
                judge_label="relationship judge",
                confidence_label="VLM judge",
            )
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.RELATION_BINARY,
            judge_method="adjudicate_relation",
            images_used=[str(path.resolve()) for path in selected],
        )

    def adjudicate_spatial_fidelity(self, request: dict) -> dict:
        return self._controlled_adjudicate(
            "adjudicate_spatial_fidelity",
            request,
        )

    def _adjudicate_spatial_fidelity_raw(
        self,
        request: dict,
    ) -> dict:
        if not isinstance(request, dict):
            raise TypeError("Spatial Fidelity judge request must be a JSON object")
        paths = [Path(str(value)).expanduser() for value in request.get("render_evidence", [])]
        selected = paths[: self.max_images]
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Spatial Fidelity render evidence does not exist: {missing}")
        if not selected:
            raise ValueError("Spatial Fidelity adjudication requires at least one rendered view")
        audit = vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.SPATIAL_FIDELITY_BINARY,
            judge_method="adjudicate_spatial_fidelity",
        )
        context = {
            **audit,
            "metric": request.get("metric"),
            "event": request.get("event"),
            "detector_evidence": request.get("detector_evidence"),
            "involved_objects": request.get("involved_objects"),
            "canonical_scene": request.get("scene_summary"),
            "natural_language_prompt": request.get("natural_language_prompt"),
            "view_names": _generic_view_names(selected),
        }
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric",
                "event",
                "detector_evidence",
                "involved_objects",
                "natural_language_prompt",
                "view_names",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Adjudicate this routed Spatial Fidelity event.\n" + context_text,
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
            for path in selected
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": SPATIAL_FIDELITY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=f"vlm_judge.spatial_fidelity.{request.get('metric') or 'event'}",
        )
        result = parse_json_object(raw)
        validate_binary_judge_response(
            result,
            judge_label="Spatial Fidelity judge",
            confidence_label="VLM judge",
        )
        return self._audit_result(
            result,
            role=VLMRole.JUDGE,
            decision_contract=DecisionContract.SPATIAL_FIDELITY_BINARY,
            judge_method="adjudicate_spatial_fidelity",
            images_used=[str(path.resolve()) for path in selected],
        )

    def _controlled_adjudicate(
        self,
        method_name: str,
        request: dict,
    ) -> dict:
        """Route direct public calls through the same strict compatibility loop."""

        # Lazy imports keep the low-level OpenAI transport independent from the
        # orchestration module at import time.  ``ControlledVLMJudge`` resolves
        # the private raw method, so this wrapper cannot recurse.
        from benchmark.visual_judge.control_config import (
            resolve_vlm_evaluation_control,
        )
        from benchmark.visual_judge.runtime import ControlledVLMJudge

        control = resolve_vlm_evaluation_control(
            existing_selector_available=False,
            judge_max_images=self.max_images,
        )
        wrapper = ControlledVLMJudge(
            self,
            control=control,
            strict=True,
        )
        call = getattr(wrapper, method_name)
        return call(request)

    def select_camera_views(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError("camera selector request must be a JSON object")
        candidates = [item for item in request.get("candidates", []) if isinstance(item, dict)]
        if not candidates:
            raise ValueError("camera selector requires at least one candidate")
        usable_candidates = [
            item
            for item in candidates
            if str(item.get("render_status") or "ok") != "blank"
        ]
        if not usable_candidates:
            raise ValueError("camera selector received no non-blank candidate previews")
        if len(usable_candidates) > self.max_images:
            raise ValueError(
                "camera selector candidate bank exceeds max_images; implicit "
                f"truncation is forbidden ({len(usable_candidates)} > "
                f"{self.max_images})"
            )
        internal_ids = [str(item.get("id") or "") for item in usable_candidates]
        if any(not value for value in internal_ids) or len(set(internal_ids)) != len(internal_ids):
            raise ValueError("camera selector candidates require unique non-empty internal IDs")
        candidate_paths = [Path(str(item.get("image_path"))).expanduser() for item in usable_candidates]
        missing = [str(path) for path in candidate_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"camera selector preview evidence does not exist: {missing}")
        selected_candidates = sorted(
            usable_candidates,
            key=_selector_candidate_order_key,
        )
        paths = [Path(str(item.get("image_path"))).expanduser() for item in selected_candidates]
        max_views = max(1, min(int(request.get("max_views") or 1), len(selected_candidates)))
        alias_to_internal = {
            f"candidate_{index:02d}": str(item.get("id"))
            for index, item in enumerate(selected_candidates)
        }
        internal_to_alias = {
            internal: alias
            for alias, internal in alias_to_internal.items()
        }
        aliases = list(alias_to_internal)
        allowed_actions = _selector_allowed_actions(request.get("allowed_actions"))
        selection_phase = str(request.get("selection_phase") or "").strip().lower()
        corrective_proposals: list[dict[str, Any]] = []
        proposal_lookup: dict[str, dict[str, Any]] = {}
        if selection_phase == "active_fallback":
            corrective_proposals, proposal_lookup = selector_safe_proposals(
                [
                    item
                    for item in request.get("corrective_proposals", [])
                    if isinstance(item, dict)
                ],
                internal_to_alias=internal_to_alias,
            )
        allow_adjustment = bool(request.get("allow_adjustment")) and bool(
            corrective_proposals
            if selection_phase == "active_fallback"
            else allowed_actions
        )
        audit = vlm_audit_metadata(
            VLMRole.VLM_CAMERA_SELECTOR,
            decision_contract=DecisionContract.CAMERA_SELECTION,
            judge_method="select_camera_views",
        )
        context = {
            **audit,
            "candidates": [
                {
                    "id": alias,
                    "pose": _minimal_selector_pose(item.get("pose")),
                }
                for alias, item in zip(aliases, selected_candidates)
            ],
            "max_views": max_views,
            "allow_adjustment": allow_adjustment,
            "allowed_actions": allowed_actions if allow_adjustment else [],
            "metric_family": _selector_metric_family(request.get("metric")),
            "preview_role": _selector_preview_role(request.get("preview_role")),
            "preview_warning_class": _selector_preview_warning_class(
                request.get("preview_visibility_warning")
                if request.get("preview_visibility_warning") is not None
                else request.get("preview_degradation")
            ),
            "color_legend": _sanitize_selector_legend(request.get("color_legend")),
        }
        if selection_phase == "active_fallback":
            context["selection_phase"] = "active_fallback"
            context["evidence_deficiency"] = _sanitize_selector_deficiency(
                request.get("evidence_deficiency")
            )
            context["corrective_proposals"] = (
                corrective_proposals if allow_adjustment else []
            )
        context_text = _budgeted_context_json(
            context,
            self.max_context_chars,
            priority_keys=(
                "vlm_role",
                "decision_contract",
                "judge_method",
                "metric_family",
                "preview_role",
                "preview_warning_class",
                "color_legend",
                "candidates",
                "max_views",
                "allow_adjustment",
                "allowed_actions",
                "selection_phase",
                "evidence_deficiency",
                "corrective_proposals",
            ),
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Select camera evidence only; do not adjudicate the event.\n" + context_text,
            }
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _selector_image_data_url(path, alias=alias)},
            }
            for alias, path in zip(aliases, paths)
        )
        raw = self.model.chat_messages(
            [
                {"role": "system", "content": CAMERA_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format_json=self.response_format_json,
            call_type=(
                "vlm_camera_pose.active_fallback"
                if selection_phase == "active_fallback"
                else "vlm_camera_pose.query_cov"
            ),
        )
        parsed = parse_json_object(raw)
        available = set(alias_to_internal)
        resolved_aliases = validate_camera_selection_response(
            parsed,
            available_view_ids=available,
            max_views=max_views,
        )
        action = parsed.get("action")
        resolved_action = None
        if action is not None:
            if not allow_adjustment or not isinstance(action, dict):
                raise ValueError("camera selector returned an action outside the adjustment contract")
            if selection_phase == "active_fallback":
                proposal_alias = str(action.get("proposal_id") or "")
                proposal = proposal_lookup.get(proposal_alias)
                if proposal is None:
                    raise ValueError(
                        "active camera selector action references an unknown "
                        "corrective proposal"
                    )
                resolved_action = {
                    "proposal_id": str(proposal.get("proposal_id") or ""),
                    "view_id": str(proposal.get("parent_view_id") or ""),
                    "type": str(proposal.get("action_primitive") or ""),
                    "family": str(proposal.get("family") or ""),
                }
            else:
                if str(action.get("view_id") or "") not in available:
                    raise ValueError("camera selector action references an unknown candidate")
                if str(action.get("type") or "") not in set(allowed_actions):
                    raise ValueError("camera selector requested an unsupported action")
                resolved_action = {
                    "view_id": alias_to_internal[str(action["view_id"])],
                    "type": str(action["type"]),
                }
        reason = parsed.get("reason")
        request_metadata = dict(self.model.last_request_metadata)
        request_metadata.update(
            {
                "selector_candidate_order_policy": "stable_pose_image_digest_v1",
                "selector_candidate_alias_policy": "per_request_sequential_alias_v1",
            }
        )
        return self._audit_result(
            {
                "selected_view_ids": [
                    alias_to_internal[value] for value in resolved_aliases
                ],
                "action": resolved_action,
                "reason": str(reason)[:1000] if reason is not None else "",
            },
            role=VLMRole.VLM_CAMERA_SELECTOR,
            decision_contract=DecisionContract.CAMERA_SELECTION,
            judge_method="select_camera_views",
            # Preserve an auditable image count without exposing local names or paths.
            images_used=aliases,
            request_metadata=request_metadata,
        )


def _normalize_evidence_aware_binary_response(
    value: dict[str, Any],
    *,
    judge_label: str,
) -> dict[str, Any]:
    if value.get("status") is not None:
        return JudgeResult.from_value(value).to_dict()
    validate_binary_judge_response(
        value,
        judge_label=judge_label,
        confidence_label="VLM judge",
    )
    return JudgeResult.from_value(
        {
            "status": value["verdict"],
            "confidence": value["confidence"],
            "reason": value["reason"],
            "defects": [],
        }
    ).to_dict()


def _generic_view_names(paths: list[Path]) -> list[str]:
    """Return request-local aliases instead of exposing local filenames."""

    return [f"image_{index:02d}" for index, _ in enumerate(paths)]


def _sanitize_outbound_view_evidence(value: Any) -> Any:
    """Remove filesystem, content-hash, provenance, and dataset linkage fields.

    Rich evidence metadata is retained in local artifacts, while the remote
    judge receives only the role, pose, target, and diagnostic legend needed
    to interpret the images.
    """

    if isinstance(value, list):
        sanitized_items = [
            _sanitize_outbound_view_evidence(item)
            for item in value
        ]
        # Judge-visible image identity is request-local and fixed-shape.  It
        # must not reveal whether a slot came from deterministic or active
        # camera search through an action-suffixed internal view ID.
        for index, item in enumerate(sanitized_items):
            if not isinstance(item, dict):
                continue
            if "view_id" in item:
                item["view_id"] = f"slot_{index:02d}"
            if "pair_id" in item:
                item["pair_id"] = f"slot_{index:02d}"
        return sanitized_items
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        normalized = key.lower()
        if (
            "path" in normalized
            or "sha256" in normalized
            or "degradation" in normalized
            or "exception" in normalized
            or normalized in {
                "hash",
                "content_hash",
                "case_id",
                "dataset_id",
                "asset_jid",
                "jid",
                "provenance",
                "geometry_provenance",
                "active_camera_fallback",
                "camera_action",
                "camera_action_protocol",
                "camera_action_parameters",
                "parent_view_id",
                "policy_source",
                "proposal_id",
                "proposal_fingerprint",
                "trajectory",
                "selection_phase",
                "active_repair",
                "active_corrective_proposal_version",
                "trigger_recommended",
                "camera_repairable",
                "repairability",
            }
        ):
            continue
        if normalized == "pose":
            sanitized[key] = _minimal_judge_pose(item)
            continue
        sanitized[key] = _sanitize_outbound_view_evidence(item)
    return sanitized


def _minimal_judge_pose(value: Any) -> dict[str, Any]:
    """Keep framing semantics while stripping camera-policy lineage."""

    pose = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    for key in ("location", "target"):
        vector = pose.get(key)
        if isinstance(vector, (list, tuple)) and len(vector) == 3:
            result[key] = [
                round(float(component), 5)
                for component in vector
            ]
    if isinstance(pose.get("lens_mm"), (int, float)):
        result["lens_mm"] = round(float(pose["lens_mm"]), 3)
    camera_type = str(pose.get("camera_type") or "").strip().upper()
    if camera_type in {"PERSP", "ORTHO"}:
        result["camera_type"] = camera_type
    return result


def build_openai_compatible_vlm_judge(config: dict[str, Any]) -> OpenAICompatibleVLMJudge:
    if not isinstance(config, dict):
        raise TypeError("VLM judge config must be a JSON object")
    if "api_key" in config:
        raise ValueError(
            "VLM judge config must not contain literal api_key; use api_key_env instead"
        )
    endpoint = config.get("endpoint") or config.get("base_url")
    model_id = config.get("model") or config.get("model_id")
    if not endpoint or not model_id:
        raise ValueError("VLM judge config requires endpoint and model")
    model = OpenAICompatibleModel(
        name=str(config.get("name") or "openai-compatible-vlm-judge"),
        endpoint=str(endpoint),
        model_id=str(model_id),
        api_key_env=str(config["api_key_env"]) if config.get("api_key_env") else None,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 2048)),
        context_length=int(config["context_length"]) if config.get("context_length") is not None else None,
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        response_format_json=bool(config.get("response_format_json", True)),
        max_retries=int(config.get("max_retries", 1)),
        retry_backoff_seconds=float(config.get("retry_backoff_seconds", 1.0)),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        send_temperature=bool(config.get("send_temperature", True)),
        require_api_key=(
            bool(config["require_api_key"])
            if config.get("require_api_key") is not None
            else None
        ),
    )
    return OpenAICompatibleVLMJudge(
        model,
        max_images=int(config.get("max_images", 6)),
        max_context_chars=int(config.get("max_context_chars", 30000)),
        response_format_json=bool(config.get("response_format_json", True)),
    )


def _check_standalone_visual_evidence(
    request: dict[str, Any],
) -> str | None:
    """Return a compatibility summary when deterministic evidence is not ready."""

    from benchmark.visual_judge.evidence_gate import (
        DeterministicEvidenceGate,
    )
    from benchmark.visual_judge.interfaces import EvidenceGateRequest
    from benchmark.visual_judge.runtime import _judge_request

    core = _judge_request(request)
    gate = DeterministicEvidenceGate().check(
        EvidenceGateRequest(
            task=core.task,
            metric=core.metric,
            target_ids=("scene",),
            scene=deepcopy(core.scene_context),
            visual_evidence=tuple(deepcopy(core.visual_evidence)),
            evidence_goal={},
            context=deepcopy(request),
        )
    )
    if gate.ready:
        return None
    reasons = list(gate.reason_codes) or [
        "technical_visual_evidence_not_ready"
    ]
    return "visual evidence is not technically ready: " + ", ".join(
        reasons
    )


def _image_data_url(path: Path) -> str:
    return _normalized_rgb_png_data_url(path, label="judge_evidence")


def _selector_image_data_url(path: Path, *, alias: str) -> str:
    """Return a metadata-free RGB PNG for the external camera selector."""

    return _normalized_rgb_png_data_url(path, label=alias)


def _normalized_rgb_png_data_url(path: Path, *, label: str) -> str:
    """Decode and re-encode one outbound image without file metadata or alpha."""

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise RuntimeError(
            "Pillow is required to sanitize outbound VLM images"
        ) from exc

    try:
        with Image.open(path) as source:
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGBA")
            flattened = Image.new("RGB", normalized.size, (255, 255, 255))
            flattened.paste(normalized, mask=normalized.getchannel("A"))
            output = BytesIO()
            flattened.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"outbound image {label} is not a valid decodable image"
        ) from exc
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _selector_candidate_order_key(candidate: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _minimal_selector_pose(candidate.get("pose")),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path = Path(str(candidate.get("image_path"))).expanduser()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selector_metric_family(value: Any) -> str:
    metric = str(value or "").strip().lower()
    allowed = {
        "collision",
        "oob",
        "support",
        "object_architecture_penetration",
    }
    if metric not in allowed:
        raise ValueError(f"camera selector does not support metric family {metric!r}")
    return metric


def _selector_preview_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in {"highlighted_focus", "rgb_fallback"} else "unspecified"


def _selector_preview_warning_class(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    warning = str(value).lower()
    if "blank" in warning:
        return "blank_preview"
    if "visib" in warning or "highlight" in warning or "coverage" in warning:
        return "incomplete_target_visibility"
    return "preview_warning"


def _selector_allowed_actions(value: Any) -> list[str]:
    allowed = {
        "orbit_left",
        "orbit_right",
        "elevate",
        "lower",
        "dolly_in",
        "dolly_out",
    }
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item) in allowed))


def _sanitize_selector_deficiency(value: Any) -> dict[str, Any]:
    """Allow only non-identifying, routing-level evidence deficits outbound."""

    source = value if isinstance(value, dict) else {}
    reason_allowlist = {
        "measured_local_visibility_insufficient",
        "required_local_view_count_missing",
        "required_entities_not_jointly_visible",
        "focus_region_out_of_frame",
        "target_occluded_or_too_small",
        "focus_region_too_small",
        "architecture_plane_not_visible",
        "redundant_local_views",
    }
    deficiencies = source.get("deficiencies")
    structured_reasons = (
        [
            str(item.get("code"))
            for item in deficiencies
            if isinstance(item, dict)
            and item.get("repairability") == "camera"
            and str(item.get("code") or "") in reason_allowlist
        ]
        if isinstance(deficiencies, list)
        else []
    )
    reasons = source.get("reason_codes")
    sanitized_reasons = (
        [
            str(reason)
            for reason in reasons
            if str(reason) in reason_allowlist
        ]
        if isinstance(reasons, list)
        else []
    )
    sanitized_reasons = list(
        dict.fromkeys(structured_reasons + sanitized_reasons)
    )
    result: dict[str, Any] = {
        "status": "insufficient",
        "reason_codes": sanitized_reasons,
    }
    for key in (
        "required_local_view_count",
        "measured_local_view_count",
        "usable_local_view_count",
    ):
        raw = source.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw <= 8:
            result[key] = raw
    utility = source.get("evidence_utility")
    if (
        isinstance(utility, (int, float))
        and not isinstance(utility, bool)
        and 0.0 <= float(utility) <= 1.0
    ):
        result["evidence_utility"] = round(float(utility), 6)
    return result


def _allowed_canonical_camera_observations(metric: Any) -> list[str]:
    return _allowed_binary_camera_observations(metric)


def _canonical_evidence_request_target_ids(
    request: dict[str, Any],
) -> tuple[str, ...]:
    group_scope = request.get("group_scope")
    if isinstance(group_scope, dict):
        members = group_scope.get("member_ids")
        if isinstance(members, list):
            values = tuple(
                dict.fromkeys(
                    str(value)
                    for value in members
                    if isinstance(value, (str, int))
                    and str(value).strip()
                )
            )
            if values:
                return values

    values: list[str] = []
    for key in ("target_object_ids", "object_ids"):
        raw = request.get(key)
        if isinstance(raw, list):
            values.extend(
                str(value)
                for value in raw
                if isinstance(value, (str, int))
                and str(value).strip()
            )
    scene = request.get("scene_summary")
    if isinstance(scene, dict):
        for item in scene.get("objects") or []:
            if not isinstance(item, dict):
                continue
            object_id = item.get("id") or item.get("object_id")
            if object_id is not None and str(object_id).strip():
                values.append(str(object_id))
    return tuple(dict.fromkeys((*values, "scene")))


def _allowed_binary_camera_observations(
    metric: Any,
    *,
    relation_type: str | None = None,
) -> list[str]:
    """Expose only controller-validated DSL tokens to the internal Judge."""

    try:
        metric_name = canonical_camera_metric(
            metric,
            relation_type=relation_type,
        )
    except ValueError:
        # target_visible belongs to every metric registry entry and remains a
        # safe technical fallback when a legacy request omits its subtype.
        return ["target_visible"]
    allowed = METRIC_CAMERA_REQUIREMENTS[
        metric_name
    ].allowed_observations
    return sorted(set(allowed) & CAMERA_OBSERVATIONS)


def _relation_type_from_request(request: dict[str, Any]) -> str | None:
    for source in (
        request.get("relation"),
        request.get("event"),
        request.get("detector_evidence"),
        request,
    ):
        if isinstance(source, str) and source.strip():
            return source.strip()
        if not isinstance(source, dict):
            continue
        for key in ("relation_type", "event_type", "predicate", "type"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        nested = source.get("relation")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _minimal_selector_pose(value: Any) -> dict[str, Any]:
    pose = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    projection = str(pose.get("camera_type") or "PERSP").strip().upper()
    result["projection"] = "orthographic" if projection == "ORTHO" else "perspective"
    for source, target in (
        ("azimuth_degrees", "azimuth_degrees"),
        ("elevation_degrees", "elevation_degrees"),
        ("lens_mm", "lens_mm"),
        ("ortho_scale", "ortho_scale"),
    ):
        raw = pose.get(source)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
            result[target] = round(float(raw), 3)
    return result


def _sanitize_selector_legend(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role.startswith("related_target_"):
            role = "related_target"
        if role not in {
            "object_a",
            "object_b",
            "primary_target",
            "related_target",
            "architecture_plane",
            "measured_support_gap",
        }:
            role = "annotation"
        entry: dict[str, Any] = {"role": role}
        color = item.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            channels: list[float] = []
            for channel in color[:3]:
                if not isinstance(channel, (int, float)) or isinstance(channel, bool):
                    channels = []
                    break
                numeric = float(channel)
                if not math.isfinite(numeric):
                    channels = []
                    break
                channels.append(round(min(1.0, max(0.0, numeric)), 4))
            if channels:
                entry["rgb"] = channels
        representation = str(item.get("representation") or "").lower()
        if "mesh" in representation:
            entry["representation"] = "mesh"
        elif any(token in representation for token in ("obb", "bbox", "proxy")):
            entry["representation"] = "proxy"
        elif any(token in representation for token in ("plane", "boundary")):
            entry["representation"] = "architecture"
        elif representation:
            entry["representation"] = "annotation"
        result.append(entry)
    return result


def _budgeted_context_json(
    context: dict[str, Any],
    max_chars: int,
    *,
    priority_keys: tuple[str, ...] = (),
) -> str:
    """Serialize a valid JSON context while preserving high-value evidence.

    Raw prefix slicing can both corrupt JSON and drop a detector packet merely
    because a large scene summary appeared first.  Required/priority fields are
    therefore considered first, optional fields are added only while they fit,
    and an oversized priority value is represented by an explicit JSON-prefix
    summary instead of truncating the entire document.
    """

    limit = max(1, int(max_chars))
    full = _compact_json(context)
    if len(full) <= limit:
        return full

    marker = {"truncated": True}
    marker_key = "_benchmark_context_budget"
    priority = [key for key in priority_keys if key in context]
    priority_set = set(priority)
    # Seed every priority key before allocating the remaining budget. This
    # prevents one oversized earlier field (for example a prompt or raw
    # relation) from consuming the budget and deleting detector evidence.
    skeleton = {**{key: None for key in priority}, marker_key: marker}
    skeleton_size = len(_compact_json(skeleton))
    per_key_extra = max(0, limit - skeleton_size) // max(len(priority), 1)
    seed_limit = len("null") + per_key_extra
    result: dict[str, Any] = {
        key: _priority_seed_value(context[key], seed_limit)
        for key in priority
    }
    if len(_compact_json({**result, marker_key: marker})) > limit:
        minimal = _compact_json({marker_key: marker})
        return minimal if len(minimal) <= limit else "{}"

    # Enrich priority fields in declared order while all other priority keys
    # remain reserved in the document.
    for key in priority:
        value = context[key]
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value
            continue
        empty_trial = {**result, key: None, marker_key: marker}
        allowance = limit - len(_compact_json(empty_trial)) + len("null")
        compacted = _truncated_json_value(value, allowance)
        if compacted is not None:
            compacted_trial = {**result, key: compacted, marker_key: marker}
            if len(_compact_json(compacted_trial)) <= limit:
                result[key] = compacted

    # Optional fields never displace reserved evidence.
    for key, value in context.items():
        if key in priority_set:
            continue
        trial = {**result, key: value, marker_key: marker}
        if len(_compact_json(trial)) <= limit:
            result[key] = value

    rendered = _compact_json({**result, marker_key: marker})
    # ``max_context_chars`` is clamped to at least 1000 by the judge, but keep
    # this helper total for direct unit use with tiny budgets.
    if len(rendered) <= limit:
        return rendered
    minimal = _compact_json({marker_key: marker})
    return minimal if len(minimal) <= limit else "{}"


def _truncated_json_value(value: Any, max_chars: int) -> dict[str, Any] | None:
    if max_chars <= 0:
        return None
    raw = _compact_json(value)
    base = {
        "_truncated": True,
        "original_chars": len(raw),
        "json_prefix": "",
    }
    if len(_compact_json(base)) > max_chars:
        return None
    low, high = 0, len(raw)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = {**base, "json_prefix": raw[:middle]}
        if len(_compact_json(candidate)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return {**base, "json_prefix": raw[:low]}


def _priority_seed_value(value: Any, max_chars: int) -> Any:
    raw = _compact_json(value)
    if len(raw) <= max_chars:
        return value
    compacted = _truncated_json_value(value, max_chars)
    if compacted is not None:
        return compacted
    marker = {"_truncated": True}
    return marker if len(_compact_json(marker)) <= max_chars else None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
